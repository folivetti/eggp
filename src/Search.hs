{-# LANGUAGE  BlockArguments #-}
{-# LANGUAGE  TupleSections #-}
{-# LANGUAGE  MultiWayIf #-}
{-# LANGUAGE  OverloadedStrings #-}
{-# LANGUAGE  BangPatterns #-}
{-# LANGUAGE  TypeSynonymInstances, FlexibleInstances #-}

module Search where

import Algorithm.EqSat.Egraph
import Algorithm.EqSat.Simplify hiding ( myCost )
import Algorithm.EqSat.Build
import Algorithm.EqSat.Queries
import Algorithm.EqSat.Info
import Algorithm.EqSat.DB
import Algorithm.SRTree.Likelihoods
import Algorithm.SRTree.ModelSelection
import Control.Lens (element, makeLenses, over, (&), (+~), (-~), (.~), (^.))
import Control.Monad (foldM, forM_, forM, when, unless, filterM, (>=>), (<=<), replicateM, replicateM_)
import Control.Monad.State.Strict

import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap
import Data.Maybe (fromJust, fromMaybe, isNothing, isJust)
import qualified Data.Vector.Unboxed as V
import Data.SRTree
import Data.SRTree.Datasets
import Data.SRTree.Eval
import Data.SRTree.Random (randomTree)
import Data.SRTree.Print
import System.Random
import qualified Data.HashSet as Set
import Data.List ( sort, maximumBy, intercalate, sortOn, intersperse, nub, zip4 )
import Data.IntSet (IntSet)
import qualified Data.IntSet as IntSet
import qualified Data.Sequence as FingerTree
import Data.Function ( on )
import qualified Data.Foldable as Foldable

import List.Shuffle ( shuffle )
import Algorithm.SRTree.NonlinearOpt
import Data.Binary ( encode, decode )
import qualified Data.ByteString.Lazy as BS
import Data.List.Split (splitOn)

import Algorithm.EqSat (runEqSat,applySingleMergeOnlyEqSat)

import GHC.IO (unsafePerformIO)
import Control.Scheduler 
import Control.Monad.IO.Unlift
import Data.SRTree (convertProtectedOps)

import Data.SRTree.Random
import Data.SRTree.Datasets
import Text.ParseSR
import Algorithm.EqSat.SearchSR
import Algorithm.SRTree.AD (ADBackEnd(..))

import Foreign.C (CInt (..), CDouble (..))
import Foreign.C.String (CString, newCString, withCString, peekCString, peekCAString, newCAString)
import Paths_eggp (version)
import System.Environment (getArgs)
import System.Exit (ExitCode (..))
import Text.Read (readMaybe)
import Data.Version (showVersion)
import Control.Exception (Exception (..), SomeException (..), handle)
import Data.Time.Clock.POSIX

data Args = Args
  { _dataset      :: String,
    _testData     :: String,
    _gens         :: Int,
    _maxSize      :: Int,
    _folds        :: Int,
    _trace        :: Bool,
    _distribution :: Distribution,
    _optIter      :: Int,
    _optRepeat    :: Int,
    _nParams      :: Int,
    _nPop         :: Int,
    _nTournament  :: Int,
    _pc           :: Double,
    _pm           :: Double,
    _nonterminals :: String,
    _dumpTo       :: String,
    _loadFrom     :: String,
    _generational :: Bool,
    _simplify     :: Bool,
    _maxtime      :: Int,
    _varnames     :: String,
    _useFracBayes :: Bool,
    _backend      :: ADBackEnd
  }
  deriving (Show)

csvHeader :: String
csvHeader = "id,view,Expression,Numpy,Math,theta,size,loss_train,loss_val,loss_test,maxloss,R2_train,R2_val,R2_test,dl_train,dl_val,dl_test"

egraphGP :: [(DataSet, DataSet)] -> [DataSet] -> Args -> StateT EGraph (StateT StdGen IO) String
egraphGP dataTrainVals dataTests args = do
  when ((not.null) (_loadFrom args)) $ (io $ BS.readFile (_loadFrom args)) >>= \eg -> put (decode eg)

  insertTerms
  unevalInit <- gets (IntSet.toList . _unevaluated . _eDB)
  fitBatch True fitFun unevalInit

  t0 <- io $ getPOSIXTime
  
  pop <- replicateM (_nPop args) $ insertRndExpr (_maxSize args) rndTerm rndNonTerm >>= canonical
  fitBatch False fitFun pop

  output <- if _trace args 
               then forM (Prelude.zip [0..] pop) $ uncurry printExpr
               else pure []

  let m = (_nPop args) `div` (_maxSize args)
      mTime = if _maxtime args < 0 then Nothing else Just (fromIntegral $ _maxtime args - 5) -- add 5 seconds slack

  (finalPop, finalOut, _) <- iterateFor (_gens args) t0 mTime (pop, output, _nPop args) $ \it (ps', out, curIx) -> do
    newPop' <- replicateM (_nPop args) (evolve ps')

    -- Batch-fit the eqsat-flagged refits (force) and the new offspring
    -- (updateIfNothing semantics) concurrently, before any fitness-based
    -- selection so Pareto ranking sees fresh fitness values.
    refitIds <- gets (IntSet.toList . _refits . _eDB)
    modify' $ over (eDB . refits) (const IntSet.empty)
    fitBatch True fitFun refitIds
    fitBatch False fitFun newPop'

    out' <- if _trace args
              then forM (Prelude.zip [curIx..] newPop') $ uncurry printExpr
              else pure []

    totSz <- gets (HashMap.size . _eNodeToEClass) -- (IntMap.size . _eClass)
    let full = totSz > max maxMem (_nPop args)
    cleanedIds <- if full then Just <$> cleanEGraph else pure Nothing
    when full cleanDB

    newPop <- if _generational args 
                 then maybe (Prelude.mapM canonical newPop') pure cleanedIds 
                 else do 
                     let n_paretos = (_nPop args) `div` (_maxSize args)
                     pareto <- if (_useFracBayes args)
                                 then getParetoFront
                                 else concat <$> (forM [1 .. _maxSize args] $ \n -> getTopFitEClassWithSize n 2)
                     -- pareto <- concat <$> (forM [1 .. _maxSize args] $ \n -> getTopFitEClassWithSize n 2)
                     -- pareto <- getParetoFront
                     let remainder = _nPop args - length pareto
                     lft <- if full
                               then getTopFitEClassThat remainder (const True)
                               else pure $ Prelude.take remainder newPop'
                     Prelude.mapM canonical (pareto <> lft)
    pure (newPop, out <> out', curIx + (_nPop args)) 

  when ((not.null) (_dumpTo args)) $ get >>= (io . BS.writeFile (_dumpTo args) . encode )
  pf <- if _trace args 
           then pure finalOut 
           else paretoFront fitFun (_maxSize args) printExpr
  pure $ unlines (csvHeader : concat pf) 
  where
    maxSize = (_maxSize args)
    maxMem = 2000000 -- running 1 iter of eqsat for each new individual will consume ~3GB
    fitFun = fitnessMV (_backend args) skipValEval shouldReparam (_optRepeat args) (_optIter args) (_distribution args) dataTrainVals
    nonTerms   = parseNonTerms (_nonterminals args)
    nFeats = length $ getX (fst $ head dataTrainVals)
    params         = if _nParams args == -1 then [param 0] else Prelude.map param [0 .. _nParams args - 1]
    shouldReparam  = _nParams args == -1
    skipValEval    = _folds args == 1
    relabel        = if shouldReparam then relabelParams else relabelParamsOrder
    terms          = if _distribution args == ROXY
                          then (var 0 : params)
                          else [var ix | ix <- [0 .. nFeats-1]] -- <> params
    uniNonTerms = [t | t <- nonTerms, isUni t]
    binNonTerms = [t | t <- nonTerms, isBin t]
    isUni (Uni _ _)   = True
    isUni _           = False
    isBin (Bin _ _ _) = True
    isBin _           = False

    sortedDLs = pickEvenly (_nPop args) $ sort [fromIntegral x * log (fromIntegral y) | x <- [2 .. _maxSize args], y <- [2 .. x]]

    pickEvenly :: Int -> [a] -> [a]
    pickEvenly m xs
        | m <= 0    = []
        | m == 1    = [head xs]
        | m >= n    = xs
        | otherwise = [xs !! (floor (fromIntegral k * step)) | k <- [0 .. m - 1]]
      where
        n = length xs
        step = fromIntegral (n - 1) / fromIntegral (m - 1)

    getParetoFront = do
      let between x (a, b) = x >= a && x <= b
          nParetos = 2
      pareto <- (concat <$> (forM (Prelude.zip sortedDLs (Prelude.tail sortedDLs)) $
        \(lo, hi) -> getTopFitEClassThat nParetos (\ec -> ((_dl . _info) ec) `between` (Just lo, Just hi))))
          >>= Prelude.mapM canonical
      pure pareto

    -- TODO: merge two or more egraphs
    cleanEGraph = do let nParetos = 10 -- (maxMem `div` 5) `div` _maxSize args
                     io . putStrLn $ "cleaning"
                     pareto <- (concat <$> (forM [1 .. _maxSize args] $ \n -> getTopFitEClassWithSize n nParetos))
                                 >>= Prelude.mapM canonical
                     infos  <- forM pareto (\c -> _info <$> getEClass c)
                     exprs  <- forM pareto getBestExpr
                     put emptyGraph
                     newIds <- fromTrees myCost $ Prelude.map relabel exprs
                     forM_ (Prelude.zip newIds (Prelude.reverse infos)) $ \(eId, info) -> do
                           let f = fromMaybe (-1.0/0.0) (_fitness info)
                           insertFitness eId f (_theta info)
                     pure newIds

    rndTerm    = do coin <- toss
                    if coin || _nParams args == 0 then randomFrom terms else randomFrom params
    rndNonTerm = randomFrom nonTerms

    iterateFor 0  _    _ xs f = pure xs
    iterateFor n t0 maxT xs f = do xs' <- f n xs
                                   t1 <- io $ getPOSIXTime
                                   let delta = t1 - t0
                                       maxT' = (subtract delta) <$> maxT
                                   case maxT' of
                                      Nothing -> iterateFor (n-1) t1 maxT' xs' f
                                      Just mt -> if mt <= 0
                                                    then pure xs
                                                    else iterateFor (n-1) t1 maxT' xs' f

    evolve xs' = do xs <- Prelude.mapM canonical xs'
                    parents <- tournament xs
                    offspring <- combine parents
                    if _nParams args == 0
                       then runEqSat myCost rewritesWithConstant 1 >> cleanDB
                       else runEqSat myCost rewritesParams 1 >> cleanDB
                    pure offspring

    tournament xs = do p1 <- applyTournament xs >>= canonical
                       p2 <- applyTournament xs >>= canonical
                       pure (p1, p2)

    applyTournament :: [EClassId] -> RndEGraph EClassId
    applyTournament xs = do challengers <- replicateM (_nTournament args) (rnd $ randomFrom xs) >>= traverse canonical
                            fits <- Prelude.map (fromMaybe (-1.0/0.0)) <$> Prelude.mapM getFitness challengers
                            pure . snd . maximumBy (compare `on` fst) $ Prelude.zip fits challengers

    combine (p1, p2) = (crossover p1 p2 >>= mutate) >>= canonical

    crossover p1 p2 = do sz <- getSize p1
                         coin <- rnd $ tossBiased (_pc args)
                         if sz == 1 || not coin
                            then rnd (randomFrom [p1, p2])
                            else do pos <- rnd $ randomRange (1, sz-1)
                                    cands <- getAllSubClasses p2
                                    tree <- getSubtree pos 0 Nothing [] cands p1
                                    fromTree myCost (relabel tree) >>= canonical

    getSubtree :: Int -> Int -> Maybe (EClassId -> ENode) -> [Maybe (EClassId -> ENode)] -> [EClassId] -> EClassId -> RndEGraph (Fix SRTree)
    getSubtree 0 sz (Just parent) mGrandParents cands p' = do
      p <- canonical p'
      candidates' <- filterM (\c -> (<maxSize-sz) <$> getSize c) cands
      candidates  <- filterM (\c -> doesNotExistGens mGrandParents (parent c)) candidates'
                       >>= traverse canonical
      if null candidates
         then getBestExpr p
         else do subtree <- rnd (randomFrom candidates)
                 getBestExpr subtree
    getSubtree pos sz parent mGrandParents cands p' = do
      p <- canonical p'
      root <- getBestENode p >>= canonize
      case root of
        EParam ix -> pure . Fix $ Param ix
        EConst x  -> pure . Fix $ Const x
        EVar   ix -> pure . Fix $ Var ix
        EUni f t' -> do t <- canonical t'
                        (Fix . Uni f) <$> getSubtree (pos-1) (sz+1) (Just (\eid -> EUni f eid)) (parent:mGrandParents) cands t
        EBin op l'' r'' ->
                      do l <- canonical l''
                         r <- canonical r''
                         szLft <- getSize l
                         szRgt <- getSize r
                         if szLft < pos
                           then do l' <- getBestExpr l
                                   r' <- getSubtree (pos-szLft-1) (sz+szLft+1) (Just (\eid -> EBin op l eid)) (parent:mGrandParents) cands r
                                   pure . Fix $ Bin op l' r'
                           else do l' <- getSubtree (pos-1) (sz+szRgt+1) (Just (\eid -> EBin op eid r)) (parent:mGrandParents) cands l
                                   r' <- getBestExpr r
                                   pure . Fix $ Bin op l' r'
        ENAry op xs -> do
          cs  <- mapM canonical xs
          szs <- mapM getSize cs
          let totalSz = sum szs
              goE [] [] _ _ = pure []
              goE (c:cs) (s:szs) acc i
                | s < pos - acc = do
                    c' <- getBestExpr c
                    cs' <- goE cs szs (acc + s) (i + 1)
                    pure (c' : cs')
                | otherwise = do
                    c' <- getSubtree (pos - acc - 1) (sz + 1 + (totalSz - s)) (Just (\eid -> ENAry op (replaceAt i eid xs))) (parent:mGrandParents) cands c
                    cs' <- mapM getBestExpr cs
                    pure (c' : cs')
          exprs <- goE cs szs 0 0
          pure $ naryTree op exprs

    getAllSubClasses p' = do
      p  <- canonical p'
      en <- getBestENode p
      case en of
        EBin _ l r -> do ls <- getAllSubClasses l
                         rs <- getAllSubClasses r
                         pure (p : (ls <> rs))
        EUni _ t   -> (p:) <$> getAllSubClasses t
        ENAry _ xs -> do xss <- mapM getAllSubClasses xs
                         pure (p : concat xss)
        _          -> pure [p]

    mutate p = do sz <- getSize p
                  coin <- rnd $ tossBiased (_pm args)
                  if coin
                     then do pos <- rnd $ randomRange (0, min sz maxSize - 1)
                             tree <- mutAt pos maxSize Nothing p
                             fromTree myCost (relabel tree) >>= canonical
                     else pure p

    peel :: Fix SRTree -> SRTree ()
    peel (Fix (Bin op l r)) = Bin op () ()
    peel (Fix (Uni f t)) = Uni f ()
    peel (Fix (Param ix)) = Param ix
    peel (Fix (Var ix)) = Var ix
    peel (Fix (Const x)) = Const x

    mutAt :: Int -> Int -> Maybe (EClassId -> ENode) -> EClassId -> RndEGraph (Fix SRTree)
    mutAt 0 sizeLeft Nothing       _ = (insertRndExpr (max 1 sizeLeft) rndTerm rndNonTerm >>= canonical) >>= getBestExpr -- we chose to mutate the root
    mutAt 0 1        _             _ = rnd $ randomFrom terms -- we don't have size left
    mutAt 0 sizeLeft (Just parent) _ = do -- we reached the mutation place
      ec    <- insertRndExpr (max 1 sizeLeft) rndTerm rndNonTerm >>= canonical -- create a random expression with the size limit
      (Fix tree) <- getBestExpr ec           --
      root  <- getBestENode ec
      exist <- canonize (parent ec) >>= doesExist
      if exist
         -- the expression `parent ec` already exists, try to fix
         then do let children = eChildren root
                 candidates <- case length children of
                                0  -> filterM (checkToken parent <=< (toENode . replaceChildren children)) (Prelude.map peel terms)
                                1 -> filterM (checkToken parent <=< (toENode . replaceChildren children)) uniNonTerms
                                2 -> filterM (checkToken parent <=< (toENode . replaceChildren children)) binNonTerms
                                _ -> pure []
                 if null candidates
                     then pure $ Fix tree -- there's no candidate, so we failed and admit defeat
                     else do newToken <- rnd (randomFrom candidates)
                             pure . Fix $ replaceChildren (childrenOf tree) newToken

         else pure . Fix $ tree

    mutAt pos sizeLeft parent p' = do
        p <- canonical p'
        root <- getBestENode p >>= canonize
        case root of
          EParam ix -> pure . Fix $ Param ix
          EConst x  -> pure . Fix $ Const x
          EVar   ix -> pure . Fix $ Var ix
          EUni f t'  -> canonical t' >>= \t -> (Fix . Uni f) <$> mutAt (pos-1) (sizeLeft-1) (Just (\eid -> EUni f eid)) t
          EBin op ln rn -> do l <- canonical ln
                              r <- canonical rn
                              szLft <- getSize l
                              szRgt <- getSize r
                              if szLft < pos
                                 then do l' <- getBestExpr l
                                         r' <- mutAt (pos-szLft-1) (sizeLeft-szLft-1) (Just (\eid -> EBin op l eid)) r
                                         pure . Fix $ Bin op l' r'
                                 else do l' <- mutAt (pos-1) (sizeLeft-szRgt-1) (Just (\eid -> EBin op eid r)) l
                                         r' <- getBestExpr r
                                         pure . Fix $ Bin op l' r'
          ENAry op xs -> do
            cs  <- mapM canonical xs
            szs <- mapM getSize cs
            let totalSz = sum szs
                goE [] [] _ _ = pure []
                goE (c:cs) (s:szs) acc i
                  | s < pos - acc = do
                      c' <- getBestExpr c
                      cs' <- goE cs szs (acc + s) (i + 1)
                      pure (c' : cs')
                  | otherwise = do
                      c' <- mutAt (pos - acc - 1) (sizeLeft - 1 - (totalSz - s)) (Just (\eid -> ENAry op (replaceAt i eid xs))) c
                      cs' <- mapM getBestExpr cs
                      pure (c' : cs')
            exprs <- goE cs szs 0 0
            pure $ naryTree op exprs


    printExpr :: Int -> EClassId -> RndEGraph [String]
    printExpr ix ec = do
        thetas' <- getTheta ec
        bestExpr <- (if _simplify args then simplifyEqSatDefault else id) <$> getBestExpr ec

        let best'   = if shouldReparam then relabelParams bestExpr else relabelParamsOrder bestExpr
            nParams = countParamsUniq best'
            nThetas = Prelude.map V.length thetas'
        (_, thetas) <- if Prelude.any (/=nParams) nThetas
                        then fitFun best'
                        else pure (1.0, thetas')

        maxLoss <- maybe 0 negate <$> getFitness ec
        ts <- forM (Data.List.zip4 [0..] dataTrainVals dataTests thetas) $ \(view, (dataTrain, dataVal), dataTest, theta) -> do
            let (x, y, mYErr) = dataTrain
                (x_val, y_val, mYErr_val) = dataVal
                (x_te, y_te, mYErr_te) = dataTest
                distribution = _distribution args

                expr      = paramsToConst (V.toList theta) best'
                showNA z  = if isNaN z then "" else show z

                n          = fromIntegral (V.length y) :: Double
                n_val      = fromIntegral (V.length y_val) :: Double
                n_te       = fromIntegral (V.length y_te) :: Double
                p          = fromIntegral (countParamsUniq best') :: Double
                f_compl    = countNodes best' * log (countUniqueTokens best')

                -- loss (NLL)
                nll_train_ = compileLoss x (buildLoss (NLL distribution) n best') y mYErr theta
                nll_val_   = compileLoss x_val (buildLoss (NLL distribution) n_val best') y_val mYErr_val theta
                nll_te_    = compileLoss x_te (buildLoss (NLL distribution) n_te best') y_te mYErr_te theta

                -- R2
                y_mean    = V.sum y / n
                y_var     = V.sum (V.map (\yi -> (yi - y_mean)^2) y) / n
                y_val_mean = V.sum y_val / n_val
                y_val_var  = V.sum (V.map (\yi -> (yi - y_val_mean)^2) y_val) / n_val
                y_te_mean  = V.sum y_te / n_te
                y_te_var   = V.sum (V.map (\yi -> (yi - y_te_mean)^2) y_te) / n_te

                mse_train = compileLoss x (buildLoss MSE n best') y Nothing theta
                r2_train_ = if y_var > 0 then 1 - mse_train / y_var else 0
                r2_val_   = if y_val_var > 0 then 1 - (compileLoss x_val (buildLoss MSE n_val best') y_val Nothing theta) / y_val_var else 0
                r2_te_    = if y_te_var > 0 then 1 - (compileLoss x_te (buildLoss MSE n_te best') y_te Nothing theta) / y_te_var else 0

                -- fractional Bayes factor
                b_train   = 1 / sqrt n
                nup       = exp (1 - log 3)
                mdl_train_ = (1 - b_train) * nll_train_ - p / 2 * log b_train + f_compl + p / 2 * log (2*pi*nup)
                b_val     = 1 / sqrt n_val
                mdl_val_   = (1 - b_val) * nll_val_ - p / 2 * log b_val + f_compl + p / 2 * log (2*pi*nup)
                b_te      = 1 / sqrt n_te
                mdl_te_    = (1 - b_te) * nll_te_ - p / 2 * log b_te + f_compl + p / 2 * log (2*pi*nup)

                vals       = intercalate ","
                           $ Prelude.map showNA [ nll_train_, nll_val_, nll_te_, maxLoss
                                                 , r2_train_, r2_val_, r2_te_
                                                 , mdl_train_, mdl_val_, mdl_te_]
                thetaStr    = intercalate ";" $ Prelude.map show (V.toList theta)
                varnames    = _varnames args
                showExprFun = if null varnames then showExpr else showExprWithVars (splitOn "," varnames)
                showLatexFun = if null varnames then showLatex else showLatexWithVars (splitOn "," varnames)
            pure $ show ix <> "," <> show view <> "," <> showExprFun expr <> "," <> "\"" <> showPython best' <> "\","
                           <> "\"$$" <> showLatexFun best' <> "$$\","
                           <> thetaStr <> "," <> show (countNodes $ convertProtectedOps expr)
                           <> "," <> vals
        pure ts

    insertTerms =
        forM terms $ \t -> do fromTree myCost t >>= canonical

    replaceAt :: Int -> EClassId -> [EClassId] -> [EClassId]
    replaceAt 0 e (_:cs) = e : cs
    replaceAt i e (c:cs) = c : replaceAt (i-1) e cs
    replaceAt _ _ []     = []
