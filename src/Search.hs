{-# LANGUAGE  BlockArguments #-}
{-# LANGUAGE  TupleSections #-}
{-# LANGUAGE  MultiWayIf #-}
{-# LANGUAGE  OverloadedStrings #-}
{-# LANGUAGE  BangPatterns #-}
{-# LANGUAGE  TypeSynonymInstances, FlexibleInstances #-}

module Search where

import Algorithm.EqSat.Egraph
import Algorithm.EqSat.Simplify
import Algorithm.EqSat.Build
import Algorithm.EqSat.Queries
import Algorithm.EqSat.Info
import Algorithm.EqSat.DB
import Algorithm.SRTree.Likelihoods
import Algorithm.SRTree.ModelSelection
import Algorithm.SRTree.Opt
import Control.Lens (element, makeLenses, over, (&), (+~), (-~), (.~), (^.))
import Control.Monad (foldM, forM_, forM, when, unless, filterM, (>=>), replicateM, replicateM_)
import Control.Monad.State.Strict
import qualified Data.IntMap.Strict as IM
import qualified Data.Map.Strict as Map
import Data.Massiv.Array as MA hiding (forM_, forM)
import Data.Maybe (fromJust, isNothing, isJust)
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
import qualified Data.IntMap as IntMap
import List.Shuffle ( shuffle )
import Algorithm.SRTree.NonlinearOpt
import Data.Binary ( encode, decode )
import qualified Data.ByteString.Lazy as BS

import Algorithm.EqSat (runEqSat,applySingleMergeOnlyEqSat)

import GHC.IO (unsafePerformIO)
import Control.Scheduler 
import Control.Monad.IO.Unlift
import Data.SRTree (convertProtectedOps)

import Data.SRTree.Random
import Data.SRTree.Datasets
import Text.ParseSR
import Algorithm.EqSat.SearchSR

import Foreign.C (CInt (..), CDouble (..))
import Foreign.C.String (CString, newCString, withCString, peekCString, peekCAString, newCAString)
import Paths_eggp (version)
import System.Environment (getArgs)
import System.Exit (ExitCode (..))
import Text.Read (readMaybe)
import Data.Version (showVersion)
import Control.Exception (Exception (..), SomeException (..), handle)

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
    _simplify     :: Bool
  }
  deriving (Show)

csvHeader :: String
csvHeader = "id,view,Expression,Numpy,theta,size,loss_train,loss_val,loss_test,maxloss,R2_train,R2_val,R2_test,mdl_train,mdl_val,mdl_test"

egraphGP :: [(DataSet, DataSet)] -> [DataSet] -> Args -> StateT EGraph (StateT StdGen IO) String
egraphGP dataTrainVals dataTests args = do
  when ((not.null) (_loadFrom args)) $ (io $ BS.readFile (_loadFrom args)) >>= \eg -> put (decode eg)

  insertTerms
  evaluateUnevaluated fitFun
  
  pop <- replicateM (_nPop args) $ do ec <- insertRndExpr (_maxSize args) rndTerm rndNonTerm >>= canonical
                                      updateIfNothing fitFun ec
                                      pure ec
  pop' <- Prelude.mapM canonical pop

  output <- if _trace args 
               then forM (Prelude.zip [0..] pop') $ uncurry printExpr
               else pure []

  let m = (_nPop args) `div` (_maxSize args)

  (finalPop, finalOut, _) <- iterateFor (_gens args) (pop', output, _nPop args) $ \it (ps', out, curIx) -> do
    newPop' <- replicateM (_nPop args) (evolve ps')

    out' <- if _trace args
              then forM (Prelude.zip [curIx..] newPop') $ uncurry printExpr
              else pure []

    totSz <- gets (Map.size . _eNodeToEClass) -- (IntMap.size . _eClass)
    let full = totSz > max maxMem (_nPop args)
    when full (cleanEGraph >> cleanDB)

    newPop <- if _generational args 
                 then Prelude.mapM canonical newPop' 
                 else do 
                     let n_paretos = (_nPop args) `div` (_maxSize args)
                     pareto <- concat <$> (forM [1 .. _maxSize args] $ \n -> getTopFitEClassWithSize n 2)
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
    fitFun = fitnessMV shouldReparam (_optRepeat args) (_optIter args) (_distribution args) dataTrainVals
    nonTerms   = parseNonTerms (_nonterminals args)
    (Sz2 _ nFeats) = MA.size (getX .fst . head $ dataTrainVals)
    params         = if _nParams args == -1 then [param 0] else Prelude.map param [0 .. _nParams args - 1]
    shouldReparam  = _nParams args == -1
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

    -- TODO: merge two or more egraphs
    cleanEGraph = do let nParetos = 10 -- (maxMem `div` 5) `div` _maxSize args
                     io . putStrLn $ "cleaning"
                     pareto <- (concat <$> (forM [1 .. _maxSize args] $ \n -> getTopFitEClassWithSize n nParetos))
                                 >>= Prelude.mapM canonical
                     infos  <- forM pareto (\c -> gets (_info . (IntMap.! c) . _eClass))
                     exprs  <- forM pareto getBestExpr
                     put emptyGraph
                     newIds <- fromTrees myCost $ Prelude.map relabel exprs
                     forM_ (Prelude.zip newIds (Prelude.reverse infos)) $ \(eId, info) ->
                         insertFitness eId (fromJust $ _fitness info) (_theta info)

    rndTerm    = do coin <- toss
                    if coin then randomFrom terms else randomFrom params
    rndNonTerm = randomFrom nonTerms

    refitChanged = do ids <- gets (_refits . _eDB) >>= Prelude.mapM canonical . Set.toList >>= pure . nub
                      modify' $ over (eDB . refits) (const Set.empty)
                      forM_ ids $ \ec -> do t <- getBestExpr ec
                                            (f, p) <- fitFun t
                                            insertFitness ec f p

    iterateFor 0 xs f = pure xs
    iterateFor n xs f = do xs' <- f n xs
                           iterateFor (n-1) xs' f

    evolve xs' = do xs <- Prelude.mapM canonical xs'
                    parents <- tournament xs
                    offspring <- combine parents
                    --applySingleMergeOnlyEqSat myCost rewritesParams >> cleanDB
                    if _nParams args == 0
                       then runEqSat myCost rewritesWithConstant 1 >> cleanDB >> refitChanged
                       else runEqSat myCost rewritesParams 1 >> cleanDB >> refitChanged
                    canonical offspring >>= updateIfNothing fitFun
                    canonical offspring
                    --pure offspring

    tournament xs = do p1 <- applyTournament xs >>= canonical
                       p2 <- applyTournament xs >>= canonical
                       pure (p1, p2)

    applyTournament :: [EClassId] -> RndEGraph EClassId
    applyTournament xs = do challengers <- replicateM (_nTournament args) (rnd $ randomFrom xs) >>= traverse canonical
                            fits <- Prelude.map fromJust <$> Prelude.mapM getFitness challengers
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
        Param ix -> pure . Fix $ Param ix
        Const x  -> pure . Fix $ Const x
        Var   ix -> pure . Fix $ Var ix
        Uni f t' -> do t <- canonical t'
                       (Fix . Uni f) <$> getSubtree (pos-1) (sz+1) (Just $ Uni f) (parent:mGrandParents) cands t
        Bin op l'' r'' ->
                      do l <- canonical l''
                         r <- canonical r''
                         szLft <- getSize l
                         szRgt <- getSize r
                         if szLft < pos
                           then do l' <- getBestExpr l
                                   r' <- getSubtree (pos-szLft-1) (sz+szLft+1) (Just $ Bin op l) (parent:mGrandParents) cands r
                                   pure . Fix $ Bin op l' r'
                           else do l' <- getSubtree (pos-1) (sz+szRgt+1) (Just (\t -> Bin op t r)) (parent:mGrandParents) cands l
                                   r' <- getBestExpr r
                                   pure . Fix $ Bin op l' r'

    getAllSubClasses p' = do
      p  <- canonical p'
      en <- getBestENode p
      case en of
        Bin _ l r -> do ls <- getAllSubClasses l
                        rs <- getAllSubClasses r
                        pure (p : (ls <> rs))
        Uni _ t   -> (p:) <$> getAllSubClasses t
        _         -> pure [p]

    mutate p = do sz <- getSize p
                  coin <- rnd $ tossBiased (_pm args)
                  if coin
                     then do pos <- rnd $ randomRange (0, sz-1)
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
    mutAt 0 sizeLeft Nothing       _ = (insertRndExpr sizeLeft rndTerm rndNonTerm >>= canonical) >>= getBestExpr -- we chose to mutate the root
    mutAt 0 1        _             _ = rnd $ randomFrom terms -- we don't have size left
    mutAt 0 sizeLeft (Just parent) _ = do -- we reached the mutation place
      ec    <- insertRndExpr sizeLeft rndTerm rndNonTerm >>= canonical -- create a random expression with the size limit
      (Fix tree) <- getBestExpr ec           --
      root  <- getBestENode ec
      exist <- canonize (parent ec) >>= doesExist
      if exist
         -- the expression `parent ec` already exists, try to fix
         then do let children = childrenOf root
                 candidates <- case length children of
                                0  -> filterM (checkToken parent . (replaceChildren children)) (Prelude.map peel terms)
                                1 -> filterM (checkToken parent . (replaceChildren children)) uniNonTerms
                                2 -> filterM (checkToken parent . (replaceChildren children)) binNonTerms
                 if null candidates
                     then pure $ Fix tree -- there's no candidate, so we failed and admit defeat
                     else do newToken <- rnd (randomFrom candidates)
                             pure . Fix $ replaceChildren (childrenOf tree) newToken

         else pure . Fix $ tree

    mutAt pos sizeLeft parent p' = do
        p <- canonical p'
        root <- getBestENode p >>= canonize
        case root of
          Param ix -> pure . Fix $ Param ix
          Const x  -> pure . Fix $ Const x
          Var   ix -> pure . Fix $ Var ix
          Uni f t'  -> canonical t' >>= \t -> (Fix . Uni f) <$> mutAt (pos-1) (sizeLeft-1) (Just $ Uni f) t
          Bin op ln rn -> do l <- canonical ln
                             r <- canonical rn
                             szLft <- getSize l
                             szRgt <- getSize r
                             if szLft < pos
                                then do l' <- getBestExpr l
                                        r' <- mutAt (pos-szLft-1) (sizeLeft-szLft-1) (Just $ Bin op l) r
                                        pure . Fix $ Bin op l' r'
                                else do l' <- mutAt (pos-1) (sizeLeft-szRgt-1) (Just (\t -> Bin op t r)) l
                                        r' <- getBestExpr r
                                        pure . Fix $ Bin op l' r'


    printExpr :: Int -> EClassId -> RndEGraph [String]
    printExpr ix ec = do
        thetas' <- gets (_theta . _info . (IM.! ec) . _eClass)
        bestExpr <- (if _simplify args then simplifyEqSatDefault else id) <$> getBestExpr ec

        let best'   = if shouldReparam then relabelParams bestExpr else relabelParamsOrder bestExpr
            nParams = countParamsUniq best'
            fromSz (MA.Sz x) = x
            nThetas = Prelude.map (fromSz . MA.size) thetas'
        (_, thetas) <- if Prelude.any (/=nParams) nThetas
                        then fitFun best'
                        else pure (1.0, thetas')

        maxLoss <- negate . fromJust <$> getFitness ec
        ts <- forM (Data.List.zip4 [0..] dataTrainVals dataTests thetas) $ \(view, (dataTrain, dataVal), dataTest, theta) -> do
            let (x, y, mYErr) = dataTrain
                (x_val, y_val, mYErr_val) = dataVal
                (x_te, y_te, mYErr_te) = dataTest
                distribution = _distribution args

                expr      = paramsToConst (MA.toList theta) best'
                showNA z  = if isNaN z then "" else show z
                r2_train  = r2 x y best' theta
                r2_val    = r2 x_val y_val best' theta
                r2_te     = r2 x_te y_te best' theta
                nll_train  = nll distribution mYErr x y best' theta
                nll_val    = nll distribution mYErr_val x_val y_val best' theta
                nll_te     = nll distribution mYErr_te x_te y_te best' theta
                mdl_train  = mdl distribution mYErr x y theta best'
                mdl_val    = mdl distribution mYErr_val x_val y_val theta best'
                mdl_te     = mdl distribution mYErr_te x_te y_te theta best'
                vals       = intercalate ","
                           $ Prelude.map showNA [ nll_train, nll_val, nll_te, maxLoss
                                                , r2_train, r2_val, r2_te
                                                , mdl_train, mdl_val, mdl_te]
                thetaStr   = intercalate ";" $ Prelude.map show (MA.toList theta)
            pure $ show ix <> "," <> show view <> "," <> showExpr expr <> "," <> "\"" <> showPython best' <> "\","
                           <> thetaStr <> "," <> show (countNodes $ convertProtectedOps expr)
                           <> "," <> vals
        pure ts

    insertTerms =
        forM terms $ \t -> do fromTree myCost t >>= canonical
