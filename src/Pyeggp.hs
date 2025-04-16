{-# LANGUAGE  BlockArguments #-}
{-# LANGUAGE  TupleSections #-}
{-# LANGUAGE  MultiWayIf #-}
{-# LANGUAGE  OverloadedStrings #-}
{-# LANGUAGE  BangPatterns #-}
{-# LANGUAGE  TypeSynonymInstances, FlexibleInstances #-}

module Pyeggp where

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
import Random
import System.Random
import qualified Data.HashSet as Set
import Data.List ( sort, maximumBy, intercalate, sortOn, intersperse, nub )
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

import Util

import Foreign.C (CInt (..), CDouble (..))
import Foreign.C.String (CString, newCString, withCString, peekCString, peekCAString, newCAString)
import Paths_Pyeggp (version)
import System.Environment (getArgs)
import System.Exit (ExitCode (..))
import Text.Read (readMaybe)
import Data.Version (showVersion)
import Control.Exception (Exception (..), SomeException (..), handle)

foreign import ccall unsafe_py_write_stdout :: CString -> IO ()

py_write_stdout :: String -> IO ()
py_write_stdout str = withCString str unsafe_py_write_stdout

foreign import ccall unsafe_py_write_stderr :: CString -> IO ()

py_write_stderr :: String -> IO ()
py_write_stderr str = withCString str unsafe_py_write_stderr

foreign export ccall hs_pyeggp_version :: IO CString

hs_pyeggp_version :: IO CString
hs_pyeggp_version =
  newCString (showVersion version)

foreign export ccall hs_pyeggp_main :: IO CInt

exitHandler :: ExitCode -> IO CInt
exitHandler ExitSuccess = return 0
exitHandler (ExitFailure n) = return (fromIntegral n)

uncaughtExceptionHandler :: SomeException -> IO CInt
uncaughtExceptionHandler (SomeException e) =
  py_write_stderr (displayException e) >> return 1

hs_pyeggp_main :: IO CInt
hs_pyeggp_main =
  handle uncaughtExceptionHandler $
    handle exitHandler $ do
      getArgs >>= \args -> do
        --out <- pyeggp args
        case "" of
          ""  -> py_write_stdout $ "wrong arguments"
          css -> py_write_stdout css
      return 0

foreign export ccall hs_pyeggp_run :: CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> IO CString

hs_pyeggp_run :: CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> IO CString
hs_pyeggp_run dataset gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams split simplify dumpTo loadFrom = do
  dataset' <- peekCString dataset
  nonterminals' <- peekCString nonterminals
  loss' <- peekCString loss
  dumpTo' <- peekCString dumpTo
  loadFrom' <- peekCString loadFrom
  out  <- pyeggp dataset' (fromIntegral gens) (fromIntegral nPop) (fromIntegral maxSize) (fromIntegral nTournament) (realToFrac pc) (realToFrac pm) nonterminals' loss' (fromIntegral optIter) (fromIntegral optRepeat) (fromIntegral nParams) (fromIntegral split) (simplify /= 0) dumpTo' loadFrom'
  newCString out

egraphGP :: [(DataSet, DataSet)] -> Args -> StateT EGraph (StateT StdGen IO) String
egraphGP dataTrainVals args = do
  when ((not.null) (_loadFrom args)) $ (io $ BS.readFile (_loadFrom args)) >>= \eg -> put (decode eg)

  pop <- replicateM (_nPop args) $ do ec <- insertRndExpr (_maxSize args) rndTerm rndNonTerm >>= canonical
                                      updateIfNothing fitFun ec
                                      pure ec
  insertTerms
  evaluateUnevaluated fitFun
  pop' <- Prelude.mapM canonical pop
  let m = (_nPop args) `div` (_maxSize args)

  finalPop <- iterateFor (_gens args) pop' $ \it ps' -> do
    newPop' <- replicateM (_nPop args) (evolve ps')

    totSz <- gets (Map.size . _eNodeToEClass) -- (IntMap.size . _eClass)
    let full = totSz > max maxMem (_nPop args)
    when full (cleanEGraph >> cleanDB)

    newPop <- do let n_paretos = (_nPop args) `div` (_maxSize args)
                 pareto <- concat <$> (forM [1 .. _maxSize args] $ \n -> getTopFitEClassWithSize n 2)
                 let remainder = _nPop args - length pareto
                 lft <- if full
                           then getTopFitEClassThat remainder (const True)
                           else pure $ Prelude.take remainder newPop'
                 Prelude.mapM canonical (pareto <> lft)
    pure newPop

  when ((not.null) (_dumpTo args)) $ get >>= (io . BS.writeFile (_dumpTo args) . encode )
  pf <- paretoFront fitFun (_maxSize args) printExpr
  pure $ "id,Expression,Numpy,theta,size\n" <> pf
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
                    if coin then Random.randomFrom terms else Random.randomFrom params
    rndNonTerm = Random.randomFrom nonTerms

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

    checkToken parent en' = do en <- canonize en'
                               mEc <- gets ((Map.!? en) . _eNodeToEClass)
                               case mEc of
                                    Nothing -> pure True
                                    Just ec -> do ec' <- canonical ec
                                                  ec'' <- canonize (parent ec')
                                                  not <$> doesExist ec''
    doesExist, doesNotExist :: ENode -> RndEGraph Bool
    doesExist en = gets ((Map.member en) . _eNodeToEClass)
    doesNotExist en = gets ((Map.notMember en) . _eNodeToEClass)

    doesNotExistGens :: [Maybe (EClassId -> ENode)] -> ENode -> RndEGraph Bool
    doesNotExistGens []              en = gets ((Map.notMember en) . _eNodeToEClass)
    doesNotExistGens (mGrand:grands) en = do b <- gets ((Map.notMember en) . _eNodeToEClass)
                                             if b
                                               then pure True
                                               else case mGrand of
                                                    Nothing -> pure False
                                                    Just gf -> do ec  <- gets ((Map.! en) . _eNodeToEClass)
                                                                  en' <- canonize (gf ec)
                                                                  doesNotExistGens grands en'

    printExpr :: Int -> EClassId -> RndEGraph String
    printExpr ix ec = do
        thetas' <- gets (_theta . _info . (IM.! ec) . _eClass)
        bestExpr <- (if _simplify args then simplifyEqSatDefault else id) <$> getBestExpr ec
        let nParams = countParamsUniq bestExpr
            fromSz (MA.Sz x) = x 
            nThetas  = Prelude.map (fromSz . MA.size) thetas'
        (_, thetas) <- if Prelude.any (/=nParams) nThetas || _simplify args
                        then fitFun bestExpr
                        else pure (1.0, thetas')

        ts <- forM (Prelude.zip dataTrainVals thetas) $ \((dataTrain, dataVal), theta) -> do
            let (x, y, mYErr) = dataTrain
                (x_val, y_val, mYErr_val) = dataVal
                distribution = _distribution args
                best'     = if shouldReparam then relabelParams bestExpr else relabelParamsOrder bestExpr
                expr      = paramsToConst (MA.toList theta) best'
                thetaStr   = intercalate ";" $ Prelude.map show (MA.toList theta)
                showFun    = showExpr expr
                showFunNp  = showPython best'
            pure $ show ix <> "," <> showFun <> "," <> "\"" <> showFunNp <> "\"" <> ","
                    <> thetaStr <> "," <> show (countNodes $ convertProtectedOps expr) <> "\n"
        pure (concat ts)
    insertTerms =
        forM terms $ \t -> do fromTree myCost t >>= canonical

data Args = Args
  { _dataset      :: String,
    _gens         :: Int,
    _maxSize      :: Int,
    _split        :: Int,
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
    _simplify     :: Bool
  }
  deriving (Show)


pyeggp :: String -> Int -> Int -> Int -> Int -> Double -> Double -> String -> String -> Int -> Int -> Int -> Int -> Bool -> String -> String -> IO String
pyeggp dataset gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams split simplify dumpTo loadFrom =
  case readMaybe loss of
       Nothing -> pure $ "Invalid loss function " <> loss
       Just l -> let arg = Args dataset gens maxSize split l optIter optRepeat nParams nPop nTournament pc pm nonterminals dumpTo loadFrom simplify
                 in eggp arg

eggp :: Args -> IO String
eggp args = do
  g    <- getStdGen
  let datasets = words (_dataset args)
  dataTrains' <- Prelude.mapM (flip loadTrainingOnly True) datasets -- load all datasets 

  let (dataTrainVals, g') = runState (Prelude.mapM (`splitData` (_split args)) dataTrains') g
      alg = evalStateT (egraphGP dataTrainVals args) emptyGraph
  evalStateT alg g'
