{-# LANGUAGE  BlockArguments #-}
{-# LANGUAGE  TupleSections #-}
{-# LANGUAGE  MultiWayIf #-}
{-# LANGUAGE  OverloadedStrings #-}
{-# LANGUAGE  BangPatterns #-}
{-# LANGUAGE  TypeSynonymInstances, FlexibleInstances #-}
{-# LANGUAGE  RankNTypes #-}

module Search where

import Algorithm.EqSat.Egraph
import Algorithm.EqSat.Simplify hiding ( myCost )
import Algorithm.EqSat.Build
import Algorithm.EqSat.Queries
import Algorithm.EqSat.Info
import Algorithm.EqSat.DB
import Algorithm.SRTree.Likelihoods
import Algorithm.SRTree.ModelSelection
import Algorithm.SRTree.Compile (compileTree, EvalTree(..))
import Algorithm.SRTree.ConfidenceIntervals (CIType(..), PType(..), paramCI, getAllProfiles, getStatsFromModel, CI(..), BasicStats(..))
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
import qualified Data.Map as Map
import qualified Data.Sequence as FingerTree
import Data.Function ( on )
import qualified Data.Foldable as Foldable

import List.Shuffle ( shuffle )
import Algorithm.SRTree.NonlinearOpt
import Data.Binary ( encode, decode )
import qualified Data.ByteString.Lazy as BS
import Data.List.Split (splitOn)

import Algorithm.EqSat (runEqSat,applySingleMergeOnlyEqSat)

import Control.Concurrent (getNumCapabilities)
import Control.Concurrent.Async (mapConcurrently)
import Control.Exception (evaluate)
import System.Timeout (timeout)
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
import Data.Version (showVersion)
import Control.Exception (Exception (..), SomeException (..), handle)
import Data.Time.Clock.POSIX

import Algorithm.EqSat.Storage.SQLite (saveGraph)
import Algorithm.EqSat.Storage.Query (getOrCreateDataset, readDatasetFit, writeDatasetFit, expressionEclass)
import Algorithm.EqSat.Storage.Types (enodeKey, parseTheta, serializeTheta)
import Algorithm.EqSat.Storage.Backend (SqlBackend)
import qualified Database.SQLite3 as SQLite
import qualified Data.Text as T
import Control.Exception (bracket)
import System.FilePath (takeFileName)

data Args = Args
  { _dataset      :: String,
    _testData     :: String,
    _gens         :: Int,
    _maxSize      :: Int,
    _folds        :: Int,
    _trace        :: Bool,
    _distribution :: Loss,
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
    _backend      :: ADBackEnd,
    _dbFile       :: String,
    _dbFitFile    :: String,
    _dbDataset    :: String,
    _dbCacheSz    :: Int,
    _dbFlushEvery :: Int
  }
  deriving (Show)

csvHeader :: Int -> String
csvHeader maxParams = "id,view,Expression,Numpy,Math,theta,size,loss_train,loss_val,loss_test,maxloss,R2_train,R2_val,R2_test,dl_train,dl_val,dl_test"
                   <> concatMap (\i -> ",t" <> show i <> "_lower,t" <> show i <> "_upper") [0 .. maxParams - 1]

forceSimp :: Fix SRTree -> Fix SRTree
forceSimp t = let s = simplifyEqSatDefault t in countNodes s `seq` s

egraphGP :: [(DataSet, DataSet)] -> [DataSet] -> Args -> StateT EGraph (StateT StdGen IO) String
egraphGP dataTrainVals dataTests args = do
  when ((not.null) (_loadFrom args)) $ (io $ BS.readFile (_loadFrom args)) >>= \eg -> put (decode eg)

  -- Load fitness cache from DB if in DB mode
  fitCache <- if not (null (_dbFitFile args))
              then io $ loadFitnessCache (_dbFitFile args) (dbDatasetName args)
              else pure HashMap.empty

  insertTerms
  unevalInit <- gets (IntSet.toList . _unevaluated . _eDB)
  fitBatchCached fitCache True fitFun unevalInit

  t0 <- io $ getPOSIXTime
  
  pop <- replicateM (_nPop args) $ insertRndExpr (_maxSize args) rndTerm rndNonTerm >>= canonical
  fitBatchCached fitCache False fitFun pop

  output <- if _trace args 
               then forM (Prelude.zip [0..] pop) $ uncurry (printExpr nFeats)
               else pure []

  let m = (_nPop args) `div` (_maxSize args)
      mTime = if _maxtime args < 0 then Nothing else Just (fromIntegral $ _maxtime args - 5) -- add 5 seconds slack

  (finalPop, finalOut, _) <- iterateFor (_gens args) t0 mTime (pop, output, _nPop args) $ \it (ps', out, curIx) -> do
    -- Phase 1: Generate offspring trees in parallel (read-only on e-graph)
    trees <- generateTrees ps'
    -- Phase 2: Batch insert all trees into the e-graph (sequential)
    newPop' <- Prelude.mapM (\t -> fromTree myCost (relabel t) >>= canonical) trees

    -- Single batch eqsat pass over all new classes
    if _nParams args == 0
       then runEqSat myCost rewritesWithConstant 1 >> cleanDB
       else runEqSat myCost rewritesParams 1 >> cleanDB
    modify' $ over (eDB . seenMatches) (const Map.empty)

    -- Batch-fit the eqsat-flagged refits (force) and the new offspring
    refitIds <- gets (IntSet.toList . _refits . _eDB)
    modify' $ over (eDB . refits) (const IntSet.empty)
    fitBatchCached fitCache True fitFun refitIds
    fitBatchCached fitCache False fitFun newPop'

    -- Store newly fitted fitness in DB
    when (not (null (_dbFitFile args))) $ do
      forM_ refitIds $ \eid -> do
        mf <- getFitness eid
        case mf of
          Just f  -> do thetas <- getTheta eid
                        let theta = if null thetas then V.empty else head thetas
                        io $ storeFitnessDB (_dbFitFile args) (dbDatasetName args) eid f theta
          Nothing -> pure ()
      forM_ newPop' $ \eid -> do
        mf <- getFitness eid
        case mf of
          Just f  -> do thetas <- getTheta eid
                        let theta = if null thetas then V.empty else head thetas
                        io $ storeFitnessDB (_dbFitFile args) (dbDatasetName args) eid f theta
          Nothing -> pure ()

    out' <- if _trace args
              then forM (Prelude.zip [curIx..] newPop') $ uncurry (printExpr nFeats)
              else pure []

    totSz <- gets (HashMap.size . _eNodeToEClass)
    let full = totSz > max maxMem (_nPop args)
    cleanedIds <- if full then Just <$> cleanEGraph else pure Nothing
    when full cleanDB

    -- Periodic sync to DB
    when (not (null (_dbFile args)) && _dbFlushEvery args > 0 && it `mod` _dbFlushEvery args == 0) $ do
      eg <- get
      io $ syncEGraphDB (_dbFile args) (dbDatasetName args) eg

    newPop <- if _generational args 
                 then maybe (Prelude.mapM canonical newPop') pure cleanedIds 
                 else do 
                     let n_paretos = (_nPop args) `div` (_maxSize args)
                     pareto <- if (_useFracBayes args)
                                 then getParetoFront
                                 else concat <$> (forM [1 .. _maxSize args] $ \n -> getTopFitEClassWithSize n 2)
                     let remainder = _nPop args - length pareto
                     lft <- if full
                               then getTopFitEClassThat remainder (const True)
                               else pure $ Prelude.take remainder newPop'
                     Prelude.mapM canonical (pareto <> lft)

    pure (newPop, out <> out', curIx + (_nPop args)) 

  -- Final sync to DB
  when (not (null (_dbFile args))) $ do
    eg <- get
    io $ syncEGraphDB (_dbFile args) (dbDatasetName args) eg

  when ((not.null) (_dumpTo args)) $ get >>= (io . BS.writeFile (_dumpTo args) . encode )

  -- When _nParams is -1, compute the actual max number of params across the
  -- Pareto front so the CSV header allocates enough CI columns.
  actualMaxP <- if _nParams args == -1
                  then computeMaxP (_maxSize args) (_distribution args) nFeats
                  else pure (_nParams args)
  let printExpr' = printExpr actualMaxP
  pf <- if _trace args 
           then pure finalOut 
           else paretoFront fitFun (_maxSize args) printExpr'
  pure $ unlines (csvHeader actualMaxP : concat pf) 
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
    terms          = if _distribution args == NLL ROXY
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

    evolve xs' = do
                xs <- Prelude.mapM canonical xs'
                parents <- tournament xs
                offspring <- combine parents
                pure offspring

    evolveDB xs' = do
                xs <- Prelude.mapM canonical xs'
                parents <- tournament xs
                offspring <- combineDB parents
                pure offspring

    tournament xs = do p1 <- applyTournament xs >>= canonical
                       p2 <- applyTournament xs >>= canonical
                       pure (p1, p2)

    applyTournament :: [EClassId] -> RndEGraph EClassId
    applyTournament xs = do challengers <- replicateM (_nTournament args) (rnd $ randomFrom xs) >>= traverse canonical
                            fits <- Prelude.map (fromMaybe (-1.0/0.0)) <$> Prelude.mapM getFitness challengers
                            pure . snd . maximumBy (compare `on` fst) $ Prelude.zip fits challengers

    combine (p1, p2) = (crossover p1 p2 >>= mutate) >>= canonical

    combineDB (p1, p2) = (crossoverDB p1 p2 >>= mutateDB) >>= canonical

    crossover p1 p2 = do sz <- getSize p1
                         coin <- rnd $ tossBiased (_pc args)
                         if sz == 1 || not coin
                            then rnd (randomFrom [p1, p2])
                            else do pos <- rnd $ randomRange (1, sz-1)
                                    cands <- getAllSubClasses p2
                                    tree <- getSubtree pos 0 Nothing [] cands p1
                                    fromTree myCost (relabel tree) >>= canonical

    crossoverDB p1 p2 = do sz <- getSize p1
                           coin <- rnd $ tossBiased (_pc args)
                           if sz == 1 || not coin
                              then rnd (randomFrom [p1, p2])
                              else do pos <- rnd $ randomRange (1, sz-1)
                                      cands <- getAllSubClasses p2
                                      tree <- getSubtree pos 0 Nothing [] cands p1
                                      eid <- fromTree myCost (relabel tree) >>= canonical
                                      en <- getBestENode eid >>= canonize
                                      let key = T.pack (enodeKey en)
                                      inDB <- io $ expressionInDB (_dbFitFile args) key
                                      if inDB
                                        then rnd (randomFrom [p1, p2])
                                        else pure eid

    -- | Like crossover but returns Fix SRTree instead of inserting into the e-graph.
    -- All operations are read-only on the e-graph snapshot.
    crossoverTree p1 p2 = do sz <- getSize p1
                             coin <- rnd $ tossBiased (_pc args)
                             if sz == 1 || not coin
                                then do p <- rnd (randomFrom [p1, p2])
                                        getBestExpr p
                                else do pos <- rnd $ randomRange (1, sz-1)
                                        cands <- getAllSubClasses p2
                                        getSubtree pos 0 Nothing [] cands p1

    -- | Like mutate but works on Fix SRTree directly without writing to the e-graph.
    mutateTree tree = do
      let sz = countNodes tree
      coin <- rnd $ tossBiased (_pm args)
      if coin
         then do pos <- rnd $ randomRange (0, min sz maxSize - 1)
                 mutAtTree pos maxSize tree
         else pure tree

    -- | Walk a Fix SRTree to a position and replace the subtree with a random expression.
    -- Unlike mutAt, this works on Fix SRTree directly and never touches the e-graph.
    mutAtTree :: Int -> Int -> Fix SRTree -> RndEGraph (Fix SRTree)
    mutAtTree 0 sizeLeft _ = do
      t <- insertRndExpr (max 1 sizeLeft) rndTerm rndNonTerm >>= canonical >>= getBestExpr
      pure t
    mutAtTree _ 1 _ = rnd $ randomFrom terms
    mutAtTree pos sizeLeft (Fix (Uni f t)) =
      (Fix . Uni f) <$> mutAtTree (pos-1) (sizeLeft-1) t
    mutAtTree pos sizeLeft (Fix (Bin op l r)) = do
      let szL = countNodes l
      if szL < pos
        then do r' <- mutAtTree (pos-szL-1) (sizeLeft-szL-1) r
                pure . Fix $ Bin op l r'
        else do l' <- mutAtTree (pos-1) (sizeLeft-1) l
                pure . Fix $ Bin op l' r
    mutAtTree _ _ tree = pure tree  -- Var, Const, Param, Y: nothing to mutate

    -- | Run offspring tree generation in parallel using mapConcurrently.
    -- Each worker gets a read-only snapshot of the e-graph, so tree extraction
    -- (tournament, crossover, mutation) is safe. No e-graph writes happen here.
    generateTrees :: [EClassId] -> RndEGraph [Fix SRTree]
    generateTrees xs = do
      nCaps <- io getNumCapabilities
      g0 <- rnd get
      let nJobs = _nPop args
          gs = [ mkStdGen (fromIntegral i * 7919 + 42) | i <- [0 .. nJobs - 1] ]
      eg <- get
      io $ mapConcurrently (\g -> runRndEGraph eg g $ do
        xs' <- Prelude.mapM canonical xs
        parents <- tournament xs'
        coin <- rnd toss
        if coin
          then crossoverTree (fst parents) (snd parents) >>= mutateTree
          else do p <- rnd (randomFrom [fst parents, snd parents])
                  getBestExpr p
        ) (Prelude.take nJobs gs)

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
        ENAry op m -> do
          let xs = expandedList m
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
                    c' <- getSubtree (pos - acc - 1) (sz + 1 + (totalSz - s)) (Just (\eid -> ENAry op (imFromList (replaceAt i eid xs)))) (parent:mGrandParents) cands c
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
        ENAry _ m -> do xss <- mapM getAllSubClasses (expandedList m)
                        pure (p : concat xss)
        _          -> pure [p]

    mutate p = do sz <- getSize p
                  coin <- rnd $ tossBiased (_pm args)
                  if coin
                     then do pos <- rnd $ randomRange (0, min sz maxSize - 1)
                             tree <- mutAt pos maxSize Nothing p
                             fromTree myCost (relabel tree) >>= canonical
                     else pure p

    mutateDB p = do sz <- getSize p
                    coin <- rnd $ tossBiased (_pm args)
                    if coin
                       then do pos <- rnd $ randomRange (0, min sz maxSize - 1)
                               tree <- mutAt pos maxSize Nothing p
                               eid <- fromTree myCost (relabel tree) >>= canonical
                               -- Check if this expression already exists in the DB
                               en <- getBestENode eid >>= canonize
                               let key = T.pack (enodeKey en)
                               inDB <- io $ expressionInDB (_dbFitFile args) key
                               if inDB
                                 then pure p  -- already explored: keep original
                                 else pure eid
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
          ENAry op m -> do
            let xs = expandedList m
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
                      c' <- mutAt (pos - acc - 1) (sizeLeft - 1 - (totalSz - s)) (Just (\eid -> ENAry op (imFromList (replaceAt i eid xs)))) c
                      cs' <- mapM getBestExpr cs
                      pure (c' : cs')
            exprs <- goE cs szs 0 0
            pure $ naryTree op exprs


    printExpr :: Int -> Int -> EClassId -> RndEGraph [String]
    printExpr actualMaxP ix ec = do
        thetas' <- getTheta ec
        bestExpr0 <- getBestExpr ec
        bestExpr <- if _simplify args
                      then do
                        res <- io $ timeout (10 * 1000000) (evaluate (forceSimp bestExpr0))
                        case res of
                          Nothing -> pure bestExpr0
                          Just s  -> pure s
                      else pure bestExpr0

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
                showNA z  = if isNaN z || isInfinite z then "NA" else show z

                n          = fromIntegral (V.length y) :: Double
                n_val      = fromIntegral (V.length y_val) :: Double
                n_te       = fromIntegral (V.length y_te) :: Double
                p          = fromIntegral (countParamsUniq best') :: Double
                f_compl    = countNodes best' * log (countUniqueTokens best')

                -- loss (NLL)
                nll_train_ = compileLoss x (buildLoss distribution n best') y mYErr theta
                nll_val_   = compileLoss x_val (buildLoss distribution n_val best') y_val mYErr_val theta
                nll_te_    = compileLoss x_te (buildLoss distribution n_te best') y_te mYErr_te theta

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

                -- Compute profile-likelihood CIs
                -- Bates (1985) profile likelihood works for MSE/least-squares:
                --   2*(MSE(theta) - MSE(theta_hat)) <= chi2_1
                -- For NLL losses, the same formula applies with the NLL.
                nSamples = V.length y
                dist = case distribution of { NLL d -> d; MSE -> LeastSquares; LOG10 -> LeastSquares; MAE -> LeastSquares; MAPE -> LeastSquares; Pinball _ -> LeastSquares; _ -> Gaussian }
                et = compileTree dist x y mYErr best'
                stats = getStatsFromModel dist mYErr x y best' theta
            profiles <- liftIO $ getAllProfiles Bates et theta (_stdErr stats) [] 0.05
            let ciVals = paramCI (Profile stats profiles) nSamples theta 0.05
                maxPExpr = actualMaxP
                ciStr = intercalate ","
                      $ Prelude.map (\(CI _ l h) -> showNA l <> "," <> showNA h) ciVals
                      ++ Prelude.replicate (2 * (maxPExpr - length ciVals)) ""

            pure $ show ix <> "," <> show view <> "," <> showExprFun expr <> "," <> "\"" <> showPython best' <> "\","
                           <> "\"$$" <> showLatexFun best' <> "$$\","
                           <> thetaStr <> "," <> show (countNodes $ convertProtectedOps expr)
                           <> "," <> vals
                           <> "," <> ciStr
        pure ts

    insertTerms =
        forM terms $ \t -> do fromTree myCost t >>= canonical

    replaceAt :: Int -> EClassId -> [EClassId] -> [EClassId]
    replaceAt 0 e (_:cs) = e : cs
    replaceAt i e (c:cs) = c : replaceAt (i-1) e cs
    replaceAt _ _ []     = []

    -- | Walk the Pareto front to find the actual max number of params
    -- (model params + distribution-specific params) across all sizes.
    computeMaxP :: Int -> Loss -> Int -> RndEGraph Int
    computeMaxP maxSize' dist nFeats' = go 1 0
      where
        distExtra = case dist of
                      NLL Gaussian -> 1
                      NLL ROXY     -> 3
                      _            -> 0
        go n acc
          | n > maxSize' = pure acc
          | otherwise = do
              ecList <- getBestExprWithSize n
              case ecList of
                ((ec, _):_) -> do
                  ec' <- canonical ec
                  bestExpr <- getBestExpr ec'
                  let best' = if shouldReparam then relabelParams bestExpr else relabelParamsOrder bestExpr
                      nP = countParamsUniq best' + distExtra
                  go (n+1) (max acc nP)
                _ -> go (n+1) acc

-- ---------------------------------------------------------------------------
-- DB helpers (used when _dbFile is non-empty)
-- ---------------------------------------------------------------------------

-- | Derive the dataset name for the DB: use _dbDataset if set, else the CSV basename.
dbDatasetName :: Args -> String
dbDatasetName args
  | not (null (_dbDataset args)) = _dbDataset args
  | otherwise                    = takeFileName (_dataset args)

-- | Open a SQLite DB, run an action, close.  SQLite-only (no Postgres).
withSqliteDB :: String -> (forall b. SqlBackend b => b -> IO a) -> IO a
withSqliteDB path k = bracket (SQLite.open (T.pack path)) SQLite.close k

-- | Load fitness cache from DB: eclass id -> (fitness, theta_text).
-- When the e-graph structure is loaded via --load-from, e-class IDs are
-- preserved so this cache is directly usable.  For cross-run keyed reuse
-- (different e-class IDs), expression_index would be needed — left as TODO.
loadFitnessCache :: String -> String -> IO (HashMap Int (Double, V.Vector Double))
loadFitnessCache "" _ = pure HashMap.empty
loadFitnessCache dbFile dataset = withSqliteDB dbFile $ \db -> do
  mdsid <- getOrCreateDataset db dataset
  fits <- readDatasetFit db mdsid
  pure $ HashMap.fromList
    [ (eid, (f, parseThetaVec th))
    | (eid, (Just f, _, _, th)) <- fits
    , not (T.null th) ]
  where
    parseThetaVec th = case parseTheta (T.unpack th) of
      []    -> V.empty
      (v:_) -> v

-- | Check if an expression's enode key exists in expression_index table.
expressionInDB :: String -> T.Text -> IO Bool
expressionInDB "" _ = pure False
expressionInDB dbFile key
  | T.null key = pure False
  | otherwise  = withSqliteDB dbFile $ \db -> do
      meid <- expressionEclass db key
      pure (meid /= Nothing)

-- | Store fitness in DB after evaluation.
storeFitnessDB :: String -> String -> EClassId -> Double -> V.Vector Double -> IO ()
storeFitnessDB "" _ _ _ _ = pure ()
storeFitnessDB dbFile dataset eid fit theta
  = withSqliteDB dbFile $ \db -> do
      dsid <- getOrCreateDataset db dataset
      writeDatasetFit db dsid eid (Just fit) Nothing (T.pack (serializeTheta [theta])) 0

-- | Sync in-memory e-graph to DB (full save for resident graphs).
syncEGraphDB :: String -> String -> EGraph -> IO ()
syncEGraphDB "" _ _ = pure ()
syncEGraphDB dbFile dataset eg
  = withSqliteDB dbFile $ \db -> do
      dsid <- getOrCreateDataset db dataset
      _ <- saveGraph db dsid eg
      pure ()

-- | Like fitBatch, but checks the fitness cache first.
-- For each e-class whose ID is in the cache, insertFitness is called directly
-- (skipping the expensive NLopt optimization).  Uncached e-classes go through
-- the normal fitFun path via the original fitBatch.
fitBatchCached :: HashMap.HashMap Int (Double, V.Vector Double)
               -> Bool
               -> (Fix SRTree -> RndEGraph (Double, [Target]))
               -> [EClassId]
               -> RndEGraph ()
fitBatchCached cache force fitFun ecs0 = do
  ecs <- Prelude.mapM canonical ecs0
  -- Phase 1: fill fitness from cache for e-classes that have a cache hit
  forM_ ecs $ \ec -> do
    mf <- getFitness ec
    when (force || mf == Nothing) $
      case HashMap.lookup ec cache of
        Just (fit, thetas) -> insertFitness ec fit [thetas]
        Nothing            -> pure ()
  -- Phase 2: run the original fitBatch — it will skip e-classes that now
  -- have fitness (from the cache or from a previous run).
  fitBatch force fitFun ecs0
