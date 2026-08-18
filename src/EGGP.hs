{-# LANGUAGE  BlockArguments #-}
{-# LANGUAGE  TupleSections #-}
{-# LANGUAGE  MultiWayIf #-}
{-# LANGUAGE  OverloadedStrings #-}
{-# LANGUAGE  BangPatterns #-}
{-# LANGUAGE  TypeSynonymInstances, FlexibleInstances #-}

module EGGP where

import Algorithm.EqSat.Egraph
import Algorithm.EqSat.Simplify
import Algorithm.EqSat.Build
import Algorithm.EqSat.Queries
import Algorithm.EqSat.Info
import Algorithm.EqSat.DB
import Algorithm.SRTree.Likelihoods
import Algorithm.SRTree.ModelSelection
import Control.Lens (element, makeLenses, over, (&), (+~), (-~), (.~), (^.))
import Control.Monad (foldM, forM_, forM, when, unless, filterM, (>=>), replicateM, replicateM_)
import Control.Monad.State.Strict
import Data.Maybe (fromJust, isNothing, isJust)
import Data.SRTree
import Data.SRTree.Datasets
import Data.SRTree.Eval
import Data.SRTree.Random (randomTree)
import Data.SRTree.Print
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
import Options.Applicative as Opt hiding (Const)

import Search
import Algorithm.EqSat.SearchSR
import Algorithm.SRTree.AD (ADBackEnd(..))
import Data.SRTree.Random
import Data.SRTree.Datasets

import Foreign.C (CInt (..), CDouble (..))
import Foreign.C.String (CString, newCString, withCString, peekCString, peekCAString, newCAString)
import Foreign.Marshal.Array (peekArray)
import Foreign.Ptr (Ptr)
import qualified Data.Vector.Unboxed as V
import qualified Data.ByteString.Char8 as B
import Paths_eggp (version)
import System.Environment (getArgs)
import System.Exit (ExitCode (..))

import Data.Version (showVersion)
import Control.Exception (Exception (..), SomeException (..), handle)

foreign import ccall unsafe_py_write_stdout :: CString -> IO ()

py_write_stdout :: String -> IO ()
py_write_stdout str = withCString str unsafe_py_write_stdout

foreign import ccall unsafe_py_write_stderr :: CString -> IO ()

py_write_stderr :: String -> IO ()
py_write_stderr str = withCString str unsafe_py_write_stderr

foreign export ccall hs_eggp_version :: IO CString

hs_eggp_version :: IO CString
hs_eggp_version =
  newCString (showVersion version)

foreign export ccall hs_eggp_main :: IO CInt

exitHandler :: ExitCode -> IO CInt
exitHandler ExitSuccess = return 0
exitHandler (ExitFailure n) = return (fromIntegral n)

uncaughtExceptionHandler :: SomeException -> IO CInt
uncaughtExceptionHandler (SomeException e) =
  py_write_stderr (displayException e) >> return 1

hs_eggp_main :: IO CInt
hs_eggp_main =
  handle uncaughtExceptionHandler $
    handle exitHandler $ do
        args <- execParser opts 
        g <- getStdGen
        let datasets = words (_dataset args)
        dataTrains' <- Prelude.mapM (flip loadTrainingOnly True) datasets -- load all datasets
        dataTests   <- if null (_testData args)
                        then pure dataTrains'
                        else Prelude.mapM (flip loadTrainingOnly True) $ words (_testData args)

        let (dataTrainVals, g') = runState (Prelude.mapM (`splitData` (_folds args)) dataTrains') g
            alg = evalStateT (egraphGP dataTrainVals dataTests args) emptyGraph
        out <- evalStateT alg g'
        py_write_stdout out

        return 0
  where
    opts = Opt.info (opt <**> helper)
            ( fullDesc <> progDesc "An implementation of GP with modified crossover and mutation\
                                   \ operators designed to exploit equality saturation and e-graphs.\
                                   \ https://arxiv.org/abs/2501.17848\n"
           <> header "eggp - E-graph Genetic Programming for Symbolic Regression." )

foreign export ccall hs_eggp_run :: CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> CString -> CInt -> CString -> CString -> CInt -> CInt -> IO CString

hs_eggp_run :: CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> CString -> CInt -> CString -> CString -> CInt -> CInt -> IO CString
hs_eggp_run dataset gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams folds maxTime simplify trace generational dumpTo loadFrom varnames' useFracBayes dbFile' dbDataset' dbCacheSz dbFlushEvery = do
  dataset' <- peekCString dataset
  nonterminals' <- peekCString nonterminals
  loss' <- peekCString loss
  dumpTo' <- peekCString dumpTo
  loadFrom' <- peekCString loadFrom
  varnames <- peekCString varnames'
  dbFile <- peekCString dbFile'
  dbDataset <- peekCString dbDataset'
  out  <- eggp_run dataset' (fromIntegral gens) (fromIntegral nPop) (fromIntegral maxSize) (fromIntegral nTournament) (realToFrac pc) (realToFrac pm) nonterminals' loss' (fromIntegral optIter) (fromIntegral optRepeat) (fromIntegral nParams) (fromIntegral folds) (fromIntegral maxTime) (simplify /= 0) (trace /= 0) (generational /= 0) dumpTo' loadFrom' varnames (useFracBayes /= 0) dbFile dbDataset (fromIntegral dbCacheSz) (fromIntegral dbFlushEvery)
  newCString out

foreign export ccall hs_eggp_run_data :: Ptr CDouble -> Ptr CInt -> CInt -> CInt -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> CString -> CInt -> CString -> CString -> CInt -> CInt -> IO CString

hs_eggp_run_data :: Ptr CDouble -> Ptr CInt -> CInt -> CInt -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> CString -> CInt -> CString -> CString -> CInt -> CInt -> IO CString
hs_eggp_run_data dataPtr nrowsPtr ndatasets ncols header params gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams folds maxTime simplify trace generational dumpTo loadFrom varnames' useFracBayes dbFile' dbDataset' dbCacheSz dbFlushEvery = do
  nonterminals' <- peekCString nonterminals
  loss' <- peekCString loss
  dumpTo' <- peekCString dumpTo
  loadFrom' <- peekCString loadFrom
  varnames <- peekCString varnames'
  header' <- peekCString header
  params' <- peekCString params
  dbFile <- peekCString dbFile'
  dbDataset <- peekCString dbDataset'
  out <- eggp_run_data dataPtr nrowsPtr (fromIntegral ndatasets) (fromIntegral ncols) header' params'
           (fromIntegral gens) (fromIntegral nPop) (fromIntegral maxSize) (fromIntegral nTournament) (realToFrac pc) (realToFrac pm) nonterminals' loss'
           (fromIntegral optIter) (fromIntegral optRepeat) (fromIntegral nParams) (fromIntegral folds) (fromIntegral maxTime) (simplify /= 0) (trace /= 0) (generational /= 0) dumpTo' loadFrom' varnames (useFracBayes /= 0) dbFile dbDataset (fromIntegral dbCacheSz) (fromIntegral dbFlushEvery)
  newCString out

opt :: Parser Args
opt = Args
   <$> strOption
       ( long "dataset"
       <> short 'd'
       <> metavar "INPUT-FILE"
       <> help "CSV dataset." )
  <*> strOption
       ( long "test"
       <> short 't'
       <> value ""
       <> showDefault
       <> help "test data")
   <*> option auto
      ( long "generations"
      <> short 'g'
      <> metavar "GENS"
      <> showDefault
      <> value 100
      <> help "Number of generations." )
  <*> option auto
       ( long "maxSize"
       <> short 's'
       <> help "max-size." )
  <*> option auto
       ( long "folds"
       <> short 'k'
       <> value 1 
       <> showDefault
       <> help "number of folds to determine the ratio of training-validation")
  <*> switch
       ( long "trace"
       <> help "print all evaluated expressions.")
  <*> option (maybeReader readLoss)
       ( long "loss"
        <> value (NLL LeastSquares)
        <> showDefault
        <> help "loss function: MSE, LOG10, MAE, MAPE, Pinball, or a distribution (Gaussian, HGaussian, Poisson, Bernoulli, ROXY, LeastSquares).")
  <*> option auto
       ( long "opt-iter"
       <> value 30
       <> showDefault
       <> help "number of iterations in parameter optimization.")
  <*> option auto
       ( long "opt-retries"
       <> value 1
       <> showDefault
       <> help "number of retries of parameter fitting.")
  <*> option auto
       ( long "number-params"
       <> value (-1)
       <> showDefault
       <> help "maximum number of parameters in the model. If this argument is absent, the number is bounded by the maximum size of the expression and there will be no repeated parameter.")
  <*> option auto
       ( long "nPop"
       <> value 100
       <> showDefault
       <> help "population size (Default: 100).")
  <*> option auto
       ( long "tournament-size"
       <> value 2
       <> showDefault
       <> help "tournament size.")
  <*> option auto
       ( long "pc"
       <> value 1.0
       <> showDefault
       <> help "probability of crossover.")
  <*> option auto
       ( long "pm"
       <> value 0.3
       <> showDefault
       <> help "probability of mutation.")
  <*> strOption
       ( long "non-terminals"
       <> value "Add,Sub,Mul,Div,PowerAbs,Recip"
       <> showDefault
       <> help "set of non-terminals to use in the search."
       )
  <*> strOption
       ( long "dump-to"
       <> value ""
       <> showDefault
       <> help "dump final e-graph to a file."
       )
  <*> strOption
       ( long "load-from"
       <> value ""
       <> showDefault
       <> help "load initial e-graph from a file."
       )
  <*> switch
       ( long "generational"
       <> help "replace the current population with the children instead of keeping the pareto front."
       )
  <*> switch
       ( long "simplify"
       <> help "simplify the expressions before displaying them."
       )
  <*> option auto
       ( long "max-time"
       <> value (-1)
       <> showDefault
       <> help "maximum allowed time budget (in seconds, -1 it will run for the number of generations)"
       )
  <*> strOption
       ( long "varnames"
       <> value ""
       <> showDefault
       <> help "comma separated variable names." )
  <*> switch
       ( long "frac-bayes-complexity"
       <> help "Use n_nodes * log unique_nodes as the second objective."
       )
  <*> option auto
       ( long "backend"
       <> value MultiThread
       <> showDefault
       <> help "AD backend: MultiThread or SingleThread." )
  <*> strOption
       ( long "db-file"
       <> value ""
       <> showDefault
       <> help "SQLite database file for persistent e-graph (empty = in-memory)." )
  <*> strOption
       ( long "db-dataset"
       <> value ""
       <> showDefault
       <> help "Dataset name for DB mode (empty = derive from CSV filename)." )
  <*> option auto
       ( long "db-cache-size"
       <> value 100000
       <> showDefault
       <> help "Max entries in the local fitness cache." )
  <*> option auto
       ( long "db-flush-every"
       <> value 0
       <> showDefault
       <> help "Flush e-graph to DB every N generations (0 = only at end)." )

eggp_run :: String -> Int -> Int -> Int -> Int -> Double -> Double -> String -> String -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> Bool -> String -> String -> String -> Bool -> String -> String -> Int -> Int -> IO String
eggp_run dataset gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams folds maxTime simplify trace generational dumpTo loadFrom varnames useFracBayes dbFile dbDataset dbCacheSz dbFlushEvery =
  case readLoss loss of
       Nothing -> pure $ "Invalid loss function " <> loss
       Just l -> let arg = Args dataset "" gens maxSize folds trace l optIter optRepeat nParams nPop nTournament pc pm nonterminals dumpTo loadFrom generational simplify maxTime varnames useFracBayes MultiThread dbFile dbDataset dbCacheSz dbFlushEvery
                 in eggp arg

eggp :: Args -> IO String
eggp args = do
  let datasets = words (_dataset args)
  dataTrains' <- Prelude.mapM (flip loadTrainingOnly True) datasets -- load all datasets 
  dataTests <- if null (_testData args)
                then pure dataTrains'
                else Prelude.mapM (flip loadTrainingOnly True) $ words (_testData args)
  eggpWithData args dataTrains' dataTests

eggpWithData :: Args -> [DataSet] -> [DataSet] -> IO String
eggpWithData args dataTrains' dataTests = do
  g    <- getStdGen
  let (dataTrainVals, g') = runState (Prelude.mapM (`splitData` (_folds args)) dataTrains') g
      alg = evalStateT (egraphGP dataTrainVals dataTests args) emptyGraph
  evalStateT alg g'

eggp_run_data :: Ptr CDouble -> Ptr CInt -> Int -> Int -> String -> String -> Int -> Int -> Int -> Int -> Double -> Double -> String -> String -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> Bool -> String -> String -> String -> Bool -> String -> String -> Int -> Int -> IO String
eggp_run_data dataPtr nrowsPtr ndatasets ncols header params gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams folds maxTime simplify trace generational dumpTo loadFrom varnames useFracBayes dbFile dbDataset dbCacheSz dbFlushEvery =
  case readLoss loss of
       Nothing -> pure $ "Invalid loss function " <> loss
       Just l -> do
         dss <- buildDataSets dataPtr nrowsPtr ndatasets ncols header params
         let arg = Args "" "" gens maxSize folds trace l optIter optRepeat nParams nPop nTournament pc pm nonterminals dumpTo loadFrom generational simplify maxTime varnames useFracBayes MultiThread dbFile dbDataset dbCacheSz dbFlushEvery
         eggpWithData arg dss dss

-- | Build a list of DataSets from a raw row-major double buffer.  The buffer
--   contains all datasets concatenated; `nrows` gives the per-dataset row
--   count.  Column selection and row ranges reuse the same `:::...` params
--   parsing as the file-based `loadDataset`, so behavior is identical.
buildDataSets :: Ptr CDouble -> Ptr CInt -> Int -> Int -> String -> String -> IO [DataSet]
buildDataSets dataPtr nrowsPtr ndatasets ncols header params = do
  nrows  <- map fromIntegral <$> peekArray ndatasets nrowsPtr
  let totalRows = sum nrows
  flat   <- V.fromList . map realToFrac <$> peekArray (totalRows * ncols) dataPtr
  let (_, prms)     = splitFileNameParams params
      headerMap     = zip (map B.strip (B.split ',' (B.pack header))) [0 .. ncols-1]
      (ixs, iy, iyErr) = getColumns headerMap (prms !! 2) (prms !! 3) (prms !! 4)
      allCols       = [ V.generate totalRows (\i -> flat V.! (i*ncols + j)) | j <- [0 .. ncols-1] ]
      colAt off len j = V.slice off len (allCols !! j)
      mkDs off nrows' =
        let (st, end) = getRows (prms !! 0) (prms !! 1) nrows'
            x  = map (colAt st end) ixs
            y  = colAt st end iy
            ye = if iyErr == -1 then Nothing else Just (colAt st end iyErr)
        in (x, y, ye)
      offs = init (scanl (+) 0 nrows)
  pure $ zipWith mkDs offs nrows
