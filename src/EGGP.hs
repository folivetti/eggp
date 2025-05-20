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
import Data.SRTree.Random
import Data.SRTree.Datasets

import Foreign.C (CInt (..), CDouble (..))
import Foreign.C.String (CString, newCString, withCString, peekCString, peekCAString, newCAString)
import Paths_eggp (version)
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

foreign export ccall hs_eggp_run :: CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> IO CString

hs_eggp_run :: CString -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CString -> CString -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CString -> CString -> IO CString
hs_eggp_run dataset gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams folds simplify trace generational dumpTo loadFrom = do
  dataset' <- peekCString dataset
  nonterminals' <- peekCString nonterminals
  loss' <- peekCString loss
  dumpTo' <- peekCString dumpTo
  loadFrom' <- peekCString loadFrom
  out  <- eggp_run dataset' (fromIntegral gens) (fromIntegral nPop) (fromIntegral maxSize) (fromIntegral nTournament) (realToFrac pc) (realToFrac pm) nonterminals' loss' (fromIntegral optIter) (fromIntegral optRepeat) (fromIntegral nParams) (fromIntegral folds) (simplify /= 0) (trace /= 0) (generational /= 0) dumpTo' loadFrom'
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
  <*> option auto
       ( long "loss"
       <> value MSE
       <> showDefault
       <> help "loss function: MSE, Gaussian, Poisson, Bernoulli.")
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

eggp_run :: String -> Int -> Int -> Int -> Int -> Double -> Double -> String -> String -> Int -> Int -> Int -> Int -> Bool -> Bool -> Bool -> String -> String -> IO String
eggp_run dataset gens nPop maxSize nTournament pc pm nonterminals loss optIter optRepeat nParams folds simplify trace generational dumpTo loadFrom =
  case readMaybe loss of
       Nothing -> pure $ "Invalid loss function " <> loss
       Just l -> let arg = Args dataset "" gens maxSize folds trace l optIter optRepeat nParams nPop nTournament pc pm nonterminals dumpTo loadFrom generational simplify
                 in eggp arg

eggp :: Args -> IO String
eggp args = do
  g    <- getStdGen
  let datasets = words (_dataset args)
  dataTrains' <- Prelude.mapM (flip loadTrainingOnly True) datasets -- load all datasets 
  dataTests <- if null (_testData args)
                then pure dataTrains'
                else Prelude.mapM (flip loadTrainingOnly True) $ words (_testData args)

  let (dataTrainVals, g') = runState (Prelude.mapM (`splitData` (_folds args)) dataTrains') g
      alg = evalStateT (egraphGP dataTrainVals dataTests args) emptyGraph
  evalStateT alg g'
