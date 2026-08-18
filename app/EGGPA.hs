{-# LANGUAGE  BlockArguments #-}
{-# LANGUAGE  TupleSections #-}
{-# LANGUAGE  MultiWayIf #-}
{-# LANGUAGE  OverloadedStrings #-}
{-# LANGUAGE  BangPatterns #-}
{-# LANGUAGE  TypeSynonymInstances, FlexibleInstances #-}

module EGGPA where

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

import System.Environment (getArgs)
import System.Exit (ExitCode (..))
import Data.Version (showVersion)
import Control.Exception (Exception (..), SomeException (..), handle)

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
       <> help "AD backend: MultiThread, SingleThread, or Accelerate." )
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
  g    <- getStdGen
  let datasets = words (_dataset args)
  dataTrains' <- Prelude.mapM (flip loadTrainingOnly True) datasets -- load all datasets 
  dataTests <- if null (_testData args)
                then pure dataTrains'
                else Prelude.mapM (flip loadTrainingOnly True) $ words (_testData args)

  let (dataTrainVals, g') = runState (Prelude.mapM (`splitData` (_folds args)) dataTrains') g
      alg = evalStateT (egraphGP dataTrainVals dataTests args) emptyGraph
  evalStateT alg g'
