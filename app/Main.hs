{-# LANGUAGE BlockArguments #-}

module Main where

import EGGPA (eggp, opt)
import Options.Applicative as Opt
import Control.Exception (Exception (..), SomeException (..), handle)
import System.Exit (ExitCode (..), exitWith)
import System.IO (hPutStrLn, stderr)
exitHandler :: ExitCode -> IO ()
exitHandler ExitSuccess = pure ()
exitHandler (ExitFailure n) = exitWith (ExitFailure n)

uncaughtExceptionHandler :: SomeException -> IO ()
uncaughtExceptionHandler (SomeException e) =
  hPutStrLn stderr (displayException e) >> exitWith (ExitFailure 1)

main :: IO ()
main =
  handle uncaughtExceptionHandler $
    handle exitHandler $ do
        args <- execParser
          (Opt.info (opt <**> helper) Opt.fullDesc)
        result <- eggp args
        putStr result
