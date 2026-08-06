{-# LANGUAGE BangPatterns #-}
-- Repro seed3: read worker exceptions after the deadlock.
import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Storable as VS
import Data.SRTree.Recursion (Fix(..))
import Data.SRTree.Random
import Data.SRTree (relabelParams, SRTree)
import Data.SRTree.Internal (SRTree(..), Op(..), Function(..))
import Algorithm.SRTree.Likelihoods (buildLoss, Loss(..), Distribution(..))
import Algorithm.SRTree.AD.Unboxed (compileTree)
import Algorithm.SRTree.AD.Accelerate (compileAccelerateTree, multiWorker)
import Control.Monad.State.Strict (evalStateT)
import Control.Monad.Identity
import System.Random (mkStdGen)
import Control.Exception (try, SomeException, evaluate, displayException)
import Control.Concurrent.MVar (tryTakeMVar)
import qualified Data.Sequence as Seq (viewl, ViewL(..))
import Data.Array.Accelerate.LLVM.Native.Target (workers)
import Data.Array.Accelerate.LLVM.Native.Execute.Scheduler (workerException)

genTerm :: Rng Identity (Fix SRTree)
genTerm = do
  coin <- toss
  if coin then Fix . Var <$> randomFrom [0..9]
          else Fix . Param <$> randomFrom [0..9]
genNonTerm :: Rng Identity (SRTree ())
genNonTerm = do
  n <- randomRange (0 :: Int, 4)
  pure $ case n of
    0 -> Bin Add () ()
    1 -> Bin Mul () ()
    2 -> Bin Div () ()
    3 -> Uni Exp ()
    _ -> Uni Log ()
genTree :: Int -> Int -> Fix SRTree
genTree seed budget = runIdentity $ evalStateT (randomTree 3 8 budget genTerm genNonTerm True) (mkStdGen seed)

main :: IO ()
main = do
    let (xss, y) = ([ VU.replicate 10000 1.0 | _ <- [0..9] ], VU.replicate 10000 2.0)
        t  = genTree 3 4
        tl = relabelParams t
        t' = buildLoss (NLL LeastSquares) 10000 tl
        ct = compileTree xss y (Just y) t'
        f  = compileAccelerateTree ct xss y
        th = VS.empty
    putStrLn "compiled, evaluating..."
    r <- try (evaluate (fst (f th))) :: IO (Either SomeException Double)
    case r of
      Left e -> do
        putStrLn ("CAUGHT: " ++ displayException e)
        mseq <- tryTakeMVar (workerException (workers multiWorker))
        case mseq of
          Nothing -> putStrLn "<no worker exceptions captured>"
          Just seqe -> mapM_ (putStrLn . (\(tid, e') -> "worker " ++ show tid ++ ": " ++ displayException e')) (Seq.viewl seqe)
      Right o -> putStrLn ("OK obj=" ++ show o)