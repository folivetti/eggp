# EGGP interpreter (Phase 2 -> fixed-shape interpreter) — status

## Goal
Replace Phase 2's per-tree Accelerate JIT with a fixed-shape, data-driven
interpreter (one kernel compile per process) so per-eval cost approaches the
MultiThread backend.

## Constraints
- srtree `cabal.project` forces `-fllvm -mavx2 -mfma -fexpose-all-unfoldings`.
- Benchmark config:
  `EGGP -d data.tsv --non-terminals add,mul,div,exp,log -s 30 -g 1 --nPop 100 --pc 0.0 --pm 1.0 --generational --opt-retries 1 --folds 1 --backend Accelerate +RTS -N4 -qn1 -RTS`
- binary: `/home/olivetti/Projects/srtools/eggp/dist-newstyle/build/.../eggp/eggp`,
  cwd `/home/olivetti/Projects/srtools/eggp`.
- `ACC_INTERP=1` gates the interpreter; stable artifacts live under `/home/olivetti/Projects/srtools/eggp/`.

## The "deadlock" — NOW RESOLVED AS A HARNESS BUG
- Repro harness drew var indices in `[0..9]` (seed3 -> `(x6 * x1)`, uses x6)
  but supplied only **4** feature columns.
- `Algorithm/SRTree/AD/Unboxed.hs:156`: `evalStatic (Var ix) = pure (xss !! ix)`.
  For x6 with 4 cols -> `Prelude.!!: index too large`.
- That forced-out-of-bounds thunk is lazily evaluated **inside an accelerate
  worker** during kernel execution; the scheduler (accelerate-llvm-native
  `Scheduler.hs`) swallows worker exceptions into `workerException` instead of
  propagating; the main thread waits forever -> `BlockedIndefinitelyOnMVar`.
- NOT a scheduler bug and NOT tree/`kDyn`-dependent. Confirmation: giving the
  harness all 10 columns makes seed3 complete: `OK obj=0.999...`.
- The earlier "seed2 pass / seed3 fail" and "all `-N`" repros were this same
  harness artifact (same 4-column / 9-range mismatch in everything run via the
  repro harness).

## Accomplished this session
- Built `harness/WExc` against a LOCAL patched copy of accelerate-llvm-native
  (`/home/olivetti/Projects/srtools/accelerate-llvm-native-debug`) that exports
  `Scheduler(workerException)`, `Target(workers)` and logs swallowed worker
  exceptions to stderr -> surfaced the real error.
- Identified and confirmed root cause; fixed by supplying enough feature columns.
- Stable harness compile recipe (cabal does NOT expose inplace project libs to
  standalone `ghc`; must pass `-package-id`):
  `cabal exec -- ghc -O2 -threaded -rtsopts \
     -package-id srtree-2.0.1.8-inplace \
     -package-id accelerate-llvm-native-1.4.0.0-inplace \
     -o harness/WExc harness/WExc.hs`

## Latest (MultiThread backend, improvement A applied)
- `Algorithm/SRTree/AD.hs`: hoisted the tree compile out of the per-theta
  evaluator for MultiThread and SingleThread (`let cts = compileTreeMulti ...`
  outside the lambda). Before, `compileTreeMulti` (theta-independent static
  columns + graphs) was recomputed on EVERY objective/gradient evaluation
  during an NLopt run; now computed once per (tree, data) and shared across
  thetas. Safe per-tree reuse (immutable cts), no cross-tree sharing.
- Measured (canonical config, `--backend MultiThread`, 2 runs each):
  N1: 12s -> ~10-11s; N4: 8s -> ~6-8s (~10-15%, noisy).
- Accelerate path unchanged (per-tree kernel reuse cache from earlier).

## Next
- Re-run the REAL EGGP config at -N4 with ACC_INTERP=1 to confirm the
  interpreter completes and measure steady per-eval cost vs Phase 2. No known
  bug now blocks -N4.
- If a genuine scheduler swallow is ever hit with in-range data, decide a
  containment (interpreted `multiWorker` single-worker, or serialize evals).
## Session: population-parallel fitness (fitBatch) — 2026-08-03
Goal: parallelize the search's population fitness evals (the search loop is
serial `RndEGraph = EGraphST (StateT StdGen IO)`; each tree fitted one at a time).

Implementation:
- `Algorithm/SRTree/AD/Unboxed.hs`: added `mtParGate :: IORef Bool` (default
  False), `setMTPopParallel :: IO () -> IO ()`; while True, `compileTreeMulti`
  forces `numChunks = 1` so the population batch owns the cores instead of
  oversubscribing with the per-tree chunk split. Exported.
- `Algorithm/EqSat/SearchSR.hs`: new `fitBatch :: Bool -> (Fix SRTree -> RndEGraph
  (Double,[Target])) -> [EClassId] -> RndEGraph ()`. Canonicalizes ecs, drops
  already-fitted unless `force`, snapshots the EGraph, splits the shared StdGen
  once into per-worker generators (global RNG draw order changes, accepted),
  `setMTPopParallel True` -> `mapConcurrently (mapM (runJob eg fitFun)) (chunk
  nCaps jobsG)` (round-robin chunks) -> `setMTPopParallel False`, then serially
  `insertFitness` the results (single serial writer, correctness preserved).
- `Search.hs`: all four fitness call sites converted to fitBatch — initial
  terms (`evaluateUnevaluated` replacement), initial `replicateM nPop` pop,
  generation loop refits (`force=True`, drained from `_refits`) + offspring
  (`force=False` = updateIfNothing semantics), run before Pareto selection so
  ranking sees fresh fitness. Removed dead `refitChanged`; evolve no longer fits.

CRITICAL BUG FOUND AND FIXED: the batch measured ~0.001s wall at N4 with 4
chunks running and NO speedup, because `runRndEGraph` returns the fitness as a
LAZY thunk. `insertFitness` forced it later, serially, in the main thread, so
all NLopt work happened on one core regardless of -N. Fix: in `runJob`, force
results to NF inside the worker: `evaluate (DeepSeq.force f)` /
`evaluate (DeepSeq.force p)` (Target = Vector Double, both NFData). After the
fix the batch wall reflects the real work and scales.

Results (real config, data.tsv, 100 pop, 1 gen, opt-retries 1):
- MultiThread: N1 ~10-13s, N2 ~7-9s, N4 ~6-7s (was ~12/N1, 6-8/N4 — modest
  gain; MT already chunk-parallel per tree internally).
- SingleThread (clean test, no inner parallelism): N1 ~27-28s -> N4 ~11-12s,
  i.e. ~2.4x scaling from the population parallelism (was flat 24s -> 23s).
- All runs rc=0, output sane (losses/R2 in normal range).

## Session: why "per-tree" parallelism shows little gain — 2026-08-04
User repro config (48s old MT -N1): `-d data.tsv --non-terminals add,mul,div,exp,log,power
-s 50 -g 1 --nPop 300 --pc 0.0 --pm 1.0 --generational --opt-retries 1 --opt-iter 100
--folds 1`.

Measured (current build, with fitBatch):
- MT -N1: 56-61s, MT -N8: 28s (2x)
- ST -N1: 160s,   ST -N8: 44-51s (3x)
So ST at -N8 (~44s) only matches MT at -N1 (~56-61s) — confirms user's observation.
MT's value is its vectorized per-even eval, not multi-core. Across-tree concurrency is a
weak lever because NLopt is serial WITHIN a tree (can only parallelize each eval call).

Phase breakdown (PHASE_DEBUG instrumentation, MT -N8): popFit ~3.6s, gen1Evolve ~12s
(serial), FitRefits ~6s, FitNew ~3.3s. The serial `gen1Evolve` cap is ENTIRELY one
`runEqSat myCost rewrites 1` call = 12.6s on the first evolve (saturating the ~300-tree
egraph); subsequent invocations are 1-3ms. runEqSat is pure serial state over the shared
egraph (Algorithm/EqSat.hs:106-135); tournament/combine are negligible (~1ms).

Strategy A/B at -N8 (env gates, since removed): round-robin chunk (current) 26-28s ==
per-tree-thread flat 28-30s, and population-serial + MT inner-parallel (old model) is
WORSE 35-37s. So round-robin across trees + single-chunk-per-tree (fitBatch) is best.
Fit-batch scaling ceiling ~3x is inherent (NLopt serial iterations + memory BW).

Conclusion: feature works and is beneficial for ST (160->44) and gives MT -N8=28s
(faster than old MT -N1). Do NOT expect larger per-tree speedup: the remaining wall is
the serial one-shot eqsat (12.6s) and the serial gen loop, neither parallelizable without
redesigning eqsat/egraph.

## Session: fixed the one-shot eqsat serial cost (variable ordering) — 2026-08-04 (cont.)

Root cause of the 12.6s serial `runEqSat myCost rewrites 1` call: the egraph is SHARED
across the whole population (`StateT EGraph` in `egraphGP`, all 300 initial trees
inserted at `Search.hs:109`), so the FIRST `runEqSat` call matches over the full
300-tree egraph while every later call matches a much-reduced/saturated one (~20-25
matches, ~0s). The `>1500`-eclass throttle never engages because `runEqSat ... 1` is
called with `maxIter=1`, and `go`'s `it == 1 || not changed` branch always returns after
exactly one iteration — throttling is only reachable at `it > 1`.

Instrumented `runEqSat`/`matchCached` (env-gated, since removed) and found the cost is
NOT in `applyMatch`/`rebuild` (~0.06s) but in match ENUMERATION
(`matchCached`/`genericJoin`, srtree/src/Algorithm/EqSat/DB.hs): the first call produces
~1750 matches, ~90% from the two simplest identity rules `x*y` and `x+y` (any mul/add
node matches). Root cause of the enumeration cost: `DB.hs:orderedVars` breaks ties (equal
`varCost`) by variable id, which puts PATTERN-VARIABLE leaves (low ids, e.g. x=0, y=1)
before the query ROOT (high id, freshly assigned, e.g. v=256). Since the root indexes the
operator trie directly, matching leaves first forces `domainX`/`intersectTries` to fold
over the ENTIRE operator trie once per candidate of the earlier variables (`O(candidates
x nodes)`), and — per `intersectTries`'s single-occurrence shortcut (DB.hs:298-304) —
the root's own domain, computed last, ignores the already-bound children entirely,
returning ALL nodes of that operator regardless of consistency.

Fix (srtree/src/Algorithm/EqSat/DB.hs, `orderedVars`): break ties by preferring a
variable that is the ROOT of some atom (an "atom header") over pattern-variable leaves,
so `domainX` on the root does ONE fold over the trie (the true set of candidate nodes),
and children are then resolved by direct trie descent instead of repeated whole-trie
folds. This only changes the TIE-BREAK order among equal-`varCost` variables — it does
not change the frequency-based primary heuristic that already handles multi-atom
patterns (shared subterms still sort first).

IMPORTANT CAVEATS actually verified (not assumed):
- The two orderings do NOT enumerate an identical match SET on the same call (verified
  via per-rule match counts on the same seed: ~21/46 rules differ in count on the first
  call, e.g. `x*y` 1738->1787 total). This means the OLD ordering was itself not a
  faithful/complete enumeration (the shortcut described above lets it over/under count
  depending on trie shape) — reordering is not a neutral optimization, so it was
  quality-gated empirically rather than assumed safe.
- Quality gate (median best `loss_train`, same seed `-s 50`, 5 runs each, MT -N8; NB: MT
  is not run-to-run deterministic due to thread scheduling / parallel fitBatch order):
  old ordering: median 153.01, range [146.11, 155.72] (spread 9.6)
  new ordering: median 151.80, range [149.88, 153.01] (spread 3.1)
  -> not worse (marginally better median, much tighter spread) - PASSES the gate.
- Wall clock (user config, clean build, single run each):
  MT -N1: 56-64s -> 49s | MT -N8: 28-31s -> 17s (was ~2x scaling, now ~2.9x)
  ST -N1: 160s   -> 125s | ST -N8: 44-51s -> 35s (was ~3x scaling, now ~3.6x)
  Shrinking the serial one-shot-eqsat bottleneck let the existing fitBatch parallelism
  scale better too (not just a flat-time win).

Change is a single ~10-line tie-break in `orderedVars`; no callers changed, no env
gates left in the tree (all debug instrumentation used to diagnose this was added and
then fully removed again).

## Session: unified SingleThread onto the vectorized AD kernel — 2026-08-04 (cont.)

Investigated remaining speedup levers after the eqsat fix. Found that the `SingleThread`
ADBackEnd was not just "MultiThread minus threads" — it called a genuinely slower
algorithm:
- `compileFunAndGrad SingleThread` -> `evalGrad` (`srtree/src/Algorithm/SRTree/AD/Unboxed.hs:206-272`,
  row-fused): outer loop over data ROWS, re-dispatching the node-kind `case` on
  EVERY row -> scalar/branchy, no SIMD across rows.
- `compileFunAndGrad MultiThread` -> `evalGradMulti` -> always `evalGradVec`
  (`Unboxed.hs:293-377`, node-outer), even at a single chunk/capability: processes rows in
  1024-row chunks, dispatching the opcode ONCE per node then running a tight straight-line
  loop over the chunk (`fwdBin`/`bwdBin`/etc, lines 494-690) with a literal lambda
  operator -- exactly what LLVM can auto-vectorize under the forced `-fllvm -mavx2 -mfma`
  build flags. (Initially assumed this kernel allocates `O(nodes*m)`; verified from source
  that it's already chunked to `stride*1024` buffers, same memory profile as `evalGrad` --
  no memory-cost tradeoff, purely a loop-shape/vectorization difference.)
- Both operate on the identical `CompiledTree` produced by `compileTree`; `evalGrad` had
  exactly one caller (the `SingleThread` branch) plus one dead commented-out reference --
  very small blast radius.

Fix (`srtree/src/Algorithm/SRTree/AD.hs`): `compileFunAndGrad SingleThread` now calls
`evalGradVec` instead of `evalGrad` (drop-in, since both consume the same `CompiledTree`).

Caveat verified (not assumed): summation order differs slightly between row-at-a-time
(`evalGrad`) and chunk-of-1024 (`evalGradVec`) accumulation -- same left-to-right row
order, but grouped into 1024-row partial sums rather than one running total, so results
are not bit-identical (though should be equally or more numerically stable). Quality-gated
empirically:
- 5 runs each, same seed `-s 50`, `--backend SingleThread +RTS -N1`:
  before: median best `loss_train` 153.01, range [152.83, 154.44]
  after:  median best `loss_train` 153.01 (identical median), range across 8 runs
          [148.91, 161.51] -- one high outlier balanced by a new best-low; verified it's
          not a correctness bug (no NaNs, converged population, just a different local
          optimum -- normal GP stochastic variance from fitBatch's async scheduling).
  -> PASSES the gate (median unchanged, spread comparable).
- Wall clock (user config, clean build):
  ST -N1: 122-147s -> 48-54s (~61% faster)
  ST -N8: 35s -> 16-18s, now equal to MT -N8 (18s) -- the ST/MT gap is fully closed,
  since population-level cross-tree parallelism (`fitBatch`) already benefited both
  backends equally; only the per-tree kernel differed.
  MT -N1 (46-48s) and MT -N8 (17-18s) unaffected, as expected (untouched code path).

## Session: phase-timed the post-fix generation loop — 2026-08-04 (cont.)

With eqsat's one-shot cost fixed, re-profiled `egraphGP` (user config, MT -N8, ~18s
total) with temporary real-IO wall-clock timers (via the existing `io :: IO a ->
RndEGraph a` escape hatch -- no `unsafePerformIO`/CSE issues like the eqsat
micro-timing, since `RndEGraph` already runs in `IO`). Instrumentation was added,
measured once, and fully removed afterward (`git diff` on `Search.hs` is empty).

Phase breakdown (single run, MT -N8):
```
insertTerms+fitBatchUneval   2.97s
insertRndExpr+fitBatchPop    3.54s
evolveLoop (300x tournament/crossover/mutate/eqsat/cleanDB)   0.64s  <- was ~12s pre-fix
fitBatchRefits               8.01s  <- now the single largest phase
fitBatchNewPop                2.99s
cleanEGraph / selection / paretoFront   ~0.26s (not triggered / cheap for -g 1)
```
Confirms the eqsat fix worked (evolveLoop dropped from ~12s to 0.64s) and reveals a new
dominant cost: `fitBatchRefits` (8.0s, ~44% of total). Investigated why it costs more
than `fitBatchNewPop` (which fits the 300 new offspring): `refitIds` (eclasses flagged
dirty by eqsat merges during the evolve loop, via `_refits . _eDB`) numbered **868**
vs 300 new offspring -- each evolve call's one-shot eqsat pass merges/touches ~2.9
*other* existing egraph classes on average (not just the offspring), so nearly 3x more
individuals get NLopt-refit than are newly created. This is a real, quantified next
optimization target (not yet acted on) -- e.g. deduplicating/capping the refit set, or
checking whether a refit's fitness actually needs recomputing before paying for NLopt.

No code changes from this session (purely diagnostic); `Search.hs` is back to its
original committed state.

## Session: fixed `Map.!` crash from egraph cleaning — 2026-08-05

**Bug:** intermittent `Map.!: given key is not an element in the map`, only with
`--generational` and only once the shared egraph crosses `maxMem = 2000000` enodes.

**Root cause:** `Search.hs` `cleanEGraph` wipes the egraph (`put emptyGraph`) when
`totSz > max maxMem (_nPop args)` and re-inserts only the per-size pareto exprs via
`fromTrees`, but returned no ids. The `--generational` branch then ran
`Prelude.mapM canonical newPop'` on pre-wipe offspring ids against the fresh empty
`_canonicalMap` -> missing key in `canonical` (`srtree/.../Egraph.hs:242`
`m IntMap.! eclassId`). Non-generational survived because when `full` it re-derives
all ids from the fresh graph (`getTopFitEClassThat`).

**Fix (src/Search.hs only):**
- `cleanEGraph` now returns `newIds` (the fresh re-inserted eclass ids).
- Generational branch uses them when `full`:
  `maybe (Prelude.mapM canonical newPop') pure cleanedIds`.
  Population after a clean = the re-inserted pareto ids (10 x maxSize); the next
  generation's `evolve` regenerates the full `_nPop`. Pareto-only survival, matching
  the existing non-generational clean semantics.

**Validation:** temporarily lowered `maxMem` to 1 to force the clean path; both
`--generational` and default branches ran 3 generations with cleaning every
generation, exit 0, valid CSV, no crash. Threshold reverted to 2000000 and rebuilt.
Real config (`-s 50 -g 1 --nPop 300 --generational`, 54s @ -N1): no cleaning, exit 0.

**Noted, not fixed:** `cleanMaps` (`srtree/.../Build.hs`) has the same latent
id-invalidation hazard but is unreachable today (only via `throttle`, which needs
`maxIter > 1`; the loop runs one-shot eqsat).

## Session: shrink `fitBatchRefits` (refit only when the class best changed) — 2026-08-05 (cont.)

Follow-up on the 2026-08-04 phase profile that showed `fitBatchRefits` at 8.0s of ~18s
(868 refits/gen). Goal per approved scope: reduce refit work (Tier 1: fewer fits, Tier 2:
warm-starts) without changing `joinData` semantics; every change gated with 5-run A/B
quality tests. All measurement instrumentation was env-gated (`EGGP_STATS=1`) and fully
removed afterward (`git diff src/Search.hs` is back to the `Map.!` fix only).

**Step 0 — instrumentation + baseline (single run, MT -N8, user config).** Counters at the
three dirty sites in `srtree/.../Build.hs` (merge / repairAnalysis / modifyEClass) classify
each refit flag by cause; `srtree/.../SearchSR.hs` counts NLopt evals and dedup. Baseline:
```
refits=744  merge(best=99,other=17) repair(best=275,other=384) modEC=1
refitJobs=744 distinctBest=738  evals=4718
```
Findings:
- Only **375/744 (50%)** refits had a changed `_best`; the other 401 were cost/size/
  `maxMaybe` churn with an unchanged best tree. Refitting those recomputes an identical
  fitness (theta is tied to `_best`) -> pure waste.
- Dedup negligible (738/744 distinct) -> abandoned idea (dedup the refit set).
- ~6.3 evals/refit -> NLopt (TNEWTON) converges far before `opt-iter=100` -> abandoned the
  idea of a `--refit-iter` budget.

**Idea 4 (kept): refit iff `_best` changed AND the class already has a fitness.**
`Build.hs` repairAnalysis + mergeClasses now insert into `_refits` only when
`bestChanged && isJust (_fitness old-class)`. The `isJust` clause drops never-fitted
intermediates: they are invisible to ranking (absent from `_fitRangeDB`) and never reported
(`paretoFront` re-verifies via safe `refit`), so skipping them is lossless. First attempt
used `bestChanged || isNothing` (744->650, kept intermediates); sharpened to
`&& isJust` (-> ~24-43).

**Gate (5 runs each, MT -N8, same config):** median best `loss_train` (proper CSV parse;
`awk` is broken by commas inside the quoted Numpy/Math columns):
```
baseline (always-refit): wall 15.96-18.33s, refits 655-910, median loss_train 152.52
idea 4 (best-changed):    wall 11.0-12.8s, refits ~35-43,   median loss_train 153.01
final clean build:        wall 11.7-14.0s,                  median loss_train 153.01
```
~30% faster; loss delta +0.49 is within the documented -N8 scheduling noise (historical
range spreads 9.6; idea4 median exactly equals the older 153.01 baseline median). PASSES.
`joinData` untouched (`Info.hs:41-64`).

**Warm-start (idea 1) — evaluated, NOT implemented.** With refit volume collapsed to ~40,
the refit phase is now ~0.3s/gen; warm-starting those fits (or the final `paretoFront`
re-verify, ~0.2s) would save <2% wall. Not worth the semantic change + a gate.

**Post-idea4 phase profile (MT -N8, ~12s, phase timers added then removed):**
```
insertTerms          1.68s        (eqsat build of initial DB)
unevalFit (9 vars)   1.78s        (0 NLopt evals; 9 bare vars, ~200ms each = compile+eval
                                   over the 103,897-row data.tsv)
initPop (300 fits)   3.7s         (1570 evals)
evolveLoop           0.66s        (tournament/crossover/mutate/one-shot eqsat)
fitBatchRefits       0.32s        (111 evals)   <- was 8.0s
fitBatchNewPop       3.8s         (1420 evals)
paretoFront          0.17s
```
Remaining wall is dominated by the required population fits (600 trees/gen) over the
~104k-row dataset (~20ms per NLopt eval); no further easy lever within the approved scope
(subsampling/batching would be a search-semantics change, out of scope).

**Files changed (final state):** `srtree/src/Algorithm/EqSat/Build.hs` (the two idea-4
refit-flag conditions; counters removed); `eggp/src/Search.hs` (`Map.!` fix only — the
refit cause/evals/phase instrumentation was removed). No env gates left in the tree.


## Session: intermittent `Map.!: given key is not an element in the map` crash — 2026-08-05

**Symptom:** steady-state (non-generational) run with `--simplify` crashed intermittently
with `Map.!: given key is not an element in the map` on:
`-d gaussian_train.csv:::y:0,1 -t gaussian_test.csv:::y:0,1 -s 15 -g 1000 --nPop 200
--non-terminals add,mul,div,log,exp --opt-retries 20 --opt-iter 10 --simplify +RTS -N8 -RTS`.
Not reproduced across 4 full runs of that config (RNG-dependent — only fires when the
simplified front expression's e-graph hits specific shapes).

**Method:** wrote a standalone stress harness (`ghc -O1 -package srtree`, not part of the
repo) generating random small trees and hammering 4 code paths per tree: `simplifyEqSatDefault`
(the `--simplify` path), `eqSat` with `rewritesParams` (the search's `evolve` path),
raw `runEqSat` + `recalculateBest`, and AD `compileFunAndGrad MultiThread`. This reproduced
the crash on the very first adversarial tree (nested `log`/`exp`, e.g.
`Log((((Exp(x1) / (x1 + 2.0)) / (x1 * x1)) * (x1 * x1)))`), then two more bugs after each fix,
in the `--simplify` path only (`simplifyEqSatDefault t = eqSat t rewrites myCost 30`,
maxIt=30 — the search's own `runEqSat myCost rewritesParams 1` never triggers any of these
because `it == 1` short-circuits before the code paths below run).

**Bug 1 — `recalculateBest`/`fillUpCosts` (`srtree/.../EqSat.hs`):** `log(exp x) :==: exp(log x)`
-style rules can create a cycle in the e-graph reachable from the root. `fillUpCosts`'s
bottom-up fixpoint never assigns a cost to a class in a cycle (no leaf anchor), so
`costs Map.! eid'` (the user's exact error) crashes on the root. Fix: made `nodeCost` total —
a missing child (still uncosted/cyclic) contributes cost 0 and a `Const 0` placeholder
expression, so every class gets a finite cost and expression by the end of the fixpoint.

**Bug 2 — `getChildrenData`/`getChildrenMinHeight` (`srtree/.../Info.hs`):** looked up raw
(non-canonicalized) e-node child ids directly in `_eClass`. `repairAnalysis` calls
`makeAnalysis` on an e-node whose children may have been merged away since it was queued;
the raw child id is no longer a key in `_eClass` -> `IntMap.!` crash. Fix: `canonical` the
child ids first.

**Bug 3 — `cleanMaps` (`srtree/.../Build.hs`), triggered by `runEqSat`'s `throttle`
(`srtree/.../EqSat.hs`) when the e-graph exceeds 1500 classes during a `maxIt > 1` run:**
`cleanMaps` replaced `_canonicalMap` with **only the identity entries**
(`IntMap.filterWithKey (\k v -> k == v)`) while `_nextId` keeps counting and `_patDB`/
`_worklist`/`_analysis`/`_unevaluated`/`_refits` still reference now-untracked stale ids.
Any subsequent `canonical` call on such a stale id crashes (`IntMap.!` on `_canonicalMap`).
This was previously noted as "latent, unreachable" (see the `--generational` session above)
because the search's `runEqSat ... 1` never throttles, but `--simplify`'s `maxIt=30` does.
Fix: `cleanMaps` now keeps `_canonicalMap` **complete** (drop the identity filter) so
`canonical` stays total for every id ever created, while `_eClass` is still pruned to live
representatives (the actual memory-saving part of `cleanMaps`).

**Bug 4 — AD `insertKey` (`srtree/.../SRTree/AD/Unboxed.hs`):** eqsat constant folding can
produce `Const NaN` nodes (e.g. `Infinity/Infinity`), which `--simplify`'s output can then
contain. `Data.Map`'s `Ord Double` is not a valid total order for `NaN` in a way that keeps
`member`/`lookup` consistent — empirically, `Map.insert (Const NaN) v m; Map.member (Const NaN) m`
can return `False` right after the insert (verified with a standalone repro: inserting two
different `NaN`-producing `Const` expressions, `member`/`lookup` on either key returned
`Nothing`/`False` even immediately after insertion). `insertKey`'s `member`-then-`lookup`
pattern hit this and crashed with `UNBOXED_GETKEYS_MISSING` (`a Map.! k`, `Map.!:` in the
original). Fix: do a single `Map.lookup`; on a miss, take the fresh id `insEntry` assigns
directly instead of looking the key back up. Repeated `NaN` nodes just get separate ids
(no CSE for them), which is harmless — their static value is recomputed identically either way.

**Validation:** custom stress harness (`ghc -O1 -package srtree`, 60 random trees x 4 exercise
paths) ran clean (`STRESS OK: no crash`) after all 4 fixes. Real user config re-run 3x
steady-state + 1x `--generational`, all `exit=0`, empty stderr, valid CSV output.

**Files changed:** `srtree/src/Algorithm/EqSat.hs` (`recalculateBest`/`fillUpCosts` total
`nodeCost`), `srtree/src/Algorithm/EqSat/Info.hs` (`getChildrenData`/`getChildrenMinHeight`
canonicalize), `srtree/src/Algorithm/EqSat/Build.hs` (`cleanMaps` keeps `_canonicalMap`
complete), `srtree/src/Algorithm/SRTree/AD/Unboxed.hs` (`insertKey` NaN-safe lookup). Also
added defensive `error`-with-context in place of a few other bare `Map.!`/`IntMap.!` calls
that were audited and found always-safe today (`matchCached`, `reprPrat`, `constrainOnVal`,
`canonical`, `getEClass`) so any future regression fails with a diagnosable message instead
of the bare containers error; added `HasCallStack` to the eqsat call chain for the same
reason. No behavior change on any path that doesn't hit the four bugs above.
