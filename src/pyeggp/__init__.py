import atexit
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, List

from ._binding import (
    unsafe_hs_pyeggp_version,
    unsafe_hs_pyeggp_main,
    unsafe_hs_pyeggp_run,
    unsafe_hs_pyeggp_init,
    unsafe_hs_pyeggp_exit,
)

VERSION: str = "1.3.0"


_hs_rts_init: bool = False
_hs_rts_lock: Lock = Lock()


def hs_rts_exit() -> None:
    global _hs_rts_lock
    with _hs_rts_lock:
        unsafe_hs_pyeggp_exit()


@contextmanager
def hs_rts_init(args: List[str] = []) -> Iterator[None]:
    global _hs_rts_init
    global _hs_rts_lock
    with _hs_rts_lock:
        if not _hs_rts_init:
            _hs_rts_init = True
            unsafe_hs_pyeggp_init(args)
            atexit.register(hs_rts_exit)
    yield None


def version() -> str:
    with hs_rts_init():
        return unsafe_hs_pyeggp_version()


def main(args: List[str] = []) -> int:
    with hs_rts_init(args):
        return unsafe_hs_pyeggp_main()

def pyeggp_run(dataset: str, gen: int, nPop: int, maxSize: int, nTournament: int, pc: float, pm: float, nonterminals: str, loss: str, optIter: int, optRepeat: int, nParams: int, split: int, simplify: int, dumpTo: str, loadFrom: str) -> str:
    with hs_rts_init():
        return unsafe_hs_pyeggp_run(dataset, gen, nPop, maxSize, nTournament, pc, pm, nonterminals, loss, optIter, optRepeat, nParams, split, simplify, dumpTo, loadFrom)

class PyEGGP():
    def __init__(self, gen = 100, nPop = 100, maxSize = 15, nTournament = 3, pc = 0.9, pm = 0.3, nonterminals = "add,sub,mul,div", loss = "MSE", optIter = 50, optRepeat = 2, nParams = -1, split = 1, simplify = False, dumpTo = "", loadFrom = ""):
        self.gen = gen
        self.nPop = nPop
        self.maxSize = maxSize
        self.nTournament = nTournament
        self.pc = pc
        self.pm = pm
        self.nonterminals = nonterminals
        self.loss = loss
        self.optIter = optIter
        self.optRepeat = optRepeat
        self.nParams = nParams
        self.split = split
        self.simplify = int(simplify)
        self.dumpTo = dumpTo
        self.loadFrom = loadFrom
    def fit(self, dataset):
        output = pyeggp_run(dataset, self.gen, self.nPop, self.maxSize, self.nTournament, self.pc, self.pm, self.nonterminals, self.loss, self.optIter, self.optRepeat, self.nParams, self.split, self.simplify, self.dumpTo, self.loadFrom)
        return output
