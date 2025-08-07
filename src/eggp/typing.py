import sys

from typing import List

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol


class Session(Protocol):
    def version(self) -> str: ...
    def main(self, args: List[str] = []) -> int: ...
    def eggp_run(self, dataset: str, gen: int, nPop: int, maxSize: int, nTournament: int, pc: float, pm: float, nonterminals: str, loss: str, optIter: int, optRepeat: int, nParams: int, split: int, max_time: int, simplify: int, trace : int, generational : int, dumpTo: str, loadFrom: str, varnames: str) -> str: ...
