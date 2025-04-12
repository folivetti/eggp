import atexit
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, List

from ._binding import (
    unsafe_hs_pyeggp_version,
    unsafe_hs_pyeggp_main,
    unsafe_hs_pyeggp_fib,
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


def pyegpp(n: List[str]) -> str:
    with hs_rts_init():
        return unsafe_hs_pyeggp(n)
