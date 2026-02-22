from typing import Any, Optional, Union

from .tacfunc import TACFunc
from .tacglobaldecl import TACGlobalDecl


# A TAC program consists of several TAC functions.
class TACProg:
    def __init__(self, funcs: list[TACFunc], decls: list[TACGlobalDecl]) -> None:
        self.funcs = funcs
        self.globalDecls = decls

    def printTo(self) -> None:
        for decl in self.globalDecls:
            decl.printTo()
        for func in self.funcs:
            func.printTo()
