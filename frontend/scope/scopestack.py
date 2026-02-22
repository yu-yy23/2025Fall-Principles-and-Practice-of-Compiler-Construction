from typing import Optional

from frontend.symbol.symbol import Symbol

from .scope import Scope, ScopeKind

class ScopeStack:
    def __init__(self, _scope: Scope) -> None:
        self.scopes: list[Scope] = [_scope]

    def push(self, _scope: Scope) -> None:
        _scope.loopDepth = self.top().loopDepth
        self.scopes.append(_scope)

    def pop(self) -> None:
        self.scopes.pop()

    def top(self) -> Scope:
        return self.scopes[-1]

    def declare(self, symbol: Symbol) -> None:
        self.top().declare(symbol)

    def lookup(self, name: str) -> Optional[Symbol]:
        for scope in reversed(self.scopes):
            symbol = scope.lookup(name)
            if symbol is not None:
                return symbol
        return None

    def openLoop(self) -> None:
        _scope = Scope(ScopeKind.LOCAL)
        self.push(_scope)
        self.top().loopDepth += 1

    def closeLoop(self) -> None:
        self.pop()

    def getLoopDepth(self) -> int:
        return self.top().loopDepth
