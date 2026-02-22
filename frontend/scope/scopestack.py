from typing import Optional

from frontend.symbol.symbol import Symbol

from .scope import Scope

class ScopeStack:
    def __init__(self, _scope: Scope) -> None:
        self.scopes: list[Scope] = [_scope]

    def push(self, _scope: Scope) -> None:
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
