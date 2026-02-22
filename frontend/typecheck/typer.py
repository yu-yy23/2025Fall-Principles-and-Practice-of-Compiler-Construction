from typing import Protocol, TypeVar

from frontend.ast.node import Node
from frontend.ast.tree import *
from frontend.ast.visitor import Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope
from frontend.scope.scopestack import ScopeStack
from frontend.type.array import ArrayType
from utils.error import *

"""
The typer phase: type check abstract syntax tree.
"""


class Typer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        for decl in program.global_decls():
            decl.accept(self, None)
        for func in program.functions_list():
            func.accept(self, None)
        return program

    def visitFunction(self, func: Function, ctx: ScopeStack) -> None:
        func.body.accept(self, ctx)
        if func.ret_t != func.body.type:
            return DecafBadReturnTypeError()

    def visitParameter(self, param: Parameter, ctx: ScopeStack) -> None:
        param.ident.type = param.var_t.type

    def visitBlock(self, block: Block, ctx: ScopeStack) -> None:
        block.type = INT
        for stmt in block.children:
            stmt.accept(self, ctx)
            if isinstance(stmt, Return):
                block.type = stmt.expr.type

    def visitReturn(self, stmt: Return, ctx: ScopeStack) -> None:
        stmt.expr.accept(self, ctx)
        stmt.type = stmt.expr.type

    def visitFor(self, stmt: For, ctx: ScopeStack) -> None:
        stmt.init.accept(self, ctx)
        stmt.cond.accept(self, ctx)
        stmt.update.accept(self, ctx)
        stmt.body.accept(self, ctx)

    def visitIf(self, stmt: If, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.body.accept(self, ctx)

    def visitBreak(self, stmt: Break, ctx: ScopeStack) -> None:
        pass

    def visitContinue(self, stmt: Continue, ctx: ScopeStack) -> None:
        pass

    def visitDeclaration(self, decl: Declaration, ctx: ScopeStack) -> None:
        if not decl.init_exprs is NULL:
            for expr in decl.init_exprs:
                expr.accept(self, ctx)
                if isinstance(decl.var_t.type, ArrayType) and expr.type != INT:
                    raise DecafTypeMismatchError()
        if (not isinstance(decl.var_t.type, ArrayType)) and decl.init_exprs is not NULL:
            if len(decl.init_exprs) > 1:
                raise DecafTypeMismatchError()
            if decl.init_exprs[0].type != decl.var_t.type:
                # print(decl.var_t, decl.init_exprs[0].type)
                raise DecafTypeMismatchError()

    def visitAssignment(self, expr: Assignment, ctx: ScopeStack) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)
        if expr.lhs.type != expr.rhs.type or expr.lhs.type != INT:
            raise DecafTypeMismatchError()
        expr.type = expr.lhs.type

    def visitUnary(self, expr: Unary, ctx: ScopeStack) -> None:
        expr.operand.accept(self, ctx)
        if expr.operand.type != INT:
            raise DecafTypeMismatchError()
        expr.type = expr.operand.type

    def visitBinary(self, expr: Binary, ctx: ScopeStack) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)
        if expr.lhs.type != expr.rhs.type or expr.lhs.type != INT:
            raise DecafTypeMismatchError()
        expr.type = expr.lhs.type

    def visitCondExpr(self, expr: ConditionExpression, ctx: ScopeStack) -> None:
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)
        if expr.then.type != expr.otherwise.type:
            raise DecafTypeMismatchError()
        expr.type = expr.then.type

    def visitIdentifier(self, ident: Identifier, ctx: ScopeStack) -> None:
        ident.type = ident.getattr("symbol").type

    def visitIntLiteral(self, expr: IntLiteral, ctx: ScopeStack) -> None:
        expr.type = INT

    def visitCall(self, expr: Call, ctx: ScopeStack) -> None:
        funcSymbol = expr.getattr("symbol")
        for (argument, paraType) in zip(expr.arguments, funcSymbol.para_type):
            argument.accept(self, ctx)
            if isinstance(argument.type, ArrayType) and isinstance(paraType, ArrayType):
                if argument.type.base != paraType.base:
                    raise DecafBadFuncCallError(expr.ident.value)
            elif argument.type != paraType:
                raise DecafBadFuncCallError(expr.ident.value)
        expr.type = funcSymbol.type

    def visitIndexExpression(self, expr: IndexExpression, ctx: ScopeStack) -> None:
        expr.base.accept(self, ctx)
        expr.index.accept(self, ctx)
        if expr.index.type != INT:
            raise DecafTypeMismatchError()
        if not isinstance(expr.base.type, ArrayType):
            raise DecafTypeMismatchError()
        expr.type = expr.base.type.base
