from typing import Protocol, TypeVar, cast

from frontend.ast.node import Node, NullType
from frontend.ast.tree import *
from frontend.ast.visitor import RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.scope.scopestack import ScopeStack
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract 
syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # Global scope. You don't have to consider it until Step 6.
        program.globalScope = GlobalScope
        ctx = ScopeStack(Scope(program.globalScope))

        program.accept(self, ctx)
        return program

    def visitProgram(self, program: Program, ctx: ScopeStack) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError

        for func in program.functions_list():
            func.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: ScopeStack) -> None:
        funcSymbol = ctx.lookup(func.ident.value)
        if funcSymbol is not None:
            if funcSymbol.parameterNum != len(func.params):
                raise DecafDeclConflictError(func.ident.value)
            for (param, paraType) in zip(func.params, funcSymbol.para_types):
                if param.var_t.type != paraType:
                    raise DecafDeclConflictError(func.ident.value)
            if funcSymbol.return_type != func.ret_t.type:
                raise DecafDeclConflictError(func.ident.value)
            if func.body is not NULL:
                raise DecafDeclConflictError(func.ident.value)
            func.setattr("symbol", funcSymbol)
            return

        scope = Scope(ScopeKind.FUNC_DECL)
        funcSymbol = FuncSymbol(func.ident.value, func.ret_t.type, scope)
        ctx.declare(funcSymbol)
        func.setattr("symbol", funcSymbol)

        ctx.push(scope)
        for param in func.params:
            funcSymbol.addParaType(param.var_t.type)
            param.accept(self, ctx)
        func.body.accept(self, ctx)
        ctx.pop()
    
    def visitParameter(self, param: Parameter, ctx: ScopeStack) -> None:
        varSymbol = ctx.top().lookup(param.ident.value)
        if varSymbol is not None:
            raise DecafDeclConflictError(param.ident.value)

        varSymbol = VarSymbol(param.ident.value, param.var_t.type)
        ctx.declare(varSymbol)
        param.setattr("symbol", varSymbol)

    def visitBlock(self, block: Block, ctx: ScopeStack) -> None:
        ctx.push(Scope(ScopeKind.LOCAL))
        for child in block:
            child.accept(self, ctx)
        ctx.pop()

    def visitReturn(self, stmt: Return, ctx: ScopeStack) -> None:
        stmt.expr.accept(self, ctx)

    def visitFor(self, stmt: For, ctx: ScopeStack) -> None:
        """
        1. Open a local scope for stmt.init.
        2. Visit stmt.init, stmt.cond, stmt.update.
        3. Open a loop in ctx (for validity checking of break/continue)
        4. Visit body of the loop.
        5. Close the loop and the local scope.
        """
        ctx.push(Scope(ScopeKind.LOCAL))

        if stmt.init is not NULL:
            stmt.init.accept(self, ctx)
        if stmt.cond is not NULL:
            stmt.cond.accept(self, ctx)
        else:
            stmt.cond = IntLiteral(1)
        if stmt.update is not NULL:
            stmt.update.accept(self, ctx)

        ctx.openLoop()

        stmt.body.accept(self, ctx)

        ctx.closeLoop()
        ctx.pop()

    def visitIf(self, stmt: If, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()

    def visitBreak(self, stmt: Break, ctx: ScopeStack) -> None:
        """
        You need to check if it is currently within the loop.
        To do this, you may need to check 'visitWhile'.

        if not in a loop:
            raise DecafBreakOutsideLoopError()
        """
        if ctx.getLoopDepth() == 0:
            raise DecafBreakOutsideLoopError()

    def visitContinue(self, stmt: Continue, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBreak.
        """
        if ctx.getLoopDepth() == 0:
            raise DecafBreakOutsideLoopError()

    def visitDeclaration(self, decl: Declaration, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        # varSymbol = ctx.top().lookup(decl.ident.value)
        varSymbol = ctx.lookupInFuncScope(decl.ident.value)
        if varSymbol is not None:
            raise DecafDeclConflictError(decl.ident.value)

        varSymbol = VarSymbol(decl.ident.value, decl.var_t.type)
        ctx.declare(varSymbol)
        decl.setattr("symbol", varSymbol)
        if not decl.init_expr is NULL:
            decl.init_expr.accept(self, ctx)


    def visitAssignment(self, expr: Assignment, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitUnary(self, expr: Unary, ctx: ScopeStack) -> None:
        expr.operand.accept(self, ctx)

    def visitBinary(self, expr: Binary, ctx: ScopeStack) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCondExpr(self, expr: ConditionExpression, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)

    def visitIdentifier(self, ident: Identifier, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        varSymbol = ctx.lookup(ident.value)
        if varSymbol is None:
            raise DecafUndefinedVarError(ident.value)

        ident.setattr("symbol", varSymbol)

    def visitIntLiteral(self, expr: IntLiteral, ctx: ScopeStack) -> None:
        value = expr.value
        if value > MAX_INT:
            raise DecafBadIntValueError(value)

    def visitCall(self, expr: Call, ctx: ScopeStack) -> None:
        funcSymbol = ctx.lookup(expr.ident.value)
        if funcSymbol is None:
            raise DecafUndefinedFuncError(expr.ident.value)
        if funcSymbol.parameterNum != len(expr.arguments):
            raise DecafBadFuncCallError(expr.ident.value)
        # for i, argument in enumerate(expr.arguments):
        #     if funcSymbol.getParaType(i) != argument.type:
        #         raise DecafBadFuncCallError(expr.ident.value)

        expr.setattr("symbol", funcSymbol)
        for argument in expr.arguments:
            argument.accept(self, ctx)
