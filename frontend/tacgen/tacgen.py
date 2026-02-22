from frontend.ast.node import Optional
from frontend.ast.tree import Function, Optional
from frontend.ast import node
from frontend.ast.tree import *
from frontend.ast.visitor import Visitor
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from utils.label.blocklabel import BlockLabel
from utils.label.funclabel import FuncLabel
from utils.tac import tacop
from utils.tac.temp import Temp
from utils.tac.tacinstr import *
from utils.tac.tacfunc import TACFunc
from utils.tac.tacprog import TACProg
from utils.tac.tacvisitor import TACVisitor


"""
The TAC generation phase: translate the abstract syntax tree into three-address code.
"""


class LabelManager:
    """
    A global label manager (just a counter).
    We use this to create unique (block) labels accross functions.
    """

    def __init__(self):
        self.nextTempLabelId = 0

    def freshLabel(self) -> BlockLabel:
        self.nextTempLabelId += 1
        return BlockLabel(str(self.nextTempLabelId))


class TACFuncEmitter:
    """
    Translates a minidecaf (AST) function into low-level TAC function.
    """

    def __init__(
        self, entry: FuncLabel, numArgs: int, labelManager: LabelManager
    ) -> None:
        self.labelManager = labelManager
        self.func = TACFunc(entry, numArgs)
        self.emitLabel(entry)
        self.nextTempId = 0

        self.continueLabelStack = []
        self.breakLabelStack = []

    # To get a fresh new temporary variable.
    def freshTemp(self) -> Temp:
        temp = Temp(self.nextTempId)
        self.nextTempId += 1
        return temp

    # To get a fresh new label (for jumping and branching, etc).
    def freshLabel(self) -> Label:
        return self.labelManager.freshLabel()

    # To count how many temporary variables have been used.
    def getUsedTemp(self) -> int:
        return self.nextTempId

    # E.g., by calling 'emitAssignment', you add an assignment instruction at the end of current function.
    def emitAssignment(self, dst: Temp, src: Temp) -> Temp:
        self.func.add(Assign(dst, src))
        return src

    def emitLoad(self, value: Union[int, str]) -> Temp:
        temp = self.freshTemp()
        self.func.add(LoadImm4(temp, value))
        return temp

    def emitUnary(self, op: UnaryOp, operand: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Unary(op, temp, operand))
        return temp

    def emitUnarySelf(self, op: UnaryOp, operand: Temp) -> None:
        self.func.add(Unary(op, operand, operand))

    def emitBinary(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Binary(op, temp, lhs, rhs))
        return temp

    def emitBinarySelf(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> None:
        self.func.add(Binary(op, lhs, lhs, rhs))

    def emitBranch(self, target: Label) -> None:
        self.func.add(Branch(target))

    def emitCondBranch(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        self.func.add(CondBranch(op, cond, target))

    def emitReturn(self, value: Optional[Temp]) -> None:
        self.func.add(Return(value))

    def emitLabel(self, label: Label) -> None:
        self.func.add(Mark(label))

    def emitMemo(self, content: str) -> None:
        self.func.add(Memo(content))

    def emitRaw(self, instr: TACInstr) -> None:
        self.func.add(instr)

    def emitEnd(self) -> TACFunc:
        if (len(self.func.instrSeq) == 0) or (not self.func.instrSeq[-1].isReturn()):
            self.func.add(Return(None))
        self.func.tempUsed = self.getUsedTemp()
        return self.func

    def emitCall(self, label: Label, args: Temp) -> None:
        temp = self.freshTemp()
        self.func.add(Call(label, args, temp))
        return temp

    def emitParameter(self, arg: Temp) -> None:
        self.func.add(Parameter(arg))
        return arg

    # To open a new loop (for break/continue statements)
    def openLoop(self, breakLabel: Label, continueLabel: Label) -> None:
        self.breakLabelStack.append(breakLabel)
        self.continueLabelStack.append(continueLabel)

    # To close the current loop.
    def closeLoop(self) -> None:
        self.breakLabelStack.pop()
        self.continueLabelStack.pop()

    # To get the label for 'break' in the current loop.
    def getBreakLabel(self) -> Label:
        return self.breakLabelStack[-1]

    # To get the label for 'continue' in the current loop.
    def getContinueLabel(self) -> Label:
        return self.continueLabelStack[-1]


class TACGen():
    # Entry of this phase
    def transform(self, program: Program) -> TACProg:
        labelManager = LabelManager()
        tacFuncs = []
        for funcName, astFunc in program.functions().items():
            # in step9, you need to use real parameter count
            emitter = TACFuncEmitter(FuncLabel(funcName), len(astFunc.params), labelManager)
            # astFunc.params.accept(self, emitter)
            # astFunc.body.accept(self, emitter)
            astFunc.accept(self, emitter)
            tacFuncs.append(emitter.emitEnd())
        return TACProg(tacFuncs)

    def visitFunction(self, func: Function, mv: TACFuncEmitter) -> None:
        for param in func.params:
            param.accept(self, mv)
        func.body.accept(self, mv)

    def visitParameter(self, param: Parameter, mv: TACFuncEmitter) -> None:
        symbol = param.getattr("symbol")
        symbol.temp = mv.freshTemp()
        mv.emitParameter(symbol.temp)

    def visitBlock(self, block: Block, mv: TACFuncEmitter) -> None:
        for child in block:
            child.accept(self, mv)

    def visitReturn(self, stmt: Return, mv: TACFuncEmitter) -> None:
        stmt.expr.accept(self, mv)
        mv.emitReturn(stmt.expr.getattr("val"))

    def visitBreak(self, stmt: Break, mv: TACFuncEmitter) -> None:
        mv.emitBranch(mv.getBreakLabel())
    
    def visitContinue(self, stmt: Continue, mv: TACFuncEmitter) -> None:
        mv.emitBranch(mv.getContinueLabel())

    def visitIdentifier(self, ident: Identifier, mv: TACFuncEmitter) -> None:
        """
        1. Set the 'val' attribute of ident as the temp variable of the 'symbol' attribute of ident.
        """
        symbol = ident.getattr("symbol")
        ident.setattr("val", symbol.temp)

    def visitDeclaration(self, decl: Declaration, mv: TACFuncEmitter) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.emitAssignment to set it.
        """
        symbol = decl.getattr("symbol")
        symbol.temp = mv.freshTemp()
        if not decl.init_expr is NULL:
            decl.init_expr.accept(self, mv)
            mv.emitAssignment(symbol.temp, decl.init_expr.getattr("val"))

    def visitAssignment(self, expr: Assignment, mv: TACFuncEmitter) -> None:
        """
        1. Visit the right hand side of expr, and get the temp variable of left hand side.
        2. Use mv.emitAssignment to emit an assignment instruction.
        3. Set the 'val' attribute of expr as the value of assignment instruction.
        """
        expr.rhs.accept(self, mv)
        lhsSymbol = expr.lhs.getattr("symbol")
        mv.emitAssignment(lhsSymbol.temp, expr.rhs.getattr("val"))
        expr.setattr("val", expr.rhs.getattr("val"))

    def visitIf(self, stmt: If, mv: TACFuncEmitter) -> None:
        stmt.cond.accept(self, mv)

        if stmt.otherwise is NULL:
            skipLabel = mv.freshLabel()
            mv.emitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.emitLabel(skipLabel)
        else:
            skipLabel = mv.freshLabel()
            exitLabel = mv.freshLabel()
            mv.emitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.emitBranch(exitLabel)
            mv.emitLabel(skipLabel)
            stmt.otherwise.accept(self, mv)
            mv.emitLabel(exitLabel)

    def visitWhile(self, stmt: While, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.emitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.emitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.emitLabel(loopLabel)
        mv.emitBranch(beginLabel)
        mv.emitLabel(breakLabel)
        mv.closeLoop()

    def visitFor(self, stmt: For, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        if stmt.init is not NULL:
            stmt.init.accept(self, mv)
        mv.emitBranch(beginLabel)

        mv.emitLabel(loopLabel)
        if stmt.update is not NULL:
            stmt.update.accept(self, mv)

        mv.emitLabel(beginLabel)
        if stmt.cond is not NULL:
            stmt.cond.accept(self, mv)
        mv.emitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)
        if stmt.body is not NULL:
            stmt.body.accept(self, mv)

        mv.emitBranch(loopLabel)
        mv.emitLabel(breakLabel)
        mv.closeLoop()

    def visitUnary(self, expr: Unary, mv: TACFuncEmitter) -> None:
        expr.operand.accept(self, mv)

        op = {
            node.UnaryOp.Neg: tacop.TacUnaryOp.NEG,
            node.UnaryOp.BitNot: tacop.TacUnaryOp.BITNOT,
            node.UnaryOp.LogicNot: tacop.TacUnaryOp.LNOT,
            # You can add unary operations here.
        }[expr.op]
        expr.setattr("val", mv.emitUnary(op, expr.operand.getattr("val")))

    def visitBinary(self, expr: Binary, mv: TACFuncEmitter) -> None:
        expr.lhs.accept(self, mv)
        expr.rhs.accept(self, mv)

        op = {
            node.BinaryOp.Add: tacop.TacBinaryOp.ADD,
            node.BinaryOp.Sub: tacop.TacBinaryOp.SUB,
            node.BinaryOp.Mul: tacop.TacBinaryOp.MUL,
            node.BinaryOp.Div: tacop.TacBinaryOp.DIV,
            node.BinaryOp.Mod: tacop.TacBinaryOp.MOD,
            node.BinaryOp.EQ: tacop.TacBinaryOp.EQU,
            node.BinaryOp.NE: tacop.TacBinaryOp.NEQ,
            node.BinaryOp.LT: tacop.TacBinaryOp.SLT,
            node.BinaryOp.LE: tacop.TacBinaryOp.LEQ,
            node.BinaryOp.GT: tacop.TacBinaryOp.SGT,
            node.BinaryOp.GE: tacop.TacBinaryOp.GEQ,
            node.BinaryOp.LogicAnd: tacop.TacBinaryOp.LAND,
            node.BinaryOp.LogicOr: tacop.TacBinaryOp.LOR,
            # You can add binary operations here.
        }[expr.op]
        expr.setattr(
            "val", mv.emitBinary(op, expr.lhs.getattr("val"), expr.rhs.getattr("val"))
        )

    def visitCondExpr(self, expr: ConditionExpression, mv: TACFuncEmitter) -> None:
        """
        1. Refer to the implementation of visitIf and visitBinary.
        """
        expr.cond.accept(self, mv)

        skipLabel = mv.freshLabel()
        exitLabel = mv.freshLabel()
        result = mv.freshTemp()

        mv.emitCondBranch(
            tacop.CondBranchOp.BEQ, expr.cond.getattr("val"), skipLabel
        )

        expr.then.accept(self, mv)
        mv.emitAssignment(result, expr.then.getattr("val"))
        mv.emitBranch(exitLabel)

        mv.emitLabel(skipLabel)
        expr.otherwise.accept(self, mv)
        mv.emitAssignment(result, expr.otherwise.getattr("val"))
        mv.emitLabel(exitLabel)

        expr.setattr("val", result)

    def visitIntLiteral(self, expr: IntLiteral, mv: TACFuncEmitter) -> None:
        expr.setattr("val", mv.emitLoad(expr.value))

    def visitCall(self, expr: Call, mv: TACFuncEmitter) -> None:
        funcSymbol = expr.getattr("symbol")
        for argument in expr.arguments:
            argument.accept(self, mv)
        expr.setattr(
            "val", mv.emitCall(funcSymbol.name, [arg.getattr("val") for arg in expr.arguments])
        )

    def visitArgument(self, arg: Expression, mv: TACFuncEmitter) -> None:
        arg.expr.accept(self, mv)
