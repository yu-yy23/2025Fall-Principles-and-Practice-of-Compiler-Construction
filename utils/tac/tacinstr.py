from enum import Enum, auto, unique
from typing import Any, Optional, Union

from utils.label.label import Label
from utils.tac.reg import Reg

from .tacop import *
from .tacvisitor import TACVisitor
from .temp import Temp

class TACInstr:
    def __init__(
        self,
        kind: InstrKind,
        dsts: list[Temp],
        srcs: list[Temp],
        label: Optional[Label],
    ) -> None:
        self.kind = kind
        self.dsts = dsts.copy()
        self.srcs = srcs.copy()
        self.label = label

    def getRead(self) -> list[int]:
        return [src.index for src in self.srcs]

    def getWritten(self) -> list[int]:
        return [dst.index for dst in self.dsts]

    def isLabel(self) -> bool:
        return self.kind is InstrKind.LABEL

    def isSequential(self) -> bool:
        return self.kind == InstrKind.SEQ

    def isReturn(self) -> bool:
        return self.kind == InstrKind.RET

    def isParameter(self) -> bool:
        return self.kind == InstrKind.PARAM

    def isGlobalDecl(self) -> bool:
        return self.kind == InstrKind.GLOBAL_DECL

    def accept(self, v: TACVisitor) -> None:
        pass


# Assignment instruction.
class Assign(TACInstr):
    def __init__(self, dst: Temp, src: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.dst = dst
        self.src = src

    def __str__(self) -> str:
        return "%s = %s" % (self.dst, self.src)

    def accept(self, v: TACVisitor) -> None:
        v.visitAssign(self)


# Loading an immediate 32-bit constant.
class LoadImm4(TACInstr):
    def __init__(self, dst: Temp, value: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.value = value

    def __str__(self) -> str:
        return "%s = %d" % (self.dst, self.value)

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadImm4(self)


# Loading a symbol's address.
class LoadSymbol(TACInstr):
    def __init__(self, dst: Temp, symbol: Any) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.symbol = symbol

    def __str__(self) -> str:
        return "%s = LOAD_SYMBOL %s" % (self.dst, str(self.symbol.name))

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadSymbol(self)


class LoadData(TACInstr):
    def __init__(self, dst: Temp, addr: Temp, ofs: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [addr], None)
        self.dst = dst
        self.addr = addr
        self.offset = ofs

    def __str__(self) -> str:
        return "%s = LOAD %s, %d" % (self.dst, str(self.addr), self.offset)

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadData(self)


class StoreData(TACInstr):
    def __init__(self, src: Temp, addr: Temp, ofs: int) -> None:
        super().__init__(InstrKind.SEQ, [], [src, addr], None)
        self.src = src
        self.addr = addr
        self.offset = ofs
    
    def __str__(self) -> str:
        return "STORE %s, %s, %d" % (str(self.src), str(self.addr), self.offset)

    def accept(self, v: TACVisitor) -> None:
        v.visitStoreData(self)


# Unary operations.
class Unary(TACInstr):
    def __init__(self, op: TacUnaryOp, dst: Temp, operand: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [operand], None)
        self.op = op
        self.dst = dst
        self.operand = operand

    def __str__(self) -> str:
        return "%s = %s %s" % (
            self.dst,
            ("-" if (self.op == TacUnaryOp.NEG) else "!"),
            self.operand,
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitUnary(self)


# Binary Operations.
class Binary(TACInstr):
    def __init__(self, op: TacBinaryOp, dst: Temp, lhs: Temp, rhs: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [lhs, rhs], None)
        self.op = op
        self.dst = dst
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        opStr = {
            TacBinaryOp.ADD: "+",
            TacBinaryOp.SUB: "-",
            TacBinaryOp.MUL: "*",
            TacBinaryOp.DIV: "/",
            TacBinaryOp.MOD: "%",
            TacBinaryOp.EQU: "==",
            TacBinaryOp.NEQ: "!=",
            TacBinaryOp.SLT: "<",
            TacBinaryOp.LEQ: "<=",
            TacBinaryOp.SGT: ">",
            TacBinaryOp.GEQ: ">=",
            TacBinaryOp.LAND: "&&",
            TacBinaryOp.LOR: "||",
        }[self.op]
        return "%s = (%s %s %s)" % (self.dst, self.lhs, opStr, self.rhs)

    def accept(self, v: TACVisitor) -> None:
        v.visitBinary(self)


# Branching instruction.
class Branch(TACInstr):
    def __init__(self, target: Label) -> None:
        super().__init__(InstrKind.JMP, [], [], target)
        self.target = target

    def __str__(self) -> str:
        return "branch %s" % str(self.target)

    def accept(self, v: TACVisitor) -> None:
        v.visitBranch(self)


# Branching with conditions.
class CondBranch(TACInstr):
    def __init__(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        super().__init__(InstrKind.COND_JMP, [], [cond], target)
        self.op = op
        self.cond = cond
        self.target = target

    def __str__(self) -> str:
        return "if (%s %s) branch %s" % (
            self.cond,
            "== 0" if self.op == CondBranchOp.BEQ else "!= 0",
            str(self.target),
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitCondBranch(self)


# Return instruction.
class Return(TACInstr):
    def __init__(self, value: Optional[Temp]) -> None:
        if value is None:
            super().__init__(InstrKind.RET, [], [], None)
        else:
            super().__init__(InstrKind.RET, [], [value], None)
        self.value = value

    def __str__(self) -> str:
        return "return" if (self.value is None) else ("return " + str(self.value))

    def accept(self, v: TACVisitor) -> None:
        v.visitReturn(self)


# Annotation (used for debugging).
class Memo(TACInstr):
    def __init__(self, msg: str) -> None:
        super().__init__(InstrKind.SEQ, [], [], None)
        self.msg = msg

    def __str__(self) -> str:
        return "memo '%s'" % self.msg

    def accept(self, v: TACVisitor) -> None:
        v.visitMemo(self)


# Label (function entry or branching target).
class Mark(TACInstr):
    def __init__(self, label: Label) -> None:
        super().__init__(InstrKind.LABEL, [], [], label)

    def __str__(self) -> str:
        return "%s:" % str(self.label)

    def accept(self, v: TACVisitor) -> None:
        v.visitMark(self)


# Function call instruction.
class Call(TACInstr):
    def __init__(self, funcLabel: str, args: list[Temp], ret: Optional[Temp]) -> None:
        super().__init__(InstrKind.SEQ, [ret] if ret is not None else [], args, funcLabel)
        self.funcLabel = funcLabel
        self.args = args
        self.ret = ret

    def __str__(self) -> str:
        argsStr = ", ".join(str(arg) for arg in self.args)
        if self.ret is not None:
            return "%s = CALL %s(%s)" % (self.ret, str(self.funcLabel), argsStr)
        else:
            return "CALL %s(%s)" % (str(self.funcLabel), argsStr)

    def accept(self, v: TACVisitor) -> None:
        v.visitCall(self)

class Parameter(TACInstr):
    def __init__(self, arg: Temp) -> None:
        super().__init__(InstrKind.PARAM, [arg], [], None)
        self.arg = arg

    def __str__(self) -> str:
        return "PARAM %s" % str(self.arg)

    def accept(self, v: TACVisitor) -> None:
        v.visitParameter(self)

class Alloc(TACInstr):
    def __init__(self, dst: Temp, size: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.size = size

    def __str__(self) -> str:
        return "%s = ALLOC %d" % (self.dst, self.size)

    def accept(self, v: TACVisitor) -> None:
        v.visitAlloc(self)
