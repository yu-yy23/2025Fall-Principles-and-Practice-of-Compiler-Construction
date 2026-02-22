import random

from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.cfg import CFG
from backend.dataflow.loc import Loc
from backend.reg.regalloc import RegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from backend.riscv.riscvasmemitter import RiscvSubroutineEmitter
from backend.subroutineinfo import SubroutineInfo
from utils.riscv import Riscv
from utils.tac.reg import Reg
from utils.tac.temp import Temp

"""
BruteRegAlloc: one kind of RegAlloc

bindings: map from temp.index to Reg

we don't need to take care of GlobalTemp here
because we can remove all the GlobalTemp in selectInstr process

1. accept：根据每个函数的 CFG 进行寄存器分配，寄存器分配结束后生成相应汇编代码
2. bind：将一个 Temp 与寄存器绑定
3. unbind：将一个 Temp 与相应寄存器解绑定
4. localAlloc：根据数据流对一个 BasicBlock 内的指令进行寄存器分配
5. allocForLoc：每一条指令进行寄存器分配
6. allocRegFor：根据数据流决定为当前 Temp 分配哪一个寄存器
"""

class BruteRegAlloc(RegAlloc):
    def __init__(self, emitter: RiscvAsmEmitter) -> None:
        super().__init__(emitter)
        self.bindings = {}
        for reg in emitter.allocatableRegs:
            reg.used = False

    def accept(self, graph: CFG, info: SubroutineInfo) -> None:
        subEmitter = RiscvSubroutineEmitter(self.emitter, info)
        # for bb in graph.iterator():
        #     print(bb.allSeq())
        #     for instr in bb.allSeq():
        #         print(instr.instr)
        #     print(bb.getLastInstr())
        for bb in graph.iterator():
            # you need to think more here
            # maybe we don't need to alloc regs for all the basic blocks
            if not graph.reachable(bb):
                continue
            if bb.label is not None:
                subEmitter.emitLabel(bb.label)
            self.localAlloc(bb, subEmitter)
        subEmitter.emitFunc()

    def bind(self, temp: Temp, reg: Reg):
        reg.used = True
        self.bindings[temp.index] = reg
        reg.occupied = True
        reg.temp = temp

    def unbind(self, temp: Temp):
        if temp.index in self.bindings:
            self.bindings[temp.index].occupied = False
            self.bindings.pop(temp.index)

    def localAlloc(self, bb: BasicBlock, subEmitter: RiscvSubroutineEmitter):
        for reg in self.emitter.allocatableRegs:
            reg.occupied = False

        # in step9, you may need to think about how to store callersave regs here
        param_idx = 0
        # print(self.emitter.printer.buffer)
        for loc in bb.allSeq():
            subEmitter.emitComment(str(loc.instr))

            if isinstance(loc.instr, Riscv.Call):
                savedRegs = []
                argRegsUsed = []

                # arguments in register
                for i in range(len(Riscv.ArgRegs)):
                    if i >= len(loc.instr.srcs):
                        break
                    argTemp = loc.instr.srcs[i]
                    argReg = Riscv.ArgRegs[i]
                    # ensure argReg is empty
                    if argReg.occupied and argReg.temp.index != argTemp.index:
                        subEmitter.emitStoreToStack(argReg)
                        self.unbind(argReg.temp)
                        savedRegs.append(argReg)
                    # load argTemp to argReg
                    if argTemp.index in self.bindings:
                        # if argTemp is already in a reg
                        argTempReg = self.bindings.get(argTemp.index)
                        if argTempReg.index != argReg.index:
                            subEmitter.emitAsm(Riscv.Move(argReg, argTempReg))
                            self.unbind(argTemp)
                    else:
                        # if argTemp is in stack
                        subEmitter.emitLoadFromStack(argReg, argTemp)
                    self.bind(argTemp, argReg)
                    argRegsUsed.append(argReg)

                for reg in argRegsUsed:
                    if reg.occupied:
                        if reg.temp.index in loc.liveOut:
                            subEmitter.emitStoreToStack(reg)
                        self.unbind(reg.temp)

                # save caller-saved regs
                for reg in Riscv.CallerSaved:
                    if reg.occupied and reg not in argRegsUsed:
                        savedRegs.append(reg)
                        subEmitter.emitStoreToStack(reg)
                        self.unbind(reg.temp)

                # arguments in stack
                for i in range(len(Riscv.ArgRegs), len(loc.instr.srcs)):
                    argTemp = loc.instr.srcs[i]
                    argTempReg = None
                    spilledTemp = None
                    if argTemp.index in self.bindings:
                        # if argTemp is already in a reg
                        argTempReg = self.bindings.get(argTemp.index)
                    
                    # considering that caller-saved regs have been stored to stack before
                    # t0 is empty here
                    if argTempReg is None:
                        argTempReg = Riscv.T0
                        subEmitter.emitLoadFromStack(argTempReg, argTemp)
                        self.bind(argTemp, argTempReg)

                    subEmitter.emitAsm(
                        Riscv.NativeStoreWord(
                            argTempReg, Riscv.SP, -4 * (len(loc.instr.srcs) - i)
                        )
                    )

                    self.unbind(argTemp)

                if len(loc.instr.srcs) > len(Riscv.ArgRegs):
                    subEmitter.emitAsm(
                        Riscv.SPAdd(
                            -4 * (len(loc.instr.srcs) - len(Riscv.ArgRegs))
                        )
                    )

                # emit the call instruction
                subEmitter.emitAsm(loc.instr)

                # save return value
                dstTemp = loc.instr.dsts[0] if len(loc.instr.dsts) > 0 else None
                dstReg = self.bindings.get(dstTemp.index, None)
                if dstTemp is not None and dstTemp.index not in self.bindings:
                    dstReg = self.allocRegFor(dstTemp, False, loc.liveOut, subEmitter)
                if dstTemp is not None and dstReg is not None:
                    subEmitter.emitAsm(Riscv.Move(dstReg, Riscv.A0))
                    self.unbind(dstTemp)
                    self.bind(dstTemp, dstReg)

                if len(loc.instr.srcs) > len(Riscv.ArgRegs):
                    subEmitter.emitAsm(
                        Riscv.SPAdd(
                            4 * (len(loc.instr.srcs) - len(Riscv.ArgRegs))
                        )
                    )
            
            elif isinstance(loc.instr, Riscv.Parameter):
                if param_idx < len(Riscv.ArgRegs):
                    self.bind(loc.instr.arg, Riscv.ArgRegs[param_idx])
                else:
                    subEmitter.offsets[loc.instr.arg.index] = 4 * (param_idx - len(Riscv.ArgRegs))
                param_idx += 1
                
            else:
                self.allocForLoc(loc, subEmitter)

        for tempindex in bb.liveOut:
            if tempindex in self.bindings:
                subEmitter.emitStoreToStack(self.bindings.get(tempindex))

        if (not bb.isEmpty()) and (bb.kind is not BlockKind.CONTINUOUS):
            self.allocForLoc(bb.locs[len(bb.locs) - 1], subEmitter)
        
        self.bindings.clear()  

    def allocForLoc(self, loc: Loc, subEmitter: RiscvSubroutineEmitter):
        instr = loc.instr
        srcRegs: list[Reg] = []
        dstRegs: list[Reg] = []

        for i in range(len(instr.srcs)):
            temp = instr.srcs[i]
            if isinstance(temp, Reg):
                srcRegs.append(temp)
            else:
                srcRegs.append(self.allocRegFor(temp, True, loc.liveIn, subEmitter))

        for i in range(len(instr.dsts)):
            temp = instr.dsts[i]
            if isinstance(temp, Reg):
                dstRegs.append(temp)
            else:
                dstRegs.append(self.allocRegFor(temp, False, loc.liveIn, subEmitter))
        instr.fillRegs(dstRegs, srcRegs)
        subEmitter.emitAsm(instr)

    def allocRegFor(
        self, temp: Temp, isRead: bool, live: set[int], subEmitter: RiscvSubroutineEmitter
    ):
        if temp.index in self.bindings:
            return self.bindings[temp.index]

        for reg in self.emitter.allocatableRegs:
            if (not reg.occupied) or (not reg.temp.index in live):
                subEmitter.emitComment(
                    "  allocate {} to {}  (read: {}):".format(
                        str(temp), str(reg), str(isRead)
                    )
                )
                if isRead:
                    subEmitter.emitLoadFromStack(reg, temp)
                if reg.occupied:
                    self.unbind(reg.temp)
                self.bind(temp, reg)
                return reg

        reg = self.emitter.allocatableRegs[
            random.randint(0, len(self.emitter.allocatableRegs) - 1)
        ]
        subEmitter.emitStoreToStack(reg)
        subEmitter.emitComment("  spill {} ({})".format(str(reg), str(reg.temp)))
        self.unbind(reg.temp)
        self.bind(temp, reg)
        subEmitter.emitComment(
            "  allocate {} to {} (read: {})".format(str(temp), str(reg), str(isRead))
        )
        if isRead:
            subEmitter.emitLoadFromStack(reg, temp)
        return reg
