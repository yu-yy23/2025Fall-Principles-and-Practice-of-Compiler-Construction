
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.cfg import CFG
from backend.dataflow.loc import Loc
from backend.reg.regalloc import RegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter, RiscvSubroutineEmitter
from backend.subroutineinfo import SubroutineInfo
from utils.riscv import Riscv
from utils.tac.reg import Reg
from utils.tac.temp import Temp


@dataclass(frozen=True, slots=True)
class Move:
    """临时变量之间的 move 指令（dst <- src），用临时变量编号表示。"""
    dst: int
    src: int


class IRCRegAlloc(RegAlloc):
    def __init__(self, emitter: RiscvAsmEmitter) -> None:
        """初始化
        确定可分配寄存器集合 colors
        准备各种数据结构
        """
        super().__init__(emitter)

        self.scratch_regs: List[Reg] = []
        for name in ["T0", "T1", "T2", "T3"]:
            r = getattr(Riscv, name, None)
            self.scratch_regs.append(r)

        reserved_idx: Set[int] = set(r.index for r in self.scratch_regs)
        for name in ["SP", "RA", "ZERO", "GP", "TP"]:
            rr = getattr(Riscv, name, None)
            reserved_idx.add(rr.index)

        self.colors: List[Reg] = [
            r for r in emitter.allocatableRegs
            if r.index not in reserved_idx
        ]
        self.K: int = len(self.colors)

        self._callee_idx: Set[int] = {r.index for r in getattr(Riscv, "CalleeSaved", [])}
        self._caller_idx: Set[int] = {r.index for r in getattr(Riscv, "CallerSaved", [])}

        self._callee_colors: List[Reg] = [r for r in self.colors if r.index in self._callee_idx]
        self._caller_colors: List[Reg] = [r for r in self.colors if (r.index in self._caller_idx and r.index not in self._callee_idx)]
        self._other_colors: List[Reg] = [r for r in self.colors if (r.index not in self._caller_idx and r.index not in self._callee_idx)]

        self.live_across_calls: Set[int] = set()

        self.temps: Dict[int, Temp] = {}
        self.spill_cost: Dict[int, int] = {}

        # 冲突图
        self.adj_set: Set[Tuple[int, int]] = set()
        self.adj_list: Dict[int, Set[int]] = {}
        self.degree: Dict[int, int] = {}

        # 和移动指令有关的集合
        self.move_list: Dict[int, Set[Move]] = {}
        self.worklist_moves: Set[Move] = set()
        self.active_moves: Set[Move] = set()
        self.coalesced_moves: Set[Move] = set()
        self.constrained_moves: Set[Move] = set()
        self.frozen_moves: Set[Move] = set()

        # 节点集合
        self.initial: Set[int] = set()
        self.simplify_worklist: Set[int] = set()
        self.freeze_worklist: Set[int] = set()
        self.spill_worklist: Set[int] = set()
        self.spilled_nodes: Set[int] = set()
        self.coalesced_nodes: Set[int] = set()
        self.colored_nodes: Set[int] = set()

        # 简化阶段用的栈
        self.select_stack: List[int] = []
        self.select_stack_set: Set[int] = set()

        # 合并用的 alias（类似并查集）
        self.alias: Dict[int, int] = {}

        # 最终分配结果：node -> Reg
        self.color_of: Dict[int, Reg] = {}

    def accept(self, graph: CFG, info: SubroutineInfo) -> None:
        """分配入口
        运行寄存器分配算法
        """
        subEmitter = RiscvSubroutineEmitter(self.emitter, info)
        # 先收集 Temp 对象与 spill 代价
        self._collect_temps_and_costs(graph)

        # 初始化每个节点的结构。
        self._init_state()

        # 构建冲突图与 move 集合
        self._build(graph)

        # 生成初始工作表
        self._mk_worklist()

        while (
            self.simplify_worklist
            or self.worklist_moves
            or self.freeze_worklist
            or self.spill_worklist
        ):
            if self.simplify_worklist:
                self._simplify()
            elif self.worklist_moves:
                self._coalesce()
            elif self.freeze_worklist:
                self._freeze()
            else:
                self._select_spill()

        # 图着色
        self._assign_colors()

        # 标记被调者保存寄存器的“使用”状态。
        self._mark_used_after_coloring()

        # 生成最终代码
        for bb in graph.iterator():
            if not graph.reachable(bb):
                continue
            if bb.label is not None:
                subEmitter.emitLabel(bb.label)
            self._emit_basic_block(bb, subEmitter)

        subEmitter.emitFunc()

    def _collect_temps_and_costs(self, graph: CFG) -> None:
        """
        扫描 CFG
        收集出现过的临时变量，并估算每个临时变量的 spill 代价
        同时记录跨调用仍然活跃的临时变量，选色时会偏向 callee-saved
        """
        self.temps.clear()
        self.spill_cost.clear()
        self.live_across_calls.clear()

        def add_temp_obj(t: Temp) -> None:
            """把一个 Temp 放进 self.temps"""
            if isinstance(t, Reg):
                return
            self.temps[t.index] = t

        def bb_weight(bb: BasicBlock) -> int:
            return 1

        def bump_cost(indices: Iterable[int], w: int) -> None:
            """把一组 temp 的 spill 代价统一加上 w"""
            for i in indices:
                self.spill_cost[i] = self.spill_cost.get(i, 0) + w

        for bb in graph.iterator():
            if not graph.reachable(bb):
                continue
            w = bb_weight(bb)

            for loc in bb.iterator():
                instr = loc.instr

                for op in getattr(instr, "srcs", []):
                    add_temp_obj(op)
                for op in getattr(instr, "dsts", []):
                    add_temp_obj(op)

                # 统计跨调用仍活跃的临时变量
                if isinstance(instr, Riscv.Call):
                    defs = set(instr.getWritten())
                    live_across = set(loc.liveOut) - defs
                    self.live_across_calls |= live_across

                bump_cost(instr.getRead(), w)
                bump_cost(instr.getWritten(), w)


    def _init_state(self) -> None:
        """把上一轮分配留下的状态清空，并为本轮的所有节点建好基础容器"""
        # 清空冲突图 / move / 工作表状态
        self.adj_set.clear()
        self.adj_list.clear()
        self.degree.clear()

        self.move_list.clear()
        self.worklist_moves.clear()
        self.active_moves.clear()
        self.coalesced_moves.clear()
        self.constrained_moves.clear()
        self.frozen_moves.clear()

        self.initial.clear()
        self.simplify_worklist.clear()
        self.freeze_worklist.clear()
        self.spill_worklist.clear()
        self.spilled_nodes.clear()
        self.coalesced_nodes.clear()
        self.colored_nodes.clear()

        self.select_stack.clear()
        self.select_stack_set.clear()

        self.alias.clear()
        self.color_of.clear()

        self.initial.update(self.temps.keys())

        # 初始化每个节点的容器
        for n in self.initial:
            self.adj_list.setdefault(n, set())
            self.move_list.setdefault(n, set())
            self.degree.setdefault(n, 0)

    def _build(self, graph: CFG) -> None:
        """根据活跃信息构建冲突图，并把 move 指令收集起来用于后续合并"""
        for bb in graph.iterator():
            if not graph.reachable(bb):
                continue

            live: Set[int] = set(bb.liveOut)

            for loc in bb.backwardIterator():
                instr = loc.instr
                defs = set(instr.getWritten())
                uses = set(instr.getRead())

                if self._is_move_instr(instr):
                    live -= uses

                    # 记录该 move 与其操作数的关联
                    mv = self._to_move(instr)
                    if mv is not None:
                        self.worklist_moves.add(mv)
                        for n in (defs | uses):
                            self.move_list.setdefault(n, set()).add(mv)

                    live |= defs

                # 对每个 def 与其后仍然 live 的所有变量建立冲突
                for d in defs:
                    for l in live:
                        self._add_edge(l, d)

                live -= defs
                live |= uses

    def _mk_worklist(self) -> None:
        """把初始节点按度数/是否和 move 相关分到不同工作表"""
        for n in list(self.initial):
            if self.degree.get(n, 0) >= self.K:
                self.spill_worklist.add(n)
            elif self._move_related(n):
                self.freeze_worklist.add(n)
            else:
                self.simplify_worklist.add(n)

    def _add_edge(self, u: int, v: int) -> None:
        """在干扰图里加一条无向边 (u, v)，并维护邻接表/度数"""
        if u == v:
            return
        if (u, v) in self.adj_set:
            return
        self.adj_set.add((u, v))
        self.adj_set.add((v, u))

        self.adj_list.setdefault(u, set()).add(v)
        self.adj_list.setdefault(v, set()).add(u)

        self.degree[u] = self.degree.get(u, 0) + 1
        self.degree[v] = self.degree.get(v, 0) + 1

    def _adjacent(self, n: int) -> Set[int]:
        """确定 n 当前的邻居，去掉已入栈的和已合并掉的节点"""
        adj = set(self.adj_list.get(n, set()))
        adj -= self.select_stack_set
        adj -= self.coalesced_nodes
        return adj

    def _node_moves(self, n: int) -> Set[Move]:
        """返回与 n 相关、且仍处于 active/worklist 的 move 集合"""
        s = set(self.move_list.get(n, set()))
        s &= (self.active_moves | self.worklist_moves)
        return s

    def _move_related(self, n: int) -> bool:
        """判断节点 n 目前是否还牵扯到 move"""
        return len(self._node_moves(n)) > 0

    def _enable_moves(self, nodes: Iterable[int]) -> None:
        """把 nodes 相关的 move 从 active 重新激活回 worklist"""
        for n in nodes:
            for m in self._node_moves(n):
                if m in self.active_moves:
                    self.active_moves.remove(m)
                    self.worklist_moves.add(m)

    def _decrement_degree(self, m: int) -> None:
        """
        节点度数减 1
        如果从 K 以上降到 K 以下，需要重新分类它的去向
        """
        d = self.degree.get(m, 0)
        self.degree[m] = d - 1
        if d == self.K:
            self._enable_moves({m} | self._adjacent(m))
            if m in self.spill_worklist:
                self.spill_worklist.remove(m)
            if self._move_related(m):
                self.freeze_worklist.add(m)
            else:
                self.simplify_worklist.add(m)

    def _add_worklist(self, u: int) -> None:
        """如果 u 已经“安全”（度数<K 且不再牵扯 move），就放进 simplify_worklist"""
        if (u not in self.coalesced_nodes) and (not self._move_related(u)) and self.degree.get(u, 0) < self.K:
            if u in self.freeze_worklist:
                self.freeze_worklist.remove(u)
            self.simplify_worklist.add(u)

    def _get_alias(self, n: int) -> int:
        """沿着 alias 链找到当前节点合并后的祖先"""
        while n in self.alias:
            n = self.alias[n]
        return n

    def _simplify(self) -> None:
        """简化：弹一个低度节点进栈，并让它的邻居度数递减"""
        n = self.simplify_worklist.pop()
        self.select_stack.append(n)
        self.select_stack_set.add(n)
        for m in self._adjacent(n):
            self._decrement_degree(m)

    def _coalesce(self) -> None:
        """合并：尝试把一条 move 的两端合并（保守合并），否则把它挂起"""
        m = next(iter(self.worklist_moves))
        self.worklist_moves.remove(m)

        u = self._get_alias(m.dst)
        v = self._get_alias(m.src)

        if u == v:
            self.coalesced_moves.add(m)
            self._add_worklist(u)
            return

        if (u, v) in self.adj_set:
            self.constrained_moves.add(m)
            self._add_worklist(u)
            self._add_worklist(v)
            return

        if self._conservative(self._adjacent(u) | self._adjacent(v)):
            self.coalesced_moves.add(m)
            self._combine(u, v)
            self._add_worklist(u)
        else:
            self.active_moves.add(m)

    def _conservative(self, nodes: Set[int]) -> bool:
        """保守合并判定：高于等于 K 度的邻居数 < K 才允许合并"""
        k = 0
        for n in nodes:
            if self.degree.get(n, 0) >= self.K:
                k += 1
        return k < self.K

    def _combine(self, u: int, v: int) -> None:
        """真正执行合并：把 v 吸收到 u 上，更新 alias/邻接/度数/move 列表等"""
        if v in self.freeze_worklist:
            self.freeze_worklist.remove(v)
        if v in self.spill_worklist:
            self.spill_worklist.remove(v)

        self.coalesced_nodes.add(v)
        self.alias[v] = u

        if v in self.live_across_calls:
            self.live_across_calls.add(u)

        self.move_list.setdefault(u, set()).update(self.move_list.get(v, set()))

        self._enable_moves([v])

        for t in self._adjacent(v):
            self._add_edge(t, u)
            self._decrement_degree(t)

        if self.degree.get(u, 0) >= self.K and u in self.freeze_worklist:
            self.freeze_worklist.remove(u)
            self.spill_worklist.add(u)

    def _freeze(self) -> None:
        """冻结一个节点：把它从 freeze_worklist 转到 simplify，并冻结相关 move"""
        u = next(iter(self.freeze_worklist))
        self.freeze_worklist.remove(u)
        self.simplify_worklist.add(u)
        self._freeze_moves(u)

    def _freeze_moves(self, u: int) -> None:
        """把 u 相关的 move 全部冻结掉；必要时让另一端节点变得可简化"""
        for m in list(self._node_moves(u)):
            x = m.dst
            y = m.src
            v = self._get_alias(y) if self._get_alias(x) == self._get_alias(u) else self._get_alias(x)

            if m in self.active_moves:
                self.active_moves.remove(m)
            if m in self.worklist_moves:
                self.worklist_moves.remove(m)
            self.frozen_moves.add(m)

            if (not self._move_related(v)) and self.degree.get(v, 0) < self.K:
                if v in self.freeze_worklist:
                    self.freeze_worklist.remove(v)
                self.simplify_worklist.add(v)

    def _select_spill(self) -> None:
        """选择一个最划算的 spill 候选，并把它推进 simplify"""
        best = None
        best_score = None
        for n in self.spill_worklist:
            deg = max(1, self.degree.get(n, 1))
            cost = self.spill_cost.get(n, 1)
            score = cost / deg
            if best is None or score < best_score:
                best = n
                best_score = score

        m = best
        self.spill_worklist.remove(m)
        self.simplify_worklist.add(m)
        self._freeze_moves(m)


    def _assign_colors(self) -> None:
        """真正着色：按栈逆序给节点分配寄存器；不行的就标记为 spilled"""
        while self.select_stack:
            n = self.select_stack.pop()
            self.select_stack_set.remove(n)

            ok_colors: List[Reg] = list(self.colors)

            for w in self.adj_list.get(n, set()):
                w = self._get_alias(w)
                if w in self.colored_nodes:
                    c = self.color_of[w]
                    ok_colors = [r for r in ok_colors if r.index != c.index]

            if not ok_colors:
                self.spilled_nodes.add(n)
                continue

            chosen = self._choose_color_with_bias(n, ok_colors)
            self.colored_nodes.add(n)
            self.color_of[n] = chosen

        # 已合并节点继承其 alias 的颜色
        for n in self.coalesced_nodes:
            a = self._get_alias(n)
            if a in self.color_of:
                self.color_of[n] = self.color_of[a]
            else:
                # 如果 alias 最终被 spill，则该节点也视为 spilled
                self.spilled_nodes.add(n)

    def _mark_used_after_coloring(self) -> None:
        """着色完后，把用到的物理寄存器标记为 used"""
        callee_by_idx: Dict[int, Reg] = {}
        for r in getattr(Riscv, "CalleeSaved", []):
            if isinstance(r, Reg):
                callee_by_idx[r.index] = r

        for r in self.color_of.values():
            if not isinstance(r, Reg):
                continue
            r.used = True
            cr = callee_by_idx.get(r.index)
            if cr is not None:
                cr.used = True

        for r in getattr(self, "scratch_regs", []):
            if isinstance(r, Reg):
                r.used = True

    def _choose_color_with_bias(self, n: int, ok_colors: List[Reg]) -> Reg:
        """
        在可用颜色里挑一个合适的
        优先：能消掉 move 的颜色
        其次：跨调用的更偏向 callee-saved
        """
        for mv in self.move_list.get(n, set()):
            other = mv.src if mv.dst == n else mv.dst
            other = self._get_alias(other)
            if other in self.color_of:
                preferred = self.color_of[other]
                for r in ok_colors:
                    if r.index == preferred.index:
                        return r

        ok_idx = {r.index for r in ok_colors}

        if n in self.live_across_calls:
            pref = self._callee_colors + self._caller_colors + self._other_colors
        else:
            pref = self._caller_colors + self._callee_colors + self._other_colors

        for r in pref:
            if r.index in ok_idx:
                return r

        return ok_colors[0]

    def _is_move_instr(self, instr) -> bool:
        """判断一条指令是不是要处理的 temp<-temp 的 move"""
        if not isinstance(instr, Riscv.Move):
            return False
        if len(getattr(instr, "srcs", [])) != 1 or len(getattr(instr, "dsts", [])) != 1:
            return False
        s0 = instr.srcs[0]
        d0 = instr.dsts[0]
        return (isinstance(s0, Temp) and isinstance(d0, Temp) and (not isinstance(s0, Reg)) and (not isinstance(d0, Reg)))

    def _to_move(self, instr) -> Optional[Move]:
        """把一条 move 指令转换成 Move(dst_idx, src_idx) 结构"""
        if not self._is_move_instr(instr):
            return None
        dst: Temp = instr.dsts[0]
        src: Temp = instr.srcs[0]
        return Move(dst=dst.index, src=src.index)


    def _emit_basic_block(self, bb: BasicBlock, subEmitter: RiscvSubroutineEmitter) -> None:
        """
        把一个基本块的指令按分配结果发射出来
        处理三件事：
        - 参数 Parameter 的批量搬运
        - Call 前后的保存恢复
        - spill-cache 的读写
        """
        param_idx = 0

        spill_cache: Dict[int, Reg] = {}
        spill_dirty: Set[int] = set()

        pending_params: List[Tuple[Riscv.Parameter, int]] = []

        def flush_params_if_any() -> None:
            """如果有一批 Parameter，就一次性发射"""
            nonlocal pending_params
            if pending_params:
                self._emit_parameters_batch(pending_params, subEmitter)
                pending_params = []

        def emit_loc(loc: Loc) -> None:
            """发射一条 Loc：识别 Parameter/Call/普通指令，并维护 spill cache"""
            nonlocal param_idx
            instr = loc.instr
            subEmitter.emitComment(str(instr))

            if isinstance(instr, Riscv.Parameter):
                pending_params.append((instr, param_idx))
                param_idx += 1
                return

            flush_params_if_any()

            if isinstance(instr, Riscv.Call):
                must = set(loc.liveOut)
                self._spill_cache_flush(spill_cache, spill_dirty, subEmitter, must_flush=must)
                spill_cache.clear()
                spill_dirty.clear()

                self._emit_call(loc, subEmitter)
                return

            self._emit_general_loc(loc, subEmitter, spill_cache, spill_dirty)

        has_term = (not bb.isEmpty()) and (bb.kind is not BlockKind.CONTINUOUS)
        term_loc = bb.locs[-1] if has_term else None
        body_locs = bb.locs[:-1] if has_term else bb.locs

        for loc in body_locs:
            emit_loc(loc)

        flush_params_if_any()

        if has_term:
            live_out = set(term_loc.liveOut)
            self._spill_cache_flush(spill_cache, spill_dirty, subEmitter, must_flush=live_out)
            emit_loc(term_loc)
            flush_params_if_any()
            return

        live_out = set(bb.liveOut)
        self._spill_cache_flush(spill_cache, spill_dirty, subEmitter, must_flush=live_out)


    def _emit_parameter(self, instr: Riscv.Parameter, param_idx: int, subEmitter: RiscvSubroutineEmitter) -> None:
        """处理单个 Parameter：把入参从 a0/a1/... 或栈槽搬到目标位置"""
        t = instr.arg
        rep = self._get_alias(t.index)

        if param_idx < len(Riscv.ArgRegs):
            incoming = Riscv.ArgRegs[param_idx]
            if rep in self.spilled_nodes:
                self._store_temp_from_reg(incoming, t, subEmitter)
            else:
                target = self.color_of.get(rep, incoming)
                if target.index != incoming.index:
                    subEmitter.emitAsm(Riscv.Move(target, incoming))
        else:
            subEmitter.offsets[t.index] = 4 * (param_idx - len(Riscv.ArgRegs))
            if rep not in self.spilled_nodes:
                target = self.color_of.get(rep, None)
                if target is not None:
                    subEmitter.emitLoadFromStack(target, t)

    def _emit_parameters_batch(
        self,
        params: List[Tuple[Riscv.Parameter, int]],
        subEmitter: RiscvSubroutineEmitter,
    ) -> None:
        """批量处理一串 Parameter"""
        pre_stores: List[Tuple[Reg, Temp]] = []
        reg_moves: List[Tuple[Reg, Reg]] = []

        for (instr, param_idx) in params:
            t = instr.arg
            rep = self._get_alias(t.index)

            if param_idx < len(Riscv.ArgRegs):
                incoming = Riscv.ArgRegs[param_idx]

                if rep in self.spilled_nodes:
                    pre_stores.append((incoming, t))
                else:
                    target = self.color_of.get(rep, incoming)
                    if target.index != incoming.index:
                        reg_moves.append((target, incoming))
            else:
                subEmitter.offsets[t.index] = 4 * (param_idx - len(Riscv.ArgRegs))
                if rep not in self.spilled_nodes:
                    target = self.color_of.get(rep, None)
                    if target is not None:
                        subEmitter.emitLoadFromStack(target, t)

        for (incoming, t) in pre_stores:
            self._store_temp_from_reg(incoming, t, subEmitter)

        self._emit_parallel_reg_moves(reg_moves, subEmitter)


    def _emit_call(self, loc: Loc, subEmitter: RiscvSubroutineEmitter) -> None:
        """发射一次函数调用：处理 caller-saved 保存/传参/返回值接收/栈上传参回收"""
        call = loc.instr
        assert isinstance(call, Riscv.Call)

        # 保存跨调用仍然活跃、且落在调用者保存寄存器里的临时变量
        defs = set(call.getWritten())
        live_across = set(loc.liveOut) - defs

        regs_to_save: List[Tuple[Reg, Temp]] = []
        for tid in live_across:
            rep = self._get_alias(tid)
            if rep in self.spilled_nodes:
                continue
            r = self.color_of.get(rep, None)
            if r is None:
                continue
            if r in Riscv.CallerSaved:
                regs_to_save.append((r, self._temp_obj(tid)))

        # 在覆盖参数寄存器之前先保存
        for (r, t) in regs_to_save:
            self._store_temp_from_reg(r, t, subEmitter)

        # 准备实参
        # 1）寄存器传参
        reg_moves: List[Tuple[Reg, Reg]] = []
        reg_loads: List[Tuple[Reg, Temp]] = []
        for i in range(min(len(Riscv.ArgRegs), len(call.srcs))):
            arg = call.srcs[i]
            dst_reg = Riscv.ArgRegs[i]
            if isinstance(arg, Reg):
                if arg.index != dst_reg.index:
                    reg_moves.append((dst_reg, arg))
            else:
                assert isinstance(arg, Temp)
                rep = self._get_alias(arg.index)
                if rep in self.spilled_nodes:
                    reg_loads.append((dst_reg, arg))
                else:
                    src_reg = self.color_of[rep]
                    if src_reg.index != dst_reg.index:
                        reg_moves.append((dst_reg, src_reg))

        self._emit_parallel_reg_moves(reg_moves, subEmitter)
        for (dst_reg, t) in reg_loads:
            subEmitter.emitLoadFromStack(dst_reg, t)

        # 2）栈上传参
        for i in range(len(Riscv.ArgRegs), len(call.srcs)):
            arg = call.srcs[i]
            store_reg: Reg
            if isinstance(arg, Reg):
                store_reg = arg
            else:
                assert isinstance(arg, Temp)
                rep = self._get_alias(arg.index)
                if rep in self.spilled_nodes:
                    store_reg = self._scratch(0)
                    subEmitter.emitLoadFromStack(store_reg, arg)
                else:
                    store_reg = self.color_of[rep]
            subEmitter.emitAsm(
                Riscv.NativeStoreWord(store_reg, Riscv.SP, -4 * (len(call.srcs) - i))
            )

        if len(call.srcs) > len(Riscv.ArgRegs):
            subEmitter.emitAsm(
                Riscv.SPAdd(-4 * (len(call.srcs) - len(Riscv.ArgRegs)))
            )

        subEmitter.emitAsm(call)

        # 在恢复调用者保存寄存器之前，把返回值写入目的临时变量
        if len(call.dsts) > 0:
            dst = call.dsts[0]
            if isinstance(dst, Temp):
                rep = self._get_alias(dst.index)
                if rep in self.spilled_nodes:
                    self._store_temp_from_reg(Riscv.A0, dst, subEmitter)
                else:
                    dst_reg = self.color_of[rep]
                    if dst_reg.index != Riscv.A0.index:
                        subEmitter.emitAsm(Riscv.Move(dst_reg, Riscv.A0))

        # 回收栈上传参占用的空间
        if len(call.srcs) > len(Riscv.ArgRegs):
            subEmitter.emitAsm(
                Riscv.SPAdd(4 * (len(call.srcs) - len(Riscv.ArgRegs)))
            )

        # 恢复之前保存的调用者保存寄存器
        for (r, t) in regs_to_save:
            subEmitter.emitLoadFromStack(r, t)

    def _emit_general_loc(
        self,
        loc: Loc,
        subEmitter: RiscvSubroutineEmitter,
        spill_cache: Dict[int, Reg],
        spill_dirty: Set[int],
    ) -> None:
        """发射普通指令（非 Parameter/Call），需要时通过 spill-cache 临时装载/回写"""
        instr = loc.instr

        if self._is_move_instr(instr):
            dst: Temp = instr.dsts[0]
            src: Temp = instr.srcs[0]
            d = self._get_alias(dst.index)
            s = self._get_alias(src.index)
            if d == s:
                return
            if (d not in self.spilled_nodes) and (s not in self.spilled_nodes):
                if self.color_of.get(d, None) is not None and self.color_of.get(s, None) is not None:
                    if self.color_of[d].index == self.color_of[s].index:
                        return

        src_regs: List[Reg] = []
        dst_regs: List[Reg] = []

        taken: Set[int] = set()

        for op in getattr(instr, "srcs", []):
            if isinstance(op, Reg):
                src_regs.append(op)
                taken.add(op.index)
                continue

            rep = self._get_alias(op.index)

            if rep in self.spilled_nodes:
                r = self._spill_cache_get_reg(
                    tid=op.index,
                    temp_obj=op,
                    subEmitter=subEmitter,
                    spill_cache=spill_cache,
                    spill_dirty=spill_dirty,
                    taken_reg_idx=taken,
                    need_load=True,
                )
                src_regs.append(r)
            else:
                r = self.color_of[rep]
                src_regs.append(r)
                taken.add(r.index)

        for op in getattr(instr, "dsts", []):
            if isinstance(op, Reg):
                dst_regs.append(op)
                taken.add(op.index)
                continue

            rep = self._get_alias(op.index)

            if rep in self.spilled_nodes:
                r = self._spill_cache_get_reg(
                    tid=op.index,
                    temp_obj=op,
                    subEmitter=subEmitter,
                    spill_cache=spill_cache,
                    spill_dirty=spill_dirty,
                    taken_reg_idx=taken,
                    need_load=False,
                )
                dst_regs.append(r)
                spill_dirty.add(op.index)
            else:
                r = self.color_of[rep]
                dst_regs.append(r)
                taken.add(r.index)

        instr.fillRegs(dst_regs, src_regs)
        subEmitter.emitAsm(instr)


    def _emit_parallel_reg_moves(self, moves: List[Tuple[Reg, Reg]], subEmitter: RiscvSubroutineEmitter) -> None:
        """并行寄存器搬运"""
        pending: List[Tuple[Reg, Reg]] = [(d, s) for (d, s) in moves if d.index != s.index]
        if not pending:
            return

        scratch = self._scratch(0)
        while pending:
            src_idx = {s.index for (_, s) in pending}
            for k, (d, s) in enumerate(pending):
                if d.index not in src_idx:
                    subEmitter.emitAsm(Riscv.Move(d, s))
                    pending.pop(k)
                    break
            else:
                d0, s0 = pending[0]
                subEmitter.emitAsm(Riscv.Move(scratch, s0))
                pending[0] = (d0, scratch)


    def _scratch(self, i: int) -> Reg:
        """
        拿第 i 个 scratch 寄存器
        不够时随便取一个可分配寄存器
        """
        if i < len(self.scratch_regs):
            return self.scratch_regs[i]
        return self.emitter.allocatableRegs[0]

    def _alloc_scratch(self, used: Dict[int, Reg]) -> Reg:
        """
        在 scratch_regs 里挑一个没被 used 占用的
        没有就用默认 scratch(0)
        """
        taken = {r.index for r in used.values()}
        for r in self.scratch_regs:
            if r.index not in taken:
                return r
        return self._scratch(0)

    def _store_temp_from_reg(self, reg: Reg, temp: Temp, subEmitter: RiscvSubroutineEmitter) -> None:
        """把 reg 当前的值按 temp 的栈槽位置存回去"""
        old_temp = getattr(reg, "temp", None)
        old_occ = getattr(reg, "occupied", False)
        try:
            reg.temp = temp
            reg.occupied = True
            subEmitter.emitStoreToStack(reg)
        finally:
            reg.temp = old_temp
            reg.occupied = old_occ

    def _temp_obj(self, idx: int) -> Temp:
        """通过编号拿到 Temp 对象"""
        t = self.temps.get(idx)
        if t is not None:
            return t
        try:
            t = Temp(idx)
        except Exception:
            t = Temp.__new__(Temp)
            setattr(t, "index", idx)
        self.temps[idx] = t
        return t

    def _spill_cache_evict_one(
        self,
        spill_cache: Dict[int, Reg],
        spill_dirty: Set[int],
        subEmitter: RiscvSubroutineEmitter,
    ) -> None:
        """spill-cache 里踢掉一个条目"""
        if not spill_cache:
            return
        tid, r = next(iter(spill_cache.items()))
        if tid in spill_dirty:
            self._store_temp_from_reg(r, self._temp_obj(tid), subEmitter)
            spill_dirty.remove(tid)
        spill_cache.pop(tid, None)

    def _spill_cache_flush(
        self,
        spill_cache: Dict[int, Reg],
        spill_dirty: Set[int],
        subEmitter: RiscvSubroutineEmitter,
        must_flush: Optional[Set[int]] = None,
    ) -> None:
        """把 spill-cache 里的脏值回写到栈"""
        if must_flush is None:
            targets = set(spill_dirty)
        else:
            targets = set(t for t in spill_dirty if t in must_flush)

        for tid in list(targets):
            r = spill_cache.get(tid)
            if r is None:
                spill_dirty.discard(tid)
                continue
            self._store_temp_from_reg(r, self._temp_obj(tid), subEmitter)
            spill_dirty.discard(tid)

    def _spill_cache_get_reg(
        self,
        tid: int,
        temp_obj: Temp,
        subEmitter: RiscvSubroutineEmitter,
        spill_cache: Dict[int, Reg],
        spill_dirty: Set[int],
        taken_reg_idx: Set[int],
        need_load: bool,
    ) -> Reg:
        """为 spilled temp 申请一个暂存寄存器（来自 scratch）"""
        if tid in spill_cache:
            r = spill_cache[tid]
            taken_reg_idx.add(r.index)
            return r

        for s in self.scratch_regs:
            if s.index in taken_reg_idx:
                continue

            victim = None
            for k, v in spill_cache.items():
                if v.index == s.index:
                    victim = k
                    break
            if victim is not None:
                if victim in spill_dirty:
                    self._store_temp_from_reg(s, self._temp_obj(victim), subEmitter)
                    spill_dirty.remove(victim)
                spill_cache.pop(victim, None)

            spill_cache[tid] = s
            taken_reg_idx.add(s.index)
            if need_load:
                subEmitter.emitLoadFromStack(s, temp_obj)
            return s

        self._spill_cache_evict_one(spill_cache, spill_dirty, subEmitter)
        return self._spill_cache_get_reg(
            tid, temp_obj, subEmitter, spill_cache, spill_dirty, taken_reg_idx, need_load
        )
