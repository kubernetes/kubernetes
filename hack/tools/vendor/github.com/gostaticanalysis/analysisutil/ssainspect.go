package analysisutil

import "golang.org/x/tools/go/ssa"

// InspectFuncs inspects functions.
func InspectFuncs(funcs []*ssa.Function, f func(i int, instr ssa.Instruction) bool) {
	for _, fun := range funcs {
		if len(fun.Blocks) == 0 {
			continue
		}
		new(instrInspector).block(fun.Blocks[0], 0, f)
	}
}

// InspectInstr inspects from i-th instruction of start block to succsessor blocks.
func InspectInstr(start *ssa.BasicBlock, i int, f func(i int, instr ssa.Instruction) bool) {
	new(instrInspector).block(start, i, f)
}

type instrInspector struct {
	done map[*ssa.BasicBlock]bool
}

func (ins *instrInspector) block(b *ssa.BasicBlock, i int, f func(i int, instr ssa.Instruction) bool) {
	if ins.done == nil {
		ins.done = map[*ssa.BasicBlock]bool{}
	}

	if b == nil || ins.done[b] || len(b.Instrs) <= i {
		return
	}

	ins.done[b] = true
	ins.instrs(i, b.Instrs[i:], f)
	for _, s := range b.Succs {
		ins.block(s, 0, f)
	}

}

func (ins *instrInspector) instrs(offset int, instrs []ssa.Instruction, f func(i int, instr ssa.Instruction) bool) {
	for i, instr := range instrs {
		if !f(offset+i, instr) {
			break
		}
	}
}
