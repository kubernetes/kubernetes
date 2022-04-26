package analysisutil

import (
	"golang.org/x/tools/go/ssa"
)

// IfInstr returns *ssa.If which is contained in the block b.
// If the block b has not any if instruction, IfInstr returns nil.
func IfInstr(b *ssa.BasicBlock) *ssa.If {
	if len(b.Instrs) == 0 {
		return nil
	}

	ifinstr, ok := b.Instrs[len(b.Instrs)-1].(*ssa.If)
	if !ok {
		return nil
	}

	return ifinstr
}

// Phi returns phi values which are contained in the block b.
func Phi(b *ssa.BasicBlock) []*ssa.Phi {
	var phis []*ssa.Phi
	for _, instr := range b.Instrs {
		if phi, ok := instr.(*ssa.Phi); ok {
			phis = append(phis, phi)
		} else {
			// no more phi
			break
		}
	}
	return phis
}

// Returns returns a slice of *ssa.Return in the function.
func Returns(v ssa.Value) []*ssa.Return {
	var fn *ssa.Function
	switch v := v.(type) {
	case *ssa.Function:
		fn = v
	case *ssa.MakeClosure:
		return Returns(v.Fn)
	default:
		return nil
	}

	var rets []*ssa.Return
	done := map[*ssa.BasicBlock]bool{}
	for _, b := range fn.Blocks {
		rets = append(rets, returnsInBlock(b, done)...)
	}
	return rets
}

func returnsInBlock(b *ssa.BasicBlock, done map[*ssa.BasicBlock]bool) (rets []*ssa.Return) {
	if done[b] {
		return nil
	}
	done[b] = true

	if b.Index != 0 && len(b.Preds) == 0 {
		return nil
	}

	if len(b.Instrs) != 0 {
		switch instr := b.Instrs[len(b.Instrs)-1].(type) {
		case *ssa.Return:
			rets = append(rets, instr)
		}
	}

	for _, s := range b.Succs {
		rets = append(rets, returnsInBlock(s, done)...)
	}

	return rets
}

// BinOp returns binary operator values which are contained in the block b.
func BinOp(b *ssa.BasicBlock) []*ssa.BinOp {
	var binops []*ssa.BinOp
	for _, instr := range b.Instrs {
		if binop, ok := instr.(*ssa.BinOp); ok {
			binops = append(binops, binop)
		}
	}
	return binops
}

// Used returns an instruction which uses the value in the instructions.
func Used(v ssa.Value, instrs []ssa.Instruction) ssa.Instruction {
	if len(instrs) == 0 || v.Referrers() == nil {
		return nil
	}

	for _, instr := range instrs {
		if used := usedInInstr(v, instr); used != nil {
			return used
		}
	}

	return nil
}

func usedInInstr(v ssa.Value, instr ssa.Instruction) ssa.Instruction {
	switch instr := instr.(type) {
	case *ssa.MakeClosure:
		return usedInClosure(v, instr)
	default:
		operands := instr.Operands(nil)
		for _, x := range operands {
			if x != nil && *x == v {
				return instr
			}
		}
	}

	switch v := v.(type) {
	case *ssa.UnOp:
		return usedInInstr(v.X, instr)
	}

	return nil
}

func usedInClosure(v ssa.Value, instr *ssa.MakeClosure) ssa.Instruction {
	fn, _ := instr.Fn.(*ssa.Function)
	if fn == nil {
		return nil
	}

	var fv *ssa.FreeVar
	for i := range instr.Bindings {
		if instr.Bindings[i] == v {
			fv = fn.FreeVars[i]
			break
		}
	}

	if fv == nil {
		return nil
	}

	for _, b := range fn.Blocks {
		if used := Used(fv, b.Instrs); used != nil {
			return used
		}
	}

	return nil
}
