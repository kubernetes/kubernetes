package functions

import (
	"honnef.co/go/tools/ssa"
)

func filterDebug(instr []ssa.Instruction) []ssa.Instruction {
	var out []ssa.Instruction
	for _, ins := range instr {
		if _, ok := ins.(*ssa.DebugRef); !ok {
			out = append(out, ins)
		}
	}
	return out
}

// IsStub reports whether a function is a stub. A function is
// considered a stub if it has no instructions or exactly one
// instruction, which must be either returning only constant values or
// a panic.
func IsStub(fn *ssa.Function) bool {
	if len(fn.Blocks) == 0 {
		return true
	}
	if len(fn.Blocks) > 1 {
		return false
	}
	instrs := filterDebug(fn.Blocks[0].Instrs)
	if len(instrs) != 1 {
		return false
	}

	switch instrs[0].(type) {
	case *ssa.Return:
		// Since this is the only instruction, the return value must
		// be a constant. We consider all constants as stubs, not just
		// the zero value. This does not, unfortunately, cover zero
		// initialised structs, as these cause additional
		// instructions.
		return true
	case *ssa.Panic:
		return true
	default:
		return false
	}
}
