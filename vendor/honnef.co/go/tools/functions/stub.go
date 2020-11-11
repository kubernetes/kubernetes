package functions

import (
	"honnef.co/go/tools/ir"
)

// IsStub reports whether a function is a stub. A function is
// considered a stub if it has no instructions or if all it does is
// return a constant value.
func IsStub(fn *ir.Function) bool {
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			switch instr.(type) {
			case *ir.Const:
				// const naturally has no side-effects
			case *ir.Panic:
				// panic is a stub if it only uses constants
			case *ir.Return:
				// return is a stub if it only uses constants
			case *ir.DebugRef:
			case *ir.Jump:
				// if there are no disallowed instructions, then we're
				// only jumping to the exit block (or possibly
				// somewhere else that's stubby?)
			default:
				// all other instructions are assumed to do actual work
				return false
			}
		}
	}
	return true
}
