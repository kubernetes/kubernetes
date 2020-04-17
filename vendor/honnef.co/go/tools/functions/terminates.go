package functions

import "honnef.co/go/tools/ssa"

// Terminates reports whether fn is supposed to return, that is if it
// has at least one theoretic path that returns from the function.
// Explicit panics do not count as terminating.
func Terminates(fn *ssa.Function) bool {
	if fn.Blocks == nil {
		// assuming that a function terminates is the conservative
		// choice
		return true
	}

	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if _, ok := block.Instrs[len(block.Instrs)-1].(*ssa.Return); ok {
			return true
		}
	}
	return false
}
