package ssafunc

import (
	"go/types"

	"github.com/gostaticanalysis/analysisutil"
	"github.com/moricho/tparallel/pkg/ssainstr"
	"golang.org/x/tools/go/ssa"
)

// IsDeferCalled returns whether the given ssa.Function calls `defer`
func IsDeferCalled(f *ssa.Function) bool {
	for _, block := range f.Blocks {
		for _, instr := range block.Instrs {
			switch instr.(type) {
			case *ssa.Defer:
				return true
			}
		}
	}
	return false
}

// IsCalled returns whether the given ssa.Function calls `fn` func
func IsCalled(f *ssa.Function, fn *types.Func) bool {
	block := f.Blocks[0]
	for _, instr := range block.Instrs {
		called := analysisutil.Called(instr, nil, fn)
		if _, ok := ssainstr.LookupCalled(instr, fn); ok || called {
			return true
		}
	}
	return false
}
