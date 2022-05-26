package tparallel

import (
	"go/types"
	"strings"

	"github.com/gostaticanalysis/analysisutil"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/ssa"

	"github.com/moricho/tparallel/pkg/ssainstr"
)

// getTestMap gets a set of a top-level test and its sub-tests
func getTestMap(ssaanalyzer *buildssa.SSA, testTyp types.Type) map[*ssa.Function][]*ssa.Function {
	testMap := map[*ssa.Function][]*ssa.Function{}

	trun := analysisutil.MethodOf(testTyp, "Run")
	for _, f := range ssaanalyzer.SrcFuncs {
		if !strings.HasPrefix(f.Name(), "Test") || !(f.Parent() == (*ssa.Function)(nil)) {
			continue
		}
		testMap[f] = []*ssa.Function{}
		for _, block := range f.Blocks {
			for _, instr := range block.Instrs {
				called := analysisutil.Called(instr, nil, trun)

				if !called && ssainstr.HasArgs(instr, types.NewPointer(testTyp)) {
					if instrs, ok := ssainstr.LookupCalled(instr, trun); ok {
						for _, v := range instrs {
							testMap[f] = appendTestMap(testMap[f], v)
						}
					}
				} else if called {
					testMap[f] = appendTestMap(testMap[f], instr)
				}
			}
		}
	}

	return testMap
}

// appendTestMap converts ssa.Instruction to ssa.Function and append it to a given sub-test slice
func appendTestMap(subtests []*ssa.Function, instr ssa.Instruction) []*ssa.Function {
	call, ok := instr.(ssa.CallInstruction)
	if !ok {
		return subtests
	}

	ssaCall := call.Value()
	for _, arg := range ssaCall.Call.Args {
		switch arg := arg.(type) {
		case *ssa.Function:
			subtests = append(subtests, arg)
		case *ssa.MakeClosure:
			fn, _ := arg.Fn.(*ssa.Function)
			subtests = append(subtests, fn)
		}
	}

	return subtests
}
