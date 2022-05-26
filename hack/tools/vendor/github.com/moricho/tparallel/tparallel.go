package tparallel

import (
	"go/types"

	"github.com/gostaticanalysis/analysisutil"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"

	"github.com/moricho/tparallel/pkg/ssafunc"
)

const doc = "tparallel detects inappropriate usage of t.Parallel() method in your Go test codes."

// Analyzer analyzes Go test codes whether they use t.Parallel() appropriately
// by using SSA (Single Static Assignment)
var Analyzer = &analysis.Analyzer{
	Name: "tparallel",
	Doc:  doc,
	Run:  run,
	Requires: []*analysis.Analyzer{
		buildssa.Analyzer,
	},
}

func run(pass *analysis.Pass) (interface{}, error) {
	ssaanalyzer := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA)

	obj := analysisutil.ObjectOf(pass, "testing", "T")
	if obj == nil {
		// skip checking
		return nil, nil
	}
	testTyp, testPkg := obj.Type(), obj.Pkg()

	p, _, _ := types.LookupFieldOrMethod(testTyp, true, testPkg, "Parallel")
	parallel, _ := p.(*types.Func)
	c, _, _ := types.LookupFieldOrMethod(testTyp, true, testPkg, "Cleanup")
	cleanup, _ := c.(*types.Func)

	testMap := getTestMap(ssaanalyzer, testTyp) // ex. {Test1: [TestSub1, TestSub2], Test2: [TestSub1, TestSub2, TestSub3], ...}
	for top, subs := range testMap {
		if len(subs) == 0 {
			continue
		}
		isParallelTop := ssafunc.IsCalled(top, parallel)
		isPararellSub := false
		for _, sub := range subs {
			isPararellSub = ssafunc.IsCalled(sub, parallel)
			if isPararellSub {
				break
			}
		}

		if ssafunc.IsDeferCalled(top) {
			useCleanup := ssafunc.IsCalled(top, cleanup)
			if isPararellSub && !useCleanup {
				pass.Reportf(top.Pos(), "%s should use t.Cleanup instead of defer", top.Name())
			}
		}

		if isParallelTop == isPararellSub {
			continue
		} else if isPararellSub {
			pass.Reportf(top.Pos(), "%s should call t.Parallel on the top level as well as its subtests", top.Name())
		} else if isParallelTop {
			pass.Reportf(top.Pos(), "%s's subtests should call t.Parallel", top.Name())
		}
	}

	return nil, nil
}
