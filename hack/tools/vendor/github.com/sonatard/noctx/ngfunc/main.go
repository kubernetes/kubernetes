package ngfunc

import (
	"go/types"

	"github.com/gostaticanalysis/analysisutil"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
)

func Run(pass *analysis.Pass) (interface{}, error) {
	ngFuncNames := []string{
		"net/http.Get",
		"net/http.Head",
		"net/http.Post",
		"net/http.PostForm",
		"(*net/http.Client).Get",
		"(*net/http.Client).Head",
		"(*net/http.Client).Post",
		"(*net/http.Client).PostForm",
	}

	ngFuncs := typeFuncs(pass, ngFuncNames)
	if len(ngFuncs) == 0 {
		return nil, nil
	}

	reportFuncs := ngCalledFuncs(pass, ngFuncs)
	report(pass, reportFuncs)

	return nil, nil
}

func ngCalledFuncs(pass *analysis.Pass, ngFuncs []*types.Func) []*Report {
	var reports []*Report

	srcFuncs := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA).SrcFuncs
	for _, sf := range srcFuncs {
		for _, b := range sf.Blocks {
			for _, instr := range b.Instrs {
				for _, ngFunc := range ngFuncs {
					if analysisutil.Called(instr, nil, ngFunc) {
						ngCalledFunc := &Report{
							Instruction: instr,
							function:    ngFunc,
						}
						reports = append(reports, ngCalledFunc)

						break
					}
				}
			}
		}
	}

	return reports
}
