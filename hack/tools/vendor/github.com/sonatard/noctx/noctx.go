package noctx

import (
	"github.com/sonatard/noctx/ngfunc"
	"github.com/sonatard/noctx/reqwithoutctx"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
)

var Analyzer = &analysis.Analyzer{
	Name: "noctx",
	Doc:  Doc,
	Run:  run,
	Requires: []*analysis.Analyzer{
		buildssa.Analyzer,
	},
}

const Doc = "noctx finds sending http request without context.Context"

func run(pass *analysis.Pass) (interface{}, error) {
	if _, err := ngfunc.Run(pass); err != nil {
		return nil, err
	}

	if _, err := reqwithoutctx.Run(pass); err != nil {
		return nil, err
	}

	return nil, nil
}
