package reqwithoutctx

import (
	"golang.org/x/tools/go/analysis"
)

func Run(pass *analysis.Pass) (interface{}, error) {
	analyzer := NewAnalyzer(pass)
	reports := analyzer.Exec()

	report(pass, reports)

	return nil, nil
}
