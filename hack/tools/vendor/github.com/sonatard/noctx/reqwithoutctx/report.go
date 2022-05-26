package reqwithoutctx

import (
	"go/token"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ssa"
)

type Report struct {
	Instruction ssa.Instruction
}

func (n *Report) Pos() token.Pos {
	return n.Instruction.Pos()
}

func (n *Report) Message() string {
	return "should rewrite http.NewRequestWithContext or add (*Request).WithContext"
}

func report(pass *analysis.Pass, reports []*Report) {
	for _, report := range reports {
		pass.Reportf(report.Pos(), report.Message())
	}
}
