package ngfunc

import (
	"fmt"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ssa"
)

type Report struct {
	Instruction ssa.Instruction
	function    *types.Func
}

func (n *Report) Pos() token.Pos {
	return n.Instruction.Pos()
}

func (n *Report) Message() string {
	return fmt.Sprintf("%s must not be called", n.function.FullName())
}

func report(pass *analysis.Pass, reports []*Report) {
	for _, report := range reports {
		pass.Reportf(report.Pos(), report.Message())
	}
}
