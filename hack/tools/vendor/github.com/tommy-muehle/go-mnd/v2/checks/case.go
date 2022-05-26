package checks

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/analysis"

	config "github.com/tommy-muehle/go-mnd/v2/config"
)

const CaseCheck = "case"

type CaseAnalyzer struct {
	pass   *analysis.Pass
	config *config.Config
}

func NewCaseAnalyzer(pass *analysis.Pass, config *config.Config) *CaseAnalyzer {
	return &CaseAnalyzer{
		pass:   pass,
		config: config,
	}
}

func (a *CaseAnalyzer) NodeFilter() []ast.Node {
	return []ast.Node{
		(*ast.CaseClause)(nil),
	}
}

func (a *CaseAnalyzer) Check(n ast.Node) {
	caseClause, ok := n.(*ast.CaseClause)
	if !ok {
		return
	}

	for _, c := range caseClause.List {
		switch x := c.(type) {
		case *ast.BasicLit:
			if a.isMagicNumber(x) {
				a.pass.Reportf(x.Pos(), reportMsg, x.Value, CaseCheck)
			}
		case *ast.BinaryExpr:
			a.checkBinaryExpr(x)
		}
	}
}

func (a *CaseAnalyzer) checkBinaryExpr(expr *ast.BinaryExpr) {
	switch x := expr.X.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(x) {
			a.pass.Reportf(x.Pos(), reportMsg, x.Value, CaseCheck)
		}
	}

	switch y := expr.Y.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(y) {
			a.pass.Reportf(y.Pos(), reportMsg, y.Value, CaseCheck)
		}
	}
}

func (a *CaseAnalyzer) isMagicNumber(l *ast.BasicLit) bool {
	return (l.Kind == token.FLOAT || l.Kind == token.INT) && !a.config.IsIgnoredNumber(l.Value)
}
