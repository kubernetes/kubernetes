package checks

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/analysis"

	config "github.com/tommy-muehle/go-mnd/v2/config"
)

const ReturnCheck = "return"

type ReturnAnalyzer struct {
	pass   *analysis.Pass
	config *config.Config
}

func NewReturnAnalyzer(pass *analysis.Pass, config *config.Config) *ReturnAnalyzer {
	return &ReturnAnalyzer{
		pass:   pass,
		config: config,
	}
}

func (a *ReturnAnalyzer) NodeFilter() []ast.Node {
	return []ast.Node{
		(*ast.ReturnStmt)(nil),
	}
}

func (a *ReturnAnalyzer) Check(n ast.Node) {
	stmt, ok := n.(*ast.ReturnStmt)
	if !ok {
		return
	}

	for _, expr := range stmt.Results {
		switch x := expr.(type) {
		case *ast.BasicLit:
			if a.isMagicNumber(x) {
				a.pass.Reportf(x.Pos(), reportMsg, x.Value, ReturnCheck)
			}
		case *ast.BinaryExpr:
			a.checkBinaryExpr(x)
		}
	}
}

func (a *ReturnAnalyzer) checkBinaryExpr(expr *ast.BinaryExpr) {
	switch x := expr.X.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(x) {
			a.pass.Reportf(x.Pos(), reportMsg, x.Value, ReturnCheck)
		}
	}

	switch y := expr.Y.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(y) {
			a.pass.Reportf(y.Pos(), reportMsg, y.Value, ReturnCheck)
		}
	}
}

func (a *ReturnAnalyzer) isMagicNumber(l *ast.BasicLit) bool {
	return (l.Kind == token.FLOAT || l.Kind == token.INT) && !a.config.IsIgnoredNumber(l.Value)
}
