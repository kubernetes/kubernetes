package checks

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/analysis"

	config "github.com/tommy-muehle/go-mnd/v2/config"
)

const OperationCheck = "operation"

type OperationAnalyzer struct {
	pass   *analysis.Pass
	config *config.Config
}

func NewOperationAnalyzer(pass *analysis.Pass, config *config.Config) *OperationAnalyzer {
	return &OperationAnalyzer{
		pass:   pass,
		config: config,
	}
}

func (a *OperationAnalyzer) NodeFilter() []ast.Node {
	return []ast.Node{
		(*ast.AssignStmt)(nil),
		(*ast.ParenExpr)(nil),
	}
}

func (a *OperationAnalyzer) Check(n ast.Node) {
	switch expr := n.(type) {
	case *ast.ParenExpr:
		switch x := expr.X.(type) {
		case *ast.BinaryExpr:
			a.checkBinaryExpr(x)
		}
	case *ast.AssignStmt:
		for _, y := range expr.Rhs {
			switch x := y.(type) {
			case *ast.BinaryExpr:
				switch xExpr := x.X.(type) {
				case *ast.BinaryExpr:
					a.checkBinaryExpr(xExpr)
				}
				switch yExpr := x.Y.(type) {
				case *ast.BinaryExpr:
					a.checkBinaryExpr(yExpr)
				}

				a.checkBinaryExpr(x)
			}
		}
	}
}

func (a *OperationAnalyzer) checkBinaryExpr(expr *ast.BinaryExpr) {
	switch x := expr.X.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(x) {
			a.pass.Reportf(x.Pos(), reportMsg, x.Value, OperationCheck)
		}
	}

	switch y := expr.Y.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(y) {
			a.pass.Reportf(y.Pos(), reportMsg, y.Value, OperationCheck)
		}
	}
}

func (a *OperationAnalyzer) isMagicNumber(l *ast.BasicLit) bool {
	return (l.Kind == token.FLOAT || l.Kind == token.INT) && !a.config.IsIgnoredNumber(l.Value)
}
