package checks

import (
	"go/ast"
	"go/token"
	"strconv"
	"sync"

	"golang.org/x/tools/go/analysis"

	"github.com/tommy-muehle/go-mnd/v2/config"
)

const ArgumentCheck = "argument"

// constantDefinitions is used to save lines (by number) which contain a constant definition.
var constantDefinitions = map[string]bool{}
var mu sync.RWMutex

type ArgumentAnalyzer struct {
	config *config.Config
	pass   *analysis.Pass
}

func NewArgumentAnalyzer(pass *analysis.Pass, config *config.Config) *ArgumentAnalyzer {
	return &ArgumentAnalyzer{
		pass:   pass,
		config: config,
	}
}

func (a *ArgumentAnalyzer) NodeFilter() []ast.Node {
	return []ast.Node{
		(*ast.GenDecl)(nil),
		(*ast.CallExpr)(nil),
	}
}

func (a *ArgumentAnalyzer) Check(n ast.Node) {
	switch expr := n.(type) {
	case *ast.CallExpr:
		a.checkCallExpr(expr)
	case *ast.GenDecl:
		if expr.Tok != token.CONST {
			return
		}

		for _, x := range expr.Specs {
			pos := a.pass.Fset.Position(x.Pos())

			mu.Lock()
			constantDefinitions[pos.Filename+":"+strconv.Itoa(pos.Line)] = true
			mu.Unlock()
		}
	}
}

func (a *ArgumentAnalyzer) checkCallExpr(expr *ast.CallExpr) {
	pos := a.pass.Fset.Position(expr.Pos())

	mu.RLock()
	ok := constantDefinitions[pos.Filename+":"+strconv.Itoa(pos.Line)]
	mu.RUnlock()

	if ok {
		return
	}

	switch f := expr.Fun.(type) {
	case *ast.SelectorExpr:
		switch prefix := f.X.(type) {
		case *ast.Ident:
			if a.config.IsIgnoredFunction(prefix.Name + "." + f.Sel.Name) {
				return
			}
		}
	case *ast.Ident:
		if a.config.IsIgnoredFunction(f.Name) {
			return
		}
	}

	for i, arg := range expr.Args {
		switch x := arg.(type) {
		case *ast.BasicLit:
			if !a.isMagicNumber(x) {
				continue
			}
			// If it's a magic number and has no previous element, report it
			if i == 0 {
				a.pass.Reportf(x.Pos(), reportMsg, x.Value, ArgumentCheck)
			} else {
				// Otherwise check all args
				switch expr.Args[i].(type) {
				case *ast.BasicLit:
					if a.isMagicNumber(x) {
						a.pass.Reportf(x.Pos(), reportMsg, x.Value, ArgumentCheck)
					}
				}
			}
		case *ast.BinaryExpr:
			a.checkBinaryExpr(x)
		}
	}
}

func (a *ArgumentAnalyzer) checkBinaryExpr(expr *ast.BinaryExpr) {
	switch x := expr.X.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(x) {
			a.pass.Reportf(x.Pos(), reportMsg, x.Value, ArgumentCheck)
		}
	}

	switch y := expr.Y.(type) {
	case *ast.BasicLit:
		if a.isMagicNumber(y) {
			a.pass.Reportf(y.Pos(), reportMsg, y.Value, ArgumentCheck)
		}
	}
}

func (a *ArgumentAnalyzer) isMagicNumber(l *ast.BasicLit) bool {
	return (l.Kind == token.FLOAT || l.Kind == token.INT) && !a.config.IsIgnoredNumber(l.Value)
}
