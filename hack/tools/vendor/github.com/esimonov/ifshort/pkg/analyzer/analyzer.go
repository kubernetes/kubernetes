package analyzer

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

var maxDeclChars, maxDeclLines int

const (
	maxDeclLinesUsage = `maximum length of variable declaration measured in number of lines, after which the linter won't suggest using short syntax.
Has precedence over max-decl-chars.`
	maxDeclCharsUsage = `maximum length of variable declaration measured in number of characters, after which the linter won't suggest using short syntax.`
)

func init() {
	Analyzer.Flags.IntVar(&maxDeclLines, "max-decl-lines", 1, maxDeclLinesUsage)
	Analyzer.Flags.IntVar(&maxDeclChars, "max-decl-chars", 30, maxDeclCharsUsage)
}

// Analyzer is an analysis.Analyzer instance for ifshort linter.
var Analyzer = &analysis.Analyzer{
	Name:     "ifshort",
	Doc:      "Checks that your code uses short syntax for if-statements whenever possible.",
	Run:      run,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspector := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.FuncDecl)(nil),
	}

	inspector.Preorder(nodeFilter, func(node ast.Node) {
		fdecl := node.(*ast.FuncDecl)

		/*if fdecl.Name.Name != "notUsed_BinaryExpressionInIndex_OK" {
			return
		}*/

		if fdecl == nil || fdecl.Body == nil {
			return
		}

		candidates := getNamedOccurrenceMap(fdecl, pass)

		for _, stmt := range fdecl.Body.List {
			candidates.checkStatement(stmt, token.NoPos)
		}

		for varName := range candidates {
			for marker, occ := range candidates[varName] {
				//  If two or more vars with the same scope marker - skip them.
				if candidates.isFoundByScopeMarker(marker) {
					continue
				}

				pass.Reportf(occ.declarationPos,
					"variable '%s' is only used in the if-statement (%s); consider using short syntax",
					varName, pass.Fset.Position(occ.ifStmtPos))
			}
		}
	})
	return nil, nil
}

func (nom namedOccurrenceMap) checkStatement(stmt ast.Stmt, ifPos token.Pos) {
	switch v := stmt.(type) {
	case *ast.AssignStmt:
		for _, el := range v.Rhs {
			nom.checkExpression(el, ifPos)
		}
		if isAssign(v.Tok) {
			for _, el := range v.Lhs {
				nom.checkExpression(el, ifPos)
			}
		}
	case *ast.DeferStmt:
		for _, a := range v.Call.Args {
			nom.checkExpression(a, ifPos)
		}
	case *ast.ExprStmt:
		switch v.X.(type) {
		case *ast.CallExpr, *ast.UnaryExpr:
			nom.checkExpression(v.X, ifPos)
		}
	case *ast.ForStmt:
		for _, el := range v.Body.List {
			nom.checkStatement(el, ifPos)
		}

		if bexpr, ok := v.Cond.(*ast.BinaryExpr); ok {
			nom.checkExpression(bexpr.X, ifPos)
			nom.checkExpression(bexpr.Y, ifPos)
		}

		nom.checkStatement(v.Post, ifPos)
	case *ast.GoStmt:
		for _, a := range v.Call.Args {
			nom.checkExpression(a, ifPos)
		}
	case *ast.IfStmt:
		for _, el := range v.Body.List {
			nom.checkStatement(el, v.If)
		}
		if elseBlock, ok := v.Else.(*ast.BlockStmt); ok {
			for _, el := range elseBlock.List {
				nom.checkStatement(el, v.If)
			}
		}

		switch cond := v.Cond.(type) {
		case *ast.UnaryExpr:
			nom.checkExpression(cond.X, v.If)
		case *ast.BinaryExpr:
			nom.checkExpression(cond.X, v.If)
			nom.checkExpression(cond.Y, v.If)
		case *ast.CallExpr:
			nom.checkExpression(cond, v.If)
		}

		if init, ok := v.Init.(*ast.AssignStmt); ok {
			for _, e := range init.Rhs {
				nom.checkExpression(e, v.If)
			}
		}
	case *ast.IncDecStmt:
		nom.checkExpression(v.X, ifPos)
	case *ast.RangeStmt:
		nom.checkExpression(v.X, ifPos)
		if v.Body != nil {
			for _, e := range v.Body.List {
				nom.checkStatement(e, ifPos)
			}
		}
	case *ast.ReturnStmt:
		for _, r := range v.Results {
			nom.checkExpression(r, ifPos)
		}
	case *ast.SendStmt:
		nom.checkExpression(v.Chan, ifPos)
		nom.checkExpression(v.Value, ifPos)
	case *ast.SwitchStmt:
		nom.checkExpression(v.Tag, ifPos)

		for _, el := range v.Body.List {
			clauses, ok := el.(*ast.CaseClause)
			if !ok {
				continue
			}

			for _, c := range clauses.List {
				switch v := c.(type) {
				case *ast.BinaryExpr:
					nom.checkExpression(v.X, ifPos)
					nom.checkExpression(v.Y, ifPos)
				case *ast.Ident:
					nom.checkExpression(v, ifPos)
				}
			}

			for _, c := range clauses.Body {
				switch v := c.(type) {
				case *ast.AssignStmt:
					for _, el := range v.Lhs {
						nom.checkExpression(el, ifPos)
					}
					for _, el := range v.Rhs {
						nom.checkExpression(el, ifPos)
					}
				case *ast.ExprStmt:
					nom.checkExpression(v.X, ifPos)
				}
			}
		}
	case *ast.SelectStmt:
		for _, el := range v.Body.List {
			clause := el.(*ast.CommClause)

			nom.checkStatement(clause.Comm, ifPos)

			for _, c := range clause.Body {
				switch v := c.(type) {
				case *ast.AssignStmt:
					for _, el := range v.Lhs {
						nom.checkExpression(el, ifPos)
					}
					for _, el := range v.Rhs {
						nom.checkExpression(el, ifPos)
					}
				case *ast.ExprStmt:
					nom.checkExpression(v.X, ifPos)
				}
			}
		}
	case *ast.LabeledStmt:
		nom.checkStatement(v.Stmt, ifPos)
	}
}

func (nom namedOccurrenceMap) checkExpression(candidate ast.Expr, ifPos token.Pos) {
	switch v := candidate.(type) {
	case *ast.BinaryExpr:
		nom.checkExpression(v.X, ifPos)
		nom.checkExpression(v.Y, ifPos)
	case *ast.CallExpr:
		for _, arg := range v.Args {
			nom.checkExpression(arg, ifPos)
		}
		nom.checkExpression(v.Fun, ifPos)
		if fun, ok := v.Fun.(*ast.SelectorExpr); ok {
			nom.checkExpression(fun.X, ifPos)
		}
	case *ast.CompositeLit:
		for _, el := range v.Elts {
			switch v := el.(type) {
			case *ast.Ident, *ast.CompositeLit:
				nom.checkExpression(v, ifPos)
			case *ast.KeyValueExpr:
				nom.checkExpression(v.Key, ifPos)
				nom.checkExpression(v.Value, ifPos)
			case *ast.SelectorExpr:
				nom.checkExpression(v.X, ifPos)
			}
		}
	case *ast.FuncLit:
		for _, el := range v.Body.List {
			nom.checkStatement(el, ifPos)
		}
	case *ast.Ident:
		if _, ok := nom[v.Name]; !ok || nom[v.Name].isEmponymousKey(ifPos) {
			return
		}

		scopeMarker1 := nom[v.Name].getScopeMarkerForPosition(v.Pos())

		delete(nom[v.Name], scopeMarker1)

		for k := range nom {
			for scopeMarker2 := range nom[k] {
				if scopeMarker1 == scopeMarker2 {
					delete(nom[k], scopeMarker2)
				}
			}
		}
	case *ast.StarExpr:
		nom.checkExpression(v.X, ifPos)
	case *ast.IndexExpr:
		nom.checkExpression(v.X, ifPos)
		switch index := v.Index.(type) {
		case *ast.BinaryExpr:
			nom.checkExpression(index.X, ifPos)
		case *ast.Ident:
			nom.checkExpression(index, ifPos)
		}
	case *ast.SelectorExpr:
		nom.checkExpression(v.X, ifPos)
	case *ast.SliceExpr:
		nom.checkExpression(v.High, ifPos)
		nom.checkExpression(v.Low, ifPos)
		nom.checkExpression(v.X, ifPos)
	case *ast.TypeAssertExpr:
		nom.checkExpression(v.X, ifPos)
	case *ast.UnaryExpr:
		nom.checkExpression(v.X, ifPos)
	}
}

func isAssign(tok token.Token) bool {
	return (tok == token.ASSIGN ||
		tok == token.ADD_ASSIGN || tok == token.SUB_ASSIGN ||
		tok == token.MUL_ASSIGN || tok == token.QUO_ASSIGN || tok == token.REM_ASSIGN ||
		tok == token.AND_ASSIGN || tok == token.OR_ASSIGN || tok == token.XOR_ASSIGN || tok == token.AND_NOT_ASSIGN ||
		tok == token.SHL_ASSIGN || tok == token.SHR_ASSIGN)
}
