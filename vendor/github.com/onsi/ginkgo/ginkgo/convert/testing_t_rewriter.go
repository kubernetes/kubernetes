package convert

import (
	"go/ast"
)

/*
 * Rewrites any other top level funcs that receive a *testing.T param
 */
func rewriteOtherFuncsToUseGinkgoT(declarations []ast.Decl) {
	for _, decl := range declarations {
		decl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}

		for _, param := range decl.Type.Params.List {
			starExpr, ok := param.Type.(*ast.StarExpr)
			if !ok {
				continue
			}

			selectorExpr, ok := starExpr.X.(*ast.SelectorExpr)
			if !ok {
				continue
			}

			xIdent, ok := selectorExpr.X.(*ast.Ident)
			if !ok || xIdent.Name != "testing" {
				continue
			}

			if selectorExpr.Sel.Name != "T" {
				continue
			}

			param.Type = newGinkgoTInterface()
		}
	}
}

/*
 * Walks all of the nodes in the file, replacing *testing.T in struct
 * and func literal nodes. eg:
 *   type foo struct { *testing.T }
 *   var bar = func(t *testing.T) { }
 */
func walkNodesInRootNodeReplacingTestingT(rootNode *ast.File) {
	ast.Inspect(rootNode, func(node ast.Node) bool {
		if node == nil {
			return false
		}

		switch node := node.(type) {
		case *ast.StructType:
			replaceTestingTsInStructType(node)
		case *ast.FuncLit:
			replaceTypeDeclTestingTsInFuncLiteral(node)
		}

		return true
	})
}

/*
 * replaces named *testing.T inside a composite literal
 */
func replaceNamedTestingTsInKeyValueExpression(kve *ast.KeyValueExpr, testingT string) {
	ident, ok := kve.Value.(*ast.Ident)
	if !ok {
		return
	}

	if ident.Name == testingT {
		kve.Value = newGinkgoTFromIdent(ident)
	}
}

/*
 * replaces *testing.T params in a func literal with GinkgoT
 */
func replaceTypeDeclTestingTsInFuncLiteral(functionLiteral *ast.FuncLit) {
	for _, arg := range functionLiteral.Type.Params.List {
		starExpr, ok := arg.Type.(*ast.StarExpr)
		if !ok {
			continue
		}

		selectorExpr, ok := starExpr.X.(*ast.SelectorExpr)
		if !ok {
			continue
		}

		target, ok := selectorExpr.X.(*ast.Ident)
		if !ok {
			continue
		}

		if target.Name == "testing" && selectorExpr.Sel.Name == "T" {
			arg.Type = newGinkgoTInterface()
		}
	}
}

/*
 * Replaces *testing.T types inside of a struct declaration with a GinkgoT
 * eg: type foo struct { *testing.T }
 */
func replaceTestingTsInStructType(structType *ast.StructType) {
	for _, field := range structType.Fields.List {
		starExpr, ok := field.Type.(*ast.StarExpr)
		if !ok {
			continue
		}

		selectorExpr, ok := starExpr.X.(*ast.SelectorExpr)
		if !ok {
			continue
		}

		xIdent, ok := selectorExpr.X.(*ast.Ident)
		if !ok {
			continue
		}

		if xIdent.Name == "testing" && selectorExpr.Sel.Name == "T" {
			field.Type = newGinkgoTInterface()
		}
	}
}
