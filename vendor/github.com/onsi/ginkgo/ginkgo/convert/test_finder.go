package convert

import (
	"go/ast"
	"regexp"
)

/*
 * Given a root node, walks its top level statements and returns
 * points to function nodes to rewrite as It statements.
 * These functions, according to Go testing convention, must be named
 * TestWithCamelCasedName and receive a single *testing.T argument.
 */
func findTestFuncs(rootNode *ast.File) (testsToRewrite []*ast.FuncDecl) {
	testNameRegexp := regexp.MustCompile("^Test[0-9A-Z].+")

	ast.Inspect(rootNode, func(node ast.Node) bool {
		if node == nil {
			return false
		}

		switch node := node.(type) {
		case *ast.FuncDecl:
			matches := testNameRegexp.MatchString(node.Name.Name)

			if matches && receivesTestingT(node) {
				testsToRewrite = append(testsToRewrite, node)
			}
		}

		return true
	})

	return
}

/*
 * convenience function that looks at args to a function and determines if its
 * params include an argument of type  *testing.T
 */
func receivesTestingT(node *ast.FuncDecl) bool {
	if len(node.Type.Params.List) != 1 {
		return false
	}

	base, ok := node.Type.Params.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}

	intermediate := base.X.(*ast.SelectorExpr)
	isTestingPackage := intermediate.X.(*ast.Ident).Name == "testing"
	isTestingT := intermediate.Sel.Name == "T"

	return isTestingPackage && isTestingT
}
