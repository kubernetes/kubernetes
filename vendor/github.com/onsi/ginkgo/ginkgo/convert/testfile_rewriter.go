package convert

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
)

/*
 * Given a file path, rewrites any tests in the Ginkgo format.
 * First, we parse the AST, and update the imports declaration.
 * Then, we walk the first child elements in the file, returning tests to rewrite.
 * A top level init func is declared, with a single Describe func inside.
 * Then the test functions to rewrite are inserted as It statements inside the Describe.
 * Finally we walk the rest of the file, replacing other usages of *testing.T
 * Once that is complete, we write the AST back out again to its file.
 */
func rewriteTestsInFile(pathToFile string) {
	fileSet := token.NewFileSet()
	rootNode, err := parser.ParseFile(fileSet, pathToFile, nil, 0)
	if err != nil {
		panic(fmt.Sprintf("Error parsing test file '%s':\n%s\n", pathToFile, err.Error()))
	}

	addGinkgoImports(rootNode)
	removeTestingImport(rootNode)

	varUnderscoreBlock := createVarUnderscoreBlock()
	describeBlock := createDescribeBlock()
	varUnderscoreBlock.Values = []ast.Expr{describeBlock}

	for _, testFunc := range findTestFuncs(rootNode) {
		rewriteTestFuncAsItStatement(testFunc, rootNode, describeBlock)
	}

	underscoreDecl := &ast.GenDecl{
		Tok:    85, // gah, magick numbers are needed to make this work
		TokPos: 14, // this tricks Go into writing "var _ = Describe"
		Specs:  []ast.Spec{varUnderscoreBlock},
	}

	imports := rootNode.Decls[0]
	tail := rootNode.Decls[1:]
	rootNode.Decls = append(append([]ast.Decl{imports}, underscoreDecl), tail...)
	rewriteOtherFuncsToUseGinkgoT(rootNode.Decls)
	walkNodesInRootNodeReplacingTestingT(rootNode)

	var buffer bytes.Buffer
	if err = format.Node(&buffer, fileSet, rootNode); err != nil {
		panic(fmt.Sprintf("Error formatting ast node after rewriting tests.\n%s\n", err.Error()))
	}

	fileInfo, err := os.Stat(pathToFile)
	if err != nil {
		panic(fmt.Sprintf("Error stat'ing file: %s\n", pathToFile))
	}

	ioutil.WriteFile(pathToFile, buffer.Bytes(), fileInfo.Mode())
}

/*
 * Given a test func named TestDoesSomethingNeat, rewrites it as
 * It("does something neat", func() { __test_body_here__ }) and adds it
 * to the Describe's list of statements
 */
func rewriteTestFuncAsItStatement(testFunc *ast.FuncDecl, rootNode *ast.File, describe *ast.CallExpr) {
	var funcIndex int = -1
	for index, child := range rootNode.Decls {
		if child == testFunc {
			funcIndex = index
			break
		}
	}

	if funcIndex < 0 {
		panic(fmt.Sprintf("Assert failed: Error finding index for test node %s\n", testFunc.Name.Name))
	}

	var block *ast.BlockStmt = blockStatementFromDescribe(describe)
	block.List = append(block.List, createItStatementForTestFunc(testFunc))
	replaceTestingTsWithGinkgoT(block, namedTestingTArg(testFunc))

	// remove the old test func from the root node's declarations
	rootNode.Decls = append(rootNode.Decls[:funcIndex], rootNode.Decls[funcIndex+1:]...)
}

/*
 * walks nodes inside of a test func's statements and replaces the usage of
 * it's named *testing.T param with GinkgoT's
 */
func replaceTestingTsWithGinkgoT(statementsBlock *ast.BlockStmt, testingT string) {
	ast.Inspect(statementsBlock, func(node ast.Node) bool {
		if node == nil {
			return false
		}

		keyValueExpr, ok := node.(*ast.KeyValueExpr)
		if ok {
			replaceNamedTestingTsInKeyValueExpression(keyValueExpr, testingT)
			return true
		}

		funcLiteral, ok := node.(*ast.FuncLit)
		if ok {
			replaceTypeDeclTestingTsInFuncLiteral(funcLiteral)
			return true
		}

		callExpr, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		replaceTestingTsInArgsLists(callExpr, testingT)

		funCall, ok := callExpr.Fun.(*ast.SelectorExpr)
		if ok {
			replaceTestingTsMethodCalls(funCall, testingT)
		}

		return true
	})
}

/*
 * rewrite t.Fail() or any other *testing.T method by replacing with T().Fail()
 * This function receives a selector expression (eg: t.Fail()) and
 * the name of the *testing.T param from the function declaration. Rewrites the
 * selector expression in place if the target was a *testing.T
 */
func replaceTestingTsMethodCalls(selectorExpr *ast.SelectorExpr, testingT string) {
	ident, ok := selectorExpr.X.(*ast.Ident)
	if !ok {
		return
	}

	if ident.Name == testingT {
		selectorExpr.X = newGinkgoTFromIdent(ident)
	}
}

/*
 * replaces usages of a named *testing.T param inside of a call expression
 * with a new GinkgoT object
 */
func replaceTestingTsInArgsLists(callExpr *ast.CallExpr, testingT string) {
	for index, arg := range callExpr.Args {
		ident, ok := arg.(*ast.Ident)
		if !ok {
			continue
		}

		if ident.Name == testingT {
			callExpr.Args[index] = newGinkgoTFromIdent(ident)
		}
	}
}
