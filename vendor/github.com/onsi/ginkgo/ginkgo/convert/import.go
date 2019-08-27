package convert

import (
	"fmt"
	"go/ast"
)

/*
 * Given the root node of an AST, returns the node containing the
 * import statements for the file.
 */
func importsForRootNode(rootNode *ast.File) (imports *ast.GenDecl, err error) {
	for _, declaration := range rootNode.Decls {
		decl, ok := declaration.(*ast.GenDecl)
		if !ok || len(decl.Specs) == 0 {
			continue
		}

		_, ok = decl.Specs[0].(*ast.ImportSpec)
		if ok {
			imports = decl
			return
		}
	}

	err = fmt.Errorf("Could not find imports for root node:\n\t%#v\n", rootNode)
	return
}

/*
 * Removes "testing" import, if present
 */
func removeTestingImport(rootNode *ast.File) {
	importDecl, err := importsForRootNode(rootNode)
	if err != nil {
		panic(err.Error())
	}

	var index int
	for i, importSpec := range importDecl.Specs {
		importSpec := importSpec.(*ast.ImportSpec)
		if importSpec.Path.Value == "\"testing\"" {
			index = i
			break
		}
	}

	importDecl.Specs = append(importDecl.Specs[:index], importDecl.Specs[index+1:]...)
}

/*
 * Adds import statements for onsi/ginkgo, if missing
 */
func addGinkgoImports(rootNode *ast.File) {
	importDecl, err := importsForRootNode(rootNode)
	if err != nil {
		panic(err.Error())
	}

	if len(importDecl.Specs) == 0 {
		// TODO: might need to create a import decl here
		panic("unimplemented : expected to find an imports block")
	}

	needsGinkgo := true
	for _, importSpec := range importDecl.Specs {
		importSpec, ok := importSpec.(*ast.ImportSpec)
		if !ok {
			continue
		}

		if importSpec.Path.Value == "\"github.com/onsi/ginkgo\"" {
			needsGinkgo = false
		}
	}

	if needsGinkgo {
		importDecl.Specs = append(importDecl.Specs, createImport(".", "\"github.com/onsi/ginkgo\""))
	}
}

/*
 * convenience function to create an import statement
 */
func createImport(name, path string) *ast.ImportSpec {
	return &ast.ImportSpec{
		Name: &ast.Ident{Name: name},
		Path: &ast.BasicLit{Kind: 9, Value: path},
	}
}
