package rule

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"

	"github.com/mgechev/revive/lint"
)

// ImportShadowingRule lints given else constructs.
type ImportShadowingRule struct{}

// Apply applies the rule to given file.
func (r *ImportShadowingRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	importNames := map[string]struct{}{}
	for _, imp := range file.AST.Imports {
		importNames[getName(imp)] = struct{}{}
	}

	fileAst := file.AST
	walker := importShadowing{
		packageNameIdent: fileAst.Name,
		importNames:      importNames,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
		alreadySeen: map[*ast.Object]struct{}{},
	}

	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ImportShadowingRule) Name() string {
	return "import-shadowing"
}

func getName(imp *ast.ImportSpec) string {
	const pathSep = "/"
	const strDelim = `"`
	if imp.Name != nil {
		return imp.Name.Name
	}

	path := imp.Path.Value
	i := strings.LastIndex(path, pathSep)
	if i == -1 {
		return strings.Trim(path, strDelim)
	}

	return strings.Trim(path[i+1:], strDelim)
}

type importShadowing struct {
	packageNameIdent *ast.Ident
	importNames      map[string]struct{}
	onFailure        func(lint.Failure)
	alreadySeen      map[*ast.Object]struct{}
}

// Visit visits AST nodes and checks if id nodes (ast.Ident) shadow an import name
func (w importShadowing) Visit(n ast.Node) ast.Visitor {
	switch n := n.(type) {
	case *ast.AssignStmt:
		if n.Tok == token.DEFINE {
			return w // analyze variable declarations of the form id := expr
		}

		return nil // skip assigns of the form id = expr (not an id declaration)
	case *ast.CallExpr, // skip call expressions (not an id declaration)
		*ast.ImportSpec,   // skip import section subtree because we already have the list of imports
		*ast.KeyValueExpr, // skip analysis of key-val expressions ({key:value}): ids of such expressions, even the same of an import name, do not shadow the import name
		*ast.ReturnStmt,   // skip skipping analysis of returns, ids in expression were already analyzed
		*ast.SelectorExpr, // skip analysis of selector expressions (anId.otherId): because if anId shadows an import name, it was already detected, and otherId does not shadows the import name
		*ast.StructType:   // skip analysis of struct type because struct fields can not shadow an import name
		return nil
	case *ast.Ident:
		if n == w.packageNameIdent {
			return nil // skip the ident corresponding to the package name of this file
		}

		id := n.Name
		if id == "_" {
			return w // skip _ id
		}

		_, isImportName := w.importNames[id]
		_, alreadySeen := w.alreadySeen[n.Obj]
		if isImportName && !alreadySeen {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Node:       n,
				Category:   "namming",
				Failure:    fmt.Sprintf("The name '%s' shadows an import name", id),
			})

			w.alreadySeen[n.Obj] = struct{}{}
		}
	}

	return w
}
