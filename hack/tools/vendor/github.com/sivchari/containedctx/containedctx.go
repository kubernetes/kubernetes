package containedctx

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const doc = "containedctx is a linter that detects struct contained context.Context field"

// Analyzer is the contanedctx analyzer
var Analyzer = &analysis.Analyzer{
	Name: "containedctx",
	Doc:  doc,
	Run:  run,
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
	},
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.StructType)(nil),
	}

	inspect.Preorder(nodeFilter, func(n ast.Node) {
		switch structTyp := n.(type) {
		case *ast.StructType:
			if structTyp.Fields.List == nil {
				return
			}
			for _, field := range structTyp.Fields.List {
				selectorExpr, ok := field.Type.(*ast.SelectorExpr)
				if !ok {
					continue
				}
				xname, ok := selectorExpr.X.(*ast.Ident)
				if !ok {
					continue
				}
				selname := selectorExpr.Sel.Name
				if xname.Name+"."+selname == "context.Context" {
					pass.Reportf(field.Pos(), "found a struct that contains a context.Context field")
				}
			}
		}
	})

	return nil, nil
}
