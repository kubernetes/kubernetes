package asciicheck

import (
	"fmt"
	"go/ast"
	"golang.org/x/tools/go/analysis"
)

func NewAnalyzer() *analysis.Analyzer {
	return &analysis.Analyzer{
		Name: "asciicheck",
		Doc:  "checks that all code identifiers does not have non-ASCII symbols in the name",
		Run:  run,
	}
}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, file := range pass.Files {
		alreadyViewed := map[*ast.Object]struct{}{}
		ast.Inspect(
			file, func(node ast.Node) bool {
				cb(pass, node, alreadyViewed)
				return true
			},
		)
	}

	return nil, nil
}

func cb(pass *analysis.Pass, n ast.Node, m map[*ast.Object]struct{}) {
	if v, ok := n.(*ast.Ident); ok {
		if _, ok := m[v.Obj]; ok {
			return
		} else {
			m[v.Obj] = struct{}{}
		}

		ch, ascii := isASCII(v.Name)
		if !ascii {
			pass.Report(
				analysis.Diagnostic{
					Pos:     v.Pos(),
					Message: fmt.Sprintf("identifier \"%s\" contain non-ASCII character: %#U", v.Name, ch),
				},
			)
		}
	}
}
