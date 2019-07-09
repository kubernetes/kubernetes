package godog

import "go/ast"

func astContexts(f *ast.File) []string {
	var contexts []string
	for _, d := range f.Decls {
		switch fun := d.(type) {
		case *ast.FuncDecl:
			for _, param := range fun.Type.Params.List {
				switch expr := param.Type.(type) {
				case *ast.StarExpr:
					switch x := expr.X.(type) {
					case *ast.Ident:
						if x.Name == "Suite" {
							contexts = append(contexts, fun.Name.Name)
						}
					case *ast.SelectorExpr:
						switch t := x.X.(type) {
						case *ast.Ident:
							if t.Name == "godog" && x.Sel.Name == "Suite" {
								contexts = append(contexts, fun.Name.Name)
							}
						}
					}
				}
			}
		}
	}
	return contexts
}
