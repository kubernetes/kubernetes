package astwalk

import (
	"go/ast"
	"go/token"
	"go/types"
)

type localDefWalker struct {
	visitor LocalDefVisitor
	info    *types.Info
}

func (w *localDefWalker) WalkFile(f *ast.File) {
	for _, decl := range f.Decls {
		decl, ok := decl.(*ast.FuncDecl)
		if !ok || !w.visitor.EnterFunc(decl) {
			continue
		}
		w.walkFunc(decl)
	}
}

func (w *localDefWalker) walkFunc(decl *ast.FuncDecl) {
	w.walkSignature(decl)
	w.walkFuncBody(decl)
}

func (w *localDefWalker) walkFuncBody(decl *ast.FuncDecl) {
	ast.Inspect(decl.Body, func(x ast.Node) bool {
		switch x := x.(type) {
		case *ast.AssignStmt:
			if x.Tok != token.DEFINE {
				return false
			}
			if len(x.Lhs) != len(x.Rhs) {
				// Multi-value assignment.
				// Invariant: there is only 1 RHS.
				for i, lhs := range x.Lhs {
					id, ok := lhs.(*ast.Ident)
					if !ok || w.info.Defs[id] == nil {
						continue
					}
					def := Name{ID: id, Kind: NameVar, Index: i}
					w.visitor.VisitLocalDef(def, x.Rhs[0])
				}
			} else {
				// Simple 1-1 assignments.
				for i, lhs := range x.Lhs {
					id, ok := lhs.(*ast.Ident)
					if !ok || w.info.Defs[id] == nil {
						continue
					}
					def := Name{ID: id, Kind: NameVar}
					w.visitor.VisitLocalDef(def, x.Rhs[i])
				}
			}
			return false

		case *ast.GenDecl:
			// Decls always introduce new names.
			for _, spec := range x.Specs {
				spec, ok := spec.(*ast.ValueSpec)
				if !ok { // Ignore type/import specs
					return false
				}
				switch {
				case len(spec.Values) == 0:
					// var-specific decls without explicit init.
					for _, id := range spec.Names {
						def := Name{ID: id, Kind: NameVar}
						w.visitor.VisitLocalDef(def, nil)
					}
				case len(spec.Names) != len(spec.Values):
					// var-specific decls that assign tuple results.
					for i, id := range spec.Names {
						def := Name{ID: id, Kind: NameVar, Index: i}
						w.visitor.VisitLocalDef(def, spec.Values[0])
					}
				default:
					// Can be either var or const decl.
					kind := NameVar
					if x.Tok == token.CONST {
						kind = NameConst
					}
					for i, id := range spec.Names {
						def := Name{ID: id, Kind: kind}
						w.visitor.VisitLocalDef(def, spec.Values[i])
					}
				}
			}
			return false
		}

		return true
	})
}

func (w *localDefWalker) walkSignature(decl *ast.FuncDecl) {
	for _, p := range decl.Type.Params.List {
		for _, id := range p.Names {
			def := Name{ID: id, Kind: NameParam}
			w.visitor.VisitLocalDef(def, nil)
		}
	}
	if decl.Type.Results != nil {
		for _, p := range decl.Type.Results.List {
			for _, id := range p.Names {
				def := Name{ID: id, Kind: NameParam}
				w.visitor.VisitLocalDef(def, nil)
			}
		}
	}
	if decl.Recv != nil && len(decl.Recv.List[0].Names) != 0 {
		def := Name{ID: decl.Recv.List[0].Names[0], Kind: NameParam}
		w.visitor.VisitLocalDef(def, nil)
	}
}
