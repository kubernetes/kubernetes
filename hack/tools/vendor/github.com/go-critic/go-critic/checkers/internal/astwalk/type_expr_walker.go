package astwalk

import (
	"go/ast"
	"go/token"
	"go/types"

	"github.com/go-toolsmith/astp"
	"github.com/go-toolsmith/typep"
)

type typeExprWalker struct {
	visitor TypeExprVisitor
	info    *types.Info
}

func (w *typeExprWalker) WalkFile(f *ast.File) {
	if !w.visitor.EnterFile(f) {
		return
	}

	for _, decl := range f.Decls {
		if decl, ok := decl.(*ast.FuncDecl); ok {
			if !w.visitor.EnterFunc(decl) {
				continue
			}
		}
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if !w.visitor.EnterFunc(decl) {
				continue
			}
			w.walkSignature(decl.Type)
			ast.Inspect(decl.Body, w.walk)
		case *ast.GenDecl:
			if decl.Tok == token.IMPORT {
				continue
			}
			ast.Inspect(decl, w.walk)
		}
	}
}

func (w *typeExprWalker) visit(x ast.Expr) bool {
	w.visitor.VisitTypeExpr(x)
	return !w.visitor.skipChilds()
}

func (w *typeExprWalker) walk(x ast.Node) bool {
	switch x := x.(type) {
	case *ast.ChanType:
		return w.visit(x)
	case *ast.ParenExpr:
		if typep.IsTypeExpr(w.info, x.X) {
			return w.visit(x)
		}
		return true
	case *ast.CallExpr:
		// Pointer conversions require parenthesis around pointer type.
		// These casts are represented as call expressions.
		// Because it's impossible for the visitor to distinguish such
		// "required" parenthesis, walker skips outmost parenthesis in such cases.
		return w.inspectInner(x.Fun)
	case *ast.SelectorExpr:
		// Like with conversions, method expressions are another special.
		return w.inspectInner(x.X)
	case *ast.StarExpr:
		if typep.IsTypeExpr(w.info, x.X) {
			return w.visit(x)
		}
		return true
	case *ast.MapType:
		return w.visit(x)
	case *ast.FuncType:
		return w.visit(x)
	case *ast.StructType:
		return w.visit(x)
	case *ast.InterfaceType:
		if !w.visit(x) {
			return false
		}
		for _, method := range x.Methods.List {
			switch x := method.Type.(type) {
			case *ast.FuncType:
				w.walkSignature(x)
			default:
				// Embedded interface.
				w.walk(x)
			}
		}
		return false
	case *ast.ArrayType:
		return w.visit(x)
	}
	return true
}

func (w *typeExprWalker) inspectInner(x ast.Expr) bool {
	parens, ok := x.(*ast.ParenExpr)
	if ok && typep.IsTypeExpr(w.info, parens.X) && astp.IsStarExpr(parens.X) {
		ast.Inspect(parens.X, w.walk)
		return false
	}
	return true
}

func (w *typeExprWalker) walkSignature(typ *ast.FuncType) {
	for _, p := range typ.Params.List {
		ast.Inspect(p.Type, w.walk)
	}
	if typ.Results != nil {
		for _, p := range typ.Results.List {
			ast.Inspect(p.Type, w.walk)
		}
	}
}
