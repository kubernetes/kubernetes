package goutil

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
)

func ResolveFunc(info *types.Info, callable ast.Expr) (ast.Expr, *types.Func) {
	switch callable := astutil.Unparen(callable).(type) {
	case *ast.Ident:
		sig, ok := info.ObjectOf(callable).(*types.Func)
		if !ok {
			return nil, nil
		}
		return nil, sig

	case *ast.SelectorExpr:
		sig, ok := info.ObjectOf(callable.Sel).(*types.Func)
		if !ok {
			return nil, nil
		}
		isMethod := sig.Type().(*types.Signature).Recv() != nil
		if _, ok := callable.X.(*ast.Ident); ok && !isMethod {
			return nil, sig
		}
		return callable.X, sig

	default:
		return nil, nil
	}
}
