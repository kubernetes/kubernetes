package lintutil

import (
	"go/ast"
	"go/token"
	"go/types"

	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/astp"
	"github.com/go-toolsmith/typep"
)

// Different utilities to make simple analysis over typed ast values flow.
//
// It's primitive and can't replace SSA, but the bright side is that
// it does not require building an additional IR eagerly.
// Expected to be used sparingly inside a few checkers.
//
// If proven really useful, can be moved to go-toolsmith library.

// IsImmutable reports whether n can be midified through any operation.
func IsImmutable(info *types.Info, n ast.Expr) bool {
	if astp.IsBasicLit(n) {
		return true
	}
	tv, ok := info.Types[n]
	return ok && !tv.Assignable() && !tv.Addressable()
}

// CouldBeMutated reports whether dst can be modified inside body.
//
// Note that it does not take already existing pointers to dst.
// An example of safe and correct usage is checking of something
// that was just defined, so the dst is a result of that definition.
func CouldBeMutated(info *types.Info, body ast.Node, dst ast.Expr) bool {
	if IsImmutable(info, dst) { // Fast path.
		return false
	}

	// We don't track pass-by-value.
	// If it's already a pointer, passing it by value
	// means that there can be a potential indirect modification.
	//
	// It's possible to be less conservative here and find at least
	// one such value pass before giving up.
	if typep.IsPointer(info.TypeOf(dst)) {
		return true
	}

	var isDst func(x ast.Expr) bool
	if dst, ok := dst.(*ast.Ident); ok {
		// Identifier can be shadowed,
		// so we need to check the object as well.
		obj := info.ObjectOf(dst)
		if obj == nil {
			return true // Being conservative
		}
		isDst = func(x ast.Expr) bool {
			id, ok := x.(*ast.Ident)
			return ok && id.Name == dst.Name && info.ObjectOf(id) == obj
		}
	} else {
		isDst = func(x ast.Expr) bool {
			return astequal.Expr(dst, x)
		}
	}

	return ContainsNode(body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.UnaryExpr:
			if n.Op == token.AND && isDst(n.X) {
				return true // Address taken
			}
		case *ast.AssignStmt:
			for _, lhs := range n.Lhs {
				if isDst(lhs) {
					return true
				}
			}
		case *ast.IncDecStmt:
			// Incremented or decremented.
			return isDst(n.X)
		}
		return false
	})
}
