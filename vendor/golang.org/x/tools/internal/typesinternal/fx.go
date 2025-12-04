// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/ast"
	"go/token"
	"go/types"
)

// NoEffects reports whether the expression has no side effects, i.e., it
// does not modify the memory state. This function is conservative: it may
// return false even when the expression has no effect.
func NoEffects(info *types.Info, expr ast.Expr) bool {
	noEffects := true
	ast.Inspect(expr, func(n ast.Node) bool {
		switch v := n.(type) {
		case nil, *ast.Ident, *ast.BasicLit, *ast.BinaryExpr, *ast.ParenExpr,
			*ast.SelectorExpr, *ast.IndexExpr, *ast.SliceExpr, *ast.TypeAssertExpr,
			*ast.StarExpr, *ast.CompositeLit, *ast.ArrayType, *ast.StructType,
			*ast.MapType, *ast.InterfaceType, *ast.KeyValueExpr:
			// No effect
		case *ast.UnaryExpr:
			// Channel send <-ch has effects
			if v.Op == token.ARROW {
				noEffects = false
			}
		case *ast.CallExpr:
			// Type conversion has no effects
			if !info.Types[v.Fun].IsType() {
				// TODO(adonovan): Add a case for built-in functions without side
				// effects (by using callsPureBuiltin from tools/internal/refactor/inline)

				noEffects = false
			}
		case *ast.FuncLit:
			// A FuncLit has no effects, but do not descend into it.
			return false
		default:
			// All other expressions have effects
			noEffects = false
		}

		return noEffects
	})
	return noEffects
}
