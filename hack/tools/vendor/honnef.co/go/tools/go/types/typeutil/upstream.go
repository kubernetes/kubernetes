package typeutil

import (
	"go/ast"
	"go/types"
	_ "unsafe"

	"golang.org/x/tools/go/types/typeutil"
)

type MethodSetCache = typeutil.MethodSetCache
type Map = typeutil.Map
type Hasher = typeutil.Hasher

func Callee(info *types.Info, call *ast.CallExpr) types.Object {
	return typeutil.Callee(info, call)
}

func IntuitiveMethodSet(T types.Type, msets *MethodSetCache) []*types.Selection {
	return typeutil.IntuitiveMethodSet(T, msets)
}

func MakeHasher() Hasher {
	return typeutil.MakeHasher()
}
