package typeutil

import (
	"go/types"
)

// Identical reports whether x and y are identical types.
// Unlike types.Identical, receivers of Signature types are not ignored.
// Unlike types.Identical, interfaces are compared via pointer equality (except for the empty interface, which gets deduplicated).
// Unlike types.Identical, structs are compared via pointer equality.
func Identical(x, y types.Type) (ret bool) {
	if !types.Identical(x, y) {
		return false
	}

	switch x := x.(type) {
	case *types.Struct:
		y, ok := y.(*types.Struct)
		if !ok {
			// should be impossible
			return true
		}
		return x == y
	case *types.Interface:
		// The issue with interfaces, typeutil.Map and types.Identical
		//
		// types.Identical, when comparing two interfaces, only looks at the set
		// of all methods, not differentiating between implicit (embedded) and
		// explicit methods.
		//
		// When we see the following two types, in source order
		//
		// type I1 interface { foo() }
		// type I2 interface { I1 }
		//
		// then we will first correctly process I1 and its underlying type. When
		// we get to I2, we will see that its underlying type is identical to
		// that of I1 and not process it again. This, however, means that we will
		// not record the fact that I2 embeds I1. If only I2 is reachable via the
		// graph root, then I1 will not be considered used.
		//
		// We choose to be lazy and compare interfaces by their
		// pointers. This will obviously miss identical interfaces,
		// but this only has a runtime cost, it doesn't affect
		// correctness.
		y, ok := y.(*types.Interface)
		if !ok {
			// should be impossible
			return true
		}
		if x.NumEmbeddeds() == 0 &&
			y.NumEmbeddeds() == 0 &&
			x.NumMethods() == 0 &&
			y.NumMethods() == 0 {
			// all truly empty interfaces are the same
			return true
		}
		return x == y
	case *types.Signature:
		y, ok := y.(*types.Signature)
		if !ok {
			// should be impossible
			return true
		}
		if x.Recv() == y.Recv() {
			return true
		}
		if x.Recv() == nil || y.Recv() == nil {
			return false
		}
		return Identical(x.Recv().Type(), y.Recv().Type())
	default:
		return true
	}
}
