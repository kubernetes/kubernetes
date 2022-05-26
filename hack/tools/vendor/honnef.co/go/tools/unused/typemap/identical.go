package typemap

import (
	"go/types"
)

// Unlike types.Identical, receivers of Signature types are not ignored.
// Unlike types.Identical, interfaces are compared via pointer equality (except for the empty interface, which gets deduplicated).
// Unlike types.Identical, structs are compared via pointer equality.
func identical0(x, y types.Type) bool {
	if x == y {
		return true
	}

	switch x := x.(type) {
	case *types.Basic:
		// Basic types are singletons except for the rune and byte
		// aliases, thus we cannot solely rely on the x == y check
		// above. See also comment in TypeName.IsAlias.
		if y, ok := y.(*types.Basic); ok {
			return x.Kind() == y.Kind()
		}

	case *types.Array:
		// Two array types are identical if they have identical element types
		// and the same array length.
		if y, ok := y.(*types.Array); ok {
			// If one or both array lengths are unknown (< 0) due to some error,
			// assume they are the same to avoid spurious follow-on errors.
			return (x.Len() < 0 || y.Len() < 0 || x.Len() == y.Len()) && identical0(x.Elem(), y.Elem())
		}

	case *types.Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*types.Slice); ok {
			return identical0(x.Elem(), y.Elem())
		}

	case *types.Struct:
		if y, ok := y.(*types.Struct); ok {
			return x == y
		}

	case *types.Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*types.Pointer); ok {
			return identical0(x.Elem(), y.Elem())
		}

	case *types.Tuple:
		// Two tuples types are identical if they have the same number of elements
		// and corresponding elements have identical types.
		if y, ok := y.(*types.Tuple); ok {
			if x.Len() == y.Len() {
				if x != nil {
					for i := 0; i < x.Len(); i++ {
						v := x.At(i)
						w := y.At(i)
						if !identical0(v.Type(), w.Type()) {
							return false
						}
					}
				}
				return true
			}
		}

	case *types.Signature:
		// Two function types are identical if they have the same number of parameters
		// and result values, corresponding parameter and result types are identical,
		// and either both functions are variadic or neither is. Parameter and result
		// names are not required to match.
		if y, ok := y.(*types.Signature); ok {

			return x.Variadic() == y.Variadic() &&
				identical0(x.Params(), y.Params()) &&
				identical0(x.Results(), y.Results()) &&
				(x.Recv() != nil && y.Recv() != nil && identical0(x.Recv().Type(), y.Recv().Type()) || x.Recv() == nil && y.Recv() == nil)
		}

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
		if y, ok := y.(*types.Interface); ok {
			if x.NumEmbeddeds() == 0 &&
				y.NumEmbeddeds() == 0 &&
				x.NumMethods() == 0 &&
				y.NumMethods() == 0 {
				// all truly empty interfaces are the same
				return true
			}
			return x == y
		}

	case *types.Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*types.Map); ok {
			return identical0(x.Key(), y.Key()) && identical0(x.Elem(), y.Elem())
		}

	case *types.Chan:
		// Two channel types are identical if they have identical value types
		// and the same direction.
		if y, ok := y.(*types.Chan); ok {
			return x.Dir() == y.Dir() && identical0(x.Elem(), y.Elem())
		}

	case *types.Named:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		if y, ok := y.(*types.Named); ok {
			return x.Obj() == y.Obj()
		}

	case nil:

	default:
		panic("unreachable")
	}

	return false
}

// Identical reports whether x and y are identical types.
// Unlike types.Identical, receivers of Signature types are not ignored.
// Unlike types.Identical, interfaces are compared via pointer equality (except for the empty interface, which gets deduplicated).
// Unlike types.Identical, structs are compared via pointer equality.
func Identical(x, y types.Type) (ret bool) {
	return identical0(x, y)
}
