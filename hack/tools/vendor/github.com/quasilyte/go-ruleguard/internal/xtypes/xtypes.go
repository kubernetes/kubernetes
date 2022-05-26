package xtypes

import (
	"go/types"
)

// Implements reports whether type v implements iface.
//
// Unlike types.Implements(), it permits X and Y named types
// to be considered identical even if their addresses are different.
func Implements(v types.Type, iface *types.Interface) bool {
	if iface.Empty() {
		return true
	}

	if v, _ := v.Underlying().(*types.Interface); v != nil {
		for i := 0; i < iface.NumMethods(); i++ {
			m := iface.Method(i)
			obj, _, _ := types.LookupFieldOrMethod(v, false, m.Pkg(), m.Name())
			switch {
			case obj == nil:
				return false
			case !Identical(obj.Type(), m.Type()):
				return false
			}
		}
		return true
	}

	// A concrete type v implements iface if it implements all methods of iface.
	for i := 0; i < iface.NumMethods(); i++ {
		m := iface.Method(i)

		obj, _, _ := types.LookupFieldOrMethod(v, false, m.Pkg(), m.Name())
		if obj == nil {
			return false
		}

		f, ok := obj.(*types.Func)
		if !ok {
			return false
		}

		if !Identical(f.Type(), m.Type()) {
			return false
		}
	}

	return true
}

// Identical reports whether x and y are identical types.
//
// Unlike types.Identical(), it permits X and Y named types
// to be considered identical even if their addresses are different.
func Identical(x, y types.Type) bool {
	return typeIdentical(x, y, nil)
}

func typeIdentical(x, y types.Type, p *ifacePair) bool {
	if x == y {
		return true
	}

	switch x := x.(type) {
	case nil:
		return false

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
			return (x.Len() < 0 || y.Len() < 0 || x.Len() == y.Len()) && typeIdentical(x.Elem(), y.Elem(), p)
		}

	case *types.Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*types.Slice); ok {
			return typeIdentical(x.Elem(), y.Elem(), p)
		}

	case *types.Struct:
		// Two struct types are identical if they have the same sequence of fields,
		// and if corresponding fields have the same names, and identical types,
		// and identical tags. Two embedded fields are considered to have the same
		// name. Lower-case field names from different packages are always different.
		if y, ok := y.(*types.Struct); ok {
			if x.NumFields() == y.NumFields() {
				for i := 0; i < x.NumFields(); i++ {
					f := x.Field(i)
					g := y.Field(i)
					if f.Embedded() != g.Embedded() || !sameID(f, g.Pkg(), g.Name()) || !typeIdentical(f.Type(), g.Type(), p) {
						return false
					}
				}
				return true
			}
		}

	case *types.Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*types.Pointer); ok {
			return typeIdentical(x.Elem(), y.Elem(), p)
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
						if !typeIdentical(v.Type(), w.Type(), p) {
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
				typeIdentical(x.Params(), y.Params(), p) &&
				typeIdentical(x.Results(), y.Results(), p)
		}

	case *types.Interface:
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types. Lower-case method names from
		// different packages are always different. The order of the methods is irrelevant.
		if y, ok := y.(*types.Interface); ok {
			if x.NumMethods() != y.NumMethods() {
				return false
			}
			// Interface types are the only types where cycles can occur
			// that are not "terminated" via named types; and such cycles
			// can only be created via method parameter types that are
			// anonymous interfaces (directly or indirectly) embedding
			// the current interface. Example:
			//
			//    type T interface {
			//        m() interface{T}
			//    }
			//
			// If two such (differently named) interfaces are compared,
			// endless recursion occurs if the cycle is not detected.
			//
			// If x and y were compared before, they must be equal
			// (if they were not, the recursion would have stopped);
			// search the ifacePair stack for the same pair.
			//
			// This is a quadratic algorithm, but in practice these stacks
			// are extremely short (bounded by the nesting depth of interface
			// type declarations that recur via parameter types, an extremely
			// rare occurrence). An alternative implementation might use a
			// "visited" map, but that is probably less efficient overall.
			q := &ifacePair{x, y, p}
			for p != nil {
				if p.identical(q) {
					return true // same pair was compared before
				}
				p = p.prev
			}
			for i := 0; i < x.NumMethods(); i++ {
				f := x.Method(i)
				g := y.Method(i)
				if f.Id() != g.Id() || !typeIdentical(f.Type(), g.Type(), q) {
					return false
				}
			}
			return true
		}

	case *types.Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*types.Map); ok {
			return typeIdentical(x.Key(), y.Key(), p) && typeIdentical(x.Elem(), y.Elem(), p)
		}

	case *types.Chan:
		// Two channel types are identical if they have identical value types
		// and the same direction.
		if y, ok := y.(*types.Chan); ok {
			return x.Dir() == y.Dir() && typeIdentical(x.Elem(), y.Elem(), p)
		}

	case *types.Named:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		y, ok := y.(*types.Named)
		if !ok {
			return false
		}
		if x.Obj() == y.Obj() {
			return true
		}
		return sameID(x.Obj(), y.Obj().Pkg(), y.Obj().Name())

	default:
		panic("unreachable")
	}

	return false
}

// An ifacePair is a node in a stack of interface type pairs compared for identity.
type ifacePair struct {
	x    *types.Interface
	y    *types.Interface
	prev *ifacePair
}

func (p *ifacePair) identical(q *ifacePair) bool {
	return (p.x == q.x && p.y == q.y) ||
		(p.x == q.y && p.y == q.x)
}

func sameID(obj types.Object, pkg *types.Package, name string) bool {
	// spec:
	// "Two identifiers are different if they are spelled differently,
	// or if they appear in different packages and are not exported.
	// Otherwise, they are the same."
	if name != obj.Name() {
		return false
	}
	// obj.Name == name
	if obj.Exported() {
		return true
	}
	// not exported, so packages must be the same (pkg == nil for
	// fields in Universe scope; this can only happen for types
	// introduced via Eval)
	if pkg == nil || obj.Pkg() == nil {
		return pkg == obj.Pkg()
	}
	// pkg != nil && obj.pkg != nil
	return pkg.Path() == obj.Pkg().Path()
}
