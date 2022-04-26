package errcheck

import (
	"fmt"
	"go/types"
)

// walkThroughEmbeddedInterfaces returns a slice of Interfaces that
// we need to walk through in order to reach the actual definition,
// in an Interface, of the method selected by the given selection.
//
// false will be returned in the second return value if:
//   - the right side of the selection is not a function
//   - the actual definition of the function is not in an Interface
//
// The returned slice will contain all the interface types that need
// to be walked through to reach the actual definition.
//
// For example, say we have:
//
//    type Inner interface {Method()}
//    type Middle interface {Inner}
//    type Outer interface {Middle}
//    type T struct {Outer}
//    type U struct {T}
//    type V struct {U}
//
// And then the selector:
//
//    V.Method
//
// We'll return [Outer, Middle, Inner] by first walking through the embedded structs
// until we reach the Outer interface, then descending through the embedded interfaces
// until we find the one that actually explicitly defines Method.
func walkThroughEmbeddedInterfaces(sel *types.Selection) ([]types.Type, bool) {
	fn, ok := sel.Obj().(*types.Func)
	if !ok {
		return nil, false
	}

	// Start off at the receiver.
	currentT := sel.Recv()

	// First, we can walk through any Struct fields provided
	// by the selection Index() method. We ignore the last
	// index because it would give the method itself.
	indexes := sel.Index()
	for _, fieldIndex := range indexes[:len(indexes)-1] {
		currentT = getTypeAtFieldIndex(currentT, fieldIndex)
	}

	// Now currentT is either a type implementing the actual function,
	// an Invalid type (if the receiver is a package), or an interface.
	//
	// If it's not an Interface, then we're done, as this function
	// only cares about Interface-defined functions.
	//
	// If it is an Interface, we potentially need to continue digging until
	// we find the Interface that actually explicitly defines the function.
	interfaceT, ok := maybeUnname(currentT).(*types.Interface)
	if !ok {
		return nil, false
	}

	// The first interface we pass through is this one we've found. We return the possibly
	// wrapping types.Named because it is more useful to work with for callers.
	result := []types.Type{currentT}

	// If this interface itself explicitly defines the given method
	// then we're done digging.
	for !explicitlyDefinesMethod(interfaceT, fn) {
		// Otherwise, we find which of the embedded interfaces _does_
		// define the method, add it to our list, and loop.
		namedInterfaceT, ok := getEmbeddedInterfaceDefiningMethod(interfaceT, fn)
		if !ok {
			// This should be impossible as long as we type-checked: either the
			// interface or one of its embedded ones must implement the method...
			panic(fmt.Sprintf("either %v or one of its embedded interfaces must implement %v", currentT, fn))
		}
		result = append(result, namedInterfaceT)
		interfaceT = namedInterfaceT.Underlying().(*types.Interface)
	}

	return result, true
}

func getTypeAtFieldIndex(startingAt types.Type, fieldIndex int) types.Type {
	t := maybeUnname(maybeDereference(startingAt))
	s, ok := t.(*types.Struct)
	if !ok {
		panic(fmt.Sprintf("cannot get Field of a type that is not a struct, got a %T", t))
	}

	return s.Field(fieldIndex).Type()
}

// getEmbeddedInterfaceDefiningMethod searches through any embedded interfaces of the
// passed interface searching for one that defines the given function. If found, the
// types.Named wrapping that interface will be returned along with true in the second value.
//
// If no such embedded interface is found, nil and false are returned.
func getEmbeddedInterfaceDefiningMethod(interfaceT *types.Interface, fn *types.Func) (*types.Named, bool) {
	for i := 0; i < interfaceT.NumEmbeddeds(); i++ {
		embedded := interfaceT.Embedded(i)
		if definesMethod(embedded.Underlying().(*types.Interface), fn) {
			return embedded, true
		}
	}
	return nil, false
}

func explicitlyDefinesMethod(interfaceT *types.Interface, fn *types.Func) bool {
	for i := 0; i < interfaceT.NumExplicitMethods(); i++ {
		if interfaceT.ExplicitMethod(i) == fn {
			return true
		}
	}
	return false
}

func definesMethod(interfaceT *types.Interface, fn *types.Func) bool {
	for i := 0; i < interfaceT.NumMethods(); i++ {
		if interfaceT.Method(i) == fn {
			return true
		}
	}
	return false
}

func maybeDereference(t types.Type) types.Type {
	p, ok := t.(*types.Pointer)
	if ok {
		return p.Elem()
	}
	return t
}

func maybeUnname(t types.Type) types.Type {
	n, ok := t.(*types.Named)
	if ok {
		return n.Underlying()
	}
	return t
}
