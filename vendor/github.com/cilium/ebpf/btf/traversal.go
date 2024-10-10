package btf

import (
	"fmt"
)

// Functions to traverse a cyclic graph of types. The below was very useful:
// https://eli.thegreenplace.net/2015/directed-graph-traversal-orderings-and-applications-to-data-flow-analysis/#post-order-and-reverse-post-order

// Visit all types reachable from root in postorder.
//
// Traversal stops if yield returns false.
//
// Returns false if traversal was aborted.
func visitInPostorder(root Type, visited map[Type]struct{}, yield func(typ Type) bool) bool {
	if _, ok := visited[root]; ok {
		return true
	}
	if visited == nil {
		visited = make(map[Type]struct{})
	}
	visited[root] = struct{}{}

	cont := children(root, func(child *Type) bool {
		return visitInPostorder(*child, visited, yield)
	})
	if !cont {
		return false
	}

	return yield(root)
}

// children calls yield on each child of typ.
//
// Traversal stops if yield returns false.
//
// Returns false if traversal was aborted.
func children(typ Type, yield func(child *Type) bool) bool {
	// Explicitly type switch on the most common types to allow the inliner to
	// do its work. This avoids allocating intermediate slices from walk() on
	// the heap.
	switch v := typ.(type) {
	case *Void, *Int, *Enum, *Fwd, *Float:
		// No children to traverse.
	case *Pointer:
		if !yield(&v.Target) {
			return false
		}
	case *Array:
		if !yield(&v.Index) {
			return false
		}
		if !yield(&v.Type) {
			return false
		}
	case *Struct:
		for i := range v.Members {
			if !yield(&v.Members[i].Type) {
				return false
			}
		}
	case *Union:
		for i := range v.Members {
			if !yield(&v.Members[i].Type) {
				return false
			}
		}
	case *Typedef:
		if !yield(&v.Type) {
			return false
		}
	case *Volatile:
		if !yield(&v.Type) {
			return false
		}
	case *Const:
		if !yield(&v.Type) {
			return false
		}
	case *Restrict:
		if !yield(&v.Type) {
			return false
		}
	case *Func:
		if !yield(&v.Type) {
			return false
		}
	case *FuncProto:
		if !yield(&v.Return) {
			return false
		}
		for i := range v.Params {
			if !yield(&v.Params[i].Type) {
				return false
			}
		}
	case *Var:
		if !yield(&v.Type) {
			return false
		}
	case *Datasec:
		for i := range v.Vars {
			if !yield(&v.Vars[i].Type) {
				return false
			}
		}
	case *declTag:
		if !yield(&v.Type) {
			return false
		}
	case *typeTag:
		if !yield(&v.Type) {
			return false
		}
	case *cycle:
		// cycle has children, but we ignore them deliberately.
	default:
		panic(fmt.Sprintf("don't know how to walk Type %T", v))
	}

	return true
}
