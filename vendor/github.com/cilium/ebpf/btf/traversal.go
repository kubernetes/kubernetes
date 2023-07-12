package btf

import (
	"fmt"

	"github.com/cilium/ebpf/internal"
)

// Functions to traverse a cyclic graph of types. The below was very useful:
// https://eli.thegreenplace.net/2015/directed-graph-traversal-orderings-and-applications-to-data-flow-analysis/#post-order-and-reverse-post-order

type postorderIterator struct {
	// Iteration skips types for which this function returns true.
	skip func(Type) bool
	// The root type. May be nil if skip(root) is true.
	root Type

	// Contains types which need to be either walked or yielded.
	types typeDeque
	// Contains a boolean whether the type has been walked or not.
	walked internal.Deque[bool]
	// The set of types which has been pushed onto types.
	pushed map[Type]struct{}

	// The current type. Only valid after a call to Next().
	Type Type
}

// postorderTraversal iterates all types reachable from root by visiting the
// leaves of the graph first.
//
// Types for which skip returns true are ignored. skip may be nil.
func postorderTraversal(root Type, skip func(Type) (skip bool)) postorderIterator {
	// Avoid allocations for the common case of a skipped root.
	if skip != nil && skip(root) {
		return postorderIterator{}
	}

	po := postorderIterator{root: root, skip: skip}
	walkType(root, po.push)

	return po
}

func (po *postorderIterator) push(t *Type) {
	if _, ok := po.pushed[*t]; ok || *t == po.root {
		return
	}

	if po.skip != nil && po.skip(*t) {
		return
	}

	if po.pushed == nil {
		// Lazily allocate pushed to avoid an allocation for Types without children.
		po.pushed = make(map[Type]struct{})
	}

	po.pushed[*t] = struct{}{}
	po.types.Push(t)
	po.walked.Push(false)
}

// Next returns true if there is another Type to traverse.
func (po *postorderIterator) Next() bool {
	for !po.types.Empty() {
		t := po.types.Pop()

		if !po.walked.Pop() {
			// Push the type again, so that we re-evaluate it in done state
			// after all children have been handled.
			po.types.Push(t)
			po.walked.Push(true)

			// Add all direct children to todo.
			walkType(*t, po.push)
		} else {
			// We've walked this type previously, so we now know that all
			// children have been handled.
			po.Type = *t
			return true
		}
	}

	// Only return root once.
	po.Type, po.root = po.root, nil
	return po.Type != nil
}

// walkType calls fn on each child of typ.
func walkType(typ Type, fn func(*Type)) {
	// Explicitly type switch on the most common types to allow the inliner to
	// do its work. This avoids allocating intermediate slices from walk() on
	// the heap.
	switch v := typ.(type) {
	case *Void, *Int, *Enum, *Fwd, *Float:
		// No children to traverse.
	case *Pointer:
		fn(&v.Target)
	case *Array:
		fn(&v.Index)
		fn(&v.Type)
	case *Struct:
		for i := range v.Members {
			fn(&v.Members[i].Type)
		}
	case *Union:
		for i := range v.Members {
			fn(&v.Members[i].Type)
		}
	case *Typedef:
		fn(&v.Type)
	case *Volatile:
		fn(&v.Type)
	case *Const:
		fn(&v.Type)
	case *Restrict:
		fn(&v.Type)
	case *Func:
		fn(&v.Type)
	case *FuncProto:
		fn(&v.Return)
		for i := range v.Params {
			fn(&v.Params[i].Type)
		}
	case *Var:
		fn(&v.Type)
	case *Datasec:
		for i := range v.Vars {
			fn(&v.Vars[i].Type)
		}
	case *declTag:
		fn(&v.Type)
	case *typeTag:
		fn(&v.Type)
	case *cycle:
		// cycle has children, but we ignore them deliberately.
	default:
		panic(fmt.Sprintf("don't know how to walk Type %T", v))
	}
}
