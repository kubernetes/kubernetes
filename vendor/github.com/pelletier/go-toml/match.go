package toml

import (
	"fmt"
)

// support function to set positions for tomlValues
// NOTE: this is done to allow ctx.lastPosition to indicate the start of any
// values returned by the query engines
func tomlValueCheck(node interface{}, ctx *queryContext) interface{} {
	switch castNode := node.(type) {
	case *tomlValue:
		ctx.lastPosition = castNode.position
		return castNode.value
	case []*TomlTree:
		if len(castNode) > 0 {
			ctx.lastPosition = castNode[0].position
		}
		return node
	default:
		return node
	}
}

// base match
type matchBase struct {
	next pathFn
}

func (f *matchBase) setNext(next pathFn) {
	f.next = next
}

// terminating functor - gathers results
type terminatingFn struct {
	// empty
}

func newTerminatingFn() *terminatingFn {
	return &terminatingFn{}
}

func (f *terminatingFn) setNext(next pathFn) {
	// do nothing
}

func (f *terminatingFn) call(node interface{}, ctx *queryContext) {
	switch castNode := node.(type) {
	case *TomlTree:
		ctx.result.appendResult(node, castNode.position)
	case *tomlValue:
		ctx.result.appendResult(node, castNode.position)
	default:
		// use last position for scalars
		ctx.result.appendResult(node, ctx.lastPosition)
	}
}

// match single key
type matchKeyFn struct {
	matchBase
	Name string
}

func newMatchKeyFn(name string) *matchKeyFn {
	return &matchKeyFn{Name: name}
}

func (f *matchKeyFn) call(node interface{}, ctx *queryContext) {
	if array, ok := node.([]*TomlTree); ok {
		for _, tree := range array {
			item := tree.values[f.Name]
			if item != nil {
				f.next.call(item, ctx)
			}
		}
	} else if tree, ok := node.(*TomlTree); ok {
		item := tree.values[f.Name]
		if item != nil {
			f.next.call(item, ctx)
		}
	}
}

// match single index
type matchIndexFn struct {
	matchBase
	Idx int
}

func newMatchIndexFn(idx int) *matchIndexFn {
	return &matchIndexFn{Idx: idx}
}

func (f *matchIndexFn) call(node interface{}, ctx *queryContext) {
	if arr, ok := tomlValueCheck(node, ctx).([]interface{}); ok {
		if f.Idx < len(arr) && f.Idx >= 0 {
			f.next.call(arr[f.Idx], ctx)
		}
	}
}

// filter by slicing
type matchSliceFn struct {
	matchBase
	Start, End, Step int
}

func newMatchSliceFn(start, end, step int) *matchSliceFn {
	return &matchSliceFn{Start: start, End: end, Step: step}
}

func (f *matchSliceFn) call(node interface{}, ctx *queryContext) {
	if arr, ok := tomlValueCheck(node, ctx).([]interface{}); ok {
		// adjust indexes for negative values, reverse ordering
		realStart, realEnd := f.Start, f.End
		if realStart < 0 {
			realStart = len(arr) + realStart
		}
		if realEnd < 0 {
			realEnd = len(arr) + realEnd
		}
		if realEnd < realStart {
			realEnd, realStart = realStart, realEnd // swap
		}
		// loop and gather
		for idx := realStart; idx < realEnd; idx += f.Step {
			f.next.call(arr[idx], ctx)
		}
	}
}

// match anything
type matchAnyFn struct {
	matchBase
}

func newMatchAnyFn() *matchAnyFn {
	return &matchAnyFn{}
}

func (f *matchAnyFn) call(node interface{}, ctx *queryContext) {
	if tree, ok := node.(*TomlTree); ok {
		for _, v := range tree.values {
			f.next.call(v, ctx)
		}
	}
}

// filter through union
type matchUnionFn struct {
	Union []pathFn
}

func (f *matchUnionFn) setNext(next pathFn) {
	for _, fn := range f.Union {
		fn.setNext(next)
	}
}

func (f *matchUnionFn) call(node interface{}, ctx *queryContext) {
	for _, fn := range f.Union {
		fn.call(node, ctx)
	}
}

// match every single last node in the tree
type matchRecursiveFn struct {
	matchBase
}

func newMatchRecursiveFn() *matchRecursiveFn {
	return &matchRecursiveFn{}
}

func (f *matchRecursiveFn) call(node interface{}, ctx *queryContext) {
	if tree, ok := node.(*TomlTree); ok {
		var visit func(tree *TomlTree)
		visit = func(tree *TomlTree) {
			for _, v := range tree.values {
				f.next.call(v, ctx)
				switch node := v.(type) {
				case *TomlTree:
					visit(node)
				case []*TomlTree:
					for _, subtree := range node {
						visit(subtree)
					}
				}
			}
		}
		f.next.call(tree, ctx)
		visit(tree)
	}
}

// match based on an externally provided functional filter
type matchFilterFn struct {
	matchBase
	Pos  Position
	Name string
}

func newMatchFilterFn(name string, pos Position) *matchFilterFn {
	return &matchFilterFn{Name: name, Pos: pos}
}

func (f *matchFilterFn) call(node interface{}, ctx *queryContext) {
	fn, ok := (*ctx.filters)[f.Name]
	if !ok {
		panic(fmt.Sprintf("%s: query context does not have filter '%s'",
			f.Pos.String(), f.Name))
	}
	switch castNode := tomlValueCheck(node, ctx).(type) {
	case *TomlTree:
		for _, v := range castNode.values {
			if tv, ok := v.(*tomlValue); ok {
				if fn(tv.value) {
					f.next.call(v, ctx)
				}
			} else {
				if fn(v) {
					f.next.call(v, ctx)
				}
			}
		}
	case []interface{}:
		for _, v := range castNode {
			if fn(v) {
				f.next.call(v, ctx)
			}
		}
	}
}
