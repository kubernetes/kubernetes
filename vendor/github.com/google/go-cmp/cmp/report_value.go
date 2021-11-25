// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp

import "reflect"

// valueNode represents a single node within a report, which is a
// structured representation of the value tree, containing information
// regarding which nodes are equal or not.
type valueNode struct {
	parent *valueNode

	Type   reflect.Type
	ValueX reflect.Value
	ValueY reflect.Value

	// NumSame is the number of leaf nodes that are equal.
	// All descendants are equal only if NumDiff is 0.
	NumSame int
	// NumDiff is the number of leaf nodes that are not equal.
	NumDiff int
	// NumIgnored is the number of leaf nodes that are ignored.
	NumIgnored int
	// NumCompared is the number of leaf nodes that were compared
	// using an Equal method or Comparer function.
	NumCompared int
	// NumTransformed is the number of non-leaf nodes that were transformed.
	NumTransformed int
	// NumChildren is the number of transitive descendants of this node.
	// This counts from zero; thus, leaf nodes have no descendants.
	NumChildren int
	// MaxDepth is the maximum depth of the tree. This counts from zero;
	// thus, leaf nodes have a depth of zero.
	MaxDepth int

	// Records is a list of struct fields, slice elements, or map entries.
	Records []reportRecord // If populated, implies Value is not populated

	// Value is the result of a transformation, pointer indirect, of
	// type assertion.
	Value *valueNode // If populated, implies Records is not populated

	// TransformerName is the name of the transformer.
	TransformerName string // If non-empty, implies Value is populated
}
type reportRecord struct {
	Key   reflect.Value // Invalid for slice element
	Value *valueNode
}

func (parent *valueNode) PushStep(ps PathStep) (child *valueNode) {
	vx, vy := ps.Values()
	child = &valueNode{parent: parent, Type: ps.Type(), ValueX: vx, ValueY: vy}
	switch s := ps.(type) {
	case StructField:
		assert(parent.Value == nil)
		parent.Records = append(parent.Records, reportRecord{Key: reflect.ValueOf(s.Name()), Value: child})
	case SliceIndex:
		assert(parent.Value == nil)
		parent.Records = append(parent.Records, reportRecord{Value: child})
	case MapIndex:
		assert(parent.Value == nil)
		parent.Records = append(parent.Records, reportRecord{Key: s.Key(), Value: child})
	case Indirect:
		assert(parent.Value == nil && parent.Records == nil)
		parent.Value = child
	case TypeAssertion:
		assert(parent.Value == nil && parent.Records == nil)
		parent.Value = child
	case Transform:
		assert(parent.Value == nil && parent.Records == nil)
		parent.Value = child
		parent.TransformerName = s.Name()
		parent.NumTransformed++
	default:
		assert(parent == nil) // Must be the root step
	}
	return child
}

func (r *valueNode) Report(rs Result) {
	assert(r.MaxDepth == 0) // May only be called on leaf nodes

	if rs.ByIgnore() {
		r.NumIgnored++
	} else {
		if rs.Equal() {
			r.NumSame++
		} else {
			r.NumDiff++
		}
	}
	assert(r.NumSame+r.NumDiff+r.NumIgnored == 1)

	if rs.ByMethod() {
		r.NumCompared++
	}
	if rs.ByFunc() {
		r.NumCompared++
	}
	assert(r.NumCompared <= 1)
}

func (child *valueNode) PopStep() (parent *valueNode) {
	if child.parent == nil {
		return nil
	}
	parent = child.parent
	parent.NumSame += child.NumSame
	parent.NumDiff += child.NumDiff
	parent.NumIgnored += child.NumIgnored
	parent.NumCompared += child.NumCompared
	parent.NumTransformed += child.NumTransformed
	parent.NumChildren += child.NumChildren + 1
	if parent.MaxDepth < child.MaxDepth+1 {
		parent.MaxDepth = child.MaxDepth + 1
	}
	return parent
}
