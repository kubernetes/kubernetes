// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

// MapNode wraps a field key and value.
type MapNode struct {
	Key   *RNode
	Value *RNode
}

// IsNilOrEmpty returns true if the MapNode is nil,
// has no value, or has a value that appears empty.
func (mn *MapNode) IsNilOrEmpty() bool {
	return mn == nil || mn.Value.IsNilOrEmpty()
}

type MapNodeSlice []*MapNode

func (m MapNodeSlice) Keys() []*RNode {
	var keys []*RNode
	for i := range m {
		if m[i] != nil {
			keys = append(keys, m[i].Key)
		}
	}
	return keys
}

func (m MapNodeSlice) Values() []*RNode {
	var values []*RNode
	for i := range m {
		if m[i] != nil {
			values = append(values, m[i].Value)
		} else {
			values = append(values, nil)
		}
	}
	return values
}
