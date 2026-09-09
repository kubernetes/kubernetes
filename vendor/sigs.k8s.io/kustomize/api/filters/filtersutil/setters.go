// Copyright 2022 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filtersutil

import (
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// SetFn is a function that accepts an RNode to possibly modify.
type SetFn func(*yaml.RNode) error

// SetScalar returns a SetFn to set a scalar value
func SetScalar(value string) SetFn {
	return SetEntry("", value, yaml.NodeTagEmpty)
}

// SetEntry returns a SetFn to set a field or a map entry to a value.
// It can be used with an empty name to set both a value and a tag on a scalar node.
// When setting only a value on a scalar node, use SetScalar instead.
func SetEntry(name, value, tag string) SetFn {
	n := &yaml.Node{
		Kind:  yaml.ScalarNode,
		Value: value,
		Tag:   tag,
	}
	return func(node *yaml.RNode) error {
		return node.PipeE(yaml.FieldSetter{
			Name:  name,
			Value: yaml.NewRNode(n),
		})
	}
}

type TrackableSetter struct {
	// SetValueCallback will be invoked each time a field is set
	setValueCallback func(name, value, tag string, node *yaml.RNode)
}

// WithMutationTracker registers a callback which will be invoked each time a field is mutated
func (s *TrackableSetter) WithMutationTracker(callback func(key, value, tag string, node *yaml.RNode)) *TrackableSetter {
	s.setValueCallback = callback
	return s
}

// SetScalar returns a SetFn to set a scalar value.
// if a mutation tracker has been registered, the tracker will be invoked each
// time a scalar is set
func (s TrackableSetter) SetScalar(value string) SetFn {
	return s.SetEntry("", value, yaml.NodeTagEmpty)
}

// SetScalarIfEmpty returns a SetFn to set a scalar value only if it isn't already set.
// If a mutation tracker has been registered, the tracker will be invoked each
// time a scalar is actually set.
func (s TrackableSetter) SetScalarIfEmpty(value string) SetFn {
	return s.SetEntryIfEmpty("", value, yaml.NodeTagEmpty)
}

// SetEntry returns a SetFn to set a field or a map entry to a value.
// It can be used with an empty name to set both a value and a tag on a scalar node.
// When setting only a value on a scalar node, use SetScalar instead.
// If a mutation tracker has been registered, the tracker will be invoked each
// time an entry is set.
func (s TrackableSetter) SetEntry(name, value, tag string) SetFn {
	origSetEntry := SetEntry(name, value, tag)
	return func(node *yaml.RNode) error {
		if s.setValueCallback != nil {
			s.setValueCallback(name, value, tag, node)
		}
		return origSetEntry(node)
	}
}

// SetEntryIfEmpty returns a SetFn to set a field or a map entry to a value only if it isn't already set.
// It can be used with an empty name to set both a value and a tag on a scalar node.
// When setting only a value on a scalar node, use SetScalar instead.
// If a mutation tracker has been registered, the tracker will be invoked each
// time an entry is actually set.
func (s TrackableSetter) SetEntryIfEmpty(key, value, tag string) SetFn {
	origSetEntry := SetEntry(key, value, tag)
	return func(node *yaml.RNode) error {
		if hasExistingValue(node, key) {
			return nil
		}
		if s.setValueCallback != nil {
			s.setValueCallback(key, value, tag, node)
		}
		return origSetEntry(node)
	}
}

func hasExistingValue(node *yaml.RNode, key string) bool {
	if node.IsNilOrEmpty() {
		return false
	}
	if err := yaml.ErrorIfInvalid(node, yaml.ScalarNode); err == nil {
		return yaml.GetValue(node) != ""
	}
	entry := node.Field(key)
	if entry.IsNilOrEmpty() {
		return false
	}
	return yaml.GetValue(entry.Value) != ""
}
