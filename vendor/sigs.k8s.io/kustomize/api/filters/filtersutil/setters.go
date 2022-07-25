package filtersutil

import (
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// SetFn is a function that accepts an RNode to possibly modify.
type SetFn func(*yaml.RNode) error

// SetScalar returns a SetFn to set a scalar value
func SetScalar(value string) SetFn {
	return func(node *yaml.RNode) error {
		return node.PipeE(yaml.FieldSetter{StringValue: value})
	}
}

// SetEntry returns a SetFn to set an entry in a map
func SetEntry(key, value, tag string) SetFn {
	n := &yaml.Node{
		Kind:  yaml.ScalarNode,
		Value: value,
		Tag:   tag,
	}
	return func(node *yaml.RNode) error {
		return node.PipeE(yaml.FieldSetter{
			Name:  key,
			Value: yaml.NewRNode(n),
		})
	}
}

type TrackableSetter struct {
	// SetValueCallback will be invoked each time a field is set
	setValueCallback func(key, value, tag string, node *yaml.RNode)
}

// WithMutationTracker registers a callback which will be invoked each time a field is mutated
func (s *TrackableSetter) WithMutationTracker(callback func(key, value, tag string, node *yaml.RNode)) {
	s.setValueCallback = callback
}

// SetScalar returns a SetFn to set a scalar value
// if a mutation tracker has been registered, the tracker will be invoked each
// time a scalar is set
func (s TrackableSetter) SetScalar(value string) SetFn {
	origSetScalar := SetScalar(value)
	return func(node *yaml.RNode) error {
		if s.setValueCallback != nil {
			s.setValueCallback("", value, "", node)
		}
		return origSetScalar(node)
	}
}

// SetEntry returns a SetFn to set an entry in a map
// if a mutation tracker has been registered, the tracker will be invoked each
// time an entry is set
func (s TrackableSetter) SetEntry(key, value, tag string) SetFn {
	origSetEntry := SetEntry(key, value, tag)
	return func(node *yaml.RNode) error {
		if s.setValueCallback != nil {
			s.setValueCallback(key, value, tag, node)
		}
		return origSetEntry(node)
	}
}
