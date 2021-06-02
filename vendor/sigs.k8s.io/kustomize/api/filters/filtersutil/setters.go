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
	if tag == yaml.NodeTagString && yaml.IsYaml1_1NonString(n) {
		n.Style = yaml.DoubleQuotedStyle
	}
	return func(node *yaml.RNode) error {
		return node.PipeE(yaml.FieldSetter{
			Name:  key,
			Value: yaml.NewRNode(n),
		})
	}
}
