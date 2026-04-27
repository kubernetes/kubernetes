// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package yamlutils

import (
	"fmt"
	"iter"
	"slices"
	"sort"
	"strconv"

	"github.com/go-openapi/swag/conv"
	"github.com/go-openapi/swag/jsonutils"
	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
	"github.com/go-openapi/swag/typeutils"
	yaml "go.yaml.in/yaml/v3"
)

var (
	_ yaml.Marshaler   = YAMLMapSlice{}
	_ yaml.Unmarshaler = &YAMLMapSlice{}
)

// YAMLMapSlice represents a YAML object, with the order of keys maintained.
//
// It is similar to [jsonutils.JSONMapSlice] and also knows how to marshal and unmarshal YAML.
//
// It behaves like an ordered map, but keys can't be accessed in constant time.
type YAMLMapSlice []YAMLMapItem

// YAMLMapItem represents the value of a key in a YAML object held by [YAMLMapSlice].
//
// It is entirely equivalent to [jsonutils.JSONMapItem], with the same limitation that
// you should not Marshal or Unmarshal directly this type, outside of a [YAMLMapSlice].
type YAMLMapItem = jsonutils.JSONMapItem

func (s YAMLMapSlice) OrderedItems() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for _, item := range s {
			if !yield(item.Key, item.Value) {
				return
			}
		}
	}
}

// SetOrderedItems implements [ifaces.SetOrdered]: it merges keys passed by the iterator argument
// into the [YAMLMapSlice].
func (s *YAMLMapSlice) SetOrderedItems(items iter.Seq2[string, any]) {
	if items == nil {
		// force receiver to be a nil slice
		*s = nil

		return
	}

	m := *s
	if len(m) > 0 {
		// update mode: short-circuited when unmarshaling fresh data structures
		idx := make(map[string]int, len(m))

		for i, item := range m {
			idx[item.Key] = i
		}

		for k, v := range items {
			idx, ok := idx[k]
			if ok {
				m[idx].Value = v

				continue
			}

			m = append(m, YAMLMapItem{Key: k, Value: v})
		}

		*s = m

		return
	}

	for k, v := range items {
		m = append(m, YAMLMapItem{Key: k, Value: v})
	}

	*s = m
}

// MarshalJSON renders this YAML object as JSON bytes.
//
// The difference with standard JSON marshaling is that the order of keys is maintained.
func (s YAMLMapSlice) MarshalJSON() ([]byte, error) {
	return jsonutils.JSONMapSlice(s).MarshalJSON()
}

// UnmarshalJSON builds this YAML object from JSON bytes.
//
// The difference with standard JSON marshaling is that the order of keys is maintained.
func (s *YAMLMapSlice) UnmarshalJSON(data []byte) error {
	js := jsonutils.JSONMapSlice(*s)

	if err := js.UnmarshalJSON(data); err != nil {
		return err
	}

	*s = YAMLMapSlice(js)

	return nil
}

// MarshalYAML produces a YAML document as bytes
//
// The difference with standard YAML marshaling is that the order of keys is maintained.
//
// It implements [yaml.Marshaler].
func (s YAMLMapSlice) MarshalYAML() (any, error) {
	if typeutils.IsNil(s) {
		return []byte("null\n"), nil
	}
	var n yaml.Node
	n.Kind = yaml.DocumentNode
	var nodes []*yaml.Node

	for _, item := range s {
		nn, err := json2yaml(item.Value)
		if err != nil {
			return nil, err
		}

		ns := []*yaml.Node{
			{
				Kind:  yaml.ScalarNode,
				Tag:   yamlStringScalar,
				Value: item.Key,
			},
			nn,
		}
		nodes = append(nodes, ns...)
	}

	n.Content = []*yaml.Node{
		{
			Kind:    yaml.MappingNode,
			Content: nodes,
		},
	}

	return yaml.Marshal(&n)
}

// UnmarshalYAML builds a YAMLMapSlice object from a YAML document [yaml.Node].
//
// It implements [yaml.Unmarshaler].
func (s *YAMLMapSlice) UnmarshalYAML(node *yaml.Node) error {
	if typeutils.IsNil(*s) {
		// allow to unmarshal with a simple var declaration (nil slice)
		*s = YAMLMapSlice{}
	}
	if node == nil {
		*s = nil
		return nil
	}

	const sensibleAllocDivider = 2
	m := slices.Grow(*s, len(node.Content)/sensibleAllocDivider)
	m = m[:0]

	for i := 0; i < len(node.Content); i += 2 {
		var nmi YAMLMapItem
		k, err := yamlStringScalarC(node.Content[i])
		if err != nil {
			return fmt.Errorf("unable to decode YAML map key: %w: %w", err, ErrYAML)
		}
		nmi.Key = k
		v, err := yamlNode(node.Content[i+1])
		if err != nil {
			return fmt.Errorf("unable to process YAML map value for key %q: %w: %w", k, err, ErrYAML)
		}
		nmi.Value = v
		m = append(m, nmi)
	}

	*s = m

	return nil
}

func json2yaml(item any) (*yaml.Node, error) {
	if typeutils.IsNil(item) {
		return &yaml.Node{
			Kind:  yaml.ScalarNode,
			Value: "null",
		}, nil
	}

	switch val := item.(type) {
	case ifaces.Ordered:
		return orderedYAML(val)

	case map[string]any:
		var n yaml.Node
		n.Kind = yaml.MappingNode
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, k := range keys {
			v := val[k]
			childNode, err := json2yaml(v)
			if err != nil {
				return nil, err
			}
			n.Content = append(n.Content, &yaml.Node{
				Kind:  yaml.ScalarNode,
				Tag:   yamlStringScalar,
				Value: k,
			}, childNode)
		}
		return &n, nil

	case []any:
		var n yaml.Node
		n.Kind = yaml.SequenceNode
		for i := range val {
			childNode, err := json2yaml(val[i])
			if err != nil {
				return nil, err
			}
			n.Content = append(n.Content, childNode)
		}
		return &n, nil
	case string:
		return &yaml.Node{
			Kind:  yaml.ScalarNode,
			Tag:   yamlStringScalar,
			Value: val,
		}, nil
	case float32:
		return floatNode(val)
	case float64:
		return floatNode(val)
	case int:
		return integerNode(val)
	case int8:
		return integerNode(val)
	case int16:
		return integerNode(val)
	case int32:
		return integerNode(val)
	case int64:
		return integerNode(val)
	case uint:
		return uintegerNode(val)
	case uint8:
		return uintegerNode(val)
	case uint16:
		return uintegerNode(val)
	case uint32:
		return uintegerNode(val)
	case uint64:
		return uintegerNode(val)
	case bool:
		return &yaml.Node{
			Kind:  yaml.ScalarNode,
			Tag:   yamlBoolScalar,
			Value: strconv.FormatBool(val),
		}, nil
	default:
		return nil, fmt.Errorf("unhandled type: %T: %w", val, ErrYAML)
	}
}

func floatNode[T conv.Float](val T) (*yaml.Node, error) {
	return &yaml.Node{
		Kind:  yaml.ScalarNode,
		Tag:   yamlFloatScalar,
		Value: conv.FormatFloat(val),
	}, nil
}

func integerNode[T conv.Signed](val T) (*yaml.Node, error) {
	return &yaml.Node{
		Kind:  yaml.ScalarNode,
		Tag:   yamlIntScalar,
		Value: conv.FormatInteger(val),
	}, nil
}

func uintegerNode[T conv.Unsigned](val T) (*yaml.Node, error) {
	return &yaml.Node{
		Kind:  yaml.ScalarNode,
		Tag:   yamlIntScalar,
		Value: conv.FormatUinteger(val),
	}, nil
}

func orderedYAML[T ifaces.Ordered](val T) (*yaml.Node, error) {
	var n yaml.Node
	n.Kind = yaml.MappingNode
	for key, value := range val.OrderedItems() {
		childNode, err := json2yaml(value)
		if err != nil {
			return nil, err
		}

		n.Content = append(n.Content, &yaml.Node{
			Kind:  yaml.ScalarNode,
			Tag:   yamlStringScalar,
			Value: key,
		}, childNode)
	}
	return &n, nil
}
