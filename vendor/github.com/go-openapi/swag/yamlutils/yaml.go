// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package yamlutils

import (
	json "encoding/json"
	"fmt"
	"strconv"

	"github.com/go-openapi/swag/jsonutils"
	yaml "go.yaml.in/yaml/v3"
)

// YAMLToJSON converts a YAML document into JSON bytes.
//
// Note: a YAML document is the output from a [yaml.Marshaler], e.g a pointer to a [yaml.Node].
//
// [YAMLToJSON] is typically called after [BytesToYAMLDoc].
func YAMLToJSON(value any) (json.RawMessage, error) {
	jm, err := transformData(value)
	if err != nil {
		return nil, err
	}

	b, err := jsonutils.WriteJSON(jm)

	return json.RawMessage(b), err
}

// BytesToYAMLDoc converts a byte slice into a YAML document.
//
// This function only supports root documents that are objects.
//
// A YAML document is a pointer to a [yaml.Node].
func BytesToYAMLDoc(data []byte) (any, error) {
	var document yaml.Node // preserve order that is present in the document
	if err := yaml.Unmarshal(data, &document); err != nil {
		return nil, err
	}
	if document.Kind != yaml.DocumentNode || len(document.Content) != 1 || document.Content[0].Kind != yaml.MappingNode {
		return nil, fmt.Errorf("only YAML documents that are objects are supported: %w", ErrYAML)
	}
	return &document, nil
}

func yamlNode(root *yaml.Node) (any, error) {
	switch root.Kind {
	case yaml.DocumentNode:
		return yamlDocument(root)
	case yaml.SequenceNode:
		return yamlSequence(root)
	case yaml.MappingNode:
		return yamlMapping(root)
	case yaml.ScalarNode:
		return yamlScalar(root)
	case yaml.AliasNode:
		return yamlNode(root.Alias)
	default:
		return nil, fmt.Errorf("unsupported YAML node type: %v: %w", root.Kind, ErrYAML)
	}
}

func yamlDocument(node *yaml.Node) (any, error) {
	if len(node.Content) != 1 {
		return nil, fmt.Errorf("unexpected YAML Document node content length: %d: %w", len(node.Content), ErrYAML)
	}
	return yamlNode(node.Content[0])
}

func yamlMapping(node *yaml.Node) (any, error) {
	const sensibleAllocDivider = 2 // nodes concatenate (key,value) sequences
	m := make(YAMLMapSlice, len(node.Content)/sensibleAllocDivider)

	if err := m.UnmarshalYAML(node); err != nil {
		return nil, err
	}

	return m, nil
}

func yamlSequence(node *yaml.Node) (any, error) {
	s := make([]any, 0)

	for i := range len(node.Content) {
		v, err := yamlNode(node.Content[i])
		if err != nil {
			return nil, fmt.Errorf("unable to decode YAML sequence value: %w: %w", err, ErrYAML)
		}
		s = append(s, v)
	}
	return s, nil
}

const ( // See https://yaml.org/type/
	yamlStringScalar = "tag:yaml.org,2002:str"
	yamlIntScalar    = "tag:yaml.org,2002:int"
	yamlBoolScalar   = "tag:yaml.org,2002:bool"
	yamlFloatScalar  = "tag:yaml.org,2002:float"
	yamlTimestamp    = "tag:yaml.org,2002:timestamp"
	yamlNull         = "tag:yaml.org,2002:null"
)

func yamlScalar(node *yaml.Node) (any, error) {
	switch node.LongTag() {
	case yamlStringScalar:
		return node.Value, nil
	case yamlBoolScalar:
		b, err := strconv.ParseBool(node.Value)
		if err != nil {
			return nil, fmt.Errorf("unable to process scalar node. Got %q. Expecting bool content: %w: %w", node.Value, err, ErrYAML)
		}
		return b, nil
	case yamlIntScalar:
		i, err := strconv.ParseInt(node.Value, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to process scalar node. Got %q. Expecting integer content: %w: %w", node.Value, err, ErrYAML)
		}
		return i, nil
	case yamlFloatScalar:
		f, err := strconv.ParseFloat(node.Value, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to process scalar node. Got %q. Expecting float content: %w: %w", node.Value, err, ErrYAML)
		}
		return f, nil
	case yamlTimestamp:
		// YAML timestamp is marshaled as string, not time
		return node.Value, nil
	case yamlNull:
		return nil, nil //nolint:nilnil
	default:
		return nil, fmt.Errorf("YAML tag %q is not supported: %w", node.LongTag(), ErrYAML)
	}
}

func yamlStringScalarC(node *yaml.Node) (string, error) {
	if node.Kind != yaml.ScalarNode {
		return "", fmt.Errorf("expecting a string scalar but got %q: %w", node.Kind, ErrYAML)
	}
	switch node.LongTag() {
	case yamlStringScalar, yamlIntScalar, yamlFloatScalar:
		return node.Value, nil
	default:
		return "", fmt.Errorf("YAML tag %q is not supported as map key: %w", node.LongTag(), ErrYAML)
	}
}

func format(t any) (string, error) {
	switch k := t.(type) {
	case string:
		return k, nil
	case uint:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint8:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint16:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint32:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint64:
		return strconv.FormatUint(k, 10), nil
	case int:
		return strconv.Itoa(k), nil
	case int8:
		return strconv.FormatInt(int64(k), 10), nil
	case int16:
		return strconv.FormatInt(int64(k), 10), nil
	case int32:
		return strconv.FormatInt(int64(k), 10), nil
	case int64:
		return strconv.FormatInt(k, 10), nil
	default:
		return "", fmt.Errorf("unexpected map key type, got: %T: %w", k, ErrYAML)
	}
}

func transformData(input any) (out any, err error) {
	switch in := input.(type) {
	case yaml.Node:
		return yamlNode(&in)
	case *yaml.Node:
		return yamlNode(in)
	case map[any]any:
		o := make(YAMLMapSlice, 0, len(in))
		for ke, va := range in {
			var nmi YAMLMapItem
			if nmi.Key, err = format(ke); err != nil {
				return nil, err
			}

			v, ert := transformData(va)
			if ert != nil {
				return nil, ert
			}
			nmi.Value = v
			o = append(o, nmi)
		}
		return o, nil
	case []any:
		len1 := len(in)
		o := make([]any, len1)
		for i := range len1 {
			o[i], err = transformData(in[i])
			if err != nil {
				return nil, err
			}
		}
		return o, nil
	}
	return input, nil
}
