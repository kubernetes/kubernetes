// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"encoding/base64"
	"sort"
	"strings"
	"unicode/utf8"
)

// SortedMapKeys returns a sorted slice of keys to the given map.
// Writing this function never gets old.
func SortedMapKeys(m map[string]string) []string {
	keys := make([]string, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	return keys
}

func (rn *RNode) LoadMapIntoConfigMapData(m map[string]string) error {
	for _, k := range SortedMapKeys(m) {
		fldName, vrN := makeConfigMapValueRNode(m[k])
		if _, err := rn.Pipe(
			LookupCreate(MappingNode, fldName),
			SetField(k, vrN)); err != nil {
			return err
		}
	}
	return nil
}

func (rn *RNode) LoadMapIntoConfigMapBinaryData(m map[string]string) error {
	for _, k := range SortedMapKeys(m) {
		_, vrN := makeConfigMapValueRNode(m[k])
		// we know this is binary data
		fldName := BinaryDataField
		if _, err := rn.Pipe(
			LookupCreate(MappingNode, fldName),
			SetField(k, vrN)); err != nil {
			return err
		}
	}
	return nil
}

func makeConfigMapValueRNode(s string) (field string, rN *RNode) {
	yN := &Node{Kind: ScalarNode}
	yN.Tag = NodeTagString
	if utf8.ValidString(s) {
		field = DataField
		yN.Value = s
	} else {
		field = BinaryDataField
		yN.Value = encodeBase64(s)
	}
	if strings.Contains(yN.Value, "\n") {
		yN.Style = LiteralStyle
	}
	return field, NewRNode(yN)
}

func (rn *RNode) LoadMapIntoSecretData(m map[string]string) error {
	mapNode, err := rn.Pipe(LookupCreate(MappingNode, DataField))
	if err != nil {
		return err
	}
	for _, k := range SortedMapKeys(m) {
		vrN := makeSecretValueRNode(m[k])
		if _, err := mapNode.Pipe(SetField(k, vrN)); err != nil {
			return err
		}
	}
	return nil
}

// In a secret, all data is base64 encoded, regardless of its conformance
// or lack thereof to UTF-8.
func makeSecretValueRNode(s string) *RNode {
	yN := &Node{Kind: ScalarNode}
	// Purposely don't use YAML tags to identify the data as being plain text or
	// binary.  It kubernetes Secrets the values in the `data` map are expected
	// to be base64 encoded, and in ConfigMaps that same can be said for the
	// values in the `binaryData` field.
	yN.Tag = NodeTagString
	yN.Value = encodeBase64(s)
	if strings.Contains(yN.Value, "\n") {
		yN.Style = LiteralStyle
	}
	return NewRNode(yN)
}

// encodeBase64 encodes s as base64 that is broken up into multiple lines
// as appropriate for the resulting length.
func encodeBase64(s string) string {
	const lineLen = 70
	encLen := base64.StdEncoding.EncodedLen(len(s))
	lines := encLen/lineLen + 1
	buf := make([]byte, encLen*2+lines)
	in := buf[0:encLen]
	out := buf[encLen:]
	base64.StdEncoding.Encode(in, []byte(s))
	k := 0
	for i := 0; i < len(in); i += lineLen {
		j := i + lineLen
		if j > len(in) {
			j = len(in)
		}
		k += copy(out[k:], in[i:j])
		if lines > 1 {
			out[k] = '\n'
			k++
		}
	}
	return string(out[:k])
}
