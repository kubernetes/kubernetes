// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"reflect"
	"strings"

	y1_1 "go.yaml.in/yaml/v2"
	y1_2 "go.yaml.in/yaml/v3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// typeToTag maps OpenAPI schema types to yaml 1.2 tags
var typeToTag = map[string]string{
	"string":  NodeTagString,
	"integer": NodeTagInt,
	"boolean": NodeTagBool,
	"number":  NodeTagFloat,
}

// FormatNonStringStyle makes sure that values which parse as non-string values in yaml 1.1
// are correctly formatted given the Schema type.
func FormatNonStringStyle(node *Node, schema spec.Schema) {
	if len(schema.Type) != 1 {
		return
	}
	t := schema.Type[0]

	if !IsYaml1_1NonString(node) {
		return
	}
	switch {
	case t == "string" && schema.Format != "int-or-string":
		if (node.Style&DoubleQuotedStyle == 0) && (node.Style&SingleQuotedStyle == 0) {
			// must quote values so they are parsed as strings
			node.Style = DoubleQuotedStyle
		}
	case t == "boolean" || t == "integer" || t == "number":
		if (node.Style&DoubleQuotedStyle != 0) || (node.Style&SingleQuotedStyle != 0) {
			// must NOT quote the values so they aren't parsed as strings
			node.Style = 0
		}
	default:
		return
	}

	// if the node tag is null, make sure we don't add any non-null tags
	// https://github.com/kptdev/kpt/issues/2321
	if node.Tag == NodeTagNull {
		// must NOT quote null values
		node.Style = 0
		return
	}
	if tag, found := typeToTag[t]; found {
		// make sure the right tag is set
		node.Tag = tag
	}
}

// IsYaml1_1NonString returns true if the value parses as a non-string value in yaml 1.1
// when unquoted.
//
// Note: yaml 1.2 uses different keywords than yaml 1.1.  Example: yaml 1.2 interprets
// `field: on` and `field: "on"` as equivalent (both strings).  However Yaml 1.1 interprets
// `field: on` as on being a bool and `field: "on"` as on being a string.
// If an input is read with `field: "on"`, and the style is changed from DoubleQuote to 0,
// it will change the type of the field from a string  to a bool.  For this reason, fields
// which are keywords in yaml 1.1 should never have their style changed, as it would break
// backwards compatibility with yaml 1.1 -- which is what is used by the Kubernetes apiserver.
func IsYaml1_1NonString(node *Node) bool {
	if node.Kind != y1_2.ScalarNode {
		// not a keyword
		return false
	}
	return IsValueNonString(node.Value)
}

func IsValueNonString(value string) bool {
	if value == "" {
		return false
	}
	if strings.Contains(value, "\n") {
		// multi-line strings will fail to unmarshal
		return false
	}
	// check if the value will unmarshal into a non-string value using a yaml 1.1 parser
	var i1 interface{}
	if err := y1_1.Unmarshal([]byte(value), &i1); err != nil {
		return false
	}
	if reflect.TypeOf(i1) != stringType {
		return true
	}

	return false
}

var stringType = reflect.TypeOf("string")
