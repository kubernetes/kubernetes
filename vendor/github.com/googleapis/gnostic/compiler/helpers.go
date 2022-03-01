// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compiler

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"

	"github.com/googleapis/gnostic/jsonschema"
	"gopkg.in/yaml.v3"
)

// compiler helper functions, usually called from generated code

// UnpackMap gets a *yaml.Node if possible.
func UnpackMap(in *yaml.Node) (*yaml.Node, bool) {
	if in == nil {
		return nil, false
	}
	return in, true
}

// SortedKeysForMap returns the sorted keys of a yamlv2.MapSlice.
func SortedKeysForMap(m *yaml.Node) []string {
	keys := make([]string, 0)
	if m.Kind == yaml.MappingNode {
		for i := 0; i < len(m.Content); i += 2 {
			keys = append(keys, m.Content[i].Value)
		}
	}
	sort.Strings(keys)
	return keys
}

// MapHasKey returns true if a yamlv2.MapSlice contains a specified key.
func MapHasKey(m *yaml.Node, key string) bool {
	if m == nil {
		return false
	}
	if m.Kind == yaml.MappingNode {
		for i := 0; i < len(m.Content); i += 2 {
			itemKey := m.Content[i].Value
			if key == itemKey {
				return true
			}
		}
	}
	return false
}

// MapValueForKey gets the value of a map value for a specified key.
func MapValueForKey(m *yaml.Node, key string) *yaml.Node {
	if m == nil {
		return nil
	}
	if m.Kind == yaml.MappingNode {
		for i := 0; i < len(m.Content); i += 2 {
			itemKey := m.Content[i].Value
			if key == itemKey {
				return m.Content[i+1]
			}
		}
	}
	return nil
}

// ConvertInterfaceArrayToStringArray converts an array of interfaces to an array of strings, if possible.
func ConvertInterfaceArrayToStringArray(interfaceArray []interface{}) []string {
	stringArray := make([]string, 0)
	for _, item := range interfaceArray {
		v, ok := item.(string)
		if ok {
			stringArray = append(stringArray, v)
		}
	}
	return stringArray
}

// SequenceNodeForNode returns a node if it is a SequenceNode.
func SequenceNodeForNode(node *yaml.Node) (*yaml.Node, bool) {
	if node.Kind != yaml.SequenceNode {
		return nil, false
	}
	return node, true
}

// BoolForScalarNode returns the bool value of a node.
func BoolForScalarNode(node *yaml.Node) (bool, bool) {
	if node == nil {
		return false, false
	}
	if node.Kind == yaml.DocumentNode {
		return BoolForScalarNode(node.Content[0])
	}
	if node.Kind != yaml.ScalarNode {
		return false, false
	}
	if node.Tag != "!!bool" {
		return false, false
	}
	v, err := strconv.ParseBool(node.Value)
	if err != nil {
		return false, false
	}
	return v, true
}

// IntForScalarNode returns the integer value of a node.
func IntForScalarNode(node *yaml.Node) (int64, bool) {
	if node == nil {
		return 0, false
	}
	if node.Kind == yaml.DocumentNode {
		return IntForScalarNode(node.Content[0])
	}
	if node.Kind != yaml.ScalarNode {
		return 0, false
	}
	if node.Tag != "!!int" {
		return 0, false
	}
	v, err := strconv.ParseInt(node.Value, 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// FloatForScalarNode returns the float value of a node.
func FloatForScalarNode(node *yaml.Node) (float64, bool) {
	if node == nil {
		return 0.0, false
	}
	if node.Kind == yaml.DocumentNode {
		return FloatForScalarNode(node.Content[0])
	}
	if node.Kind != yaml.ScalarNode {
		return 0.0, false
	}
	if (node.Tag != "!!int") && (node.Tag != "!!float") {
		return 0.0, false
	}
	v, err := strconv.ParseFloat(node.Value, 64)
	if err != nil {
		return 0.0, false
	}
	return v, true
}

// StringForScalarNode returns the string value of a node.
func StringForScalarNode(node *yaml.Node) (string, bool) {
	if node == nil {
		return "", false
	}
	if node.Kind == yaml.DocumentNode {
		return StringForScalarNode(node.Content[0])
	}
	switch node.Kind {
	case yaml.ScalarNode:
		switch node.Tag {
		case "!!int":
			return node.Value, true
		case "!!str":
			return node.Value, true
		case "!!timestamp":
			return node.Value, true
		case "!!null":
			return "", true
		default:
			return "", false
		}
	default:
		return "", false
	}
}

// StringArrayForSequenceNode converts a sequence node to an array of strings, if possible.
func StringArrayForSequenceNode(node *yaml.Node) []string {
	stringArray := make([]string, 0)
	for _, item := range node.Content {
		v, ok := StringForScalarNode(item)
		if ok {
			stringArray = append(stringArray, v)
		}
	}
	return stringArray
}

// MissingKeysInMap identifies which keys from a list of required keys are not in a map.
func MissingKeysInMap(m *yaml.Node, requiredKeys []string) []string {
	missingKeys := make([]string, 0)
	for _, k := range requiredKeys {
		if !MapHasKey(m, k) {
			missingKeys = append(missingKeys, k)
		}
	}
	return missingKeys
}

// InvalidKeysInMap returns keys in a map that don't match a list of allowed keys and patterns.
func InvalidKeysInMap(m *yaml.Node, allowedKeys []string, allowedPatterns []*regexp.Regexp) []string {
	invalidKeys := make([]string, 0)
	if m == nil || m.Kind != yaml.MappingNode {
		return invalidKeys
	}
	for i := 0; i < len(m.Content); i += 2 {
		key := m.Content[i].Value
		found := false
		// does the key match an allowed key?
		for _, allowedKey := range allowedKeys {
			if key == allowedKey {
				found = true
				break
			}
		}
		if !found {
			// does the key match an allowed pattern?
			for _, allowedPattern := range allowedPatterns {
				if allowedPattern.MatchString(key) {
					found = true
					break
				}
			}
			if !found {
				invalidKeys = append(invalidKeys, key)
			}
		}
	}
	return invalidKeys
}

// NewNullNode creates a new Null node.
func NewNullNode() *yaml.Node {
	node := &yaml.Node{
		Kind: yaml.ScalarNode,
		Tag:  "!!null",
	}
	return node
}

// NewMappingNode creates a new Mapping node.
func NewMappingNode() *yaml.Node {
	return &yaml.Node{
		Kind:    yaml.MappingNode,
		Content: make([]*yaml.Node, 0),
	}
}

// NewSequenceNode creates a new Sequence node.
func NewSequenceNode() *yaml.Node {
	node := &yaml.Node{
		Kind:    yaml.SequenceNode,
		Content: make([]*yaml.Node, 0),
	}
	return node
}

// NewScalarNodeForString creates a new node to hold a string.
func NewScalarNodeForString(s string) *yaml.Node {
	return &yaml.Node{
		Kind:  yaml.ScalarNode,
		Tag:   "!!str",
		Value: s,
	}
}

// NewSequenceNodeForStringArray creates a new node to hold an array of strings.
func NewSequenceNodeForStringArray(strings []string) *yaml.Node {
	node := &yaml.Node{
		Kind:    yaml.SequenceNode,
		Content: make([]*yaml.Node, 0),
	}
	for _, s := range strings {
		node.Content = append(node.Content, NewScalarNodeForString(s))
	}
	return node
}

// NewScalarNodeForBool creates a new node to hold a bool.
func NewScalarNodeForBool(b bool) *yaml.Node {
	return &yaml.Node{
		Kind:  yaml.ScalarNode,
		Tag:   "!!bool",
		Value: fmt.Sprintf("%t", b),
	}
}

// NewScalarNodeForFloat creates a new node to hold a float.
func NewScalarNodeForFloat(f float64) *yaml.Node {
	return &yaml.Node{
		Kind:  yaml.ScalarNode,
		Tag:   "!!float",
		Value: fmt.Sprintf("%g", f),
	}
}

// NewScalarNodeForInt creates a new node to hold an integer.
func NewScalarNodeForInt(i int64) *yaml.Node {
	return &yaml.Node{
		Kind:  yaml.ScalarNode,
		Tag:   "!!int",
		Value: fmt.Sprintf("%d", i),
	}
}

// PluralProperties returns the string "properties" pluralized.
func PluralProperties(count int) string {
	if count == 1 {
		return "property"
	}
	return "properties"
}

// StringArrayContainsValue returns true if a string array contains a specified value.
func StringArrayContainsValue(array []string, value string) bool {
	for _, item := range array {
		if item == value {
			return true
		}
	}
	return false
}

// StringArrayContainsValues returns true if a string array contains all of a list of specified values.
func StringArrayContainsValues(array []string, values []string) bool {
	for _, value := range values {
		if !StringArrayContainsValue(array, value) {
			return false
		}
	}
	return true
}

// StringValue returns the string value of an item.
func StringValue(item interface{}) (value string, ok bool) {
	value, ok = item.(string)
	if ok {
		return value, ok
	}
	intValue, ok := item.(int)
	if ok {
		return strconv.Itoa(intValue), true
	}
	return "", false
}

// Description returns a human-readable represention of an item.
func Description(item interface{}) string {
	value, ok := item.(*yaml.Node)
	if ok {
		return jsonschema.Render(value)
	}
	return fmt.Sprintf("%+v", item)
}

// Display returns a description of a node for use in error messages.
func Display(node *yaml.Node) string {
	switch node.Kind {
	case yaml.ScalarNode:
		switch node.Tag {
		case "!!str":
			return fmt.Sprintf("%s (string)", node.Value)
		}
	}
	return fmt.Sprintf("%+v (%T)", node, node)
}

// Marshal creates a yaml version of a structure in our preferred style
func Marshal(in *yaml.Node) []byte {
	clearStyle(in)
	//bytes, _ := yaml.Marshal(&yaml.Node{Kind: yaml.DocumentNode, Content: []*yaml.Node{in}})
	bytes, _ := yaml.Marshal(in)

	return bytes
}

func clearStyle(node *yaml.Node) {
	node.Style = 0
	for _, c := range node.Content {
		clearStyle(c)
	}
}
