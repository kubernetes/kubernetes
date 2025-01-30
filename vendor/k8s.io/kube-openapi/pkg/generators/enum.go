/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package generators

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

const tagEnumType = "enum"
const enumTypeDescriptionHeader = "Possible enum values:"

type enumValue struct {
	Name    string
	Value   string
	Comment string
}

type enumType struct {
	Name   types.Name
	Values []*enumValue
}

// enumMap is a map from the name to the matching enum type.
type enumMap map[types.Name]*enumType

type enumContext struct {
	enumTypes enumMap
}

func newEnumContext(c *generator.Context) *enumContext {
	return &enumContext{enumTypes: parseEnums(c)}
}

// EnumType checks and finds the enumType for a given type.
// If the given type is a known enum type, returns the enumType, true
// Otherwise, returns nil, false
func (ec *enumContext) EnumType(t *types.Type) (enum *enumType, isEnum bool) {
	// if t is a pointer, use its underlying type instead
	if t.Kind == types.Pointer {
		t = t.Elem
	}
	enum, ok := ec.enumTypes[t.Name]
	return enum, ok
}

// ValueStrings returns all possible values of the enum type as strings
// the results are sorted and quoted as Go literals.
func (et *enumType) ValueStrings() []string {
	var values []string
	for _, value := range et.Values {
		// use "%q" format to generate a Go literal of the string const value
		values = append(values, fmt.Sprintf("%q", value.Value))
	}
	sort.Strings(values)
	return values
}

// DescriptionLines returns a description of the enum in this format:
//
// Possible enum values:
//   - `"value1"` description 1
//   - `"value2"` description 2
func (et *enumType) DescriptionLines() []string {
	if len(et.Values) == 0 {
		return nil
	}
	var lines []string
	for _, value := range et.Values {
		lines = append(lines, value.Description())
	}
	sort.Strings(lines)
	// Prepend an empty string to initiate a new paragraph.
	return append([]string{"", enumTypeDescriptionHeader}, lines...)
}

func parseEnums(c *generator.Context) enumMap {
	// First, find the builtin "string" type
	stringType := c.Universe.Type(types.Name{Name: "string"})

	// find all enum types.
	enumTypes := make(enumMap)
	for _, p := range c.Universe {
		for _, t := range p.Types {
			if isEnumType(stringType, t) {
				if _, ok := enumTypes[t.Name]; !ok {
					enumTypes[t.Name] = &enumType{
						Name: t.Name,
					}
				}
			}
		}
	}

	// find all enum values from constants, and try to match each with its type.
	for _, p := range c.Universe {
		for _, c := range p.Constants {
			enumType := c.Underlying
			if _, ok := enumTypes[enumType.Name]; ok {
				value := &enumValue{
					Name:    c.Name.Name,
					Value:   *c.ConstValue,
					Comment: strings.Join(c.CommentLines, " "),
				}
				enumTypes[enumType.Name].addIfNotPresent(value)
			}
		}
	}

	return enumTypes
}

func (et *enumType) addIfNotPresent(value *enumValue) {
	// If we already have an enum case with the same value, then ignore this new
	// one. This can happen if an enum aliases one from another package and
	// re-exports the cases.
	for i, existing := range et.Values {
		if existing.Value == value.Value {

			// Take the value of the longer comment (or some other deterministic tie breaker)
			if len(existing.Comment) < len(value.Comment) || (len(existing.Comment) == len(value.Comment) && existing.Comment > value.Comment) {
				et.Values[i] = value
			}

			return
		}
	}
	et.Values = append(et.Values, value)
}

// Description returns the description line for the enumValue
// with the format:
//   - `"FooValue"` is the Foo value
func (ev *enumValue) Description() string {
	comment := strings.TrimSpace(ev.Comment)
	// The comment should starts with the type name, trim it first.
	comment = strings.TrimPrefix(comment, ev.Name)
	// Trim the possible space after previous step.
	comment = strings.TrimSpace(comment)
	// The comment may be multiline, cascade all consecutive whitespaces.
	comment = whitespaceRegex.ReplaceAllString(comment, " ")
	return fmt.Sprintf(" - `%q` %s", ev.Value, comment)
}

// isEnumType checks if a given type is an enum by the definition
// An enum type should be an alias of string and has tag '+enum' in its comment.
// Additionally, pass the type of builtin 'string' to check against.
func isEnumType(stringType *types.Type, t *types.Type) bool {
	return t.Kind == types.Alias && t.Underlying == stringType && hasEnumTag(t)
}

func hasEnumTag(t *types.Type) bool {
	return gengo.ExtractCommentTags("+", t.CommentLines)[tagEnumType] != nil
}

// whitespaceRegex is the regex for consecutive whitespaces.
var whitespaceRegex = regexp.MustCompile(`\s+`)
