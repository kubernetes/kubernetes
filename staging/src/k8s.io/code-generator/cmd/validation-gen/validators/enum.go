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

package validators

import (
	"cmp"
	"fmt"
	"slices"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

const enumTagName = "k8s:enum"

func init() {
	RegisterTagValidator(&enumTagValidator{})
}

type enumTagValidator struct {
	enumContext *enumContext
}

func (etv *enumTagValidator) Init(cfg Config) {
	etv.enumContext = newEnumContext(cfg.GengoContext)
}

func (enumTagValidator) TagName() string {
	return enumTagName
}

var enumTagValidScopes = sets.New(ScopeType)

func (enumTagValidator) ValidScopes() sets.Set[Scope] {
	return enumTagValidScopes
}

var (
	enumValidator = types.Name{Package: libValidationPkg, Name: "Enum"}
)

var setsNew = types.Name{Package: "k8s.io/apimachinery/pkg/util/sets", Name: "New"}

func (etv *enumTagValidator) GetValidations(context Context, _ codetags.Tag) (Validations, error) {
	// NOTE: typedefs to pointers are not supported, so we should never see a pointer here.
	if t := util.NativeType(context.Type); t != types.String {
		return Validations{}, fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}

	var result Validations

	if enum, ok := etv.enumContext.EnumType(context.Type); ok {
		// TODO: Avoid the "local" here. This was added to to avoid errors caused when the package is an empty string.
		//       The correct package would be the output package but is not known here. This does not show up in generated code.
		// TODO: Append a consistent hash suffix to avoid generated name conflicts?
		supportVarName := PrivateVar{Name: "SymbolsFor" + context.Type.Name.Name, Package: "local"}
		supportVar := Variable(supportVarName, Function(enumTagName, DefaultFlags, setsNew, enum.ValueArgs()...).WithTypeArgs(enum.Name))
		result.AddVariable(supportVar)
		fn := Function(enumTagName, DefaultFlags, enumValidator, supportVarName)
		result.AddFunction(fn)
	}

	return result, nil
}

func (etv *enumTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         etv.TagName(),
		Scopes:      etv.ValidScopes().UnsortedList(),
		Description: "Indicates that a string type is an enum. All const values of this type are considered values in the enum.",
	}
}

func (et *enumType) ValueArgs() []any {
	var values []any
	for _, value := range et.SymbolConstants() {
		values = append(values, value)
	}
	return values
}

func (et *enumType) SymbolConstants() []Identifier {
	var values []Identifier
	for _, value := range et.Values {
		values = append(values, Identifier(value.Name))
	}
	slices.SortFunc(values, func(a, b Identifier) int {
		return cmp.Compare(a.Name, b.Name)
	})
	return values
}

// TODO: Everything below this comment is copied from kube-openapi's enum.go.

type enumValue struct {
	Name    types.Name
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

func parseEnums(c *generator.Context) enumMap {
	// find all enum types.
	enumTypes := make(enumMap)
	for _, p := range c.Universe {
		for _, t := range p.Types {
			if isEnumType(t) {
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
					Name:    c.Name,
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

// isEnumType checks if a given type is an enum by the definition
// An enum type should be an alias of string and has tag '+enum' in its comment.
// Additionally, pass the type of builtin 'string' to check against.
func isEnumType(t *types.Type) bool {
	return t.Kind == types.Alias && t.Underlying == types.String && hasEnumTag(t)
}

func hasEnumTag(t *types.Type) bool {
	return codetags.Extract("+", t.CommentLines)[enumTagName] != nil
}
