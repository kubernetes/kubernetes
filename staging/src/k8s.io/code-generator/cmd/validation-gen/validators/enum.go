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
	"k8s.io/gengo/v2/types"
)

const (
	enumTagName        = "k8s:enum"
	enumExcludeTagName = "k8s:enumExclude"
)

func init() {
	RegisterTagValidator(&enumTagValidator{})
	RegisterTagValidator(&enumExcludeTagValidator{})
}

type enumExcludeTagValidator struct {
}

func (*enumExcludeTagValidator) Init(_ Config) {
}

func (*enumExcludeTagValidator) TagName() string {
	return enumExcludeTagName
}

var enumExcludeValidScope = sets.New(ScopeConst)

func (*enumExcludeTagValidator) ValidScopes() sets.Set[Scope] {
	return enumExcludeValidScope
}

func (*enumExcludeTagValidator) GetValidations(_ Context, _ codetags.Tag) (Validations, error) {
	return Validations{}, nil
}

func (eetv *enumExcludeTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            eetv.TagName(),
		StabilityLevel: Alpha,
		Scopes:         eetv.ValidScopes().UnsortedList(),
		Description: `Indicates that an constant value is not part of an enum, even if the constant's type is tagged with k8s:enum.
May be conditionally excluded via +k8s:ifEnabled(Option)=+k8s:enumExclude or +k8s:ifDisabled(Option)=+k8s:enumExclude.
If multiple +k8s:ifEnabled/+k8s:ifDisabled tags are used, the value is excluded if any of the exclude conditions are met.`,
	}
}

type enumTagValidator struct {
	validator Validator
}

func (etv *enumTagValidator) Init(cfg Config) {
	etv.validator = cfg.Validator
}

func (enumTagValidator) TagName() string {
	return enumTagName
}

var enumTagValidScopes = sets.New(ScopeType)

func (enumTagValidator) ValidScopes() sets.Set[Scope] {
	return enumTagValidScopes
}

var (
	enumValidator     = types.Name{Package: libValidationPkg, Name: "Enum"}
	enumExclusionType = types.Name{Package: libValidationPkg, Name: "EnumExclusion"}
	setsNew           = types.Name{Package: "k8s.io/apimachinery/pkg/util/sets", Name: "New"}
)

func (etv *enumTagValidator) GetValidations(context Context, _ codetags.Tag) (Validations, error) {
	// NOTE: typedefs to pointers are not supported, so we should never see a pointer here.
	if t := util.NativeType(context.Type); t != types.String {
		return Validations{}, fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}

	enum := &enumType{Name: context.Type.Name}
	for _, c := range context.Constants {
		var exclusions []enumExclude
		isExcluded := false
		for _, tag := range c.Tags {
			switch tag.Name {
			case enumExcludeTagName:
				isExcluded = true
			case ifEnabledTag, ifDisabledTag:
				if tag.ValueTag != nil && tag.ValueTag.Name == enumExcludeTagName {
					if option, ok := tag.PositionalArg(); ok {
						exclusions = append(exclusions, enumExclude{
							excludeWhen: tag.Name == ifEnabledTag,
							option:      option.Value,
						})
					}
				}
			}
		}
		if isExcluded {
			continue
		}

		value := &enumValue{
			Name:       c.Constant.Name,
			Value:      *c.Constant.ConstValue,
			Comment:    strings.Join(c.Constant.CommentLines, " "),
			Exclusions: exclusions,
		}
		enum.addIfNotPresent(value)
	}

	// Sort the values for the codegen that happens later.
	slices.SortFunc(enum.Values, func(a, b *enumValue) int {
		return cmp.Compare(a.Name.Name, b.Name.Name)
	})
	for _, v := range enum.Values {
		slices.SortFunc(v.Exclusions, func(a, b enumExclude) int {
			if a.excludeWhen == b.excludeWhen {
				return cmp.Compare(a.option, b.option)
			}
			if a.excludeWhen {
				return 1
			}
			return -1
		})
	}

	var result Validations

	// TODO: Avoid the "local" here. This was added to avoid errors caused when the package is an empty string.
	//       The correct package would be the output package but is not known here. This does not show up in generated code.
	// TODO: Append a consistent hash suffix to avoid generated name conflicts?
	symbolsVarName := PrivateVar{Name: "SymbolsFor" + context.Type.Name.Name, Package: "local"}
	var allValues []any
	var exclusionRules []any
	for _, v := range enum.Values {
		allValues = append(allValues, Identifier(v.Name))
		for _, exclusion := range v.Exclusions {
			exclusionRules = append(exclusionRules, StructLiteral{
				Type:     enumExclusionType,
				TypeArgs: []*types.Type{context.Type},
				Fields: []StructLiteralField{
					{"Value", Identifier(v.Name)},
					{"Option", exclusion.option},
					{"ExcludeWhen", exclusion.excludeWhen},
				},
			})
		}
	}
	initFn := Function("setsNew", DefaultFlags, setsNew, allValues...)
	if len(allValues) == 0 {
		initFn = initFn.WithTypeArgs(enum.Name)
	}
	result.AddVariable(Variable(symbolsVarName, initFn))
	var exclusions any = Literal("nil")
	if len(exclusionRules) > 0 {
		exclusionsVar := PrivateVar{Name: "ExclusionsFor" + context.Type.Name.Name, Package: "local"}
		result.AddVariable(Variable(exclusionsVar, SliceLiteral{
			ElementType:     enumExclusionType,
			ElementTypeArgs: []*types.Type{context.Type},
			Elements:        exclusionRules,
		}))
		exclusions = exclusionsVar
	}
	fn := Function(enumTagName, DefaultFlags, enumValidator, symbolsVarName, exclusions)
	result.AddFunction(fn)

	return result, nil
}

func (etv *enumTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            etv.TagName(),
		StabilityLevel: Beta,
		Scopes:         etv.ValidScopes().UnsortedList(),
		Description:    "Indicates that a string type is an enum. All constant values of this type are considered values in the enum unless excluded using +k8s:enumExclude.",
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

type enumType struct {
	Name   types.Name
	Values []*enumValue
}

type enumValue struct {
	Name       types.Name
	Value      string
	Comment    string
	Exclusions []enumExclude
}

type enumExclude struct {
	// excludeWhen determines the condition for exclusion.
	// If true, the value is excluded if the option is present.
	// If false, the value is excluded if the option is NOT present.
	excludeWhen bool
	// option is the name of the feature option that controls the exclusion.
	option string
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
