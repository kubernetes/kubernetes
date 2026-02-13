/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"slices"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/parser/tags"
	"k8s.io/gengo/v2/types"
)

const (
	discriminatorTagName = "k8s:discriminator"
	memberTagName        = "k8s:member"
)

func init() {
	RegisterTagValidator(&modeTagValidator{modeDefinitions})
	RegisterTagValidator(&memberTagValidator{modeDefinitions, nil})
	RegisterTypeValidator(&modeTypeOrFieldValidator{modeDefinitions})
	RegisterFieldValidator(&modeTypeOrFieldValidator{modeDefinitions})
}

// modeDefinitions stores all mode definitions found by tag validators.
// Key is the struct path.
var modeDefinitions = map[string]modeGroups{}

type modeGroups map[string]*modeGroup

type modeGroup struct {
	name                string
	discriminatorMember *types.Member
	// members maps field names to their rules in this mode group.
	members map[string]*fieldModeRules
}

type fieldModeRules struct {
	member *types.Member
	rules  []modeRule
}

type modeRule struct {
	value       string
	validations Validations
}

func (mg modeGroups) getOrCreate(name string) *modeGroup {
	if name == "" {
		name = "default"
	}
	g, ok := mg[name]
	if !ok {
		g = &modeGroup{
			name:    name,
			members: make(map[string]*fieldModeRules),
		}
		mg[name] = g
	}
	return g
}

type modeTagValidator struct {
	shared map[string]modeGroups
}

func (mtv *modeTagValidator) Init(_ Config) {}

func (mtv *modeTagValidator) TagName() string {
	return discriminatorTagName
}

func (mtv *modeTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField)
}

func (mtv *modeTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if util.NativeType(context.Type).Kind == types.Pointer {
		return Validations{}, fmt.Errorf("can only be used on non-pointer types")
	}

	if t := util.NonPointer(util.NativeType(context.Type)); t.Kind != types.Builtin || (t.Name.Name != "string" && t.Name.Name != "bool" && !types.IsInteger(t)) {
		return Validations{}, fmt.Errorf("can only be used on string, bool or integer types (%s)", rootTypeString(context.Type, t))
	}

	if mtv.shared[context.ParentPath.String()] == nil {
		mtv.shared[context.ParentPath.String()] = make(modeGroups)
	}
	modeName := ""
	if nameArg, ok := tag.NamedArg("name"); ok {
		modeName = nameArg.Value
	}
	group := mtv.shared[context.ParentPath.String()].getOrCreate(modeName)
	if group.discriminatorMember != nil && group.discriminatorMember != context.Member {
		return Validations{}, fmt.Errorf("duplicate discriminator: %q", modeName)
	}
	group.discriminatorMember = context.Member

	return Validations{}, nil
}

func (mtv *modeTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mtv.TagName(),
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         mtv.ValidScopes().UnsortedList(),
		Description:    "Indicates that this field is a discriminator for state-based validation.",
		Args: []TagArgDoc{{
			Name:        "name",
			Description: "<string>",
			Docs:        "the name of the discriminator group, if more than one exists",
			Type:        codetags.ArgTypeString,
		}},
	}
}

type memberTagValidator struct {
	shared    map[string]modeGroups
	validator TagValidationExtractor
}

func (mtv *memberTagValidator) Init(cfg Config) {
	mtv.validator = cfg.TagValidator
}

func (mtv *memberTagValidator) TagName() string {
	return memberTagName
}

func (mtv *memberTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField)
}

var disallowedPayloadTags = sets.New(
	listTypeTagName,
	ListMapKeyTagName,
	unionDiscriminatorTagName,
	unionMemberTagName,
	zeroOrOneOfMemberTagName,
	discriminatorTagName,
	memberTagName,
)

func (mtv *memberTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if tag.ValueTag == nil {
		return Validations{}, fmt.Errorf("missing required payload")
	}

	if disallowedPayloadTags.Has(tag.ValueTag.Name) {
		return Validations{}, fmt.Errorf("unsupported payload tag: %q", tag.ValueTag.Name)
	}

	modeName := ""
	if modeArg, ok := tag.NamedArg("discriminator"); ok {
		modeName = modeArg.Value
	}

	value := ""
	if valArg, ok := tag.NamedArg("value"); ok {
		value = valArg.Value
	} else if len(tag.Args) > 0 && tag.Args[0].Name == "" {
		// Positional argument
		value = tag.Args[0].Value
	} else {
		return Validations{}, fmt.Errorf("missing required value")
	}

	if mtv.shared[context.ParentPath.String()] == nil {
		mtv.shared[context.ParentPath.String()] = make(modeGroups)
	}
	group := mtv.shared[context.ParentPath.String()].getOrCreate(modeName)

	fieldName := context.Member.Name
	if rules, ok := group.members[fieldName]; ok {
		if rules.member != context.Member {
			return Validations{}, fmt.Errorf("internal error: member mismatch for field %q", fieldName)
		}
	} else {
		group.members[fieldName] = &fieldModeRules{
			member: context.Member,
		}
	}

	payloadValidations, err := mtv.validator.ExtractTagValidations(context, *tag.ValueTag)
	if err != nil {
		return Validations{}, err
	}

	group.members[fieldName].rules = append(group.members[fieldName].rules, modeRule{
		value:       value,
		validations: payloadValidations,
	})

	return Validations{}, nil
}

func (mtv *memberTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mtv.TagName(),
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         mtv.ValidScopes().UnsortedList(),
		Description:    "Indicates that this field's validation depends on a discriminator.",
		Args: []TagArgDoc{{
			Name:        "", // positional
			Description: "<string>",
			Docs:        "the value of the discriminator for which this validation applies",
			Type:        codetags.ArgTypeString,
		}, {
			Name:        "discriminator",
			Description: "<string>",
			Docs:        "the name of the discriminator group",
			Type:        codetags.ArgTypeString,
		}, {
			Name:        "value",
			Description: "<string>",
			Docs:        "the value of the discriminator for which this validation applies",
			Type:        codetags.ArgTypeString,
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
}

type modeTypeOrFieldValidator struct {
	shared map[string]modeGroups
}

func (modeTypeOrFieldValidator) Init(_ Config) {}

func (modeTypeOrFieldValidator) Name() string {
	return "modeTypeOrFieldValidator"
}

func (mtfv *modeTypeOrFieldValidator) GetValidations(context Context) (Validations, error) {
	// Extract the most concrete type possible.
	if k := util.NonPointer(util.NativeType(context.Type)).Kind; k != types.Struct {
		return Validations{}, nil
	}

	groups, ok := mtfv.shared[context.Path.String()]
	if !ok || len(groups) == 0 {
		return Validations{}, nil
	}

	var result Validations

	// Sort group names for deterministic output
	groupNames := make([]string, 0, len(groups))
	for name := range groups {
		groupNames = append(groupNames, name)
	}
	slices.Sort(groupNames)

	for _, gn := range groupNames {
		group := groups[gn]
		if group.discriminatorMember == nil {
			if len(group.members) > 0 {
				if gn == "default" {
					return Validations{}, fmt.Errorf("missing discriminator")
				}
				return Validations{}, fmt.Errorf("missing discriminator for group %q", gn)
			}
			continue
		}

		fieldNames := make([]string, 0, len(group.members))
		for name := range group.members {
			fieldNames = append(fieldNames, name)
		}
		slices.Sort(fieldNames)

		for _, fn := range fieldNames {
			rules := group.members[fn]
			v, err := mtfv.generateModeFieldValidation(context, group, rules)
			if err != nil {
				return Validations{}, err
			}
			result.Add(v)
		}
	}

	return result, nil
}

func (mtfv *modeTypeOrFieldValidator) generateModeFieldValidation(context Context, group *modeGroup, rules *fieldModeRules) (Validations, error) {
	fieldType := rules.member.Type

	// Use the nilable form to handle missing values.
	nilableFieldType := fieldType
	fieldExprPrefix := ""
	if !util.IsNilableType(nilableFieldType) {
		nilableFieldType = types.PointerTo(nilableFieldType)
		fieldExprPrefix = "&"
	}

	// Get the JSON name of the field
	jsonName := rules.member.Name
	if jt, ok := tags.LookupJSON(*rules.member); ok {
		jsonName = jt.Name
	}

	// Default validation is Forbidden
	defaultForbidden, err := mtfv.getForbiddenValidation(fieldType)
	if err != nil {
		return Validations{}, err
	}

	// Prepare ModalRules
	// Aggregate rules by value
	rulesByValue := make(map[string]Validations)
	var values []string
	for _, rule := range rules.rules {
		if _, ok := rulesByValue[rule.value]; !ok {
			values = append(values, rule.value)
		}
		v := rulesByValue[rule.value]
		v.Add(rule.validations)
		rulesByValue[rule.value] = v
	}
	slices.Sort(values)

	var modalRules []any
	for _, val := range values {
		ruleValidations := rulesByValue[val]

		wrapper := MultiWrapperFunction{
			Functions: ruleValidations.Functions,
			ObjType:   nilableFieldType,
		}

		modalRules = append(modalRules, StructLiteral{
			Type:     types.Name{Package: libValidationPkg, Name: "ModalRule"},
			TypeArgs: []*types.Type{nilableFieldType},
			Fields: []StructLiteralField{
				{Name: "Value", Value: val},
				{Name: "Validation", Value: wrapper},
			},
		})
	}

	modalValidator := types.Name{Package: libValidationPkg, Name: "Modal"}

	rulesSlice := SliceLiteral{
		ElementType:     types.Name{Package: libValidationPkg, Name: "ModalRule"},
		ElementTypeArgs: []*types.Type{nilableFieldType},
		Elements:        modalRules,
	}

	// getValue extractor
	getValue := FunctionLiteral{
		Parameters: []ParamResult{{Name: "obj", Type: types.PointerTo(context.Type)}},
		Results:    []ParamResult{{Type: nilableFieldType}},
		Body:       fmt.Sprintf("return %sobj.%s", fieldExprPrefix, rules.member.Name),
	}

	// getDiscriminator extractor
	discriminatorType := group.discriminatorMember.Type
	getDiscriminator := FunctionLiteral{
		Parameters: []ParamResult{{Name: "obj", Type: types.PointerTo(context.Type)}},
		Results:    []ParamResult{{Type: discriminatorType}},
		Body:       fmt.Sprintf("return obj.%s", group.discriminatorMember.Name),
	}

	// Calculate equivArg for ratcheting the value
	var equivArg any
	if util.IsDirectComparable(util.NonPointer(util.NativeType(fieldType))) {
		equivArg = Identifier(validateDirectEqualPtr)
	} else {
		equivArg = Identifier(validateSemanticDeepEqual)
	}

	fn := Function(discriminatorTagName, DefaultFlags, modalValidator,
		Literal(fmt.Sprintf("%q", jsonName)),
		getValue,
		getDiscriminator,
		equivArg,
		defaultForbidden,
		rulesSlice,
	)

	return Validations{Functions: []FunctionGen{fn}}, nil
}

func (mtfv *modeTypeOrFieldValidator) getForbiddenValidation(t *types.Type) (any, error) {
	var forbidden types.Name
	nt := util.NativeType(t)
	switch nt.Kind {
	case types.Slice:
		forbidden = types.Name{Package: libValidationPkg, Name: "ForbiddenSlice"}
	case types.Map:
		forbidden = types.Name{Package: libValidationPkg, Name: "ForbiddenMap"}
	case types.Pointer:
		forbidden = types.Name{Package: libValidationPkg, Name: "ForbiddenPointer"}
	case types.Struct:
		return nil, fmt.Errorf("modal member fields of struct type must be pointers")
	default:
		forbidden = types.Name{Package: libValidationPkg, Name: "ForbiddenValue"}
	}

	fg := Function(forbiddenTagName, DefaultFlags, forbidden)

	// Use the nilable form to match standard validation function signatures.
	wrapperObjType := t
	if !util.IsNilableType(t) {
		wrapperObjType = types.PointerTo(t)
	}

	return MultiWrapperFunction{
		Functions: []FunctionGen{fg},
		ObjType:   wrapperObjType,
	}, nil
}
