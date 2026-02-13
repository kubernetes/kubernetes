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
	"regexp"
	"slices"
	"strconv"

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

// validGroupNameRegex restricts discriminator group names to identifiers that
// start with a letter and contain only alphanumeric characters and underscores.
var validGroupNameRegex = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9_]*$`)

func init() {
	RegisterTagValidator(&discriminatorTagValidator{discriminatorDefinitions})
	RegisterTagValidator(&memberTagValidator{discriminatorDefinitions, nil})
	RegisterTypeValidator(&discriminatorFieldValidator{discriminatorDefinitions})
	RegisterFieldValidator(&discriminatorFieldValidator{discriminatorDefinitions})
}

// discriminatorDefinitions stores all discriminator definitions found by tag validators.
// Key is the struct path.
var discriminatorDefinitions = map[string]discriminatorGroups{}

type discriminatorGroups map[string]*discriminatorGroup

type discriminatorGroup struct {
	name                string
	discriminatorMember *types.Member
	// members maps field names to their rules in this discriminator group.
	members map[string]*fieldMemberRules
}

type fieldMemberRules struct {
	member *types.Member
	rules  []memberRule
}

type memberRule struct {
	value       string
	validations Validations
}

func (mg discriminatorGroups) getOrCreate(name string) *discriminatorGroup {
	if name == "" {
		name = "default"
	}
	g, ok := mg[name]
	if !ok {
		g = &discriminatorGroup{
			name:    name,
			members: make(map[string]*fieldMemberRules),
		}
		mg[name] = g
	}
	return g
}

type discriminatorTagValidator struct {
	shared map[string]discriminatorGroups
}

func (mtv *discriminatorTagValidator) Init(_ Config) {}

func (mtv *discriminatorTagValidator) TagName() string {
	return discriminatorTagName
}

func (mtv *discriminatorTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField)
}

func (mtv *discriminatorTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if util.NativeType(context.Type).Kind == types.Pointer {
		return Validations{}, fmt.Errorf("can only be used on non-pointer types")
	}

	if t := util.NonPointer(util.NativeType(context.Type)); t.Kind != types.Builtin || (t.Name.Name != "string" && t.Name.Name != "bool" && !types.IsInteger(t)) {
		return Validations{}, fmt.Errorf("can only be used on string, bool or integer types (%s)", rootTypeString(context.Type, t))
	}

	if mtv.shared[context.ParentPath.String()] == nil {
		mtv.shared[context.ParentPath.String()] = make(discriminatorGroups)
	}
	groupName := ""
	if nameArg, ok := tag.NamedArg("name"); ok {
		groupName = nameArg.Value
	}
	if groupName != "" && !validGroupNameRegex.MatchString(groupName) {
		return Validations{}, fmt.Errorf("discriminator group name must match %s, got %q", validGroupNameRegex.String(), groupName)
	}
	if groupName == "default" {
		return Validations{}, fmt.Errorf("discriminator group name %q is reserved", groupName)
	}
	group := mtv.shared[context.ParentPath.String()].getOrCreate(groupName)
	if group.discriminatorMember != nil && group.discriminatorMember != context.Member {
		return Validations{}, fmt.Errorf("duplicate discriminator: %q", groupName)
	}
	group.discriminatorMember = context.Member

	return Validations{}, nil
}

func (mtv *discriminatorTagValidator) Docs() TagDoc {
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
	shared    map[string]discriminatorGroups
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

	groupName := ""
	if modeArg, ok := tag.NamedArg("discriminator"); ok {
		groupName = modeArg.Value
	}
	if groupName == "default" {
		return Validations{}, fmt.Errorf("discriminator group name %q is reserved", groupName)
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
		mtv.shared[context.ParentPath.String()] = make(discriminatorGroups)
	}
	group := mtv.shared[context.ParentPath.String()].getOrCreate(groupName)

	fieldName := context.Member.Name
	if rules, ok := group.members[fieldName]; ok {
		if rules.member != context.Member {
			return Validations{}, fmt.Errorf("internal error: member mismatch for field %q", fieldName)
		}
	} else {
		group.members[fieldName] = &fieldMemberRules{
			member: context.Member,
		}
	}

	payloadValidations, err := mtv.validator.ExtractTagValidations(context, *tag.ValueTag)
	if err != nil {
		return Validations{}, err
	}

	group.members[fieldName].rules = append(group.members[fieldName].rules, memberRule{
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

type discriminatorFieldValidator struct {
	shared map[string]discriminatorGroups
}

func (discriminatorFieldValidator) Init(_ Config) {}

func (discriminatorFieldValidator) Name() string {
	return "discriminatorFieldValidator"
}

func (mtfv *discriminatorFieldValidator) GetValidations(context Context) (Validations, error) {
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
			v, err := mtfv.generateMemberFieldValidation(context, group, rules)
			if err != nil {
				return Validations{}, err
			}
			result.Add(v)
		}
	}

	return result, nil
}

func (mtfv *discriminatorFieldValidator) generateMemberFieldValidation(context Context, group *discriminatorGroup, rules *fieldMemberRules) (Validations, error) {
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

	// Prepare DiscriminatedRules
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

	discriminatorType := group.discriminatorMember.Type

	var discriminatedRules []any
	for _, val := range values {
		ruleValidations := rulesByValue[val]

		wrapper := MultiWrapperFunction{
			Functions: ruleValidations.Functions,
			ObjType:   nilableFieldType,
		}

		// Convert the string tag value to the appropriate typed Go literal
		// for the discriminator type.
		typedValue, err := convertDiscriminatorValue(val, discriminatorType)
		if err != nil {
			return Validations{}, fmt.Errorf("invalid discriminator value %q: %w", val, err)
		}

		discriminatedRules = append(discriminatedRules, StructLiteral{
			Type:     types.Name{Package: libValidationPkg, Name: "DiscriminatedRule"},
			TypeArgs: []*types.Type{nilableFieldType, discriminatorType},
			Fields: []StructLiteralField{
				{Name: "Value", Value: typedValue},
				{Name: "Validation", Value: wrapper},
			},
		})
	}

	discriminatedValidator := types.Name{Package: libValidationPkg, Name: "Discriminated"}

	rulesSlice := SliceLiteral{
		ElementType:     types.Name{Package: libValidationPkg, Name: "DiscriminatedRule"},
		ElementTypeArgs: []*types.Type{nilableFieldType, discriminatorType},
		Elements:        discriminatedRules,
	}

	// getValue extractor
	getValue := FunctionLiteral{
		Parameters: []ParamResult{{Name: "obj", Type: types.PointerTo(context.Type)}},
		Results:    []ParamResult{{Type: nilableFieldType}},
		Body:       fmt.Sprintf("return %sobj.%s", fieldExprPrefix, rules.member.Name),
	}

	// getDiscriminator extractor
	getDiscriminator := FunctionLiteral{
		Parameters: []ParamResult{{Name: "obj", Type: types.PointerTo(context.Type)}},
		Results:    []ParamResult{{Type: discriminatorType}},
		Body:       fmt.Sprintf("return obj.%s", group.discriminatorMember.Name),
	}

	// directComparable is used to determine whether we can use the direct
	// comparison operator "==" or need to use the semantic DeepEqual when
	// looking up and comparing correlated list elements for validation ratcheting.
	var equivArg any
	if util.IsDirectComparable(util.NonPointer(util.NativeType(fieldType))) {
		equivArg = Identifier(validateDirectEqualPtr)
	} else {
		equivArg = Identifier(validateSemanticDeepEqual)
	}

	fn := Function(discriminatorTagName, DefaultFlags, discriminatedValidator,
		Literal(fmt.Sprintf("%q", jsonName)),
		getValue,
		getDiscriminator,
		equivArg,
		defaultForbidden,
		rulesSlice,
	)

	return Validations{Functions: []FunctionGen{fn}}, nil
}

func (mtfv *discriminatorFieldValidator) getForbiddenValidation(t *types.Type) (any, error) {
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
		return nil, fmt.Errorf("discriminated member fields of struct type must be pointers")
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

// convertDiscriminatorValue converts a string tag value to the appropriate
// typed Go literal for the given discriminator type.
func convertDiscriminatorValue(val string, discType *types.Type) (any, error) {
	nt := util.NonPointer(util.NativeType(discType))
	if nt.Kind != types.Builtin {
		return nil, fmt.Errorf("unsupported discriminator type: %s", nt.Name.Name)
	}

	switch nt.Name.Name {
	case "string":
		return val, nil
	case "bool":
		b, err := strconv.ParseBool(val)
		if err != nil {
			return nil, fmt.Errorf("cannot parse %q as bool: %w", val, err)
		}
		return b, nil
	default:
		if types.IsInteger(nt) {
			i, err := strconv.ParseInt(val, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("cannot parse %q as integer: %w", val, err)
			}
			return int(i), nil
		}
		return nil, fmt.Errorf("unsupported discriminator type: %s", nt.Name.Name)
	}
}
