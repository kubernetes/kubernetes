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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/parser/tags"
	"k8s.io/gengo/v2/types"
)

const (
	modeDiscriminatorTagName = "k8s:modeDiscriminator"
	ifModeTagName            = "k8s:ifMode"
)

// validGroupNameRegex restricts discriminator group names to identifiers that
// start with a letter and contain only alphanumeric characters and underscores.
var validGroupNameRegex = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9_]*$`)

func init() {
	RegisterTagValidator(&modeDiscriminatorTagValidator{})
	RegisterTagValidator(&ifModeTagValidator{})
	RegisterAggregateEmitter(AggregateModesOrder, modeAggregateEmitter{})
}

// modeAggregateEmitter generates discriminated-mode validations from the
// discriminator metadata collected off modeDiscriminator and ifMode tags.
type modeAggregateEmitter struct{}

func (modeAggregateEmitter) Name() string { return "modes" }

func (modeAggregateEmitter) GenerateGroups(context Context, metadata SchemaMetadata) ([]EmittedGroup, error) {
	v, err := generateModeValidations(context, metadata)
	if err != nil {
		return nil, err
	}
	if v.Empty() {
		return nil, nil
	}
	return []EmittedGroup{{Validations: v}}, nil
}

// discriminatorDefinitions stores all discriminator definitions found by tag validators.
// Key is the struct path.
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
	value          string
	validations    Validations
	stabilityLevel ValidationStabilityLevel
	conditions     Conditions
}

func (dg *discriminatorGroup) merge(other *discriminatorGroup) error {
	if other == nil {
		return nil
	}
	if other.discriminatorMember != nil {
		if dg.discriminatorMember != nil && dg.discriminatorMember != other.discriminatorMember {
			return fmt.Errorf("duplicate discriminator: %q", dg.name)
		}
		dg.discriminatorMember = other.discriminatorMember
	}
	if len(other.members) > 0 {
		if dg.members == nil {
			dg.members = make(map[string]*fieldMemberRules)
		}
		for fieldName, fmr := range other.members {
			if existingFmr, ok := dg.members[fieldName]; ok {
				if fmr.member != nil {
					existingFmr.member = fmr.member
				}
				for _, r := range fmr.rules {
					existingFmr.rules = append(existingFmr.rules, r.DeepCopy())
				}
			} else {
				dg.members[fieldName] = fmr.DeepCopy()
			}
		}
	}
	return nil
}

func (dg *discriminatorGroup) DeepCopy() *discriminatorGroup {
	if dg == nil {
		return nil
	}
	out := &discriminatorGroup{
		name:                dg.name,
		discriminatorMember: dg.discriminatorMember,
	}
	if dg.members != nil {
		out.members = make(map[string]*fieldMemberRules, len(dg.members))
		for k, v := range dg.members {
			out.members[k] = v.DeepCopy()
		}
	}
	return out
}

func (fmr *fieldMemberRules) DeepCopy() *fieldMemberRules {
	if fmr == nil {
		return nil
	}
	out := &fieldMemberRules{
		member: fmr.member,
	}
	if fmr.rules != nil {
		out.rules = make([]memberRule, len(fmr.rules))
		for i, r := range fmr.rules {
			out.rules[i] = r.DeepCopy()
		}
	}
	return out
}

func (mr memberRule) DeepCopy() memberRule {
	out := mr
	out.validations = mr.validations.Clone()
	return out
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

func (mg discriminatorGroups) SortedList() []*discriminatorGroup {
	if len(mg) == 0 {
		return nil
	}
	keys := make([]string, 0, len(mg))
	for k := range mg {
		keys = append(keys, k)
	}
	// Sort group names for deterministic output
	slices.Sort(keys)

	list := make([]*discriminatorGroup, 0, len(keys))
	for _, k := range keys {
		list = append(list, mg[k])
	}
	return list
}

type modeDiscriminatorTagValidator struct {
	extractor Extractor
}

func (mdtv *modeDiscriminatorTagValidator) Init(cfg Config) {
	mdtv.extractor = cfg.Extractor
}

func (mdtv *modeDiscriminatorTagValidator) TagName() string {
	return modeDiscriminatorTagName
}

func (mdtv *modeDiscriminatorTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeType, ScopeField)
}

// collectModes gathers discriminator metadata. Validations are emitted by modeAggregateEmitter during type validation extraction.
func (mdtv *modeDiscriminatorTagValidator) collectModes(context Context, tag codetags.Tag) (discriminatorGroups, error) {
	if context.Member == nil {
		return nil, nil
	}
	member := context.Member
	groups := make(discriminatorGroups)

	if tag.Name == modeDiscriminatorTagName {
		if util.NativeType(member.Type).Kind == types.Pointer {
			return nil, fmt.Errorf("can only be used on non-pointer types")
		}
		if t := util.NonPointer(util.NativeType(member.Type)); t.Kind != types.Builtin || (t.Name.Name != "string" && t.Name.Name != "bool") {
			return nil, fmt.Errorf("can only be used on string or bool types (%s)", rootTypeString(member.Type, t))
		}
		groupName := ""
		if nameArg, ok := tag.NamedArg("modality"); ok {
			groupName = nameArg.Value
		}
		if groupName != "" && !validGroupNameRegex.MatchString(groupName) {
			return nil, fmt.Errorf("discriminator group name must match %s, got %q", validGroupNameRegex.String(), groupName)
		}
		if groupName == "default" {
			return nil, fmt.Errorf("discriminator group name %q is reserved", groupName)
		}
		group := groups.getOrCreate(groupName)
		if group.discriminatorMember != nil && group.discriminatorMember != member {
			return nil, fmt.Errorf("duplicate discriminator: %q", groupName)
		}
		group.discriminatorMember = member
	}

	if tag.Name == ifModeTagName {
		if tag.ValueTag == nil {
			return nil, fmt.Errorf("missing required payload")
		}
		groupName := ""
		if modeArg, ok := tag.NamedArg("modality"); ok {
			groupName = modeArg.Value
		}
		if groupName == "default" {
			return nil, fmt.Errorf("discriminator group name %q is reserved", groupName)
		}
		value := ""
		if valArg, ok := tag.NamedArg("mode"); ok {
			value = valArg.Value
		} else if len(tag.Args) > 0 && tag.Args[0].Name == "" {
			value = tag.Args[0].Value
		} else {
			return nil, fmt.Errorf("missing required mode")
		}

		group := groups.getOrCreate(groupName)
		fieldName := member.Name
		if rules, ok := group.members[fieldName]; ok {
			if rules.member != member {
				return nil, fmt.Errorf("internal error: member mismatch for field %q", fieldName)
			}
		} else {
			group.members[fieldName] = &fieldMemberRules{member: member}
		}

		payloadValidations, err := mdtv.extractor.ExtractTagValidations(context, SchemaMetadata{}, *tag.ValueTag)
		if err != nil {
			return nil, err
		}

		group.members[fieldName].rules = append(group.members[fieldName].rules, memberRule{
			value:          value,
			validations:    payloadValidations,
			stabilityLevel: context.StabilityLevel,
			conditions:     context.Conditions,
		})
	}

	if len(groups) == 0 {
		return nil, nil
	}

	return groups, nil
}

func (mdtv *modeDiscriminatorTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	modes, err := mdtv.collectModes(context, tag)
	if err != nil || len(modes) == 0 {
		return SchemaMetadata{}, err
	}
	res := SchemaMetadata{}
	// Mode groups belong to the enclosing struct: key by its path, which is
	// the parent of the member declaring the tag.
	rootPath := context.ParentPath
	if rootPath == nil {
		rootPath = context.Path
	}
	node := res.GetOrCreateNode(nodeKeyFor(rootPath))
	for _, g := range modes.SortedList() {
		node.Modes = append(node.Modes, Conditional[*discriminatorGroup]{
			Conditions:     context.Conditions,
			StabilityLevel: context.StabilityLevel,
			Payload:        g,
		})
	}
	return res, nil
}

func generateModeValidations(context Context, metadata SchemaMetadata) (Validations, error) {
	modes, err := metadata.Modes()
	if err != nil {
		return Validations{}, err
	}
	if len(modes) == 0 {
		return Validations{}, nil
	}

	hasUnemitted := false
	for _, m := range modes {
		emittedKey := fmt.Sprintf("mode:%s", m.name)
		if !metadata.MarkEmitted(emittedKey) {
			hasUnemitted = true
		}
	}
	if !hasUnemitted {
		return Validations{}, nil
	}

	return getDiscriminatorValidations(modes, context)
}

func (mdtv *modeDiscriminatorTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mdtv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(mdtv.ValidScopes()),
		Description:    "Indicates that this field is a discriminator for state-based validation.",
		Args: []TagArgDoc{{
			Name:        "modality",
			Description: "<string>",
			Docs:        "the name of the discriminator group, if more than one exists",
			Type:        codetags.ArgTypeString,
		}},
	}
}

type ifModeTagValidator struct{}

func (imtv *ifModeTagValidator) Init(_ Config) {}

func (imtv *ifModeTagValidator) TagName() string {
	return ifModeTagName
}

func (imtv *ifModeTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField)
}

func (imtv *ifModeTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	mdtv := &modeDiscriminatorTagValidator{extractor: globalRegistry}
	return mdtv.CollectMetadata(context, tag)
}

func (imtv *ifModeTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            imtv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(imtv.ValidScopes()),
		Description:    "Indicates that this field's validation depends on a mode discriminator.",
		Args: []TagArgDoc{{
			Name:        "", // positional
			Description: "<string>",
			Docs:        "the value of the mode discriminator for which this validation applies",
			Type:        codetags.ArgTypeString,
		}, {
			Name:        "modality",
			Description: "<string>",
			Docs:        "the name of the discriminator group",
			Type:        codetags.ArgTypeString,
		}, {
			Name:        "mode",
			Description: "<string>",
			Docs:        "the mode value for which this validation applies",
			Type:        codetags.ArgTypeString,
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
}

func getDiscriminatorValidations(groups discriminatorGroups, context Context) (Validations, error) {
	if k := util.NonPointer(util.NativeType(context.Type)).Kind; k != types.Struct {
		return Validations{}, nil
	}

	if len(groups) == 0 {
		return Validations{}, nil
	}

	var result Validations
	structType := util.NonPointer(util.NativeType(context.Type))

	for _, group := range groups.SortedList() {
		gn := group.name
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
			v, err := generateMemberFieldValidation(structType, group, rules)
			if err != nil {
				return Validations{}, err
			}
			result.Add(v)
		}
	}

	return result, nil
}

func generateMemberFieldValidation(structType *types.Type, group *discriminatorGroup, rules *fieldMemberRules) (Validations, error) {
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

	// Default-Forbidden runs at structPath.Child(jsonName) inside the
	// Discriminated runtime; attribute it to "." + jsonName.
	defaultForbidden, err := getForbiddenValidation(fieldType, "."+jsonName)
	if err != nil {
		return Validations{}, err
	}

	commonCond := uniformConditions(rules.rules)

	// Prepare DiscriminatedRules
	// Mark each rule's validation functions with their stability level before
	// aggregating by value, so that different rules for the same value can
	// carry different stability levels.
	rulesByValue := make(map[string]*Validations)
	var values []string
	for _, rule := range rules.rules {
		// Track unique discriminator values in order of first appearance.
		if _, ok := rulesByValue[rule.value]; !ok {
			rulesByValue[rule.value] = &Validations{}
			values = append(values, rule.value)
		}
		ruleCond := rule.conditions
		if !commonCond.Empty() {
			ruleCond = Conditions{}
		}
		v, err := FinalizeGroup(Context{Type: nilableFieldType}, EmittedGroup{
			Validations:    rule.validations,
			Conditions:     ruleCond,
			StabilityLevel: rule.stabilityLevel,
		})
		if err != nil {
			return Validations{}, err
		}
		// Accumulate this rule's validations with others that share the same discriminator value.
		rulesByValue[rule.value].Add(v)
	}
	slices.Sort(values)

	// When all rules share the same stability level, the default-forbidden
	// (which fires for unrecognized discriminator values) should also be marked
	// with that level so its errors carry the same stability annotation.
	if uniformLevel := uniformStabilityLevel(rules.rules); uniformLevel != "" {
		mwf, ok := defaultForbidden.(MultiWrapperFunction)
		if !ok {
			return Validations{}, fmt.Errorf("internal error: defaultForbidden is not a MultiWrapperFunction")
		}
		marked := make([]FunctionGen, len(mwf.Functions))
		for i, f := range mwf.Functions {
			marked[i] = f.WithStabilityLevel(uniformLevel)
		}
		mwf.Functions = marked
		defaultForbidden = mwf
	}

	discriminatorType := group.discriminatorMember.Type
	var discriminatedRules []any
	for _, val := range values {
		wrapper := MultiWrapperFunction{
			Functions: rulesByValue[val].Functions,
			ObjType:   nilableFieldType,
			// Per-mode rules also run at structPath.Child(jsonName).
			PathFragment: "." + jsonName,
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
		Parameters: []ParamResult{{Name: "obj", Type: types.PointerTo(structType)}},
		Results:    []ParamResult{{Type: nilableFieldType}},
		Body:       fmt.Sprintf("return %sobj.%s", fieldExprPrefix, rules.member.Name),
	}

	// getDiscriminator extractor
	getDiscriminator := FunctionLiteral{
		Parameters: []ParamResult{{Name: "obj", Type: types.PointerTo(structType)}},
		Results:    []ParamResult{{Type: discriminatorType}},
		Body:       fmt.Sprintf("return obj.%s", group.discriminatorMember.Name),
	}

	// directComparable is used to determine whether we can use the direct
	// comparison operator "==" or need to use the semantic DeepEqual when
	// looking up and comparing correlated list elements for validation ratcheting.
	var equivArg any
	if util.IsDirectComparable(util.NonPointer(util.NativeType(fieldType))) {
		equivArg = Identifier(validateDirectEqual)
	} else {
		equivArg = Identifier(validateSemanticDeepEqual)
	}

	fn := Function(modeDiscriminatorTagName, DefaultFlags, discriminatedValidator,
		Literal(fmt.Sprintf("%q", jsonName)),
		getValue,
		getDiscriminator,
		equivArg,
		defaultForbidden,
		rulesSlice,
	)
	// Stability levels are already set on the wrapped validation functions, so
	// skip the level wrapping in the upstream. Processing the stability level
	// in the upstream will override the stability levels of the wrapped validators.
	fn.StabilityLevelSelfManaged = true

	return FinalizeGroup(Context{Type: structType}, EmittedGroup{
		Validations: Validations{Functions: []FunctionGen{fn}},
		Conditions:  commonCond,
	})
}

// uniformConditions returns the common conditions if all rules share
// the same ones, or Conditions{} if they differ.
func uniformConditions(rules []memberRule) Conditions {
	if len(rules) == 0 {
		return Conditions{}
	}
	cond := rules[0].conditions
	for i := 1; i < len(rules); i++ {
		if rules[i].conditions.Compare(cond) != 0 {
			return Conditions{}
		}
	}
	return cond
}

// uniformStabilityLevel returns the common stability level if all rules share
// the same one, or "" if they differ.
func uniformStabilityLevel(rules []memberRule) ValidationStabilityLevel {
	level := rules[0].stabilityLevel
	for i := 1; i < len(rules); i++ {
		if rules[i].stabilityLevel != level {
			return ""
		}
	}
	return level
}

// getForbiddenValidation returns a MultiWrapperFunction wrapping the runtime
// validate.Forbidden* call appropriate for t's kind. pathFragment is the
// wrapper's PathFragment (see MultiWrapperFunction).
func getForbiddenValidation(t *types.Type, pathFragment string) (any, error) {
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

	fg := Function(forbiddenTagName, DefaultFlags, forbidden).
		WithEmits(Emission{field.ErrorTypeForbidden, "", ""})

	// Use the nilable form to match standard validation function signatures.
	wrapperObjType := t
	if !util.IsNilableType(t) {
		wrapperObjType = types.PointerTo(t)
	}

	return MultiWrapperFunction{
		Functions:    []FunctionGen{fg},
		ObjType:      wrapperObjType,
		PathFragment: pathFragment,
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
		b, err := util.ParseBool(val)
		if err != nil {
			return nil, fmt.Errorf("cannot parse %q as bool: %w", val, err)
		}
		return b, nil
	default:
		return nil, fmt.Errorf("unsupported discriminator type: %s", nt.Name.Name)
	}
}
