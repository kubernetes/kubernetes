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
	"fmt"
	"regexp"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/parser/tags"
	"k8s.io/gengo/v2/types"
)

var discriminatedUnionValidator = types.Name{Package: libValidationPkg, Name: "DiscriminatedUnion"}
var unionValidator = types.Name{Package: libValidationPkg, Name: "Union"}

var newDiscriminatedUnionMember = types.Name{Package: libValidationPkg, Name: "NewDiscriminatedUnionMember"}
var newDiscriminatedUnionMembership = types.Name{Package: libValidationPkg, Name: "NewDiscriminatedUnionMembership"}
var newUnionMember = types.Name{Package: libValidationPkg, Name: "NewUnionMember"}
var newUnionMembership = types.Name{Package: libValidationPkg, Name: "NewUnionMembership"}
var unionVariablePrefix = "unionMembershipFor"

func init() {
	RegisterTagValidator(unionDiscriminatorTagValidator{})
	RegisterTagValidator(unionMemberTagValidator{})
	RegisterAggregateEmitter(AggregateUnionsOrder, unionAggregateEmitter{})
}

const (
	unionDiscriminatorTagName = "k8s:unionDiscriminator"
	unionMemberTagName        = "k8s:unionMember"
)

type unionDiscriminatorTagValidator struct{}

func (unionDiscriminatorTagValidator) Init(_ Config) {}

func (unionDiscriminatorTagValidator) TagName() string {
	return unionDiscriminatorTagName
}

var unionTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal)

func (unionDiscriminatorTagValidator) ValidScopes() sets.Set[Scope] {
	return unionTagValidScopes
}

func (udtv unionDiscriminatorTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            udtv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(udtv.ValidScopes()),
		Description:    "Indicates that this field is the discriminator for a union.",
		Args: []TagArgDoc{{
			Name:        "union",
			Description: "<string>",
			Docs:        "the name of the union, if more than one exists",
			Type:        codetags.ArgTypeString,
		}},
	}
}

type unionMemberTagValidator struct{}

func (unionMemberTagValidator) Init(_ Config) {}

func (unionMemberTagValidator) TagName() string {
	return unionMemberTagName
}

func (unionMemberTagValidator) ValidScopes() sets.Set[Scope] {
	return unionTagValidScopes
}

// unionAggregateEmitter generates union enforcement functions from the union
// metadata collected off member and discriminator tags.
type unionAggregateEmitter struct{}

func (unionAggregateEmitter) Name() string { return "unions" }

func (unionAggregateEmitter) GenerateGroups(context Context, metadata SchemaMetadata) ([]EmittedGroup, error) {
	u1, err := generateUnionGroupsForType(metadata, false, unionVariablePrefix,
		unionMemberTagName, unionValidator, discriminatedUnionValidator,
		Emission{field.ErrorTypeInvalid, "union", ""})
	if err != nil {
		return nil, err
	}
	u2, err := generateUnionGroupsForType(metadata, true, zeroOrOneOfVariablePrefix,
		zeroOrOneOfMemberTagName, zeroOrOneOfUnionValidator, types.Name{},
		Emission{field.ErrorTypeInvalid, "zeroOrOneOf", ""})
	if err != nil {
		return nil, err
	}
	return append(u1, u2...), nil
}

func generateUnionGroupsForType(metadata SchemaMetadata, isZeroOrOneOf bool, varPrefix string, memberTagName string, defaultValidator types.Name, discriminatedValidator types.Name, emit Emission) ([]EmittedGroup, error) {
	sortedUnions := metadata.SortedUnions()
	if len(sortedUnions) == 0 {
		return nil, nil
	}

	type unionGroupKey struct {
		condKey string
		pKey    string
	}
	type unionGroupVal struct {
		cond   Conditions
		unions unions
	}
	grouped := map[unionGroupKey]unionGroupVal{}
	hasUnemitted := false
	for _, uDef := range sortedUnions {
		if uDef.isZeroOrOneOf == isZeroOrOneOf {
			var pStr string
			if uDef.parentPath != nil {
				pStr = uDef.parentPath.String()
			}
			emittedKey := fmt.Sprintf("union:%v:%s:%s:%s", isZeroOrOneOf, uDef.name, pStr, uDef.conditions.Key())
			if !metadata.MarkEmitted(emittedKey) {
				hasUnemitted = true
			}
			key := unionGroupKey{
				condKey: uDef.conditions.Key(),
				pKey:    pStr,
			}
			val := grouped[key]
			if val.unions == nil {
				val.cond = uDef.conditions
				val.unions = unions{}
			}
			val.unions[uDef.name] = uDef
			grouped[key] = val
		}
	}

	keys := make([]unionGroupKey, 0, len(grouped))
	for k := range grouped {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b unionGroupKey) int {
		if cmp := grouped[a].cond.Compare(grouped[b].cond); cmp != 0 {
			return cmp
		}
		return strings.Compare(a.pKey, b.pKey)
	})

	if !hasUnemitted {
		return nil, nil
	}

	var groups []EmittedGroup
	for _, key := range keys {
		group := grouped[key]
		cond := group.cond
		targetUnions := group.unions
		targetUnionsList := targetUnions.SortedList()

		var targetPath *field.Path
		var targetType *types.Type
		if len(targetUnionsList) > 0 {
			uDef := targetUnionsList[0]
			targetPath = uDef.parentPath
			targetType = uDef.parentType
		}

		var targetStabilityLevel ValidationStabilityLevel
		for _, u := range targetUnionsList {
			if u.stabilityLevel != "" {
				if targetStabilityLevel != "" && targetStabilityLevel != u.stabilityLevel {
					return nil, fmt.Errorf("conflicting stability levels %q and %q for union target path %q", targetStabilityLevel, u.stabilityLevel, targetPath)
				}
				targetStabilityLevel = u.stabilityLevel
			}
		}

		result, err := processUnionValidations(targetPath, targetType, targetUnions, varPrefix,
			memberTagName, defaultValidator, discriminatedValidator,
			emit)
		if err != nil {
			return nil, err
		}

		groups = append(groups, EmittedGroup{
			Validations:    result,
			Conditions:     cond,
			StabilityLevel: targetStabilityLevel,
			TargetPath:     targetPath,
			TargetType:     targetType,
			Hoist:          true,
		})
	}

	return groups, nil
}

func (umtv unionMemberTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            umtv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(umtv.ValidScopes()),
		Description:    "Indicates that this field is a member of a union.",
		Args: []TagArgDoc{{
			Name:        "union",
			Description: "<string>",
			Docs:        "the name of the union, if more than one exists",
			Type:        codetags.ArgTypeString,
		}, {
			Name:        "memberName",
			Description: "<string>",
			Docs:        "the discriminator value for this member",
			Default:     "the field's name",
			Type:        codetags.ArgTypeString,
		}},
	}
}

// union defines how a union validation will be generated. Unions can be
// composed of either a set of struct fields (with an optional disctriminator),
// or a set of list items (stored as selection criteria).
type union struct {
	parentPath     *field.Path
	parentType     *types.Type
	name           string
	isZeroOrOneOf  bool
	conditions     Conditions
	stabilityLevel ValidationStabilityLevel

	// members provides field information about all the members of the union.
	// Each item provides a fieldName and discriminatorValue pair, where the
	// name identifies the field or selector (for use in errors) and the
	// discriminatorValue indicates the value which should be used in a
	// discriminated union to name this member.
	members []unionMember

	// fieldMembers describes all the members of a struct-field union.  This is
	// mutually exclusive with itemMembers.
	fieldMembers []*types.Member

	// discriminator is the name of the discriminator field.
	discriminator *string
	// discriminatorMember describes the discriminator field.
	discriminatorMember *types.Member

	// itemMembers stores selection criteria for all the members of a list-item
	// union. This is mutually exclusive with fieldMembers. The map key is the
	// "field name" (eg: `field[{"name": "succeeded"}]`), and the value is a
	// list of selection criteria.
	itemMembers map[string][]ListSelectorTerm
}

// Discriminator returns the discriminator field name or nil if undiscriminated.
func (u *union) Discriminator() *string {
	return u.discriminator
}

// DiscriminatorMember returns the discriminator member or nil if undiscriminated.
func (u *union) DiscriminatorMember() *types.Member {
	return u.discriminatorMember
}

// Members returns the members of the union.
func (u *union) Members() []unionMember {
	return u.members
}

// FieldMembers returns the struct field members of the union.
func (u *union) FieldMembers() []*types.Member {
	return u.fieldMembers
}

// ItemMembers returns the list item selector criteria of the union.
func (u *union) ItemMembers() map[string][]ListSelectorTerm {
	return u.itemMembers
}

// Merge merges another union definition into u.
func (u *union) Merge(other *union) error {
	if other == nil {
		return nil
	}
	if other.discriminator != nil {
		u.discriminator = other.discriminator
		u.discriminatorMember = other.discriminatorMember
	}
	if other.stabilityLevel != "" {
		if u.stabilityLevel != "" && u.stabilityLevel != other.stabilityLevel {
			pStr := ""
			if u.parentPath != nil {
				pStr = u.parentPath.String()
			}
			return fmt.Errorf("conflicting stability levels %q and %q for union target path %q", u.stabilityLevel, other.stabilityLevel, pStr)
		}
		u.stabilityLevel = other.stabilityLevel
	}
	u.conditions = u.conditions.Merge(other.conditions)
	u.isZeroOrOneOf = u.isZeroOrOneOf || other.isZeroOrOneOf
	u.members = append(u.members, other.members...)
	u.fieldMembers = append(u.fieldMembers, other.fieldMembers...)
	if other.itemMembers != nil {
		if u.itemMembers == nil {
			u.itemMembers = make(map[string][]ListSelectorTerm)
		}
		for itemKey, terms := range other.itemMembers {
			u.itemMembers[itemKey] = append(u.itemMembers[itemKey], terms...)
		}
	}
	return nil
}

func (u *union) DeepCopy() *union {
	if u == nil {
		return nil
	}
	out := *u
	if u.members != nil {
		out.members = append([]unionMember(nil), u.members...)
	}
	if u.fieldMembers != nil {
		out.fieldMembers = append([]*types.Member(nil), u.fieldMembers...)
	}
	if u.discriminator != nil {
		d := *u.discriminator
		out.discriminator = &d
	}
	if u.itemMembers != nil {
		out.itemMembers = make(map[string][]ListSelectorTerm, len(u.itemMembers))
		for k, v := range u.itemMembers {
			out.itemMembers[k] = append([]ListSelectorTerm(nil), v...)
		}
	}
	return &out
}

type unionMember struct {
	fieldName          string
	discriminatorValue string
}

// FieldName returns the field name of the union member.
func (m unionMember) FieldName() string {
	return m.fieldName
}

// DiscriminatorValue returns the discriminator value string for this member.
func (m unionMember) DiscriminatorValue() string {
	return m.discriminatorValue
}

// Unions are comprised of multiple tags that need to share information.
// For field-based unions: tags are on struct fields, validation is on the struct
// For item-based unions: tags are on list items (via +k8s:item), validation is on the list
// "shared" maps from field path strings (key) to union definitions (value)
// key examples:
//   - struct union: "MyStruct" (validation on the struct type)
//   - list union: "Pipeline.Tasks" (validation on the list field)
//
// unions are keyed by ParentPath for struct fields (ScopeField), or Path for others.
// TODO: Add support for map items once map item validation is implemented
// unions represents all the unions for a go struct.
type unions map[string]*union

// newUnion initializes a new union instance
func newUnion() *union {
	return &union{
		// slice fields can be nil
		itemMembers: make(map[string][]ListSelectorTerm),
	}
}

// getOrCreate gets a union by name, or initializes a new union by the given name.
func (us unions) getOrCreate(name string) *union {
	var u *union
	var ok bool
	if u, ok = us[name]; !ok {
		u = newUnion()
		u.name = name
		us[name] = u
	}
	return u
}

func (us unions) SortedList() []*union {
	if len(us) == 0 {
		return nil
	}
	keys := make([]string, 0, len(us))
	for k := range us {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	list := make([]*union, 0, len(keys))
	for _, k := range keys {
		list = append(list, us[k])
	}
	return list
}

func processUnionValidations(structPath *field.Path, parentType *types.Type, unions unions, varPrefix string,
	tagName string, undiscriminatedValidator types.Name, discriminatedValidator types.Name, emits Emission,
) (Validations, error) {
	result := Validations{}

	// Sort the keys for stable output.
	keys := make([]string, 0, len(unions))
	for k := range unions {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, unionName := range keys {
		u := unions[unionName]
		if len(u.fieldMembers) > 0 || u.discriminator != nil || len(u.itemMembers) > 0 {
			if len(u.fieldMembers) > 0 && len(u.itemMembers) > 0 {
				return Validations{}, fmt.Errorf("cannot have both field members and item members")
			}
			nativeType := util.NonPointer(util.NativeType(parentType))
			if nativeType.Kind == types.Struct && len(u.itemMembers) > 0 {
				return Validations{}, fmt.Errorf("struct type cannot have item members")
			}
			if nativeType.Kind == types.Slice && len(u.fieldMembers) > 0 {
				return Validations{}, fmt.Errorf("slice type cannot have field members")
			}

			// TODO: Avoid the "local" here. This was added to avoid errors caused when the package is an empty string.
			//       The correct package would be the output package but is not known here. This does not show up in generated code.
			// TODO: Append a consistent hash suffix to avoid generated name conflicts?
			varBaseName := sanitizeName(structPath.String() + "_" + unionName) // unionName can be ""
			supportVarName := PrivateVar{Name: varPrefix + "_" + varBaseName, Package: "local"}

			var extractorArgs []any
			ptrType := types.PointerTo(parentType)

			// Handle field unions
			for _, member := range u.fieldMembers {
				extractor := createMemberExtractor(ptrType, member)
				extractorArgs = append(extractorArgs, extractor)
			}

			// Handle list item unions for lists
			if nativeType.Kind == types.Slice && len(u.itemMembers) > 0 {
				elemType := util.NonPointer(nativeType.Elem)

				// Sort keys for stable output
				keys := make([]string, 0, len(u.itemMembers))
				for key := range u.itemMembers {
					keys = append(keys, key)
				}
				slices.Sort(keys)

				for _, fullPath := range keys {
					selector := u.itemMembers[fullPath]
					extractor, err := createItemExtractor(parentType, elemType, selector)
					if err != nil {
						return Validations{}, err
					}
					extractorArgs = append(extractorArgs, extractor)
				}
			}

			if u.discriminator != nil {
				supportVar := Variable(supportVarName,
					Function(tagName, DefaultFlags, newDiscriminatedUnionMembership,
						append([]any{*u.discriminator}, getMemberArgs(u, parentType, true)...)...))
				result.Variables = append(result.Variables, supportVar)

				discriminatorExtractor := FunctionLiteral{
					Parameters: []ParamResult{{Name: "obj", Type: ptrType}},
					Results:    []ParamResult{{Type: types.String}},
					Body:       fmt.Sprintf("if obj == nil {return \"\"}; return string(obj.%s)", u.discriminatorMember.Name), // Cast to string
				}

				extraArgs := append([]any{supportVarName, discriminatorExtractor}, extractorArgs...)
				fn := Function(tagName, DefaultFlags, discriminatedValidator, extraArgs...).
					WithEmits(emits)
				result.Functions = append(result.Functions, fn)
			} else {
				supportVar := Variable(supportVarName, Function(tagName, DefaultFlags, newUnionMembership, getMemberArgs(u, parentType, false)...))
				result.Variables = append(result.Variables, supportVar)

				extraArgs := append([]any{supportVarName}, extractorArgs...)
				fn := Function(tagName, DefaultFlags, undiscriminatedValidator, extraArgs...).
					WithEmits(emits)
				result.Functions = append(result.Functions, fn)
			}
		}
	}

	return result, nil
}

func createMemberExtractor(ptrType *types.Type, member *types.Member) FunctionLiteral {
	extractor := FunctionLiteral{
		Parameters: []ParamResult{{Name: "obj", Type: ptrType}},
		Results:    []ParamResult{{Type: types.Bool}},
	}
	nt := util.NativeType(member.Type)
	switch nt.Kind {
	case types.Pointer:
		extractor.Body = fmt.Sprintf("if obj == nil {return false}; return obj.%s != nil", member.Name)
	case types.Map, types.Slice:
		extractor.Body = fmt.Sprintf("if obj == nil {return false}; return len(obj.%s) != 0", member.Name)
	case types.Builtin:
		extractor.Body = fmt.Sprintf("if obj == nil {return false}; var z %s; return obj.%s != z", member.Type, member.Name)
	default:
		// This should be caught before we get here, but JIC.
		extractor.Body = fmt.Sprintf("if obj == nil {return false}; return false /* unsupported union member kind: %s */", nt.Kind)
	}
	return extractor
}

// createItemExtractor creates an extractor function for list item union
// members. It generates code that loops through the list to check if an item
// matching the criteria exists.
func createItemExtractor(listType *types.Type, elemType *types.Type, selector []ListSelectorTerm) (FunctionLiteral, error) {
	var criteria []keyValuePair
	for _, term := range selector {
		criteria = append(criteria, keyValuePair{
			key:   term.Field,
			value: fmt.Sprint(term.Value),
		})
	}

	// Sort for stable output.
	slices.SortFunc(criteria, func(a, b keyValuePair) int {
		return strings.Compare(a.key, b.key)
	})

	condition, err := buildMatchConditions(elemType, criteria, "list[i]")
	if err != nil {
		return FunctionLiteral{}, err
	}

	extractor := FunctionLiteral{
		Parameters: []ParamResult{{Name: "list", Type: listType}},
		Results:    []ParamResult{{Type: types.Bool}},
		Body: fmt.Sprintf(
			`for i := range list {
				if %s {
					return true
				}
			 }
			 return false`, condition),
	}

	return extractor, nil
}

// Shared between unionDiscriminatorTagValidator and unionMemberTagValidator.
// Configure discriminator for the union. Validations are emitted by the union member validator.
// processDiscriminatorValidations processes union discriminator tags. It is a
// free function, rather than a method so that it can be called from other
// union-like tags.
func processDiscriminatorValidations(shared map[string]unions, context Context, tag codetags.Tag) error {
	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	if t := util.NonPointer(util.NativeType(context.Type)); t != types.String {
		return fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}
	parentKey := ""
	if context.ParentPath != nil {
		parentKey = context.ParentPath.String()
	}
	if shared[parentKey] == nil {
		shared[parentKey] = unions{}
	}
	unionArg, _ := tag.NamedArg("union") // optional
	u := shared[parentKey].getOrCreate(unionArg.Value)
	u.parentPath = context.ParentPath
	u.parentType = context.ParentType
	if !context.Conditions.Empty() {
		u.conditions = u.conditions.Merge(context.Conditions)
	}
	if context.StabilityLevel != "" {
		if u.stabilityLevel != "" && u.stabilityLevel != context.StabilityLevel {
			return fmt.Errorf("conflicting stability levels %q and %q for union target path %q", u.stabilityLevel, context.StabilityLevel, context.ParentPath)
		}
		u.stabilityLevel = context.StabilityLevel
	}

	var discriminatorFieldName string
	if context.Member != nil {
		if jsonAnnotation, ok := tags.LookupJSON(*context.Member); ok {
			discriminatorFieldName = jsonAnnotation.Name
			u.discriminator = &discriminatorFieldName
			u.discriminatorMember = context.Member
		}
	}

	return nil
}

// processMemberValidations processes union member tags for fields and list
// items.  It is a free function, rather than a method so that it can be called
// from other union-like tags.
func processMemberValidations(shared map[string]unions, context Context, tag codetags.Tag) error {
	switch context.Scope {
	case ScopeField:
		return processFieldMemberValidations(shared, context, tag)
	case ScopeListVal:
		return processListMemberValidations(shared, context, tag)
	}
	return fmt.Errorf("can only be used on fields and list items: %v", context.Scope)
}

// processFieldMemberValidations processes union member tags for struct fields.
// It is a free function, rather than a method so that it can be called from
// other union-like tags.
func processFieldMemberValidations(shared map[string]unions, context Context, tag codetags.Tag) error {
	if context.Member == nil {
		return fmt.Errorf("struct-field union member has no member info in context")
	}
	nt := util.NativeType(context.Member.Type)
	switch nt.Kind {
	case types.Pointer, types.Map, types.Slice, types.Builtin:
		// OK
	default:
		// In particular non-pointer structs are not supported.
		return fmt.Errorf("can only be used on nilable and primitive types (%s)", nt.Kind)
	}

	jsonTag, ok := tags.LookupJSON(*context.Member)
	if !ok {
		return fmt.Errorf("field %q is a union member but has no JSON struct field tag", context.Member.Name)
	}
	fieldName := jsonTag.Name
	if len(fieldName) == 0 {
		return fmt.Errorf("field %q is a union member but has no JSON name", context.Member.Name)
	}

	parentKey := ""
	if context.ParentPath != nil {
		parentKey = context.ParentPath.String()
	}
	if shared[parentKey] == nil {
		shared[parentKey] = unions{}
	}

	// See if the tag specified a member name.
	memberName := context.Member.Name                        // default
	if memberNameArg, ok := tag.NamedArg("memberName"); ok { // optional
		memberName = memberNameArg.Value
	}

	unionArg, _ := tag.NamedArg("union") // optional
	u := shared[parentKey].getOrCreate(unionArg.Value)
	u.parentPath = context.ParentPath
	u.parentType = context.ParentType
	if !context.Conditions.Empty() {
		u.conditions = u.conditions.Merge(context.Conditions)
	}
	if context.StabilityLevel != "" {
		if u.stabilityLevel != "" && u.stabilityLevel != context.StabilityLevel {
			return fmt.Errorf("conflicting stability levels %q and %q for union target path %q", u.stabilityLevel, context.StabilityLevel, context.ParentPath)
		}
		u.stabilityLevel = context.StabilityLevel
	}
	u.members = append(u.members, unionMember{fieldName, memberName})

	u.fieldMembers = append(u.fieldMembers, context.Member)

	return nil
}

// processListMemberValidations processes union member tags for list items.  It
// is a free function, rather than a method so that it can be called from other
// union-like tags.
func processListMemberValidations(shared map[string]unions, context Context, tag codetags.Tag) error {
	if context.ListSelector == nil {
		return fmt.Errorf("list-item union member has no list selector in context")
	}

	// It's not really a "field", but close enough. We don't really NEED the
	// field name, since it is present in the error message, but it is more
	// human-friendly. eg: `field[{"name": "succeeded"}]`
	fieldName := lastPathElement(context.Path)

	parentKey := ""
	if context.ParentPath != nil {
		parentKey = context.ParentPath.String()
	}
	if shared[parentKey] == nil {
		shared[parentKey] = unions{}
	}

	// See if the tag specified a member name.
	memberName := ""
	if memberNameArg, ok := tag.NamedArg("memberName"); ok { // optional
		memberName = memberNameArg.Value
	}

	unionArg, _ := tag.NamedArg("union") // optional
	u := shared[parentKey].getOrCreate(unionArg.Value)
	u.parentPath = context.ParentPath
	u.parentType = context.ParentType
	if !context.Conditions.Empty() {
		u.conditions = u.conditions.Merge(context.Conditions)
	}
	if context.StabilityLevel != "" {
		if u.stabilityLevel != "" && u.stabilityLevel != context.StabilityLevel {
			return fmt.Errorf("conflicting stability levels %q and %q for union target path %q", u.stabilityLevel, context.StabilityLevel, context.ParentPath)
		}
		u.stabilityLevel = context.StabilityLevel
	}
	u.members = append(u.members, unionMember{fieldName, memberName})

	if _, found := u.itemMembers[fieldName]; found {
		return fmt.Errorf("list-item union member %q already exists", fieldName)
	}
	u.itemMembers[fieldName] = context.ListSelector

	return nil
}

func lastPathElement(path *field.Path) string {
	parts := strings.Split(path.String(), ".")
	if len(parts) > 0 {
		return parts[len(parts)-1]
	}
	return ""
}

// getMemberArgs gets a list of arguments which construct union members.
func getMemberArgs(u *union, _ *types.Type, discrim bool) []any {
	members := make([]any, 0, len(u.members))
	for _, f := range u.members {
		fieldName := f.fieldName
		memberName := f.discriminatorValue
		if discrim {
			members = append(members, Function("unused", 0, newDiscriminatedUnionMember, fieldName, memberName))
		} else {
			members = append(members, Function("unused", 0, newUnionMember, fieldName))
		}
	}
	return members
}

// sanitizeName converts a string into a valid Go identifier
func sanitizeName(name string) string {
	name = strings.ReplaceAll(name, ".", "_")
	re := regexp.MustCompile(`[^a-zA-Z0-9_]`)
	return re.ReplaceAllString(name, "_")
}

func (udtv unionDiscriminatorTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	return collectUnionTagMetadata(context, tag)
}

func (umtv unionMemberTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	return collectUnionTagMetadata(context, tag)
}

func (zomtv zeroOrOneOfMemberTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	return collectUnionTagMetadata(context, tag)
}

func collectUnionTagMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	shared := map[string]unions{}
	var err error
	switch tag.Name {
	case unionDiscriminatorTagName:
		err = processDiscriminatorValidations(shared, context, tag)
	case unionMemberTagName:
		err = processMemberValidations(shared, context, tag)
	case zeroOrOneOfMemberTagName:
		err = processZeroOrOneOfMemberTag(shared, context, tag)
	}
	if err != nil {
		return SchemaMetadata{}, err
	}

	res := SchemaMetadata{}
	keys := make([]string, 0, len(shared))
	for k := range shared {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	// Union fragments from all members of a struct must land on the same
	// node so that they merge into one union: key by the enclosing struct's
	// path, which is the parent of the member declaring the tag.
	rootPath := context.ParentPath
	if rootPath == nil {
		rootPath = context.Path
	}
	for _, k := range keys {
		uMap := shared[k]
		node := res.GetOrCreateNode(nodeKeyFor(rootPath))
		for _, uDef := range uMap.SortedList() {
			node.Unions = append(node.Unions, Conditional[*union]{
				Conditions:     uDef.conditions,
				StabilityLevel: uDef.stabilityLevel,
				Payload:        uDef,
			})
		}
	}
	return res, nil
}

func processZeroOrOneOfMemberTag(shared map[string]unions, context Context, tag codetags.Tag) error {
	unionArg, _ := tag.NamedArg("union") // optional
	key := ""
	if context.ParentPath != nil {
		key = context.ParentPath.String()
	}
	if shared[key] == nil {
		shared[key] = unions{}
	}
	u := shared[key].getOrCreate(unionArg.Value)
	u.isZeroOrOneOf = true
	return processMemberValidations(shared, context, tag)
}
