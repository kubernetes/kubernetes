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
	// Unions are comprised of multiple tags that need to share information.
	// For field-based unions: tags are on struct fields, validation is on the struct
	// For item-based unions: tags are on list items (via +k8s:item), validation is on the list

	// "shared" maps from field path strings (key) to union definitions (value)
	// key examples:
	//   - struct union: "MyStruct" (validation on the struct type)
	//   - list union: "Pipeline.Tasks" (validation on the list field)
	shared := map[string]unions{}
	RegisterTypeValidator(unionTypeOrFieldValidator{shared})
	RegisterFieldValidator(unionTypeOrFieldValidator{shared})
	RegisterTagValidator(unionDiscriminatorTagValidator{shared})
	RegisterTagValidator(unionMemberTagValidator{shared})
}

type unionTypeOrFieldValidator struct {
	shared map[string]unions
}

func (unionTypeOrFieldValidator) Init(_ Config) {}

func (unionTypeOrFieldValidator) Name() string {
	return "unionTypeOrFieldValidator"
}

func (utfv unionTypeOrFieldValidator) GetValidations(context Context) (Validations, error) {
	// TODO: Add support for map items once map item validation is implemented

	// Extract the most concrete type possible.
	if k := util.NonPointer(util.NativeType(context.Type)).Kind; k != types.Struct && k != types.Slice {
		return Validations{}, nil
	}

	unions := utfv.shared[context.Path.String()]
	if len(unions) == 0 {
		return Validations{}, nil
	}

	return processUnionValidations(context, unions, unionVariablePrefix,
		unionMemberTagName, unionValidator, discriminatedUnionValidator)
}

const (
	unionDiscriminatorTagName = "k8s:unionDiscriminator"
	unionMemberTagName        = "k8s:unionMember"
)

type unionDiscriminatorTagValidator struct {
	shared map[string]unions
}

func (unionDiscriminatorTagValidator) Init(_ Config) {}

func (unionDiscriminatorTagValidator) TagName() string {
	return unionDiscriminatorTagName
}

// Shared between unionDiscriminatorTagValidator and unionMemberTagValidator.
var unionTagValidScopes = sets.New(ScopeField, ScopeListVal)

func (unionDiscriminatorTagValidator) ValidScopes() sets.Set[Scope] {
	return unionTagValidScopes
}

func (udtv unionDiscriminatorTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	err := processDiscriminatorValidations(udtv.shared, context, tag)
	if err != nil {
		return Validations{}, err
	}
	// This tag does not actually emit any validations, it just accumulates
	// information. The validation is done by the unionTypeOrFieldValidator.
	return Validations{}, nil
}

func (udtv unionDiscriminatorTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         udtv.TagName(),
		Scopes:      udtv.ValidScopes().UnsortedList(),
		Description: "Indicates that this field is the discriminator for a union.",
		Args: []TagArgDoc{{
			Name:        "union",
			Description: "<string>",
			Docs:        "the name of the union, if more than one exists",
			Type:        codetags.ArgTypeString,
		}},
	}
}

type unionMemberTagValidator struct {
	shared map[string]unions
}

func (unionMemberTagValidator) Init(_ Config) {}

func (unionMemberTagValidator) TagName() string {
	return unionMemberTagName
}

func (unionMemberTagValidator) ValidScopes() sets.Set[Scope] {
	return unionTagValidScopes
}

func (umtv unionMemberTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	err := processMemberValidations(umtv.shared, context, tag)
	if err != nil {
		return Validations{}, err
	}
	// This tag does not actually emit any validations, it just accumulates
	// information. The validation is done by the unionTypeOrFieldValidator.
	return Validations{}, nil
}

func (umtv unionMemberTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         umtv.TagName(),
		Scopes:      umtv.ValidScopes().UnsortedList(),
		Description: "Indicates that this field is a member of a union.",
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

// union defines how a union validation will be generated, based
// on +k8s:unionMember and +k8s:unionDiscriminator tags found in a go struct.
type union struct {
	// fields provides field information about all the members of the union.
	// Each item provides a fieldName and memberName pair, where [0] identifies
	// the field name and [1] identifies the union member Name. fields is index
	// aligned with fieldMembers.
	// If member name is not set, it defaults to the go struct field name.
	fields []unionMember
	// fieldMembers describes all the members of the union.
	fieldMembers []*types.Member

	// discriminator is the name of the discriminator field
	discriminator *string
	// discriminatorMember describes the discriminator field.
	discriminatorMember *types.Member

	// itemMatchers stores matcher criteria for list item unions.
	// key is the virtual field path (eg: "<path>/Pipeline.Tasks[{"name": "succeeded"}]"),
	// value is the parsed matcher map (eg: {"name": "succeeded"}).
	// Represents union members that are list items matching specific criteria
	itemMatchers map[string][]ListSelectorTerm
}

type unionMember struct {
	fieldName          string
	discriminatorValue string
}

// unions represents all the unions for a go struct.
type unions map[string]*union

// newUnion initializes a new union instance
func newUnion() *union {
	return &union{
		// slice fields can be nil
		itemMatchers: make(map[string][]ListSelectorTerm),
	}
}

// getOrCreate gets a union by name, or initializes a new union by the given name.
func (us unions) getOrCreate(name string) *union {
	var u *union
	var ok bool
	if u, ok = us[name]; !ok {
		u = newUnion()
		us[name] = u
	}
	return u
}

func processUnionValidations(context Context, unions unions, varPrefix string,
	tagName string, undiscriminatedValidator types.Name, discriminatedValidator types.Name,
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
		if len(u.fieldMembers) > 0 || u.discriminator != nil || len(u.itemMatchers) > 0 {
			if len(u.fieldMembers) > 0 && len(u.itemMatchers) > 0 {
				return Validations{}, fmt.Errorf("cannot have both field members and item matchers")
			}
			nativeType := util.NonPointer(util.NativeType(context.Type))
			if nativeType.Kind == types.Struct && len(u.itemMatchers) > 0 {
				return Validations{}, fmt.Errorf("struct type cannot have item matchers")
			}
			if nativeType.Kind == types.Slice && len(u.fieldMembers) > 0 {
				return Validations{}, fmt.Errorf("slice type cannot have field members")
			}

			// TODO: Avoid the "local" here. This was added to avoid errors caused when the package is an empty string.
			//       The correct package would be the output package but is not known here. This does not show up in generated code.
			// TODO: Append a consistent hash suffix to avoid generated name conflicts?
			varBaseName := sanitizeName(context.Path.String() + "_" + unionName) // unionName can be ""
			supportVarName := PrivateVar{Name: varPrefix + "_" + varBaseName, Package: "local"}

			var extractorArgs []any
			ptrType := types.PointerTo(context.Type)

			// Handle field unions
			for _, member := range u.fieldMembers {
				extractor := createMemberExtractor(ptrType, member)
				extractorArgs = append(extractorArgs, extractor)
			}

			// Handle list item unions for lists
			if nativeType.Kind == types.Slice && len(u.itemMatchers) > 0 {
				elemType := util.NonPointer(nativeType.Elem)

				// Sort matcher paths for stable output
				matcherPaths := make([]string, 0, len(u.itemMatchers))
				for path := range u.itemMatchers {
					matcherPaths = append(matcherPaths, path)
				}
				slices.Sort(matcherPaths)

				for _, fullPath := range matcherPaths {
					matcher := u.itemMatchers[fullPath]
					extractor, err := createItemExtractor(context.Type, elemType, matcher)
					if err != nil {
						return Validations{}, err
					}
					extractorArgs = append(extractorArgs, extractor)
				}
			}

			if u.discriminator != nil {
				supportVar := Variable(supportVarName,
					Function(tagName, DefaultFlags, newDiscriminatedUnionMembership,
						append([]any{*u.discriminator}, getDisplayFields(u, context, true)...)...))
				result.Variables = append(result.Variables, supportVar)

				discriminatorExtractor := FunctionLiteral{
					Parameters: []ParamResult{{Name: "obj", Type: ptrType}},
					Results:    []ParamResult{{Type: types.String}},
					Body:       fmt.Sprintf("if obj == nil {return \"\"}; return string(obj.%s)", u.discriminatorMember.Name), // Cast to string
				}

				extraArgs := append([]any{supportVarName, discriminatorExtractor}, extractorArgs...)
				fn := Function(tagName, DefaultFlags, discriminatedValidator, extraArgs...)
				result.Functions = append(result.Functions, fn)
			} else {
				supportVar := Variable(supportVarName, Function(tagName, DefaultFlags, newUnionMembership, getDisplayFields(u, context, false)...))
				result.Variables = append(result.Variables, supportVar)

				extraArgs := append([]any{supportVarName}, extractorArgs...)
				fn := Function(tagName, DefaultFlags, undiscriminatedValidator, extraArgs...)
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
	case types.Pointer, types.Map, types.Slice:
		extractor.Body = fmt.Sprintf("if obj == nil {return false}; return obj.%s != nil", member.Name)
	case types.Builtin:
		extractor.Body = fmt.Sprintf("if obj == nil {return false}; var z %s; return obj.%s != z", member.Type, member.Name)
	default:
		// This should be caught before we get here, but JIC.
		extractor.Body = fmt.Sprintf("if obj == nil {return false}; return false /* unsupported union member kind: %s */", nt.Kind)
	}
	return extractor
}

// createItemExtractor creates an extractor function for list item union members.
// It generates code that loops through the list to check if an item matching the criteria exists.
func createItemExtractor(listType *types.Type, elemType *types.Type, matcher []ListSelectorTerm) (FunctionLiteral, error) {
	var criteria []keyValuePair
	for _, term := range matcher {
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

// processDiscriminatorValidations processes union discriminator tags. It is a
// free function, rather than a method so that it can be called from other
// union-like tags.
func processDiscriminatorValidations(shared map[string]unions, context Context, tag codetags.Tag) error {
	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	if t := util.NonPointer(util.NativeType(context.Type)); t != types.String {
		return fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}
	if shared[context.ParentPath.String()] == nil {
		shared[context.ParentPath.String()] = unions{}
	}
	unionArg, _ := tag.NamedArg("union") // optional
	u := shared[context.ParentPath.String()].getOrCreate(unionArg.Value)

	var discriminatorFieldName string
	if jsonAnnotation, ok := tags.LookupJSON(*context.Member); ok {
		discriminatorFieldName = jsonAnnotation.Name
		u.discriminator = &discriminatorFieldName
		u.discriminatorMember = context.Member
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
	nt := util.NativeType(context.Member.Type)
	switch nt.Kind {
	case types.Pointer, types.Map, types.Slice, types.Builtin:
		// OK
	default:
		// In particular non-pointer structs are not supported.
		return fmt.Errorf("can only be used on nilable and primitive types (%s)", nt.Kind)
	}
	if context.Member == nil {
		return fmt.Errorf("struct-field union member has no member info in context")
	}

	jsonTag, ok := tags.LookupJSON(*context.Member)
	if !ok {
		return fmt.Errorf("field %q is a union member but has no JSON struct field tag", context.Member)
	}
	fieldName := jsonTag.Name
	if len(fieldName) == 0 {
		return fmt.Errorf("field %q is a union member but has no JSON name", context.Member)
	}

	if shared[context.ParentPath.String()] == nil {
		shared[context.ParentPath.String()] = unions{}
	}

	// See if the tag specified a member name.
	memberName := context.Member.Name                        // default
	if memberNameArg, ok := tag.NamedArg("memberName"); ok { // optional
		memberName = memberNameArg.Value
	}

	unionArg, _ := tag.NamedArg("union") // optional
	u := shared[context.ParentPath.String()].getOrCreate(unionArg.Value)
	u.fields = append(u.fields, unionMember{fieldName, memberName})

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

	// It's not really a "field", but close enough.
	fieldName := context.Path.String() // eg: "<path>/Pipeline.Tasks[{"name": "succeeded"}]"

	if shared[context.ParentPath.String()] == nil {
		shared[context.ParentPath.String()] = unions{}
	}

	// See if the tag specified a member name.
	memberName := ""
	if memberNameArg, ok := tag.NamedArg("memberName"); ok { // optional
		memberName = memberNameArg.Value
	}

	unionArg, _ := tag.NamedArg("union") // optional
	u := shared[context.ParentPath.String()].getOrCreate(unionArg.Value)
	u.fields = append(u.fields, unionMember{fieldName, memberName})

	if _, found := u.itemMatchers[fieldName]; found {
		return fmt.Errorf("list-item union member %q already exists", fieldName)
	}
	u.itemMatchers[fieldName] = context.ListSelector

	return nil
}

// getDisplayFields formats union field names for user-friendly error messages.
// For list item unions, it converts paths like "<path>/Pipeline.Tasks[{\"name\": \"succeeded\"}]"
// to readable formats like "Tasks[{\"name\": \"succeeded\"}]".
func getDisplayFields(u *union, context Context, discrim bool) []any {
	displayFields := make([]any, len(u.fields))
	listFieldName := context.Path.String()
	pathParts := strings.Split(listFieldName, ".")
	if len(pathParts) > 0 {
		listFieldName = pathParts[len(pathParts)-1]
	}
	for i, f := range u.fields {
		fieldName := f.fieldName
		memberName := f.discriminatorValue
		if _, isItem := u.itemMatchers[fieldName]; isItem {
			// Extract the JSON part from the input
			bracketIndex := strings.Index(fieldName, "[")
			if bracketIndex != -1 {
				jsonPart := fieldName[bracketIndex:]
				fieldName = listFieldName + jsonPart
			}
		}
		if discrim {
			displayFields[i] = Function("xx", 0, newDiscriminatedUnionMember, fieldName, memberName)
		} else {
			displayFields[i] = Function("xx", 0, newUnionMember, fieldName)
		}
	}
	return displayFields
}

// sanitizeName converts a string into a valid Go identifier
func sanitizeName(name string) string {
	name = strings.ReplaceAll(name, ".", "_")
	re := regexp.MustCompile(`[^a-zA-Z0-9_]`)
	return re.ReplaceAllString(name, "_")
}
