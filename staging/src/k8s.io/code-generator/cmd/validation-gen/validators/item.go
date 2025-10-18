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
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	itemTagName = "k8s:item"
)

func init() {
	// Processing item tags requires the list metadata.
	RegisterTagValidator(&itemTagValidator{listByPath: globalListMeta})
}

type keyValuePair struct {
	key       string
	value     any
	valueType codetags.ValueType
}

type itemTagValidator struct {
	validator  Validator
	listByPath map[string]*listMetadata
}

func (itv *itemTagValidator) Init(cfg Config) {
	itv.validator = cfg.Validator
}

func (itemTagValidator) TagName() string {
	return itemTagName
}

var itemTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (itemTagValidator) ValidScopes() sets.Set[Scope] {
	return itemTagValidScopes
}

// LateTagValidator indicates that this validator has to run AFTER the listType
// and listMapKey tags.
func (itemTagValidator) LateTagValidator() {}

var (
	validateSliceItem = types.Name{Package: libValidationPkg, Name: "SliceItem"}
)

func (itv *itemTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// TODO: Support regular maps with syntax like:
	// +k8s:item("map-key")=+k8s:immutable

	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	nt := util.NonPointer(util.NativeType(context.Type))
	if nt.Kind != types.Slice {
		return Validations{}, fmt.Errorf("can only be used on list types")
	}
	elemT := util.NonPointer(util.NativeType(nt.Elem))
	if elemT.Kind != types.Struct {
		return Validations{}, fmt.Errorf("can only be used on lists of structs")
	}
	if tag.ValueType != codetags.ValueTypeTag || tag.ValueTag == nil {
		return Validations{}, fmt.Errorf("requires a validation tag as its value payload")
	}

	// For fields, list metadata can fall back to the type.
	// For types, list metadata must be defined on the type itself.
	listMeta, ok := itv.listByPath[context.Path.String()]
	if !ok {
		if context.Scope == ScopeField {
			typePath := context.Type.String()
			listMeta, ok = itv.listByPath[typePath]
		}
	}

	// Fields inherit list metadata from typedefs, but not vice-versa.
	// If we find no listMetadata then something is wrong.
	// The item tag requires map semantics, which can come from either listType=map or unique=map
	if !ok || listMeta.semantic != semanticMap || len(listMeta.keyMembers) == 0 {
		return Validations{}, fmt.Errorf("found items with no list metadata - item tags require listType=map or unique=map with listMapKey")
	}

	// Parse key-value pairs from named args.
	criteria, err := criteriaFromArgs(tag.Args)
	if err != nil {
		return Validations{}, err
	}
	if len(criteria) == 0 {
		return Validations{}, fmt.Errorf("no selection criteria was specified")
	}
	if narg, nkey := len(criteria), len(listMeta.keyNames); narg != nkey {
		return Validations{}, fmt.Errorf("number of arguments (%d) does not match number of listMapKey fields (%d)", narg, nkey)
	}

	// Validate that all listMapKeys are provided and types match
	foundKeys := make(map[string]bool)
	sortedCriteria := make([]keyValuePair, 0, len(listMeta.keyNames))
	for _, keyName := range listMeta.keyNames {
		for _, pair := range criteria {
			if pair.key == keyName {
				member := util.GetMemberByJSON(elemT, pair.key)
				if member == nil {
					return Validations{}, fmt.Errorf("no field with JSON name %q", pair.key)
				}
				if err := validateTypeMatch(member.Type, pair.value); err != nil {
					return Validations{}, fmt.Errorf("key %q: %w", pair.key, err)
				}
				foundKeys[keyName] = true
				sortedCriteria = append(sortedCriteria, pair)
				break
			}
		}
	}
	if len(foundKeys) != len(listMeta.keyNames) {
		missing := []string{}
		for _, k := range listMeta.keyNames {
			if !foundKeys[k] {
				missing = append(missing, k)
			}
		}
		return Validations{}, fmt.Errorf("missing required listMapKey fields: %v", missing)
	}
	criteria = sortedCriteria

	result := Validations{}

	// Extract validations from the stored tag
	itemKey := selectorString(criteria)
	itemPath := context.Path.Key(itemKey)
	itemSelector := generateSelector(criteria)
	subContext := Context{
		Scope:        ScopeListVal,
		Type:         elemT,
		Path:         itemPath,
		ListSelector: itemSelector,
		ParentPath:   context.Path,
	}

	validations, err := itv.validator.ExtractValidations(subContext, *tag.ValueTag)
	if err != nil {
		return Validations{}, err
	}

	result.Variables = append(result.Variables, validations.Variables...)

	// matchArg is the function that is used to select the item in new and
	// old lists.
	matchArg, err := createMatchFn(elemT, criteria)
	if err != nil {
		return Validations{}, err
	}

	// equivArg is the function that is used to compare the correlated
	// elements in the old and new lists, for ratcheting.
	var equivArg any

	// directComparable is used to determine whether we can use the direct
	// comparison operator "==" or need to use the semantic DeepEqual when
	// looking up and comparing correlated list elements for validation ratcheting.
	directComparable := util.IsDirectComparable(util.NonPointer(util.NativeType(elemT)))
	if directComparable {
		equivArg = Identifier(validateDirectEqual)
	} else {
		equivArg = Identifier(validateSemanticDeepEqual)
	}

	for _, vfn := range validations.Functions {
		f := Function(itemTagName, vfn.Flags, validateSliceItem, matchArg, equivArg, WrapperFunction{vfn, elemT})
		f.Cohort = itemKey
		result.AddFunction(f)
	}

	return result, nil
}

func criteriaFromArgs(args []codetags.Arg) ([]keyValuePair, error) {
	criteria := []keyValuePair{}
	keysSeen := sets.Set[string]{}

	for _, arg := range args {
		if arg.Name == "" {
			return nil, fmt.Errorf("all arguments must be named")
		}
		if keysSeen.Has(arg.Name) {
			return nil, fmt.Errorf("duplicate argument: %q", arg.Name)
		}
		keysSeen.Insert(arg.Name)

		parsedValue, valueType, err := parseTypedValue(arg.Value, arg.Type)
		if err != nil {
			return nil, fmt.Errorf("invalid value for argument %q: %w", arg.Name, err)
		}

		criteria = append(criteria, keyValuePair{
			key:       arg.Name,
			value:     parsedValue,
			valueType: valueType,
		})
	}
	return criteria, nil
}

// parseTypedValue parses a value based on its detected type
func parseTypedValue(value string, argType codetags.ArgType) (any, codetags.ValueType, error) {
	switch argType {
	case codetags.ArgTypeString:
		return value, codetags.ValueTypeString, nil
	case codetags.ArgTypeInt:
		intVal, err := strconv.Atoi(value)
		if err != nil {
			return nil, "", fmt.Errorf("invalid integer: %w", err)
		}
		return intVal, codetags.ValueTypeInt, nil
	case codetags.ArgTypeBool:
		boolVal, err := strconv.ParseBool(value)
		if err != nil {
			return nil, "", fmt.Errorf("invalid boolean: %w", err)
		}
		return boolVal, codetags.ValueTypeBool, nil
	default:
		return nil, "", fmt.Errorf("unsupported argument type: %q", argType)
	}
}

func (itv itemTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            itv.TagName(),
		StabilityLevel: Alpha,
		Scopes:         itv.ValidScopes().UnsortedList(),
		Description: "Declares a validation for an item of a slice declared as a +k8s:listType=map. " +
			"The item to match is declared by providing field-value pair arguments. All key fields must be specified.",
		Usage: "+k8s:item(stringKey: \"value\", intKey: 42, boolKey: true)=<validation-tag>",
		Docs: "Arguments must be named with the JSON names of the list-map key fields. " +
			"Values can be strings, integers, or booleans. " +
			"For example: +k8s:item(name: \"myname\", priority: 10, enabled: true)=<chained-validation-tag>",
		AcceptsUnknownArgs: true,
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "The tag to evaluate for the matching list item.",
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
	return doc
}

// validateTypeMatch ensures the provided value matches the field's type
func validateTypeMatch(fieldType *types.Type, value any) error {
	nativeType := util.NonPointer(util.NativeType(fieldType))

	switch {
	case nativeType == types.String:
		if _, ok := value.(string); !ok {
			return fmt.Errorf("expected string value")
		}
	case nativeType == types.Bool:
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected bool value")
		}
	case types.IsInteger(nativeType):
		if _, ok := value.(int); !ok {
			return fmt.Errorf("expected integer value")
		}
	default:
		return fmt.Errorf("unsupported field type: %s", nativeType.String())
	}
	return nil
}

func buildMatchConditions(elemT *types.Type, criteria []keyValuePair, itemRef string) (string, error) {
	var conditions []string

	for _, fld := range criteria {
		member := util.GetMemberByJSON(elemT, fld.key)
		if member == nil {
			return "", fmt.Errorf("no field with JSON name %q", fld.key)
		}
		// Generate the comparison based on the field's actual type
		rhs, err := generateComparisonRHS(member, fld.value)
		if err != nil {
			return "", err
		}
		if member.Type.Kind == types.Pointer {
			conditions = append(conditions, fmt.Sprintf("%s.%s != nil && *%s.%s == %s", itemRef, member.Name, itemRef, member.Name, rhs))
		} else {
			conditions = append(conditions, fmt.Sprintf("%s.%s == %s", itemRef, member.Name, rhs))
		}
	}

	return strings.Join(conditions, " && "), nil
}

func createMatchFn(elemT *types.Type, criteria []keyValuePair) (FunctionLiteral, error) {
	condition, err := buildMatchConditions(elemT, criteria, "item")
	if err != nil {
		return FunctionLiteral{}, err
	}

	return FunctionLiteral{
		Parameters: []ParamResult{{"item", types.PointerTo(elemT)}},
		Results:    []ParamResult{{"", types.Bool}},
		Body:       fmt.Sprintf("return %s", condition),
	}, nil
}

func generateComparisonRHS(member *types.Member, value any) (string, error) {
	memberType := util.NonPointer(util.NativeType(member.Type))
	switch {
	case memberType == types.String:
		strVal, ok := value.(string)
		if !ok {
			return "", fmt.Errorf("type mismatch, field is string but value is not: %v", value)
		}
		return fmt.Sprintf("%q", strVal), nil

	case memberType == types.Bool:
		boolVal, ok := value.(bool)
		if !ok {
			return "", fmt.Errorf("type mismatch, field is bool but value is not: %v", value)
		}
		return fmt.Sprintf("%t", boolVal), nil

	case types.IsInteger(memberType):
		intVal, ok := value.(int)
		if !ok {
			return "", fmt.Errorf("type mismatch, field is int but value is not: %v", value)
		}
		return fmt.Sprintf("%d", intVal), nil

	default:
		return "", fmt.Errorf("unsupported type %s for field %s", memberType.String(), member.Name)
	}
}

// selectorString creates a field path for list map items using a JSON-like
// syntax. The format is {"key": "value", "key2": 42, "key3": true} with quoted
// keys and appropriately formatted values.
func selectorString(criteria []keyValuePair) string {
	var pairs []string
	for _, fld := range criteria {
		valueStr := formatValueForPath(fld.value)
		pairs = append(pairs, fmt.Sprintf("%q: %s", fld.key, valueStr))
	}
	return fmt.Sprintf("{%s}", strings.Join(pairs, ", "))
}

func formatValueForPath(value any) string {
	switch v := value.(type) {
	case string:
		return fmt.Sprintf("%q", v)
	case int:
		return fmt.Sprintf("%d", v)
	case bool:
		return fmt.Sprintf("%t", v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func generateSelector(criteria []keyValuePair) []ListSelectorTerm {
	terms := make([]ListSelectorTerm, len(criteria))
	for i, pair := range criteria {
		terms[i] = ListSelectorTerm{
			Field: pair.key,
			Value: pair.value,
		}
	}
	return terms
}
