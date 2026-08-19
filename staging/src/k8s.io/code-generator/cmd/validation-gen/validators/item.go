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
	RegisterTagValidator(&itemTagValidator{})
}

type keyValuePair struct {
	key       string
	value     any
	valueType codetags.ValueType
}

type itemTagValidator struct {
	extractor Extractor
}

func (itv *itemTagValidator) Init(cfg Config) {
	itv.extractor = cfg.Extractor
}

func (itemTagValidator) TagName() string {
	return itemTagName
}

var itemTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (itemTagValidator) ValidScopes() sets.Set[Scope] {
	return itemTagValidScopes
}

type itemSubContext struct {
	context  Context
	elemT    *types.Type
	criteria []keyValuePair
	itemKey  string
}

func (itv *itemTagValidator) makeSubContext(context Context, tag codetags.Tag) (itemSubContext, error) {
	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	nt := util.NonPointer(util.NativeType(context.Type))
	if nt.Kind != types.Slice {
		return itemSubContext{}, fmt.Errorf("can only be used on list types")
	}
	elemT := util.NonPointer(util.NativeType(nt.Elem))
	if elemT.Kind != types.Struct {
		return itemSubContext{}, fmt.Errorf("can only be used on lists of structs")
	}
	if tag.ValueType != codetags.ValueTypeTag || tag.ValueTag == nil {
		return itemSubContext{}, fmt.Errorf("requires a validation tag as its value payload")
	}

	// Parse key-value pairs from named args.
	criteria, err := criteriaFromArgs(tag.Args)
	if err != nil {
		return itemSubContext{}, err
	}
	if len(criteria) == 0 {
		return itemSubContext{}, fmt.Errorf("no selection criteria was specified")
	}

	// Extract validations from the stored tag
	itemKey := selectorString(criteria)
	itemPath := context.Path.Key(itemKey)
	itemSelector := generateSelector(criteria)
	subContext := Context{
		Scope:          ScopeListVal,
		Type:           elemT,
		Path:           itemPath,
		ListSelector:   itemSelector,
		ParentPath:     context.Path,
		ParentType:     context.Type,
		StabilityLevel: context.StabilityLevel,
		Conditions:     context.Conditions,
	}
	return itemSubContext{
		context:  subContext,
		elemT:    elemT,
		criteria: criteria,
		itemKey:  itemKey,
	}, nil
}

type itemMetadata struct {
	tag        codetags.Tag
	criteria   []keyValuePair
	itemKey    string
	elemT      *types.Type
	subContext Context
	conditions Conditions
	level      ValidationStabilityLevel
}


func (itv *itemTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	sc, err := itv.makeSubContext(context, tag)
	if err != nil {
		return SchemaMetadata{}, err
	}
	meta, err := globalRegistry.ExtractMetadata(sc.context, *tag.ValueTag)
	if err != nil {
		return SchemaMetadata{}, err
	}

	im := &itemMetadata{
		tag:        tag,
		criteria:   sc.criteria,
		itemKey:    sc.itemKey,
		elemT:      sc.elemT,
		subContext: sc.context,
		conditions: context.Conditions,
		level:      context.StabilityLevel,
	}

	node := meta.GetOrCreateNode(nodeKeyFor(context.Path))
	node.Items = append(node.Items, Conditional[*itemMetadata]{
		Conditions:     context.Conditions,
		StabilityLevel: context.StabilityLevel,
		Payload:        im,
	})
	return meta, nil
}

var (
	validateValSliceItem = types.Name{Package: libValidationPkg, Name: "ValSliceItem"}
	validatePtrSliceItem = types.Name{Package: libValidationPkg, Name: "PtrSliceItem"}
)

func (itv *itemTagValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (EmittedGroup, error) {
	if context.Scope != ScopeField && context.Scope != ScopeType {
		return EmittedGroup{}, nil
	}
	nt := util.NonPointer(util.NativeType(context.Type))
	if nt == nil || (nt.Kind != types.Slice && nt.Kind != types.Array) {
		return EmittedGroup{}, nil
	}

	var items []Conditional[*itemMetadata]
	if node, ok := metadata.Nodes[nodeKeyFor(context.Path)]; ok {
		items = node.Items
	}
	if len(items) == 0 {
		return EmittedGroup{}, nil
	}

	hasUnemitted := false
	for _, itemCond := range items {
		if itemCond.Payload != nil {
			emittedKey := fmt.Sprintf("item:%s:%s:%s", context.Path.String(), itemCond.Payload.itemKey, itemCond.Conditions.Key())
			if !metadata.MarkEmitted(emittedKey) {
				hasUnemitted = true
			}
		}
	}
	if !hasUnemitted {
		return EmittedGroup{}, nil
	}

	listMeta := GetListMetadataFromSchema(context, metadata)
	result := Validations{}
	validateFunc := validateValSliceItem
	if nt.Elem.Kind == types.Pointer {
		validateFunc = validatePtrSliceItem
	}

	for _, itemCond := range items {
		im := itemCond.Payload
		if im == nil {
			continue
		}
		validations, err := itv.extractor.ExtractTagValidations(im.subContext, metadata, *im.tag.ValueTag)
		if err != nil {
			return EmittedGroup{}, err
		}

		if len(validations.Functions) == 0 {
			continue
		}

		matchArg, equivArg, err := itv.prepareArgsWithMeta(context, im.criteria, im.elemT, listMeta)
		if err != nil {
			return EmittedGroup{}, err
		}

		// Because item tags are evaluated in the context of the list field, the "parent"
		// of a list element is conceptually the list itself, which maps to the field phase.
		itemValidations := Validations{}
		for _, vfn := range validations.Functions {
			f := Function(itemTagName, vfn.Flags, validateFunc, matchArg, equivArg, WrapperFunction{Function: vfn, ObjType: im.elemT})
			f.Cohort = im.itemKey
			itemValidations.AddFunction(f)
		}
		finalized, err := FinalizeGroup(context, EmittedGroup{
			Validations:    itemValidations,
			Conditions:     itemCond.Conditions,
			StabilityLevel: itemCond.StabilityLevel,
		})
		if err != nil {
			return EmittedGroup{}, err
		}
		result.Add(finalized)
	}

	return EmittedGroup{Validations: result}, nil
}

func (itv *itemTagValidator) prepareArgsWithMeta(context Context, criteria []keyValuePair, elemT *types.Type, listMeta *listMetadata) (matchArg any, equivArg any, err error) {
	if listMeta == nil || listMeta.semantic != semanticMap || len(listMeta.keyMembers) == 0 {
		return nil, nil, fmt.Errorf("found items with no list metadata - item tags require listType=map or unique=map with listMapKey")
	}

	if len(criteria) != len(listMeta.keyNames) {
		return nil, nil, fmt.Errorf("number of arguments (%d) does not match number of listMapKey fields (%d)", len(criteria), len(listMeta.keyNames))
	}

	foundKeys := make(map[string]bool)
	sortedCriteria := make([]keyValuePair, 0, len(listMeta.keyNames))
	for _, keyName := range listMeta.keyNames {
		for _, pair := range criteria {
			if pair.key == keyName {
				member := util.GetMemberByJSON(elemT, pair.key)
				if member == nil {
					return nil, nil, fmt.Errorf("no field with JSON name %q", pair.key)
				}
				// We don't need to validate type match here, as it's done during generateComparisonRHS
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
		return nil, nil, fmt.Errorf("missing required listMapKey fields: %v", missing)
	}
	matchArg, err = createMatchFn(elemT, sortedCriteria)
	if err != nil {
		return nil, nil, err
	}
	directComparable := util.IsDirectComparable(util.NonPointer(util.NativeType(elemT)))
	if directComparable {
		equivArg = Identifier(validateDirectEqual)
	} else {
		equivArg = Identifier(validateSemanticDeepEqual)
	}
	return matchArg, equivArg, nil
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
		intVal, err := util.ParseInt(value)
		if err != nil {
			return nil, "", fmt.Errorf("invalid integer: %w", err)
		}
		return intVal, codetags.ValueTypeInt, nil
	case codetags.ArgTypeBool:
		boolVal, err := util.ParseBool(value)
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
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(itv.ValidScopes()),
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
