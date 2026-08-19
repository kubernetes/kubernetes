/*
Copyright 2024 The Kubernetes Authors.

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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	listTypeTagName     = "k8s:listType"
	ListMapKeyTagName   = "k8s:listMapKey"
	uniqueTagName       = "k8s:unique"
	customUniqueTagName = "k8s:customUnique"
)

func init() {
	RegisterTagValidator(listTypeTagValidator{})
	RegisterTagValidator(listMapKeyTagValidator{})
	RegisterTagValidator(uniqueTagValidator{})
	RegisterTagValidator(customUniqueTagValidator{})
}

// This applies to all tags in this file.
var listTagsValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

type listOwnership string

const (
	ownershipSingle listOwnership = "single" // from listType=atomic
	ownershipShared listOwnership = "shared" // from listType=set/map
)

type listSemantic string

const (
	semanticAtomic listSemantic = "atomic" // No uniqueness check
	semanticSet    listSemantic = "set"    // uniqueness check
	semanticMap    listSemantic = "map"    // uniqueness check based on key(s)
)

// listMetadata collects information about a single list with map or set semantics.
type listMetadata struct {
	ownership  listOwnership // For now we don't use it for generation.
	semantic   listSemantic
	keyMembers []*types.Member // For semantic == map.
	keyNames   []string        // For semantic == map.
	// customUnique indicates that k8s:customUnique is set on this list.
	// It disables generation of uniqueness validation for this list.
	customUnique   bool
	stabilityLevel ValidationStabilityLevel
	conditions     Conditions
	targetPath     *field.Path
	listType       *types.Type
}

func (lm *listMetadata) merge(other *listMetadata) error {
	if other == nil {
		return nil
	}
	if other.ownership != "" {
		if lm.ownership != "" && lm.ownership != other.ownership {
			return fmt.Errorf("listType cannot be specified more than once")
		}
		lm.ownership = other.ownership
	}
	if other.semantic != "" {
		if lm.semantic != "" && lm.semantic != other.semantic && lm.semantic != semanticAtomic && other.semantic != semanticAtomic {
			return fmt.Errorf("unique tag is redundant for listType=%q", lm.semantic)
		}
		if lm.semantic == "" || lm.semantic == semanticAtomic {
			lm.semantic = other.semantic
		}
	}
	if len(other.keyMembers) > 0 {
		lm.keyMembers = append(lm.keyMembers, other.keyMembers...)
	}
	if len(other.keyNames) > 0 {
		lm.keyNames = append(lm.keyNames, other.keyNames...)
	}
	lm.sortKeys()
	if other.customUnique {
		lm.customUnique = true
	}
	if other.stabilityLevel != "" {
		lm.stabilityLevel = other.stabilityLevel
	}
	if !other.conditions.Empty() {
		lm.conditions = lm.conditions.Merge(other.conditions)
	}
	if lm.targetPath == nil {
		lm.targetPath = other.targetPath
	}
	if lm.listType == nil {
		lm.listType = other.listType
	}
	return nil
}

func (lm *listMetadata) sortKeys() {
	if len(lm.keyNames) <= 1 {
		return
	}
	type keyPair struct {
		name   string
		member *types.Member
	}
	pairs := make([]keyPair, len(lm.keyNames))
	for i := range lm.keyNames {
		pairs[i] = keyPair{lm.keyNames[i], lm.keyMembers[i]}
	}
	slices.SortFunc(pairs, func(a, b keyPair) int {
		return strings.Compare(a.name, b.name)
	})
	for i, p := range pairs {
		lm.keyNames[i] = p.name
		lm.keyMembers[i] = p.member
	}
}

func (lm *listMetadata) check() error {
	// Check some fundamental constraints on list tags.

	// If we have listMapKey but no map semantics, that's an error
	if len(lm.keyMembers) > 0 && lm.semantic != semanticMap {
		return fmt.Errorf("found listMapKey without listType=map or unique=map")
	}

	// If we have map semantics but no keys, that's an error
	if lm.semantic == semanticMap && len(lm.keyMembers) == 0 {
		return fmt.Errorf("found listType=map or unique=map without listMapKey")
	}

	// listType is mandatory.
	if lm.ownership == "" {
		return fmt.Errorf("listType must be specified - use listType=atomic, listType=set, or listType=map")
	}

	return nil
}


// makeListMapMatchFunc generates a function that compares two list-map
// elements by their list-map key fields.
func (lm *listMetadata) makeListMapMatchFunc(t *types.Type) FunctionLiteral {
	if lm.semantic != semanticMap {
		panic("makeListMapMatchFunc called on a non-map list")
	}

	matchFn := FunctionLiteral{
		Parameters: []ParamResult{{"a", types.PointerTo(util.NonPointer(t))}, {"b", types.PointerTo(util.NonPointer(t))}},
		Results:    []ParamResult{{"", types.Bool}},
	}
	buf := strings.Builder{}
	buf.WriteString("return ")

	for i, memb := range lm.keyMembers {
		if i > 0 {
			buf.WriteString(" && ")
		}
		fldName := memb.Name

		if memb.Type.Kind == types.Pointer {
			// Dereference pointers for comparison.
			// This is tricky because they could be nil.
			// Two keys are equal if all their fields are equal.
			// For pointer fields, that means either both are nil,
			// or neither is nil and the pointed-to values are equal.
			buf.WriteString(fmt.Sprintf("((a.%s == nil && b.%s == nil) || (a.%s != nil && b.%s != nil && *a.%s == *b.%s))", fldName, fldName, fldName, fldName, fldName, fldName))
		} else {
			buf.WriteString(fmt.Sprintf("a.%s == b.%s", fldName, fldName))
		}
	}
	matchFn.Body = buf.String()
	return matchFn
}

type listTypeTagValidator struct{}

func (listTypeTagValidator) Init(Config) {}

func (listTypeTagValidator) TagName() string {
	return listTypeTagName
}

func (listTypeTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (lttv listTypeTagValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (EmittedGroup, error) {
	lm := GetListMetadataFromSchema(context, metadata)
	if lm == nil {
		return EmittedGroup{}, nil
	}
	return generateListValidations(lm, context.Type)
}

func (lttv listTypeTagValidator) collectLists(context Context, tag codetags.Tag) (map[NodePath]*listMetadata, error) {
	byPath := map[NodePath]*listMetadata{}
	cond, level := context.Conditions, context.StabilityLevel
	itemPathStr := nodeKeyFor(context.Path)
	itemTargetPath := context.Path
	itemListType := context.Type

	var err error
	switch tag.Name {
	case listTypeTagName:
		err = processListTypeTag(byPath, itemPathStr, itemTargetPath, itemListType, tag, cond, level)
	case ListMapKeyTagName:
		err = processListMapKeyTag(byPath, itemPathStr, itemTargetPath, itemListType, tag, cond, level)
	case uniqueTagName:
		err = processUniqueTag(byPath, itemPathStr, itemTargetPath, itemListType, tag, cond, level)
	case customUniqueTagName:
		err = processCustomUniqueTag(byPath, itemPathStr, itemTargetPath, itemListType, tag, cond, level)
	}
	if err != nil {
		return nil, err
	}
	if len(byPath) == 0 {
		return nil, nil
	}
	return byPath, nil
}

func (lttv listTypeTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	lists, err := lttv.collectLists(context, tag)
	if err != nil || len(lists) == 0 {
		return SchemaMetadata{}, err
	}
	res := SchemaMetadata{}
	keys := make([]NodePath, 0, len(lists))
	for k := range lists {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		lm := lists[k]
		node := res.GetOrCreateNode(k)
		node.Lists = append(node.Lists, Conditional[*listMetadata]{
			Conditions:     lm.conditions,
			StabilityLevel: lm.stabilityLevel,
			Payload:        lm,
		})
	}
	return res, nil
}

func generateListValidations(lm *listMetadata, contextType *types.Type) (EmittedGroup, error) {
	if err := lm.check(); err != nil {
		return EmittedGroup{}, err
	}

	result := Validations{}
	if lm.customUnique {
		// Uniqueness validation is disabled in generated validation for this list.
		// It would defer to handwritten validation to check the uniqueness.
		result.AddComment("Uniqueness validation is implemented via custom, handwritten validation")
		return EmittedGroup{Validations: result}, nil
	}

	nt := util.NativeType(lm.listType)

	if lm.semantic == semanticSet {
		// Only compare primitive values when possible. Slices and maps are not
		// comparable, and structs might hold pointer fields, which are directly
		// comparable but not what we need.
		//
		// TODO: There are some fields which are declared as maps which do not
		// enforce uniqueness in manual validation. Those either need to not be
		// maps or we need to allow types to opt-out from this validation. SSA
		// is also not able to handle these well.
		matchArg := validateSemanticDeepEqual
		if util.IsDirectComparable(util.NonPointer(util.NativeType(nt.Elem))) {
			matchArg = validateDirectEqual
		}
		validateFunc := validateValSliceUnique
		if nt.Elem.Kind == types.Pointer {
			validateFunc = validatePtrSliceUnique
		}
		comment := "lists with set semantics require unique values"
		f := Function("listValidator", DefaultFlags, validateFunc, Identifier(matchArg)).
			WithComment(comment).
			WithEmits(Emission{field.ErrorTypeDuplicate, "", "[*]"})
		if lm.stabilityLevel != "" {
			f = f.WithStabilityLevel(lm.stabilityLevel)
		}
		result.AddFunction(f)
	}

	if lm.semantic == semanticMap {
		matchArg := lm.makeListMapMatchFunc(nt.Elem)
		validateFunc := validateValSliceUnique
		if nt.Elem.Kind == types.Pointer {
			validateFunc = validatePtrSliceUnique
		}
		comment := "lists with map semantics require unique keys"

		f := Function("listValidator", DefaultFlags, validateFunc, matchArg).
			WithComment(comment).
			WithStabilityLevel(lm.stabilityLevel).
			WithEmits(Emission{field.ErrorTypeDuplicate, "", "[*]"})
		result.AddFunction(f)
	}

	if lm.stabilityLevel != "" {
		for i, fn := range result.Functions {
			if !fn.StabilityLevelSelfManaged {
				fn.StabilityLevel = lm.stabilityLevel
				result.Functions[i] = fn
			}
		}
	}

	return EmittedGroup{
		Validations:    result,
		StabilityLevel: lm.stabilityLevel,
	}, nil
}

func (lttv listTypeTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            lttv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(lttv.ValidScopes()),
		Description:    "Declares a list field's semantic type and ownership behavior. atomic: single ownership, set: shared ownership with uniqueness, map: shared ownership with key-based uniqueness.",
		Payloads: []TagPayloadDoc{{
			Description: "<type>",
			Docs:        "atomic | map | set",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
	return doc
}

type listMapKeyTagValidator struct{}

func (listMapKeyTagValidator) Init(Config) {}

func (listMapKeyTagValidator) TagName() string {
	return ListMapKeyTagName
}

func (listMapKeyTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (lmktv listMapKeyTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	return listTypeTagValidator{}.CollectMetadata(context, tag)
}

func (lmktv listMapKeyTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            lmktv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(lmktv.ValidScopes()),
		Description:    "Declares a named sub-field of a list's value-type to be part of the list-map key.",
		Payloads: []TagPayloadDoc{{
			Description: "<field-json-name>",
			Docs:        "The name of the field.",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
	return doc
}

type uniqueTagValidator struct{}

func (uniqueTagValidator) Init(Config) {}

func (uniqueTagValidator) TagName() string {
	return uniqueTagName
}

func (uniqueTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (utv uniqueTagValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (EmittedGroup, error) {
	lm := GetListMetadataFromSchema(context, metadata)
	if lm == nil || lm.ownership != "" {
		return EmittedGroup{}, nil
	}
	return generateListValidations(lm, context.Type)
}

func (utv uniqueTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	return listTypeTagValidator{}.CollectMetadata(context, tag)
}

func (utv uniqueTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            utv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(utv.ValidScopes()),
		Description:    "Declares that a list field's elements are unique. This tag can be used with listType=atomic to add uniqueness constraints, or independently to specify uniqueness semantics.",
		Payloads: []TagPayloadDoc{{
			Description: "<type>",
			Docs:        "map | set",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
	return doc

}

type customUniqueTagValidator struct{}

func (customUniqueTagValidator) Init(Config) {}

func (customUniqueTagValidator) TagName() string {
	return customUniqueTagName
}

func (customUniqueTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (cutv customUniqueTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	return listTypeTagValidator{}.CollectMetadata(context, tag)
}

func (cutv customUniqueTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            cutv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(cutv.ValidScopes()),
		Description:    "Indicates that uniqueness validation for this list is implemented via custom, handwritten validation. This disables generation of uniqueness validation for this list.",
		Payloads:       nil,
	}
	return doc
}

var (
	validateValSliceUnique    = types.Name{Package: libValidationPkg, Name: "ValSliceUnique"}
	validatePtrSliceUnique    = types.Name{Package: libValidationPkg, Name: "PtrSliceUnique"}
	validateSemanticDeepEqual = types.Name{Package: libValidationPkg, Name: "SemanticDeepEqual"}
	validateDirectEqual       = types.Name{Package: libValidationPkg, Name: "DirectEqual"}
)



// GetListMetadataFromSchema retrieves listMetadata from a pre-collected SchemaMetadata tree.
func GetListMetadataFromSchema(context Context, sm SchemaMetadata) *listMetadata {
	targetPath := context.Path
	if context.Scope == ScopeListVal || (context.Type != nil && util.NativeType(context.Type).Kind != types.Slice && util.NativeType(context.Type).Kind != types.Array) {
		if context.ParentPath != nil {
			targetPath = context.ParentPath
		}
	}

	if node, ok := sm.Nodes[nodeKeyFor(targetPath)]; ok && len(node.Lists) > 0 {
		return node.Lists[0].Payload
	}
	return nil
}

func processListTypeTag(byPath map[NodePath]*listMetadata, pathStr NodePath, targetPath *field.Path, listType *types.Type, tag codetags.Tag, cond Conditions, level ValidationStabilityLevel) error {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(listType)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return fmt.Errorf("can only be used on list types (%s)", t.Kind)
	}

	lm := byPath[pathStr]
	if lm == nil {
		lm = &listMetadata{targetPath: targetPath, listType: listType}
		byPath[pathStr] = lm
	}
	if level != "" {
		lm.stabilityLevel = level
	}
	if !cond.Empty() {
		lm.conditions = lm.conditions.Merge(cond)
	}
	if lm.ownership != "" {
		return fmt.Errorf("listType cannot be specified more than once")
	}

	switch tag.Value {
	case "atomic":
		lm.ownership = ownershipSingle
		// Do not overwrite a more specific semantic from uniqueTagValidator
		// If uniqueTagValidator has run for `unique=set` or `unique=map`,
		// lm.semantic will be non-empty and non-atomic.
		if lm.semantic == "" {
			lm.semantic = semanticAtomic
		}
	case "set":
		lm.ownership = ownershipShared
		// If uniqueTagValidator has run for `unique=set` or `unique=map`,
		// lm.semantic will be non-empty and non-atomic.
		if lm.semantic != "" && lm.semantic != semanticAtomic {
			return fmt.Errorf("unique tag is redundant for listType=%q", tag.Value)
		}
		lm.semantic = semanticSet
	case "map":
		lm.ownership = ownershipShared
		// If uniqueTagValidator has run for `unique=set` or `unique=map`,
		// lm.semantic will be non-empty and non-atomic.
		if lm.semantic != "" && lm.semantic != semanticAtomic {
			return fmt.Errorf("unique tag is redundant for listType=%q", tag.Value)
		}
		if util.NonPointer(util.NativeType(t.Elem)).Kind != types.Struct {
			return fmt.Errorf("only lists of structs can be list-maps")
		}
		lm.semantic = semanticMap
	default:
		return fmt.Errorf("unknown list type %q", tag.Value)
	}
	return nil
}

func processListMapKeyTag(byPath map[NodePath]*listMetadata, pathStr NodePath, targetPath *field.Path, listType *types.Type, tag codetags.Tag, cond Conditions, level ValidationStabilityLevel) error {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(listType)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return fmt.Errorf("can only be used on list types (%s)", t.Kind)
	}
	structT := util.NonPointer(util.NativeType(t.Elem))
	if structT.Kind != types.Struct {
		return fmt.Errorf("only lists of structs can be list-maps")
	}

	var memb *types.Member
	if m := util.GetMemberByJSON(structT, tag.Value); m == nil {
		return fmt.Errorf("no field for JSON name %q", tag.Value)
	} else {
		keyType := m.Type
		if keyType.Kind == types.Pointer {
			keyType = keyType.Elem
		}
		if util.NativeType(keyType).Kind != types.Builtin {
			return fmt.Errorf("only primitive types and pointers to primitive types can be list-map keys, not %s", m.Type.String())
		}
		memb = m
	}

	lm := byPath[pathStr]
	if lm == nil {
		lm = &listMetadata{targetPath: targetPath, listType: listType}
		byPath[pathStr] = lm
	}
	if lm.stabilityLevel == "" && level != "" {
		lm.stabilityLevel = level
	}
	if !cond.Empty() {
		lm.conditions = lm.conditions.Merge(cond)
	}
	lm.keyMembers = append(lm.keyMembers, memb)
	lm.keyNames = append(lm.keyNames, tag.Value)
	lm.sortKeys()
	return nil
}

func processUniqueTag(byPath map[NodePath]*listMetadata, pathStr NodePath, targetPath *field.Path, listType *types.Type, tag codetags.Tag, cond Conditions, level ValidationStabilityLevel) error {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(listType)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return fmt.Errorf("can only be used on list types (%s)", t.Kind)
	}

	lm := byPath[pathStr]
	if lm == nil {
		lm = &listMetadata{targetPath: targetPath, listType: listType}
		byPath[pathStr] = lm
	}

	if level != "" {
		lm.stabilityLevel = level
	}
	if !cond.Empty() {
		lm.conditions = lm.conditions.Merge(cond)
	}

	if lm.ownership != "" && lm.ownership != ownershipSingle {
		return fmt.Errorf("unique tag may not be used with listType=set or listType=map")
	}

	if lm.semantic != "" && lm.semantic != semanticAtomic {
		return fmt.Errorf("unique tag cannot be specified more than once")
	}

	switch tag.Value {
	case "set":
		lm.semantic = semanticSet
	case "map":
		if util.NonPointer(util.NativeType(t.Elem)).Kind != types.Struct {
			return fmt.Errorf("only lists of structs can be list-maps")
		}
		lm.semantic = semanticMap
	default:
		return fmt.Errorf("unknown unique type %q", tag.Value)
	}
	return nil
}

func processCustomUniqueTag(byPath map[NodePath]*listMetadata, pathStr NodePath, targetPath *field.Path, listType *types.Type, tag codetags.Tag, cond Conditions, level ValidationStabilityLevel) error {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(listType)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return fmt.Errorf("can only be used on list types (%s)", t.Kind)
	}

	lm := byPath[pathStr]
	if lm == nil {
		lm = &listMetadata{targetPath: targetPath, listType: listType}
		byPath[pathStr] = lm
	}

	if level != "" {
		lm.stabilityLevel = level
	}
	if !cond.Empty() {
		lm.conditions = lm.conditions.Merge(cond)
	}
	lm.customUnique = true
	return nil
}
