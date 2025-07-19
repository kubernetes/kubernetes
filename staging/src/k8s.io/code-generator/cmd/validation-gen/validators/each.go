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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	listTypeTagName   = "k8s:listType"
	ListMapKeyTagName = "k8s:listMapKey"
	eachValTagName    = "k8s:eachVal"
	eachKeyTagName    = "k8s:eachKey"
)

// We keep the eachVal and eachKey validators around because the main
// code-generation logic calls them directly.  We could move them into the main
// pkg, but it's easier and cleaner to leave them here.
var globalEachVal *eachValTagValidator
var globalEachKey *eachKeyTagValidator

func init() {
	// Lists with list-map semantics are comprised of multiple tags, which need
	// to share metadata about the list between them.
	listMeta := map[string]*listMetadata{} // keyed by the field or type path

	// Accumulate list metadata via tags.
	RegisterTagValidator(listTypeTagValidator{byPath: listMeta})
	RegisterTagValidator(listMapKeyTagValidator{byPath: listMeta})

	// Finish work on the accumulated list metadata.
	RegisterFieldValidator(listValidator{byPath: listMeta})
	RegisterTypeValidator(listValidator{byPath: listMeta})

	// List-map item validator uses shared listType and listMapKey information
	itemMeta := make(map[string]*itemMetadata) // keyed by the fieldpath

	// Accumulate item metadata via tags.
	RegisterTagValidator(&itemTagValidator{byPath: itemMeta})

	// Finish work on the accumulated item metadata.
	RegisterTypeValidator(&itemValidator{
		listByPath: listMeta,
		itemByPath: itemMeta,
	})
	RegisterFieldValidator(&itemValidator{
		listByPath: listMeta,
		itemByPath: itemMeta,
	})

	// Iterating values of lists and maps is a special tag, which can be called
	// directly by the code-generator logic.
	globalEachVal = &eachValTagValidator{byPath: listMeta, validator: nil}
	RegisterTagValidator(globalEachVal)

	// Iterating keys of maps is a special tag, which can be called directly by
	// the code-generator logic.
	globalEachKey = &eachKeyTagValidator{validator: nil}
	RegisterTagValidator(globalEachKey)
}

// This applies to all tags in this file.
var listTagsValidScopes = sets.New(ScopeAny)

// listMetadata collects information about a single list with map or set semantics.
type listMetadata struct {
	// These will be checked for correctness elsewhere.
	declaredAsAtomic bool
	declaredAsSet    bool
	declaredAsMap    bool
	keyFields        []string // iff declaredAsMap
	keyNames         []string // iff declaredAsMap
}

// makeListMapMatchFunc generates a function that compares two list-map
// elements by their list-map key fields.
func (lm *listMetadata) makeListMapMatchFunc(t *types.Type) FunctionLiteral {
	if !lm.declaredAsMap {
		panic("makeListMapMatchFunc called on a non-map list")
	}
	// If no keys are defined, we will throw a good error later.

	matchFn := FunctionLiteral{
		Parameters: []ParamResult{{"a", t}, {"b", t}},
		Results:    []ParamResult{{"", types.Bool}},
	}
	buf := strings.Builder{}
	buf.WriteString("return ")
	// Note: this does not handle pointer fields, which are not
	// supposed to be used as listMap keys.
	for i, fld := range lm.keyFields {
		if i > 0 {
			buf.WriteString(" && ")
		}
		buf.WriteString(fmt.Sprintf("a.%s == b.%s", fld, fld))
	}
	matchFn.Body = buf.String()
	return matchFn
}

type listTypeTagValidator struct {
	byPath map[string]*listMetadata
}

func (listTypeTagValidator) Init(Config) {}

func (listTypeTagValidator) TagName() string {
	return listTypeTagName
}

func (listTypeTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (lttv listTypeTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(context.Type)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return Validations{}, fmt.Errorf("can only be used on list types (%s)", t.Kind)
	}

	switch tag.Value {
	case "atomic":
		// We don't do much with atomic, but this ensures no conflicts between
		// tags on typedefs and tags on fields which use those typedefs.
		if lttv.byPath[context.Path.String()] == nil {
			lttv.byPath[context.Path.String()] = &listMetadata{}
		}
		lm := lttv.byPath[context.Path.String()]
		lm.declaredAsAtomic = true
	case "set":
		if lttv.byPath[context.Path.String()] == nil {
			lttv.byPath[context.Path.String()] = &listMetadata{}
		}
		lm := lttv.byPath[context.Path.String()]
		lm.declaredAsSet = true
		// NOTE: we validate uniqueness in the listValidator.
	case "map":
		// NOTE: maps of pointers are not supported, so we should never see a pointer here.
		if util.NativeType(t.Elem).Kind != types.Struct {
			return Validations{}, fmt.Errorf("only lists of structs can be list-maps")
		}

		// Save the fact that this list is a map.
		if lttv.byPath[context.Path.String()] == nil {
			lttv.byPath[context.Path.String()] = &listMetadata{}
		}
		lm := lttv.byPath[context.Path.String()]
		lm.declaredAsMap = true
		// NOTE: we validate uniqueness of the keys in the listValidator.
	default:
		return Validations{}, fmt.Errorf("unknown list type %q", tag.Value)
	}

	// This tag doesn't generate any validations.  It just accumulates
	// information for other tags to use.
	return Validations{}, nil
}

func (lttv listTypeTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:         lttv.TagName(),
		Scopes:      lttv.ValidScopes().UnsortedList(),
		Description: "Declares a list field's semantic type.",
		Payloads: []TagPayloadDoc{{
			Description: "<type>",
			Docs:        "atomic | map | set",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
	return doc
}

type listMapKeyTagValidator struct {
	byPath map[string]*listMetadata
}

func (listMapKeyTagValidator) Init(Config) {}

func (listMapKeyTagValidator) TagName() string {
	return ListMapKeyTagName
}

func (listMapKeyTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (lmktv listMapKeyTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(context.Type)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return Validations{}, fmt.Errorf("can only be used on list types (%s)", t.Kind)
	}
	// NOTE: lists of pointers are not supported, so we should never see a pointer here.
	if util.NativeType(t.Elem).Kind != types.Struct {
		return Validations{}, fmt.Errorf("only lists of structs can be list-maps")
	}

	var fieldName string
	if memb := util.GetMemberByJSON(util.NativeType(t.Elem), tag.Value); memb == nil {
		return Validations{}, fmt.Errorf("no field for JSON name %q", tag.Value)
	} else if k := util.NativeType(memb.Type).Kind; k != types.Builtin {
		return Validations{}, fmt.Errorf("only primitive types can be list-map keys (%s)", k)
	} else {
		fieldName = memb.Name
	}

	if lmktv.byPath[context.Path.String()] == nil {
		lmktv.byPath[context.Path.String()] = &listMetadata{}
	}
	lm := lmktv.byPath[context.Path.String()]
	lm.keyFields = append(lm.keyFields, fieldName)
	lm.keyNames = append(lm.keyNames, tag.Value)

	// This tag doesn't generate any validations.  It just accumulates
	// information for other tags to use.
	return Validations{}, nil
}

func (lmktv listMapKeyTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:         lmktv.TagName(),
		Scopes:      lmktv.ValidScopes().UnsortedList(),
		Description: "Declares a named sub-field of a list's value-type to be part of the list-map key.",
		Payloads: []TagPayloadDoc{{
			Description: "<field-json-name>",
			Docs:        "The name of the field.",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
	return doc
}

type listValidator struct {
	byPath map[string]*listMetadata
}

func (listValidator) Init(_ Config) {}

func (listValidator) Name() string {
	return "listValidator"
}

var (
	validateUnique = types.Name{Package: libValidationPkg, Name: "Unique"}
)

func (lv listValidator) GetValidations(context Context) (Validations, error) {
	lm := lv.byPath[context.Path.String()]
	if err := lv.check(lm); err != nil {
		return Validations{}, err
	}
	// NOTE: We don't really support list-of-list or map-of-list, so this does
	// not consider the case of ScopeListVal or ScopeMapVal. If we want to
	// support those, we need to look at this and make sure the paths work the
	// way we need.
	if context.Scope == ScopeField {
		tm := lv.byPath[context.Type.String()]
		if lm != nil && tm != nil {
			return Validations{}, fmt.Errorf("found list metadata for both a field and its type: %s", context.Path)
		}
		// For the purpose of emitting validations, we can use the
		// type's metadata if the field's metadata is not set.
		//
		// TypeValidators happen before FieldValidators, so if we end
		// up here, we can rely on this having been checked already.
		if lm == nil && tm != nil {
			lm = tm
		}
	}
	if lm == nil {
		// TODO(thockin): enable this once the whole codebase is converted or
		// if we only run against fields which are opted-in.
		// if context.Type.Kind == types.Slice || context.Type.Kind == types.Array {
		// 	return Validations{}, fmt.Errorf("found list field without a listType")
		// }
		return Validations{}, nil
	}

	result := Validations{}

	// Generate uniqueness checks for lists with higher-order semantics.
	nt := util.NativeType(context.Type)
	if lm.declaredAsSet {
		// Only compare primitive values when possible. Slices and maps are not
		// comparable, and structs might hold pointer fields, which are directly
		// comparable but not what we need.
		//
		// NOTE: lists of pointers are not supported, so we should never see a pointer here.
		matchArg := validateSemanticDeepEqual
		if util.IsDirectComparable(util.NonPointer(util.NativeType(nt.Elem))) {
			matchArg = validateDirectEqual
		}
		f := Function("listValidator", DefaultFlags, validateUnique, Identifier(matchArg))
		result.AddFunction(f)
	}
	// TODO: enable the following once we have a way to either opt-out from this validation
	// or settle the decision on how to handle the ratcheting cases.
	// if lm.declaredAsMap {
	// TODO: There are some fields which are declared as maps which do not
	// enforce uniqueness in manual validation. Those either need to not be
	// maps or we need to allow types to opt-out from this validation.  SSA
	// is also not able to handle these well.

	// matchArg := lm.makeListMapMatchFunc(nt.Elem)
	// f := Function("listValidator", DefaultFlags, validateUnique, matchArg).
	// 	WithComment("listType=map requires unique keys")
	// result.AddFunction(f)
	// }

	return result, nil
}

// make sure a given listMetadata makes sense.
func (lv listValidator) check(lm *listMetadata) error {
	if lm != nil {
		// Check some fundamental constraints on list tags.
		decls := []string{}
		if lm.declaredAsAtomic {
			decls = append(decls, "atomic")
		}
		if lm.declaredAsSet {
			decls = append(decls, "set")
		}
		if lm.declaredAsMap {
			decls = append(decls, "map")
		}
		if len(decls) > 1 {
			return fmt.Errorf("listType cannot have multiple types (%s)", strings.Join(decls, ", "))
		}
		if lm.declaredAsMap && len(lm.keyFields) == 0 {
			return fmt.Errorf("found listType=map without listMapKey")
		}
		if len(lm.keyFields) > 0 && !lm.declaredAsMap {
			return fmt.Errorf("found listMapKey without listType=map")
		}
		// Check for missing listType (after the other checks so the more specific errors take priority)
		if len(decls) == 0 {
			return fmt.Errorf("found list metadata without a listType")
		}
	}
	return nil
}

type eachValTagValidator struct {
	byPath    map[string]*listMetadata
	validator Validator
}

func (evtv *eachValTagValidator) Init(cfg Config) {
	evtv.validator = cfg.Validator
}

func (eachValTagValidator) TagName() string {
	return eachValTagName
}

func (eachValTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

// LateTagValidator indicates that this validator has to run AFTER the listType
// and listMapKey tags.
func (eachValTagValidator) LateTagValidator() {}

var (
	validateEachSliceVal      = types.Name{Package: libValidationPkg, Name: "EachSliceVal"}
	validateEachMapVal        = types.Name{Package: libValidationPkg, Name: "EachMapVal"}
	validateSemanticDeepEqual = types.Name{Package: libValidationPkg, Name: "SemanticDeepEqual"}
	validateDirectEqual       = types.Name{Package: libValidationPkg, Name: "DirectEqual"}
)

func (evtv eachValTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// NOTE: pointers to lists and maps are not supported, so we should never see a pointer here.
	t := context.Type
	nt := util.NativeType(t)
	switch nt.Kind {
	case types.Slice, types.Array, types.Map:
	default:
		return Validations{}, fmt.Errorf("can only be used on list or map types (%s)", t.Kind)
	}

	elemContext := Context{
		Type:       nt.Elem,
		ParentPath: context.Path,
		Path:       context.Path.Key("*"),
	}
	switch nt.Kind {
	case types.Slice, types.Array:
		elemContext.Scope = ScopeListVal
	case types.Map:
		elemContext.Scope = ScopeMapVal
	}
	if tag.ValueTag == nil {
		return Validations{}, fmt.Errorf("missing validation tag")
	}
	if validations, err := evtv.validator.ExtractValidations(elemContext, *tag.ValueTag); err != nil {
		return Validations{}, err
	} else {
		if validations.Empty() && !validations.OpaqueKeyType && !validations.OpaqueValType && !validations.OpaqueType {
			return Validations{}, fmt.Errorf("no validation functions found")
		}
		if len(validations.Variables) > 0 {
			return Validations{}, fmt.Errorf("variable generation is not supported")
		}
		// Pass the real (possibly alias) type.
		return evtv.getValidations(context.Path, t, validations)
	}
}

// t is expected to be the top-most type of the list or map. For example, if
// this is a typedef to a list, this is the alias type, not the underlying
// type.
func (evtv eachValTagValidator) getValidations(fldPath *field.Path, t *types.Type, validations Validations) (Validations, error) {
	switch util.NativeType(t).Kind {
	case types.Slice, types.Array:
		return evtv.getListValidations(fldPath, t, validations)
	case types.Map:
		return evtv.getMapValidations(t, validations)
	}
	return Validations{}, fmt.Errorf("non-iterable type: %v", t)
}

// ForEachVal returns a validation that applies a function to each element of
// a list or map. The type argument is expected to be the top-most type of the
// list or map. For example, if this is a typedef to a list, this is the alias
// type, not the underlying type.
func ForEachVal(fldPath *field.Path, t *types.Type, fn FunctionGen) (Validations, error) {
	return globalEachVal.getValidations(fldPath, t, Validations{Functions: []FunctionGen{fn}})
}

// t is expected to be the top-most type of the list. For example, if this is a
// typedef to a list, this is the alias type, not the underlying type.
func (evtv eachValTagValidator) getListValidations(fldPath *field.Path, t *types.Type, validations Validations) (Validations, error) {
	result := Validations{}
	result.OpaqueValType = validations.OpaqueType

	// This type is a "late" validator, so it runs after all the keys are
	// registered.  See LateTagValidator() above.
	listMetadata := evtv.byPath[fldPath.String()]
	if listMetadata == nil {
		// If we don't have metadata for this field, we might have it for the
		// field's type.
		listMetadata = evtv.byPath[t.String()]
	}

	nt := util.NativeType(t)

	// matchArg is the function that is used to lookup the correlated element in the old list.
	var matchArg any = Literal("nil")

	// equivArg is the function that is used to compare the correlated elements in the old and new lists.
	// It would be "nil" if the matchArg is a full comparison function.
	var equivArg any = Literal("nil")

	// directComparable is used to determine whether we can use the direct
	// comparison operator "==" or need to use the semantic DeepEqual when
	// looking up and comparing correlated list elements for validation ratcheting.
	directComparable := util.IsDirectComparable(util.NonPointer(util.NativeType(nt.Elem)))

	switch {
	case listMetadata != nil && listMetadata.declaredAsMap:
		// For listType=map, we use key to lookup the correlated element in the old list.
		// And use equivFunc to compare the correlated elements in the old and new lists.
		matchArg = listMetadata.makeListMapMatchFunc(nt.Elem)
		if directComparable {
			equivArg = Identifier(validateDirectEqual)
		} else {
			equivArg = Identifier(validateSemanticDeepEqual)
		}
	case listMetadata != nil && listMetadata.declaredAsSet:
		// For listType=set, matchArg is the equivalence check, so equivArg is nil.
		if directComparable {
			matchArg = Identifier(validateDirectEqual)
		} else {
			matchArg = Identifier(validateSemanticDeepEqual)
		}
	default:
		// For non-map and non-set list, we don't lookup the correlated element in the old list.
		// The matchArg and equivArg are both nil.
	}
	for _, vfn := range validations.Functions {
		f := Function(eachValTagName, vfn.Flags, validateEachSliceVal, matchArg, equivArg, WrapperFunction{vfn, nt.Elem})
		result.AddFunction(f)
	}

	return result, nil
}

// t is expected to be the top-most type of the map. For example, if this is a
// typedef to a map, this is the alias type, not the underlying type.
func (evtv eachValTagValidator) getMapValidations(t *types.Type, validations Validations) (Validations, error) {
	result := Validations{}
	result.OpaqueValType = validations.OpaqueType

	nt := util.NativeType(t)
	equivArg := Identifier(validateSemanticDeepEqual)
	if util.IsDirectComparable(util.NonPointer(util.NativeType(nt.Elem))) {
		equivArg = Identifier(validateDirectEqual)
	}
	for _, vfn := range validations.Functions {
		f := Function(eachValTagName, vfn.Flags, validateEachMapVal, equivArg, WrapperFunction{vfn, nt.Elem})
		result.AddFunction(f)
	}

	return result, nil
}

func (evtv eachValTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:         evtv.TagName(),
		Scopes:      evtv.ValidScopes().UnsortedList(),
		Description: "Declares a validation for each value in a map or list.",
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "The tag to evaluate for each value.",
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
	return doc
}

type eachKeyTagValidator struct {
	validator Validator
}

func (ektv *eachKeyTagValidator) Init(cfg Config) {
	ektv.validator = cfg.Validator
}

func (eachKeyTagValidator) TagName() string {
	return eachKeyTagName
}

func (eachKeyTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

var (
	validateEachMapKey = types.Name{Package: libValidationPkg, Name: "EachMapKey"}
)

func (ektv eachKeyTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(context.Type)
	if t.Kind != types.Map {
		return Validations{}, fmt.Errorf("can only be used on map types (%s)", t.Kind)
	}

	elemContext := Context{
		Scope:      ScopeMapKey,
		Type:       t.Elem,
		ParentPath: context.Path,
		Path:       context.Path.Child("(keys)"),
	}

	if validations, err := ektv.validator.ExtractValidations(elemContext, *tag.ValueTag); err != nil {
		return Validations{}, err
	} else {
		if len(validations.Variables) > 0 {
			return Validations{}, fmt.Errorf("variable generation is not supported")
		}

		return ektv.getValidations(t, validations)
	}
}

func (ektv eachKeyTagValidator) getValidations(t *types.Type, validations Validations) (Validations, error) {
	result := Validations{}
	result.OpaqueKeyType = validations.OpaqueType
	for _, vfn := range validations.Functions {
		f := Function(eachKeyTagName, vfn.Flags, validateEachMapKey, WrapperFunction{vfn, t.Key})
		result.AddFunction(f)
	}
	return result, nil
}

// ForEachKey returns a validation that applies a function to each key of
// a map.
func ForEachKey(_ *field.Path, t *types.Type, fn FunctionGen) (Validations, error) {
	return globalEachKey.getValidations(t, Validations{Functions: []FunctionGen{fn}})
}

func (ektv eachKeyTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:         ektv.TagName(),
		Scopes:      ektv.ValidScopes().UnsortedList(),
		Description: "Declares a validation for each value in a map or list.",
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "The tag to evaluate for each value.",
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
	return doc
}
