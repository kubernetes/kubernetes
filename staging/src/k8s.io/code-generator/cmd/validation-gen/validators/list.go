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
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	listTypeTagName   = "k8s:listType"
	ListMapKeyTagName = "k8s:listMapKey"
	uniqueTagName     = "k8s:unique"
)

// globalListMeta is shared between list-related validators.
var globalListMeta = map[string]*listMetadata{} // keyed by the field or type path

func init() {
	// Accumulate list metadata via tags.
	RegisterTagValidator(listTypeTagValidator{byPath: globalListMeta})
	RegisterTagValidator(listMapKeyTagValidator{byPath: globalListMeta})
	RegisterTagValidator(uniqueTagValidator{byPath: globalListMeta})

	// Finish work on the accumulated list metadata.
	RegisterFieldValidator(listValidator{byPath: globalListMeta})
	RegisterTypeValidator(listValidator{byPath: globalListMeta})
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
	ownership listOwnership // For now we don't use it for generation.
	semantic  listSemantic
	keyFields []string // For semantic == map.
	keyNames  []string // For semantic == map.
}

// makeListMapMatchFunc generates a function that compares two list-map
// elements by their list-map key fields.
func (lm *listMetadata) makeListMapMatchFunc(t *types.Type) FunctionLiteral {
	if lm.semantic != semanticMap {
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

	lm := lttv.byPath[context.Path.String()]
	if lm == nil {
		lm = &listMetadata{}
		lttv.byPath[context.Path.String()] = lm
	}
	if lm.ownership != "" {
		return Validations{}, fmt.Errorf("listType cannot be specified more than once")
	}

	switch tag.Value {
	case "atomic":
		lm.ownership = ownershipSingle
		// Do not overwrite a more specific semantic from uniqueTagValidator
		if lm.semantic == "" {
			lm.semantic = semanticAtomic
		}
	case "set":
		lm.ownership = ownershipShared
		// If uniqueTagValidator has run, lm.semantic will be non-empty and non-atomic.
		if lm.semantic != "" && lm.semantic != semanticAtomic {
			return Validations{}, fmt.Errorf("unique tag can only be used with listType=atomic")
		}
		lm.semantic = semanticSet
	case "map":
		lm.ownership = ownershipShared
		// If uniqueTagValidator has run, lm.semantic will be non-empty and non-atomic.
		if lm.semantic != "" && lm.semantic != semanticAtomic {
			return Validations{}, fmt.Errorf("unique tag can only be used with listType=atomic")
		}
		if util.NativeType(t.Elem).Kind != types.Struct {
			return Validations{}, fmt.Errorf("only lists of structs can be list-maps")
		}
		lm.semantic = semanticMap
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
		Description: "Declares a list field's semantic type and ownership behavior. atomic: single ownership, set: shared ownership with uniqueness, map: shared ownership with key-based uniqueness.",
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

	lm := lmktv.byPath[context.Path.String()]
	if lm == nil {
		lm = &listMetadata{}
		lmktv.byPath[context.Path.String()] = lm
	}
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

type uniqueTagValidator struct {
	byPath map[string]*listMetadata
}

func (uniqueTagValidator) Init(Config) {}

func (uniqueTagValidator) TagName() string {
	return uniqueTagName
}

func (uniqueTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (utv uniqueTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := util.NativeType(context.Type)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return Validations{}, fmt.Errorf("can only be used on list types (%s)", t.Kind)
	}

	lm := utv.byPath[context.Path.String()]
	if lm == nil {
		lm = &listMetadata{}
		utv.byPath[context.Path.String()] = lm
	}

	// If listType has already run and set a non-atomic ownership, this is an error.
	if lm.ownership != "" && lm.ownership != ownershipSingle {
		return Validations{}, fmt.Errorf("unique tag can only be used with listType=atomic")
	}

	if lm.semantic != "" && lm.semantic != semanticAtomic {
		return Validations{}, fmt.Errorf("unique tag cannot be specified more than once")
	}

	switch tag.Value {
	case "set":
		lm.semantic = semanticSet
	case "map":
		if util.NativeType(t.Elem).Kind != types.Struct {
			return Validations{}, fmt.Errorf("only lists of structs can be list-maps")
		}
		lm.semantic = semanticMap
	default:
		return Validations{}, fmt.Errorf("unknown unique type %q", tag.Value)
	}

	// This tag doesn't generate any validations.  It just accumulates
	// information for other tags to use.
	return Validations{}, nil
}

func (utv uniqueTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:         utv.TagName(),
		Scopes:      utv.ValidScopes().UnsortedList(),
		Description: "Declares that a list field's elements are unique. This tag can be used with listType=atomic to add uniqueness constraints, or independently to specify uniqueness semantics.",
		Payloads: []TagPayloadDoc{{
			Description: "<type>",
			Docs:        "map | set",
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
	validateUnique            = types.Name{Package: libValidationPkg, Name: "Unique"}
	validateSemanticDeepEqual = types.Name{Package: libValidationPkg, Name: "SemanticDeepEqual"}
	validateDirectEqual       = types.Name{Package: libValidationPkg, Name: "DirectEqual"}
)

func (lv listValidator) GetValidations(context Context) (Validations, error) {
	nt := util.NativeType(context.Type)
	if nt.Kind != types.Slice && nt.Kind != types.Array {
		return Validations{}, nil
	}

	// Look up the list metadata which is defined on this field or type.
	lm := lv.byPath[context.Path.String()]

	// NOTE: We don't really support list-of-list or map-of-list, so this does
	// not consider the case of ScopeListVal or ScopeMapVal. If we want to
	// support those, we need to look at this and make sure the paths work the
	// way we need.
	if context.Scope == ScopeField {
		// If this is a field, look up the list metadata for the type.
		// TypeValidators happen before FieldValidators, so this is safe.
		tm := lv.byPath[context.Type.String()]
		if lm != nil && tm != nil {
			return Validations{}, fmt.Errorf("found list metadata for both a field and its type: %s", context.Path)
		}
		// TODO(thockin): enable this once the whole codebase is converted or
		// if we only run against fields which are opted-in.
		// if lm == nil && tm == nil {
		// 	 return Validations{}, fmt.Errorf("found a list field without list metadata")
		// }
	}

	if lm == nil {
		// If we don't have metadata for this field, we might have it for the
		// field's type.
		return Validations{}, nil
	}

	// Do this after the above - if we only get one error, the one(s) above
	// this are more important.
	if err := lv.check(lm); err != nil {
		return Validations{}, err
	}

	result := Validations{}

	// Generate uniqueness checks for lists with higher-order semantics.
	if lm.semantic == semanticSet {
		// Only compare primitive values when possible. Slices and maps are not
		// comparable, and structs might hold pointer fields, which are directly
		// comparable but not what we need.
		//
		// NOTE: lists of pointers are not supported, so we should never see a pointer here.
		matchArg := validateSemanticDeepEqual
		if util.IsDirectComparable(util.NonPointer(util.NativeType(nt.Elem))) {
			matchArg = validateDirectEqual
		}
		comment := "lists with set semantics require unique values"
		f := Function("listValidator", DefaultFlags, validateUnique, Identifier(matchArg)).
			WithComment(comment)
		result.AddFunction(f)
	}
	// TODO: Replace with the following once we have a way to either opt-out from
	// uniqueness validation of list-maps or settle the decision on how to handle
	// the ratcheting cases of it.
	// if lm.semantic == semanticMap {
	if lm.semantic == semanticMap && lm.ownership == ownershipSingle {
		// TODO: There are some fields which are declared as maps which do not
		// enforce uniqueness in manual validation. Those either need to not be
		// maps or we need to allow types to opt-out from this validation.  SSA
		// is also not able to handle these well.
		matchArg := lm.makeListMapMatchFunc(nt.Elem)
		comment := "lists with map semantics require unique keys"

		f := Function("listValidator", DefaultFlags, validateUnique, matchArg).
			WithComment(comment)
		result.AddFunction(f)
	}

	return result, nil
}

// make sure a given listMetadata makes sense.
func (lv listValidator) check(lm *listMetadata) error {
	// Check some fundamental constraints on list tags.

	// If we have listMapKey but no map semantics, that's an error
	if len(lm.keyFields) > 0 && lm.semantic != semanticMap {
		return fmt.Errorf("found listMapKey without listType=map or unique=map")
	}

	// If we have map semantics but no keys, that's an error
	if lm.semantic == semanticMap && len(lm.keyFields) == 0 {
		return fmt.Errorf("found listType=map or unique=map without listMapKey")
	}

	// listType is mandatory.
	if lm.ownership == "" {
		return fmt.Errorf("listType must be specified - use listType=atomic, listType=set, or listType=map")
	}

	return nil
}
