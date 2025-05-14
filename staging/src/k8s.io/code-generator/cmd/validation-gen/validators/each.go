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
	// to share information between them.
	shared := map[string]*listMetadata{} // keyed by the fieldpath
	RegisterTagValidator(listTypeTagValidator{byFieldPath: shared})
	RegisterTagValidator(listMapKeyTagValidator{byFieldPath: shared})

	globalEachVal = &eachValTagValidator{byFieldPath: shared, validator: nil}
	RegisterTagValidator(globalEachVal)

	globalEachKey = &eachKeyTagValidator{validator: nil}
	RegisterTagValidator(globalEachKey)
}

// This applies to all tags in this file.
var listTagsValidScopes = sets.New(ScopeAny)

// listMetadata collects information about a single list with map or set semantics.
type listMetadata struct {
	// These will be checked for correctness elsewhere.
	declaredAsMap bool
	declaredAsSet bool
	keyFields     []string
}

type listTypeTagValidator struct {
	byFieldPath map[string]*listMetadata
}

func (listTypeTagValidator) Init(Config) {}

func (listTypeTagValidator) TagName() string {
	return listTypeTagName
}

func (listTypeTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

var (
	validateUniqueByCompare = types.Name{Package: libValidationPkg, Name: "UniqueByCompare"}
	validateUniqueByReflect = types.Name{Package: libValidationPkg, Name: "UniqueByReflect"}
)

func (lttv listTypeTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := NativeType(context.Type)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return Validations{}, fmt.Errorf("can only be used on list types")
	}

	switch payload {
	case "atomic":
		// Allowed but no special handling.
	case "set":
		if lttv.byFieldPath[context.Path.String()] == nil {
			lttv.byFieldPath[context.Path.String()] = &listMetadata{}
		}
		lm := lttv.byFieldPath[context.Path.String()]
		lm.declaredAsSet = true
		// Only compare primitive values when possible. Slices and maps are not
		// comparable, and structs might hold pointer fields, which are directly
		// comparable but not what we need.
		//
		// NOTE: lists of pointers are not supported, so we should never see a pointer here.
		if NonPointer(NativeType(t.Elem)).Kind == types.Builtin {
			return Validations{Functions: []FunctionGen{Function(listTypeTagName, DefaultFlags, validateUniqueByCompare)}}, nil
		}
		return Validations{Functions: []FunctionGen{Function(listTypeTagName, DefaultFlags, validateUniqueByReflect)}}, nil
	case "map":
		// NOTE: maps of pointers are not supported, so we should never see a pointer here.
		if NativeType(t.Elem).Kind != types.Struct {
			return Validations{}, fmt.Errorf("only lists of structs can be list-maps")
		}

		// Save the fact that this list is a map.
		if lttv.byFieldPath[context.Path.String()] == nil {
			lttv.byFieldPath[context.Path.String()] = &listMetadata{}
		}
		lm := lttv.byFieldPath[context.Path.String()]
		lm.declaredAsMap = true
	default:
		return Validations{}, fmt.Errorf("unknown list type %q", payload)
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
	}
	return doc
}

type listMapKeyTagValidator struct {
	byFieldPath map[string]*listMetadata
}

func (listMapKeyTagValidator) Init(Config) {}

func (listMapKeyTagValidator) TagName() string {
	return ListMapKeyTagName
}

func (listMapKeyTagValidator) ValidScopes() sets.Set[Scope] {
	return listTagsValidScopes
}

func (lmktv listMapKeyTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := NativeType(context.Type)
	if t.Kind != types.Slice && t.Kind != types.Array {
		return Validations{}, fmt.Errorf("can only be used on list types")
	}
	// NOTE: lists of pointers are not supported, so we should never see a pointer here.
	if NativeType(t.Elem).Kind != types.Struct {
		return Validations{}, fmt.Errorf("only lists of structs can be list-maps")
	}

	var fieldName string
	if memb := getMemberByJSON(NativeType(t.Elem), payload); memb == nil {
		return Validations{}, fmt.Errorf("no field for JSON name %q", payload)
	} else if k := NativeType(memb.Type).Kind; k != types.Builtin {
		return Validations{}, fmt.Errorf("only primitive types can be list-map keys (%s)", k)
	} else {
		fieldName = memb.Name
	}

	if lmktv.byFieldPath[context.Path.String()] == nil {
		lmktv.byFieldPath[context.Path.String()] = &listMetadata{}
	}
	lm := lmktv.byFieldPath[context.Path.String()]
	lm.keyFields = append(lm.keyFields, fieldName)

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
	}
	return doc
}

type eachValTagValidator struct {
	byFieldPath map[string]*listMetadata
	validator   Validator
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

// LateTagValidator indicatesa that validator has to run after the listType and
// listMapKey tags.
func (eachValTagValidator) LateTagValidator() {}

var (
	validateEachSliceVal      = types.Name{Package: libValidationPkg, Name: "EachSliceVal"}
	validateEachMapVal        = types.Name{Package: libValidationPkg, Name: "EachMapVal"}
	validateSemanticDeepEqual = types.Name{Package: libValidationPkg, Name: "SemanticDeepEqual"}
	validateDirectEqual       = types.Name{Package: libValidationPkg, Name: "DirectEqual"}
)

func (evtv eachValTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	// NOTE: pointers to lists and maps are not supported, so we should never see a pointer here.
	t := NativeType(context.Type)
	switch t.Kind {
	case types.Slice, types.Array, types.Map:
	default:
		return Validations{}, fmt.Errorf("can only be used on list or map types")
	}

	fakeComments := []string{payload}
	elemContext := Context{
		Type:   t.Elem,
		Parent: t,
		Path:   context.Path.Key("*"),
	}
	switch t.Kind {
	case types.Slice, types.Array:
		elemContext.Scope = ScopeListVal
	case types.Map:
		elemContext.Scope = ScopeMapVal
	}
	if validations, err := evtv.validator.ExtractValidations(elemContext, fakeComments); err != nil {
		return Validations{}, err
	} else {
		if len(validations.Variables) > 0 {
			return Validations{}, fmt.Errorf("variable generation is not supported")
		}
		return evtv.getValidations(context.Path, t, validations)
	}
}

func (evtv eachValTagValidator) getValidations(fldPath *field.Path, t *types.Type, validations Validations) (Validations, error) {
	switch t.Kind {
	case types.Slice, types.Array:
		return evtv.getListValidations(fldPath, t, validations)
	case types.Map:
		return evtv.getMapValidations(t, validations)
	}
	return Validations{}, fmt.Errorf("non-iterable type: %v", t)
}

// ForEachVal returns a validation that applies a function to each element of
// a list or map.
func ForEachVal(fldPath *field.Path, t *types.Type, fn FunctionGen) (Validations, error) {
	return globalEachVal.getValidations(fldPath, t, Validations{Functions: []FunctionGen{fn}})
}

func (evtv eachValTagValidator) getListValidations(fldPath *field.Path, t *types.Type, validations Validations) (Validations, error) {
	result := Validations{}
	result.OpaqueValType = validations.OpaqueType

	var listMetadata *listMetadata
	if lm, found := evtv.byFieldPath[fldPath.String()]; found {
		if lm.declaredAsSet && lm.declaredAsMap {
			return Validations{}, fmt.Errorf("listType cannot be both set and map")
		}
		if lm.declaredAsMap && len(lm.keyFields) == 0 {
			return Validations{}, fmt.Errorf("found listType=map without listMapKey")
		}
		if len(lm.keyFields) > 0 && !lm.declaredAsMap {
			return Validations{}, fmt.Errorf("found listMapKey without listType=map")
		}
		// Check for missing listType (after the other checks so the more specific errors take priority)
		if !lm.declaredAsSet && !lm.declaredAsMap {
			return Validations{}, fmt.Errorf("found list metadata without a listType")
		}
		listMetadata = lm
	}

	for _, vfn := range validations.Functions {
		var cmpArg any = Literal("nil")
		if listMetadata != nil {
			if listMetadata.declaredAsMap {
				cmpFn := FunctionLiteral{
					Parameters: []ParamResult{{"a", t.Elem}, {"b", t.Elem}},
					Results:    []ParamResult{{"", types.Bool}},
				}
				buf := strings.Builder{}
				buf.WriteString("return ")
				// Note: this does not handle pointer fields, which are not
				// supposed to be used as listMap keys.
				for i, fld := range listMetadata.keyFields {
					if i > 0 {
						buf.WriteString(" && ")
					}
					buf.WriteString(fmt.Sprintf("a.%s == b.%s", fld, fld))
				}
				cmpFn.Body = buf.String()
				cmpArg = cmpFn
			} else if listMetadata.declaredAsSet {
				// Emit the cmpArg as a simple comparison when possible.
				// Slices and maps are not comparable, and structs might hold
				// pointer fields, which are directly comparable but not what we need.
				//
				// Note: This compares the pointee, not the pointer itself.
				if NonPointer(NativeType(t.Elem)).Kind == types.Builtin {
					cmpArg = Identifier(validateDirectEqual)
				} else {
					cmpArg = Identifier(validateSemanticDeepEqual)
				}
			}
		}
		f := Function(eachValTagName, vfn.Flags, validateEachSliceVal, cmpArg, WrapperFunction{vfn, t.Elem})
		result.Functions = append(result.Functions, f)
	}

	return result, nil
}

func (evtv eachValTagValidator) getMapValidations(t *types.Type, validations Validations) (Validations, error) {
	result := Validations{}
	result.OpaqueValType = validations.OpaqueType

	for _, vfn := range validations.Functions {
		f := Function(eachValTagName, vfn.Flags, validateEachMapVal, WrapperFunction{vfn, t.Elem})
		result.Functions = append(result.Functions, f)
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

func (ektv eachKeyTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	t := NativeType(context.Type)
	if t.Kind != types.Map {
		return Validations{}, fmt.Errorf("can only be used on map types")
	}

	fakeComments := []string{payload}
	elemContext := Context{
		Scope:  ScopeMapKey,
		Type:   t.Elem,
		Parent: t,
		Path:   context.Path.Child("(keys)"),
	}
	if validations, err := ektv.validator.ExtractValidations(elemContext, fakeComments); err != nil {
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
		result.Functions = append(result.Functions, f)
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
	}
	return doc
}
