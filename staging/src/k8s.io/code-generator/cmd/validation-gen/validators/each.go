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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	eachValTagName = "k8s:eachVal"
	eachKeyTagName = "k8s:eachKey"
)

// We keep the eachVal and eachKey validators around because the main
// code-generation logic calls them directly.  We could move them into the main
// pkg, but it's easier and cleaner to leave them here.
var globalEachVal *eachValTagValidator
var globalEachKey *eachKeyTagValidator

func init() {
	// Iterating values of lists and maps is a special tag, which can be called
	// directly by the code-generator logic.
	globalEachVal = &eachValTagValidator{byPath: globalListMeta, validator: nil}
	RegisterTagValidator(globalEachVal)

	// Iterating keys of maps is a special tag, which can be called directly by
	// the code-generator logic.
	globalEachKey = &eachKeyTagValidator{validator: nil}
	RegisterTagValidator(globalEachKey)
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
	validateEachSliceVal   = types.Name{Package: libValidationPkg, Name: "EachSliceVal"}
	validateEachMapVal     = types.Name{Package: libValidationPkg, Name: "EachMapVal"}
	validateDirectEqualPtr = types.Name{Package: libValidationPkg, Name: "DirectEqualPtr"}
)

func (evtv eachValTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// NOTE: pointers to lists and maps are not supported, so we should never see a pointer here.
	t := context.Type
	nt := util.NativeType(t)
	switch nt.Kind {
	case types.Slice, types.Array, types.Map:
	default:
		return Validations{}, fmt.Errorf("can only be used on list or map types (%s)", nt.Kind)
	}

	elemContext := Context{
		// Scope is initialized below.
		Type:       nt.Elem,
		Path:       context.Path.Key("(vals)"),
		Member:     nil, // NA for list/map values
		ParentPath: context.Path,
	}
	switch nt.Kind {
	case types.Slice, types.Array:
		elemContext.Scope = ScopeListVal
		elemContext.ListSelector = []ListSelectorTerm{} // empty == "all"
	case types.Map:
		elemContext.Scope = ScopeMapVal
		// TODO: We may need map selectors at some point.
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

	if listMetadata != nil {
		switch listMetadata.semantic {
		case semanticMap:
			// For listType=map, we use key to lookup the correlated element in the old list.
			// And use equivFunc to compare the correlated elements in the old and new lists.
			matchArg = listMetadata.makeListMapMatchFunc(nt.Elem)
			if directComparable {
				equivArg = Identifier(validateDirectEqual)
			} else {
				equivArg = Identifier(validateSemanticDeepEqual)
			}
		case semanticSet:
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
	}

	for _, vfn := range validations.Functions {
		comm := vfn.Comments
		vfn.Comments = nil
		f := Function(eachValTagName, vfn.Flags, validateEachSliceVal, matchArg, equivArg, WrapperFunction{vfn, nt.Elem}).WithComments(comm...)
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
		comm := vfn.Comments
		vfn.Comments = nil
		f := Function(eachValTagName, vfn.Flags, validateEachMapVal, equivArg, WrapperFunction{vfn, nt.Elem}).WithComments(comm...)
		result.AddFunction(f)
	}

	return result, nil
}

func (evtv eachValTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            evtv.TagName(),
		StabilityLevel: Alpha,
		Scopes:         evtv.ValidScopes().UnsortedList(),
		Description:    "Declares a validation for each value in a map or list.",
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
	t := context.Type
	nt := util.NativeType(t)
	if nt.Kind != types.Map {
		return Validations{}, fmt.Errorf("can only be used on map types (%s)", nt.Kind)
	}

	elemContext := Context{
		Scope:      ScopeMapKey,
		Type:       nt.Elem,
		Path:       context.Path.Key("(keys)"),
		Member:     nil, // NA for map keys
		ParentPath: context.Path,
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
	nt := util.NativeType(t)
	result := Validations{}
	result.OpaqueKeyType = validations.OpaqueType
	for _, vfn := range validations.Functions {
		comm := vfn.Comments
		vfn.Comments = nil
		f := Function(eachKeyTagName, vfn.Flags, validateEachMapKey, WrapperFunction{vfn, nt.Key}).WithComments(comm...)
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
		Tag:            ektv.TagName(),
		Scopes:         ektv.ValidScopes().UnsortedList(),
		StabilityLevel: Alpha,
		Description:    "Declares a validation for each value in a map or list.",
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "The tag to evaluate for each key.",
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
	return doc
}
