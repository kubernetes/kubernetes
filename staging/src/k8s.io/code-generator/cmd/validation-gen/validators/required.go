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
	"encoding/json"
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/types"
)

const (
	requiredTagName  = "k8s:required"
	optionalTagName  = "k8s:optional"
	forbiddenTagName = "k8s:forbidden"
	defaultTagName   = "default" // TODO: this should evenually be +k8s:default
)

func init() {
	RegisterTagValidator(requirednessTagValidator{requirednessRequired})
	RegisterTagValidator(requirednessTagValidator{requirednessOptional})
	RegisterTagValidator(requirednessTagValidator{requirednessForbidden})
}

// requirednessTagValidator implements multiple modes of requiredness.
type requirednessTagValidator struct {
	mode requirednessMode
}

type requirednessMode string

const (
	requirednessRequired  requirednessMode = requiredTagName
	requirednessOptional  requirednessMode = optionalTagName
	requirednessForbidden requirednessMode = forbiddenTagName
)

func (requirednessTagValidator) Init(_ Config) {}

func (rtv requirednessTagValidator) TagName() string {
	return string(rtv.mode)
}

var requirednessTagValidScopes = sets.New(ScopeField)

func (requirednessTagValidator) ValidScopes() sets.Set[Scope] {
	return requirednessTagValidScopes
}

func (rtv requirednessTagValidator) GetValidations(context Context, _ []string, _ string) (Validations, error) {
	switch rtv.mode {
	case requirednessRequired:
		return rtv.doRequired(context)
	case requirednessOptional:
		return rtv.doOptional(context)
	case requirednessForbidden:
		return rtv.doForbidden(context)
	}
	panic(fmt.Sprintf("unknown requiredness mode: %q", rtv.mode))
}

var (
	requiredValueValidator   = types.Name{Package: libValidationPkg, Name: "RequiredValue"}
	requiredPointerValidator = types.Name{Package: libValidationPkg, Name: "RequiredPointer"}
	requiredSliceValidator   = types.Name{Package: libValidationPkg, Name: "RequiredSlice"}
	requiredMapValidator     = types.Name{Package: libValidationPkg, Name: "RequiredMap"}
)

// TODO: It might be valuable to have a string payload for when requiredness is
// conditional (e.g. required when <otherfield> is specified).
func (rtv requirednessTagValidator) doRequired(context Context) (Validations, error) {
	// Most validators don't care whether the value they are validating was
	// originally defined as a value-type or a pointer-type in the API.  This
	// one does.  Since Go doesn't do partial specialization of templates, we
	// do manual dispatch here.
	switch unaliasType(context.Type).Kind {
	case types.Slice:
		return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredSliceValidator)}}, nil
	case types.Map:
		return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredMapValidator)}}, nil
	case types.Pointer:
		return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredPointerValidator)}}, nil
	case types.Struct:
		// The +k8s:required tag on a non-pointer struct is not supported.
		// If you encounter this error and believe you have a valid use case
		// for forbiddening a non-pointer struct, please let us know! We need
		// to understand your scenario to determine if we need to adjust
		// this behavior or provide alternative validation mechanisms.
		return Validations{}, fmt.Errorf("non-pointer structs cannot use the %q tag", requiredTagName)
	}
	return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredValueValidator)}}, nil
}

var (
	optionalValueValidator   = types.Name{Package: libValidationPkg, Name: "OptionalValue"}
	optionalPointerValidator = types.Name{Package: libValidationPkg, Name: "OptionalPointer"}
	optionalSliceValidator   = types.Name{Package: libValidationPkg, Name: "OptionalSlice"}
	optionalMapValidator     = types.Name{Package: libValidationPkg, Name: "OptionalMap"}
)

func (rtv requirednessTagValidator) doOptional(context Context) (Validations, error) {
	// All of our tags are expressed from the perspective of a client of the
	// API, but the code we generate is for the server. Optional is tricky.
	//
	// A field which is marked as optional and does not have a default is
	// strictly optional. A client is allowed to not set it and the server will
	// not give it a default value. Code which consumes it must handle that it
	// might not have any value at all.
	//
	// A field which is marked as optional but has a default is optional to
	// clients, but required to the server. A client is allowed to not set it
	// but the server will give it a default value. Code which consumes it can
	// assume that it always has a value.
	//
	// One special case must be handled: optional non-pointer fields with
	// default values. If the default is not the zero value for the type, then
	// the zero value is used to decide whether to assign the default value,
	// and so must be out of bounds; we can proceed as above.
	//
	// But if the default is the zero value, then the zero value is obviously
	// valid, and the fact that the field is optional is meaningless - there is
	// no way to tell the difference between a client not setting it (yielding
	// the zero value) and a client setting it to the zero value.
	//
	// TODO: handle default=ref(...)
	// TODO: handle manual defaulting
	if hasDefault, zeroDefault, err := rtv.hasZeroDefault(context); err != nil {
		return Validations{}, err
	} else if hasDefault {
		if !isNilableType(context.Type) && zeroDefault {
			return Validations{Comments: []string{"optional value-type fields with zero-value defaults are purely documentation"}}, nil
		}
		validations, err := rtv.doRequired(context)
		if err != nil {
			return Validations{}, err
		}
		for i, fn := range validations.Functions {
			validations.Functions[i] = fn.WithComment("optional fields with default values are effectively required")
		}
		return validations, nil
	}

	// Most validators don't care whether the value they are validating was
	// originally defined as a value-type or a pointer-type in the API.  This
	// one does.  Since Go doesn't do partial specialization of templates, we
	// do manual dispatch here.
	switch unaliasType(context.Type).Kind {
	case types.Slice:
		return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalSliceValidator)}}, nil
	case types.Map:
		return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalMapValidator)}}, nil
	case types.Pointer:
		return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalPointerValidator)}}, nil
	case types.Struct:
		// The +k8s:optional tag on a non-pointer struct is not supported.
		// If you encounter this error and believe you have a valid use case
		// for forbiddening a non-pointer struct, please let us know! We need
		// to understand your scenario to determine if we need to adjust
		// this behavior or provide alternative validation mechanisms.
		return Validations{}, fmt.Errorf("non-pointer structs cannot use the %q tag", optionalTagName)
	}
	return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalValueValidator)}}, nil
}

// hasZeroDefault returns whether the field has a default value and whether
// that default value is the zero value for the field's type.
func (rtv requirednessTagValidator) hasZeroDefault(context Context) (bool, bool, error) {
	t := realType(context.Type)
	// This validator only applies to fields, so Member must be valid.
	tagsByName, err := gengo.ExtractFunctionStyleCommentTags("+", []string{defaultTagName}, context.Member.CommentLines)
	if err != nil {
		return false, false, fmt.Errorf("failed to read tags: %w", err)
	}

	tags, hasDefault := tagsByName[defaultTagName]
	if !hasDefault {
		return false, false, nil
	}
	if len(tags) == 0 {
		return false, false, fmt.Errorf("+default tag with no value")
	}
	if len(tags) > 1 {
		return false, false, fmt.Errorf("+default tag with multiple values: %q", tags)
	}

	payload := tags[0].Value
	var defaultValue any
	if err := json.Unmarshal([]byte(payload), &defaultValue); err != nil {
		return false, false, fmt.Errorf("failed to parse default value %q: %w", payload, err)
	}
	if defaultValue == nil {
		return false, false, fmt.Errorf("failed to parse default value %q: unmarshalled to nil", payload)
	}

	zero, found := typeZeroValue[t.String()]
	if !found {
		return false, false, fmt.Errorf("unknown zero-value for type %s", t.String())
	}

	return true, reflect.DeepEqual(defaultValue, zero), nil
}

// This is copied from defaulter-gen.
// TODO: move this back to gengo as Type.ZeroValue()?
var typeZeroValue = map[string]any{
	"uint":        0.,
	"uint8":       0.,
	"uint16":      0.,
	"uint32":      0.,
	"uint64":      0.,
	"int":         0.,
	"int8":        0.,
	"int16":       0.,
	"int32":       0.,
	"int64":       0.,
	"byte":        0.,
	"float64":     0.,
	"float32":     0.,
	"bool":        false,
	"time.Time":   "",
	"string":      "",
	"integer":     0.,
	"number":      0.,
	"boolean":     false,
	"[]byte":      "", // base64 encoded characters
	"interface{}": interface{}(nil),
	"any":         interface{}(nil),
}

var (
	forbiddenValueValidator   = types.Name{Package: libValidationPkg, Name: "ForbiddenValue"}
	forbiddenPointerValidator = types.Name{Package: libValidationPkg, Name: "ForbiddenPointer"}
	forbiddenSliceValidator   = types.Name{Package: libValidationPkg, Name: "ForbiddenSlice"}
	forbiddenMapValidator     = types.Name{Package: libValidationPkg, Name: "ForbiddenMap"}
)

// TODO: It might be valuable to have a string payload for when forbidden is
// conditional (e.g. forbidden when <option> is disabled).
func (requirednessTagValidator) doForbidden(context Context) (Validations, error) {
	// Forbidden is weird.  Each of these emits two checks, which are polar
	// opposites.  If the field fails the forbidden check, it will
	// short-circuit and not run the optional check.  If it passes the
	// forbidden check, it must not be specified, so it will "fail" the
	// optional check and short-circuit (but without error).  Why?  For
	// example, this prevents any further validation from trying to run on a
	// nil pointer.
	switch unaliasType(context.Type).Kind {
	case types.Slice:
		return Validations{
			Functions: []FunctionGen{
				Function(forbiddenTagName, ShortCircuit, forbiddenSliceValidator),
				Function(forbiddenTagName, ShortCircuit|NonError, optionalSliceValidator),
			},
		}, nil
	case types.Map:
		return Validations{
			Functions: []FunctionGen{
				Function(forbiddenTagName, ShortCircuit, forbiddenMapValidator),
				Function(forbiddenTagName, ShortCircuit|NonError, optionalMapValidator),
			},
		}, nil
	case types.Pointer:
		return Validations{
			Functions: []FunctionGen{
				Function(forbiddenTagName, ShortCircuit, forbiddenPointerValidator),
				Function(forbiddenTagName, ShortCircuit|NonError, optionalPointerValidator),
			},
		}, nil
	case types.Struct:
		// The +k8s:forbidden tag on a non-pointer struct is not supported.
		// If you encounter this error and believe you have a valid use case
		// for forbiddening a non-pointer struct, please let us know! We need
		// to understand your scenario to determine if we need to adjust
		// this behavior or provide alternative validation mechanisms.
		return Validations{}, fmt.Errorf("non-pointer structs cannot use the %q tag", forbiddenTagName)
	}
	return Validations{
		Functions: []FunctionGen{
			Function(forbiddenTagName, ShortCircuit, forbiddenValueValidator),
			Function(forbiddenTagName, ShortCircuit|NonError, optionalValueValidator),
		},
	}, nil
}

func (rtv requirednessTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:    rtv.TagName(),
		Scopes: rtv.ValidScopes().UnsortedList(),
	}

	switch rtv.mode {
	case requirednessRequired:
		doc.Description = "Indicates that a field must be specified by clients."
	case requirednessOptional:
		doc.Description = "Indicates that a field is optional to clients."
	case requirednessForbidden:
		doc.Description = "Indicates that a field may not be specified."
	default:
		panic(fmt.Sprintf("unknown requiredness mode: %q", rtv.mode))
	}

	return doc
}
