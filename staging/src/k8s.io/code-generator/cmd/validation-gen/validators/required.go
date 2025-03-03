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
	"k8s.io/gengo/v2/types"
)

const (
	requiredTagName  = "k8s:required"
	optionalTagName  = "k8s:optional"
	forbiddenTagName = "k8s:forbidden"
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
	if context.Type.Kind == types.Alias {
		panic("alias type should already have been unwrapped")
	}
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
func (requirednessTagValidator) doRequired(context Context) (Validations, error) {
	// Most validators don't care whether the value they are validating was
	// originally defined as a value-type or a pointer-type in the API.  This
	// one does.  Since Go doesn't do partial specialization of templates, we
	// do manual dispatch here.
	switch context.Type.Kind {
	case types.Slice:
		return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredSliceValidator)}}, nil
	case types.Map:
		return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredMapValidator)}}, nil
	case types.Pointer:
		return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredPointerValidator)}}, nil
	case types.Struct:
		// The +required tag on a non-pointer struct is only for documentation.
		// We don't perform validation here and defer the validation to
		// the struct's fields.
		return Validations{Comments: []string{"required non-pointer structs are purely documentation"}}, nil
	}
	return Validations{Functions: []FunctionGen{Function(requiredTagName, ShortCircuit, requiredValueValidator)}}, nil
}

var (
	optionalValueValidator   = types.Name{Package: libValidationPkg, Name: "OptionalValue"}
	optionalPointerValidator = types.Name{Package: libValidationPkg, Name: "OptionalPointer"}
	optionalSliceValidator   = types.Name{Package: libValidationPkg, Name: "OptionalSlice"}
	optionalMapValidator     = types.Name{Package: libValidationPkg, Name: "OptionalMap"}
)

func (requirednessTagValidator) doOptional(context Context) (Validations, error) {
	// Most validators don't care whether the value they are validating was
	// originally defined as a value-type or a pointer-type in the API.  This
	// one does.  Since Go doesn't do partial specialization of templates, we
	// do manual dispatch here.
	switch context.Type.Kind {
	case types.Slice:
		return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalSliceValidator)}}, nil
	case types.Map:
		return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalMapValidator)}}, nil
	case types.Pointer:
		return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalPointerValidator)}}, nil
	case types.Struct:
		// Specifying that a non-pointer struct is optional doesn't actually
		// make sense technically almost ever, and is better described as a
		// union inside the struct. It does, however, make sense as
		// documentation.
		return Validations{Comments: []string{"optional non-pointer structs are purely documentation"}}, nil
	}
	return Validations{Functions: []FunctionGen{Function(optionalTagName, ShortCircuit|NonError, optionalValueValidator)}}, nil
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
	switch context.Type.Kind {
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
		// The +forbidden tag on a non-pointer struct is not supported.
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
