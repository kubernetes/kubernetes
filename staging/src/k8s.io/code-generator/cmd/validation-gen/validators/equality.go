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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	neqTagName = "k8s:neq"
)

func init() {
	RegisterTagValidator(neqTagValidator{})
}

type neqTagValidator struct{}

func (neqTagValidator) Init(_ Config) {}

func (neqTagValidator) TagName() string {
	return neqTagName
}

var neqTagValidScopes = sets.New(ScopeAny)

func (neqTagValidator) ValidScopes() sets.Set[Scope] {
	return neqTagValidScopes
}

var (
	neqValidator = types.Name{Package: libValidationPkg, Name: "NEQ"}
)

func (v neqTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	t := util.NonPointer(util.NativeType(context.Type))
	if !util.IsDirectComparable(t) {
		return Validations{}, fmt.Errorf("can only be used on comparable types (e.g. string, int, bool), but got %s", rootTypeString(context.Type, t))
	}

	if tag.ValueType == codetags.ValueTypeNone {
		return Validations{}, fmt.Errorf("missing required payload")
	}

	var disallowedValue any
	var err error

	switch {
	case t == types.String:
		if tag.ValueType != codetags.ValueTypeString {
			return Validations{}, fmt.Errorf("type mismatch: field is a string, but payload is of type %s", tag.ValueType)
		}
		disallowedValue = tag.Value
	case t == types.Bool:
		if tag.ValueType != codetags.ValueTypeBool {
			return Validations{}, fmt.Errorf("type mismatch: field is a bool, but payload is of type %s", tag.ValueType)
		}
		disallowedValue, err = strconv.ParseBool(tag.Value)
		if err != nil {
			return Validations{}, fmt.Errorf("invalid bool value for payload: %w", err)
		}
	case types.IsInteger(t):
		if tag.ValueType != codetags.ValueTypeInt {
			return Validations{}, fmt.Errorf("type mismatch: field is an integer, but payload is of type %s", tag.ValueType)
		}
		disallowedValue, err = strconv.Atoi(tag.Value)
		if err != nil {
			return Validations{}, fmt.Errorf("invalid integer value for payload: %w", err)
		}
	default:
		return Validations{}, fmt.Errorf("unsupported type for 'neq' tag: %s", t.Name)
	}

	fn := Function(v.TagName(), DefaultFlags, neqValidator, disallowedValue)
	return Validations{Functions: []FunctionGen{fn}}, nil
}

func (v neqTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:              v.TagName(),
		Scopes:           v.ValidScopes().UnsortedList(),
		Description:      "Verifies the field's value is not equal to a specific disallowed value. Supports string, integer, and boolean types.",
		PayloadsRequired: true,
		PayloadsType:     codetags.ValueTypeRaw,
		Payloads: []TagPayloadDoc{{
			Description: "<value>",
			Docs:        "The disallowed value. The parser will infer the type (string, int, bool).",
		}},
	}
}
