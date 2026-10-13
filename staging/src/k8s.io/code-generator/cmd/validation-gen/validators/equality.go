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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	eqTagName  = "eq"
	neqTagName = "neq"
)

func init() {
	RegisterTagValidator(eqTagValidator{})
	RegisterTagValidator(neqTagValidator{})
}

type eqTagValidator struct{}

func (eqTagValidator) Init(_ Config) {}

func (eqTagValidator) TagName() string {
	return eqTagName
}

var eqTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (eqTagValidator) ValidScopes() sets.Set[Scope] {
	return eqTagValidScopes
}

var (
	eqValidator = types.Name{Package: libValidationPkg, Name: "EQ"}
)

func (v eqTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	expectedValue, err := parseComparableValue(context, tag)
	if err != nil {
		return Validations{}, err
	}

	fn := Function(v.TagName(), DefaultFlags, eqValidator, expectedValue).
		WithEmits(Emission{field.ErrorTypeNotSupported, "", ""})
	return Validations{Functions: []FunctionGen{fn}}, nil
}

func (v eqTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:              v.TagName(),
		StabilityLevel:   TagStabilityLevelAlpha,
		Scopes:           sets.List(v.ValidScopes()),
		Description:      "Verifies the field's value is equal to a specific value. Supports string, integer, and boolean types.",
		PayloadsRequired: true,
		PayloadsType:     codetags.ValueTypeRaw,
		Payloads: []TagPayloadDoc{{
			Description: "<value>",
			Docs:        "The expected value. The parser will infer the type (string, int, bool).",
		}},
	}
}

type neqTagValidator struct{}

func (neqTagValidator) Init(_ Config) {}

func (neqTagValidator) TagName() string {
	return neqTagName
}

var neqTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (neqTagValidator) ValidScopes() sets.Set[Scope] {
	return neqTagValidScopes
}

var (
	neqValidator = types.Name{Package: libValidationPkg, Name: "NEQ"}
)

func (v neqTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	disallowedValue, err := parseComparableValue(context, tag)
	if err != nil {
		return Validations{}, err
	}

	fn := Function(v.TagName(), DefaultFlags, neqValidator, disallowedValue).
		WithEmits(Emission{field.ErrorTypeInvalid, "neq", ""})
	return Validations{Functions: []FunctionGen{fn}}, nil
}

// parseComparableValue parses an eq or neq payload for the field's type.
func parseComparableValue(context Context, tag codetags.Tag) (any, error) {
	t := util.NonPointer(util.NativeType(context.Type))
	if !util.IsDirectComparable(t) {
		return nil, fmt.Errorf("can only be used on comparable types (e.g. string, int, bool), but got %s", rootTypeString(context.Type, t))
	}

	if tag.ValueType == codetags.ValueTypeNone {
		return nil, fmt.Errorf("missing required payload")
	}

	var value any
	var err error

	switch {
	case t == types.String:
		if tag.ValueType != codetags.ValueTypeString {
			return nil, fmt.Errorf("type mismatch: field is a string, but payload is of type %s", tag.ValueType)
		}
		value = tag.Value
	case t == types.Bool:
		if tag.ValueType != codetags.ValueTypeBool {
			return nil, fmt.Errorf("type mismatch: field is a bool, but payload is of type %s", tag.ValueType)
		}
		value, err = util.ParseBool(tag.Value)
		if err != nil {
			return nil, fmt.Errorf("invalid bool value for payload: %w", err)
		}
	case types.IsInteger(t):
		if tag.ValueType != codetags.ValueTypeInt {
			return nil, fmt.Errorf("type mismatch: field is an integer, but payload is of type %s", tag.ValueType)
		}
		value, err = util.ParseInt(tag.Value)
		if err != nil {
			return nil, fmt.Errorf("invalid integer value for payload: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported type: %s", t.Name)
	}

	return value, nil
}

func (v neqTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:              v.TagName(),
		StabilityLevel:   TagStabilityLevelAlpha,
		Scopes:           sets.List(v.ValidScopes()),
		Description:      "Verifies the field's value is not equal to a specific disallowed value. Supports string, integer, and boolean types.",
		PayloadsRequired: true,
		PayloadsType:     codetags.ValueTypeRaw,
		Payloads: []TagPayloadDoc{{
			Description: "<value>",
			Docs:        "The disallowed value. The parser will infer the type (string, int, bool).",
		}},
	}
}
