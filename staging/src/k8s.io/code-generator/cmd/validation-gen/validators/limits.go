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
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	minItemsTagName      = "k8s:minItems"
	maxItemsTagName      = "k8s:maxItems"
	minimumTagName       = "k8s:minimum"
	maximumTagName       = "k8s:maximum"
	minLengthTagName     = "k8s:minLength"
	maxLengthTagName     = "k8s:maxLength"
	maxBytesTagName      = "k8s:maxBytes"
	maxPropertiesTagName = "k8s:maxProperties"
)

func init() {
	RegisterTagValidator(minItemsTagValidator{})
	RegisterTagValidator(maxItemsTagValidator{})
	RegisterTagValidator(maxPropertiesTagValidator{})
	RegisterTagValidator(minimumTagValidator{})
	RegisterTagValidator(maximumTagValidator{})
	RegisterTagValidator(minLengthTagValidator{})
	RegisterTagValidator(maxLengthTagValidator{})
	RegisterTagValidator(maxBytesTagValidator{})
}

type maxLengthTagValidator struct{}

func (maxLengthTagValidator) Init(_ Config) {}

func (maxLengthTagValidator) TagName() string {
	return maxLengthTagName
}

var maxLengthTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (maxLengthTagValidator) ValidScopes() sets.Set[Scope] {
	return maxLengthTagValidScopes
}

var maxLengthValidator = types.Name{Package: libValidationPkg, Name: "MaxLength"}

func (maxLengthTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	if t := util.NonPointer(util.NativeType(context.Type)); t != types.String {
		return Validations{}, fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := util.ParseInt(tag.Value)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}
	if intVal < 0 {
		return result, fmt.Errorf("must be greater than or equal to zero")
	}
	result.AddFunction(Function(maxLengthTagName, DefaultFlags, maxLengthValidator, intVal))
	return result, nil
}

func (mltv maxLengthTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mltv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(mltv.ValidScopes()),
		Description: `Indicates that a string field has a limit on its length in characters.
		This could allow up to 4*N bytes if multi-byte characters are used.
		If you want to limit length of bytes specifically, use maxBytes.`,
		Payloads: []TagPayloadDoc{{
			Description: "<non-negative integer>",
			Docs:        "This field must be no more than X characters long.",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}

type maxBytesTagValidator struct{}

func (maxBytesTagValidator) Init(_ Config) {}

func (maxBytesTagValidator) TagName() string {
	return maxBytesTagName
}

var maxBytesTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (maxBytesTagValidator) ValidScopes() sets.Set[Scope] {
	return maxBytesTagValidScopes
}

var maxBytesValidator = types.Name{Package: libValidationPkg, Name: "MaxBytes"}

func (maxBytesTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	if t := util.NonPointer(util.NativeType(context.Type)); t != types.String {
		return Validations{}, fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := util.ParseInt(tag.Value)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}
	if intVal < 0 {
		return result, fmt.Errorf("must be greater than or equal to zero")
	}
	result.AddFunction(Function(maxBytesTagName, DefaultFlags, maxBytesValidator, intVal))
	return result, nil
}

func (mbtv maxBytesTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mbtv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(mbtv.ValidScopes()),
		Description: `Indicates that a string field has a limit on its length in bytes.
		This could only allow as few as N/4 multi-byte characters.
		If you want to limit length of characters specifically, use maxLength.`,
		Payloads: []TagPayloadDoc{{
			Description: "<non-negative integer>",
			Docs:        "This field must be no more than X bytes long.",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}

type minItemsTagValidator struct{}

func (minItemsTagValidator) Init(_ Config) {}

func (minItemsTagValidator) TagName() string {
	return minItemsTagName
}

var minItemsTagValidScopes = sets.New(
	ScopeType,
	ScopeField,
	ScopeListVal,
	ScopeMapVal,
)

func (minItemsTagValidator) ValidScopes() sets.Set[Scope] {
	return minItemsTagValidScopes
}

var minItemsValidator = types.Name{Package: libValidationPkg, Name: "MinItems"}

func (minItemsTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	if t := util.NativeType(context.Type); t.Kind != types.Slice && t.Kind != types.Array {
		return Validations{}, fmt.Errorf("can only be used on list types (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := util.ParseInt(tag.Value)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}
	if intVal < 0 {
		return result, fmt.Errorf("must be greater than or equal to zero")
	}
	result.AddFunction(Function(minItemsTagName, DefaultFlags, minItemsValidator, intVal))
	return result, nil
}

func (mitv minItemsTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mitv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(mitv.ValidScopes()),
		Description:    "Indicates that a list has a minimum size.",
		Payloads: []TagPayloadDoc{{
			Description: "<non-negative integer>",
			Docs:        "This list must be at least X items long.",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}

type maxItemsTagValidator struct{}

func (maxItemsTagValidator) Init(_ Config) {}

func (maxItemsTagValidator) TagName() string {
	return maxItemsTagName
}

var maxItemsTagValidScopes = sets.New(
	ScopeType,
	ScopeField,
	ScopeListVal,
	ScopeMapVal,
)

func (maxItemsTagValidator) ValidScopes() sets.Set[Scope] {
	return maxItemsTagValidScopes
}

var maxItemsValidator = types.Name{Package: libValidationPkg, Name: "MaxItems"}

func (maxItemsTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// NOTE: pointers to lists are not supported, so we should never see a pointer here.
	if t := util.NativeType(context.Type); t.Kind != types.Slice && t.Kind != types.Array {
		return Validations{}, fmt.Errorf("can only be used on list types (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := util.ParseInt(tag.Value)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}
	if intVal < 0 {
		return result, fmt.Errorf("must be greater than or equal to zero")
	}
	// Note: maxItems short-circuits other validations for safety.
	result.AddFunction(Function(maxItemsTagName, ShortCircuit, maxItemsValidator, intVal))
	return result, nil
}

func (mitv maxItemsTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mitv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(mitv.ValidScopes()),
		Description:    "Indicates that a list has a limit on its size.",
		Payloads: []TagPayloadDoc{{
			Description: "<non-negative integer>",
			Docs:        "This list must be no more than X items long.",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}

type maxPropertiesTagValidator struct{}

func (maxPropertiesTagValidator) Init(_ Config) {}

func (maxPropertiesTagValidator) TagName() string {
	return maxPropertiesTagName
}

var maxPropertiesTagValidScopes = sets.New(
	ScopeType,
	ScopeField,
)

func (maxPropertiesTagValidator) ValidScopes() sets.Set[Scope] {
	return maxPropertiesTagValidScopes
}

var maxPropertiesValidator = types.Name{Package: libValidationPkg, Name: "MaxProperties"}

func (maxPropertiesTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// NOTE: pointers to maps are not supported, so we should never see a pointer here.
	t := util.NativeType(context.Type)
	if t.Kind != types.Map {
		return Validations{}, fmt.Errorf("can only be used on map types (%s)", rootTypeString(context.Type, t))
	}
	keyType := util.NativeType(t.Key)
	if keyType.Kind != types.Builtin || keyType.Name.Name != "string" {
		return Validations{}, fmt.Errorf("can only be used on map types with string-based keys (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := util.ParseInt(tag.Value)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}
	if intVal < 0 {
		return result, fmt.Errorf("must be greater than or equal to zero")
	}
	if intVal > 100000 {
		return result, fmt.Errorf("must be less than or equal to 100000")
	}
	// Note: maxProperties short-circuits other validations.
	result.AddFunction(Function(maxPropertiesTagName, ShortCircuit, maxPropertiesValidator, intVal))
	return result, nil
}

func (mptv maxPropertiesTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mptv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(mptv.ValidScopes()),
		Description:    "maxProperties provides a limit on properties of an object as defined by JSON schema. In Kubernetes it may only be used to constrain the number of elements on a field defined as a golang map.",
		Payloads: []TagPayloadDoc{{
			Description: "<non-negative integer>",
			Docs:        "This map must have no more than X properties (where X <= 100000).",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}

type minimumTagValidator struct{}

func (minimumTagValidator) Init(_ Config) {}

func (minimumTagValidator) TagName() string {
	return minimumTagName
}

var minimumTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (minimumTagValidator) ValidScopes() sets.Set[Scope] {
	return minimumTagValidScopes
}

var minimumValidator = types.Name{Package: libValidationPkg, Name: "Minimum"}

func (minimumTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	t := util.NonPointer(util.NativeType(context.Type))
	if !types.IsInteger(t) {
		return result, fmt.Errorf("can only be used on integer types (%s)", rootTypeString(context.Type, t))
	}

	bitSize, err := intBitSize(t)
	if err != nil {
		return result, err
	}
	if isUnsignedInt(t) {
		uintVal, err := util.ParseUnsignedInt(tag.Value, bitSize)
		if err != nil {
			return result, fmt.Errorf("failed to parse tag payload: %w", err)
		}
		result.AddFunction(Function(minimumTagName, DefaultFlags, minimumValidator, uintVal))
	} else {
		intVal, err := util.ParseSignedInt(tag.Value, bitSize)
		if err != nil {
			return result, fmt.Errorf("failed to parse tag payload: %w", err)
		}
		result.AddFunction(Function(minimumTagName, DefaultFlags, minimumValidator, intVal))
	}
	return result, nil
}

func (mtv minimumTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mtv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(mtv.ValidScopes()),
		Description:    "Indicates that a numeric field has a minimum value.",
		Payloads: []TagPayloadDoc{{
			Description: "<integer>",
			Docs:        "This field must be greater than or equal to X.",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}

type maximumTagValidator struct{}

func (maximumTagValidator) Init(_ Config) {}

func (maximumTagValidator) TagName() string {
	return maximumTagName
}

var maximumTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (maximumTagValidator) ValidScopes() sets.Set[Scope] {
	return maximumTagValidScopes
}

var maximumValidator = types.Name{Package: libValidationPkg, Name: "Maximum"}

func (maximumTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	t := util.NonPointer(util.NativeType(context.Type))
	if !types.IsInteger(t) {
		return result, fmt.Errorf("can only be used on integer types (%s)", rootTypeString(context.Type, t))
	}

	bitSize, err := intBitSize(t)
	if err != nil {
		return result, err
	}
	if isUnsignedInt(t) {
		uintVal, err := util.ParseUnsignedInt(tag.Value, bitSize)
		if err != nil {
			return result, fmt.Errorf("failed to parse tag payload: %w", err)
		}
		result.AddFunction(Function(maximumTagName, DefaultFlags, maximumValidator, uintVal))
	} else {
		intVal, err := util.ParseSignedInt(tag.Value, bitSize)
		if err != nil {
			return result, fmt.Errorf("failed to parse tag payload: %w", err)
		}
		result.AddFunction(Function(maximumTagName, DefaultFlags, maximumValidator, intVal))
	}
	return result, nil
}

func (mtv maximumTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mtv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(mtv.ValidScopes()),
		Description:    "Indicates that a numeric field has a maximum value.",
		Payloads: []TagPayloadDoc{{
			Description: "<integer>",
			Docs:        "This field must be less than or equal to X.",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}

type minLengthTagValidator struct{}

func (minLengthTagValidator) Init(_ Config) {}

func (minLengthTagValidator) TagName() string {
	return minLengthTagName
}

var minLengthTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (minLengthTagValidator) ValidScopes() sets.Set[Scope] {
	return minLengthTagValidScopes
}

var minLengthValidator = types.Name{Package: libValidationPkg, Name: "MinLength"}

func (minLengthTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	if t := util.NonPointer(util.NativeType(context.Type)); t != types.String {
		return result, fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := util.ParseInt(tag.Value)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}

	// Usage of `+k8s:minLength=0` is useful as a semantic representation of the fact that
	// the minimum valid length of the field was considered and that it intentionally doesn't
	// have a minimum length constraint.
	// Because a minimum length of `0` is semantically equivalent to not performing
	// any validation, we only add a validation function to the result if the
	// minimum length constraint is > 0.
	if intVal > 0 {
		result.AddFunction(Function(minLengthTagName, DefaultFlags, minLengthValidator, intVal))
	}
	return result, nil
}

func (mltv minLengthTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            mltv.TagName(),
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         sets.List(mltv.ValidScopes()),
		Description: `Indicates that a string field has a minimum length for its value in characters.
		This means that the minimum size in bytes is a range from X to 4X if multi-byte characters are allowed.
		`,
		Payloads: []TagPayloadDoc{{
			Description: "<integer>",
			Docs:        "This field must be at least X characters long.",
		}},
		PayloadsType:     codetags.ValueTypeInt,
		PayloadsRequired: true,
	}
}
