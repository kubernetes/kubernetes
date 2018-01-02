// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validate

import (
	"reflect"
	"testing"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/stretchr/testify/assert"
)

// common validations: enum, allOf, anyOf, oneOf, not, definitions

func maxError(param *spec.Parameter) *errors.Validation {
	return errors.ExceedsMaximum(param.Name, param.In, *param.Maximum, param.ExclusiveMaximum)
}

func minError(param *spec.Parameter) *errors.Validation {
	return errors.ExceedsMinimum(param.Name, param.In, *param.Minimum, param.ExclusiveMinimum)
}

func multipleOfError(param *spec.Parameter) *errors.Validation {
	return errors.NotMultipleOf(param.Name, param.In, *param.MultipleOf)
}

func makeFloat(data interface{}) float64 {
	val := reflect.ValueOf(data)
	knd := val.Kind()
	switch {
	case knd >= reflect.Int && knd <= reflect.Int64:
		return float64(val.Int())
	case knd >= reflect.Uint && knd <= reflect.Uint64:
		return float64(val.Uint())
	default:
		return val.Float()
	}
}

func TestNumberParameterValidation(t *testing.T) {

	values := [][]interface{}{
		[]interface{}{23, 49, 56, 21, 14, 35, 28, 7, 42},
		[]interface{}{uint(23), uint(49), uint(56), uint(21), uint(14), uint(35), uint(28), uint(7), uint(42)},
		[]interface{}{float64(23), float64(49), float64(56), float64(21), float64(14), float64(35), float64(28), float64(7), float64(42)},
	}

	for _, v := range values {
		factorParam := spec.QueryParam("factor")
		factorParam.WithMaximum(makeFloat(v[1]), false)
		factorParam.WithMinimum(makeFloat(v[3]), false)
		factorParam.WithMultipleOf(makeFloat(v[7]))
		factorParam.WithEnum(v[3], v[6], v[8], v[1])
		factorParam.Typed("number", "double")
		validator := NewParamValidator(factorParam, strfmt.Default)

		// MultipleOf
		err := validator.Validate(v[0])
		assert.True(t, err.HasErrors())
		assert.EqualError(t, multipleOfError(factorParam), err.Errors[0].Error())

		// Maximum
		err = validator.Validate(v[1])
		assert.True(t, err == nil || err.IsValid())
		if err != nil {
			assert.Empty(t, err.Errors)
		}
		err = validator.Validate(v[2])

		assert.True(t, err.HasErrors())
		assert.EqualError(t, maxError(factorParam), err.Errors[0].Error())

		// ExclusiveMaximum
		factorParam.ExclusiveMaximum = true
		// requires a new items validator because this is set a creation time
		validator = NewParamValidator(factorParam, strfmt.Default)
		err = validator.Validate(v[1])
		assert.True(t, err.HasErrors())
		assert.EqualError(t, maxError(factorParam), err.Errors[0].Error())

		// Minimum
		err = validator.Validate(v[3])
		assert.True(t, err == nil || err.IsValid())
		err = validator.Validate(v[4])
		assert.True(t, err.HasErrors())
		assert.EqualError(t, minError(factorParam), err.Errors[0].Error())

		// ExclusiveMinimum
		factorParam.ExclusiveMinimum = true
		// requires a new items validator because this is set a creation time
		validator = NewParamValidator(factorParam, strfmt.Default)
		err = validator.Validate(v[3])
		assert.True(t, err.HasErrors())
		assert.EqualError(t, minError(factorParam), err.Errors[0].Error())

		// Enum
		err = validator.Validate(v[5])
		assert.True(t, err.HasErrors())
		assert.EqualError(t, enumFail(factorParam, v[5]), err.Errors[0].Error())

		err = validator.Validate(v[6])
		assert.True(t, err == nil || err.IsValid())
	}

	// Not required in a parameter or items
	// AllOf
	// AnyOf
	// OneOf
	// Not
	// Definitions
}

func maxLengthError(param *spec.Parameter) *errors.Validation {
	return errors.TooLong(param.Name, param.In, *param.MaxLength)
}

func minLengthError(param *spec.Parameter) *errors.Validation {
	return errors.TooShort(param.Name, param.In, *param.MinLength)
}

func patternFail(param *spec.Parameter) *errors.Validation {
	return errors.FailedPattern(param.Name, param.In, param.Pattern)
}

func enumFail(param *spec.Parameter, data interface{}) *errors.Validation {
	return errors.EnumFail(param.Name, param.In, data, param.Enum)
}

func TestStringParameterValidation(t *testing.T) {
	nameParam := spec.QueryParam("name").AsRequired().WithMinLength(3).WithMaxLength(5).WithPattern(`^[a-z]+$`).Typed("string", "")
	nameParam.WithEnum("aaa", "bbb", "ccc")
	validator := NewParamValidator(nameParam, strfmt.Default)

	// required
	err := validator.Validate("")
	assert.True(t, err.HasErrors())
	assert.EqualError(t, requiredError(nameParam), err.Errors[0].Error())
	// MaxLength
	err = validator.Validate("abcdef")
	assert.True(t, err.HasErrors())
	assert.EqualError(t, maxLengthError(nameParam), err.Errors[0].Error())
	// MinLength
	err = validator.Validate("a")
	assert.True(t, err.HasErrors())
	assert.EqualError(t, minLengthError(nameParam), err.Errors[0].Error())
	// Pattern
	err = validator.Validate("a394")
	assert.True(t, err.HasErrors())
	assert.EqualError(t, patternFail(nameParam), err.Errors[0].Error())

	// Enum
	err = validator.Validate("abcde")
	assert.True(t, err.HasErrors())
	assert.EqualError(t, enumFail(nameParam, "abcde"), err.Errors[0].Error())

	// Valid passes
	err = validator.Validate("bbb")
	assert.True(t, err == nil || err.IsValid())

	// Not required in a parameter or items
	// AllOf
	// AnyOf
	// OneOf
	// Not
	// Definitions
}

func minItemsError(param *spec.Parameter) *errors.Validation {
	return errors.TooFewItems(param.Name, param.In, *param.MinItems)
}
func maxItemsError(param *spec.Parameter) *errors.Validation {
	return errors.TooManyItems(param.Name, param.In, *param.MaxItems)
}
func duplicatesError(param *spec.Parameter) *errors.Validation {
	return errors.DuplicateItems(param.Name, param.In)
}

func TestArrayParameterValidation(t *testing.T) {
	tagsParam := spec.QueryParam("tags").CollectionOf(stringItems, "").WithMinItems(1).WithMaxItems(5).UniqueValues()
	tagsParam.WithEnum([]string{"a", "a", "a"}, []string{"b", "b", "b"}, []string{"c", "c", "c"})
	validator := NewParamValidator(tagsParam, strfmt.Default)

	// MinItems
	err := validator.Validate([]string{})
	assert.True(t, err.HasErrors())
	assert.EqualError(t, minItemsError(tagsParam), err.Errors[0].Error())
	// MaxItems
	err = validator.Validate([]string{"a", "b", "c", "d", "e", "f"})
	assert.True(t, err.HasErrors())
	assert.EqualError(t, maxItemsError(tagsParam), err.Errors[0].Error())
	// UniqueItems
	err = validator.Validate([]string{"a", "a"})
	assert.True(t, err.HasErrors())
	assert.EqualError(t, duplicatesError(tagsParam), err.Errors[0].Error())

	// Enum
	err = validator.Validate([]string{"a", "b", "c"})
	assert.True(t, err.HasErrors())
	assert.EqualError(t, enumFail(tagsParam, []string{"a", "b", "c"}), err.Errors[0].Error())

	// Items
	strItems := spec.NewItems().WithMinLength(3).WithMaxLength(5).WithPattern(`^[a-z]+$`).Typed("string", "")
	tagsParam = spec.QueryParam("tags").CollectionOf(strItems, "").WithMinItems(1).WithMaxItems(5).UniqueValues()
	validator = NewParamValidator(tagsParam, strfmt.Default)
	err = validator.Validate([]string{"aa", "bbb", "ccc"})
	assert.True(t, err.HasErrors())
	assert.EqualError(t, minLengthErrorItems("tags.0", tagsParam.In, strItems), err.Errors[0].Error())

	// Not required in a parameter or items
	// Additional items
	// AllOf
	// AnyOf
	// OneOf
	// Not
	// Definitions
}
