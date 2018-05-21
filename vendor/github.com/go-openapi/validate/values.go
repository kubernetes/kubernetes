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
	"regexp"
	"unicode/utf8"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/swag"
)

// Enum validates if the data is a member of the enum
func Enum(path, in string, data interface{}, enum interface{}) *errors.Validation {
	val := reflect.ValueOf(enum)
	if val.Kind() != reflect.Slice {
		return nil
	}

	var values []interface{}
	for i := 0; i < val.Len(); i++ {
		ele := val.Index(i)
		enumValue := ele.Interface()
		if data != nil {
			if reflect.DeepEqual(data, enumValue) {
				return nil
			}
			actualType := reflect.TypeOf(enumValue)
			if actualType == nil {
				continue
			}
			expectedValue := reflect.ValueOf(data)
			if expectedValue.IsValid() && expectedValue.Type().ConvertibleTo(actualType) {
				// Attempt comparison after type conversion
				if reflect.DeepEqual(expectedValue.Convert(actualType).Interface(), enumValue) {
					return nil
				}
			}
		}
		values = append(values, enumValue)
	}
	return errors.EnumFail(path, in, data, values)
}

// MinItems validates that there are at least n items in a slice
func MinItems(path, in string, size, min int64) *errors.Validation {
	if size < min {
		return errors.TooFewItems(path, in, min)
	}
	return nil
}

// MaxItems validates that there are at most n items in a slice
func MaxItems(path, in string, size, max int64) *errors.Validation {
	if size > max {
		return errors.TooManyItems(path, in, max)
	}
	return nil
}

// UniqueItems validates that the provided slice has unique elements
func UniqueItems(path, in string, data interface{}) *errors.Validation {
	val := reflect.ValueOf(data)
	if val.Kind() != reflect.Slice {
		return nil
	}
	var unique []interface{}
	for i := 0; i < val.Len(); i++ {
		v := val.Index(i).Interface()
		for _, u := range unique {
			if reflect.DeepEqual(v, u) {
				return errors.DuplicateItems(path, in)
			}
		}
		unique = append(unique, v)
	}
	return nil
}

// MinLength validates a string for minimum length
func MinLength(path, in, data string, minLength int64) *errors.Validation {
	strLen := int64(utf8.RuneCount([]byte(data)))
	if strLen < minLength {
		return errors.TooShort(path, in, minLength)
	}
	return nil
}

// MaxLength validates a string for maximum length
func MaxLength(path, in, data string, maxLength int64) *errors.Validation {
	strLen := int64(utf8.RuneCount([]byte(data)))
	if strLen > maxLength {
		return errors.TooLong(path, in, maxLength)
	}
	return nil
}

// Required validates an interface for requiredness
func Required(path, in string, data interface{}) *errors.Validation {
	val := reflect.ValueOf(data)
	if val.IsValid() {
		if reflect.DeepEqual(reflect.Zero(val.Type()).Interface(), val.Interface()) {
			return errors.Required(path, in)
		}
		return nil
	}
	return errors.Required(path, in)
}

// RequiredString validates a string for requiredness
func RequiredString(path, in, data string) *errors.Validation {
	if data == "" {
		return errors.Required(path, in)
	}
	return nil
}

// RequiredNumber validates a number for requiredness
func RequiredNumber(path, in string, data float64) *errors.Validation {
	if data == 0 {
		return errors.Required(path, in)
	}
	return nil
}

// Pattern validates a string against a regular expression
func Pattern(path, in, data, pattern string) *errors.Validation {
	re := regexp.MustCompile(pattern)
	if !re.MatchString(data) {
		return errors.FailedPattern(path, in, pattern)
	}
	return nil
}

// MaximumInt validates if a number is smaller than a given maximum
func MaximumInt(path, in string, data, max int64, exclusive bool) *errors.Validation {
	if (!exclusive && data > max) || (exclusive && data >= max) {
		return errors.ExceedsMaximumInt(path, in, max, exclusive)
	}
	return nil
}

// MaximumUint validates if a number is smaller than a given maximum
func MaximumUint(path, in string, data, max uint64, exclusive bool) *errors.Validation {
	if (!exclusive && data > max) || (exclusive && data >= max) {
		return errors.ExceedsMaximumUint(path, in, max, exclusive)
	}
	return nil
}

// Maximum validates if a number is smaller than a given maximum
func Maximum(path, in string, data, max float64, exclusive bool) *errors.Validation {
	if (!exclusive && data > max) || (exclusive && data >= max) {
		return errors.ExceedsMaximum(path, in, max, exclusive)
	}
	return nil
}

// Minimum validates if a number is smaller than a given minimum
func Minimum(path, in string, data, min float64, exclusive bool) *errors.Validation {
	if (!exclusive && data < min) || (exclusive && data <= min) {
		return errors.ExceedsMinimum(path, in, min, exclusive)
	}
	return nil
}

// MinimumInt validates if a number is smaller than a given minimum
func MinimumInt(path, in string, data, min int64, exclusive bool) *errors.Validation {
	if (!exclusive && data < min) || (exclusive && data <= min) {
		return errors.ExceedsMinimumInt(path, in, min, exclusive)
	}
	return nil
}

// MinimumUint validates if a number is smaller than a given minimum
func MinimumUint(path, in string, data, min uint64, exclusive bool) *errors.Validation {
	if (!exclusive && data < min) || (exclusive && data <= min) {
		return errors.ExceedsMinimumUint(path, in, min, exclusive)
	}
	return nil
}

// MultipleOf validates if the provided number is a multiple of the factor
func MultipleOf(path, in string, data, factor float64) *errors.Validation {
	mult := data / factor
	if factor < 1 {
		mult = 1 / factor * data
	}
	if !swag.IsFloat64AJSONInteger(mult) {
		return errors.NotMultipleOf(path, in, factor)
	}
	return nil
}

// FormatOf validates if a string matches a format in the format registry
func FormatOf(path, in, format, data string, registry strfmt.Registry) *errors.Validation {
	if registry == nil {
		registry = strfmt.Default
	}
	if ok := registry.ContainsName(format); !ok {
		return errors.InvalidTypeName(format)
	}
	if ok := registry.Validates(format, data); !ok {
		return errors.InvalidType(path, in, format, data)
	}

	return nil
}
