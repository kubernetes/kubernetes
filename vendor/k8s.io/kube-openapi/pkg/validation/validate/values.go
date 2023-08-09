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
	"fmt"
	"reflect"
	"unicode/utf8"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/validation/errors"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
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
			if actualType == nil { // Safeguard. Frankly, I don't know how we may get a nil
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
		return errors.TooFewItems(path, in, min, size)
	}
	return nil
}

// MaxItems validates that there are at most n items in a slice
func MaxItems(path, in string, size, max int64) *errors.Validation {
	if size > max {
		return errors.TooManyItems(path, in, max, size)
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
		return errors.TooShort(path, in, minLength, data)
	}
	return nil
}

// MaxLength validates a string for maximum length
func MaxLength(path, in, data string, maxLength int64) *errors.Validation {
	strLen := int64(utf8.RuneCount([]byte(data)))
	if strLen > maxLength {
		return errors.TooLong(path, in, maxLength, data)
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

// Pattern validates a string against a regular expression
func Pattern(path, in, data, pattern string) *errors.Validation {
	re, err := compileRegexp(pattern)
	if err != nil {
		return errors.FailedPattern(path, in, fmt.Sprintf("%s, but pattern is invalid: %s", pattern, err.Error()), data)
	}
	if !re.MatchString(data) {
		return errors.FailedPattern(path, in, pattern, data)
	}
	return nil
}

// MaximumInt validates if a number is smaller than a given maximum
func MaximumInt(path, in string, data, max int64, exclusive bool) *errors.Validation {
	if (!exclusive && data > max) || (exclusive && data >= max) {
		return errors.ExceedsMaximumInt(path, in, max, exclusive, data)
	}
	return nil
}

// MaximumUint validates if a number is smaller than a given maximum
func MaximumUint(path, in string, data, max uint64, exclusive bool) *errors.Validation {
	if (!exclusive && data > max) || (exclusive && data >= max) {
		return errors.ExceedsMaximumUint(path, in, max, exclusive, data)
	}
	return nil
}

// Maximum validates if a number is smaller than a given maximum
func Maximum(path, in string, data, max float64, exclusive bool) *errors.Validation {
	if (!exclusive && data > max) || (exclusive && data >= max) {
		return errors.ExceedsMaximum(path, in, max, exclusive, data)
	}
	return nil
}

// Minimum validates if a number is smaller than a given minimum
func Minimum(path, in string, data, min float64, exclusive bool) *errors.Validation {
	if (!exclusive && data < min) || (exclusive && data <= min) {
		return errors.ExceedsMinimum(path, in, min, exclusive, data)
	}
	return nil
}

// MinimumInt validates if a number is smaller than a given minimum
func MinimumInt(path, in string, data, min int64, exclusive bool) *errors.Validation {
	if (!exclusive && data < min) || (exclusive && data <= min) {
		return errors.ExceedsMinimumInt(path, in, min, exclusive, data)
	}
	return nil
}

// MinimumUint validates if a number is smaller than a given minimum
func MinimumUint(path, in string, data, min uint64, exclusive bool) *errors.Validation {
	if (!exclusive && data < min) || (exclusive && data <= min) {
		return errors.ExceedsMinimumUint(path, in, min, exclusive, data)
	}
	return nil
}

// MultipleOf validates if the provided number is a multiple of the factor
func MultipleOf(path, in string, data, factor float64) *errors.Validation {
	// multipleOf factor must be positive
	if factor < 0 {
		return errors.MultipleOfMustBePositive(path, in, factor)
	}
	var mult float64
	if factor < 1 {
		mult = 1 / factor * data
	} else {
		mult = data / factor
	}
	if !swag.IsFloat64AJSONInteger(mult) {
		return errors.NotMultipleOf(path, in, factor, data)
	}
	return nil
}

// MultipleOfInt validates if the provided integer is a multiple of the factor
func MultipleOfInt(path, in string, data int64, factor int64) *errors.Validation {
	// multipleOf factor must be positive
	if factor < 0 {
		return errors.MultipleOfMustBePositive(path, in, factor)
	}
	mult := data / factor
	if mult*factor != data {
		return errors.NotMultipleOf(path, in, factor, data)
	}
	return nil
}

// MultipleOfUint validates if the provided unsigned integer is a multiple of the factor
func MultipleOfUint(path, in string, data, factor uint64) *errors.Validation {
	mult := data / factor
	if mult*factor != data {
		return errors.NotMultipleOf(path, in, factor, data)
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

// MaximumNativeType provides native type constraint validation as a facade
// to various numeric types versions of Maximum constraint check.
//
// Assumes that any possible loss conversion during conversion has been
// checked beforehand.
//
// NOTE: currently, the max value is marshalled as a float64, no matter what,
// which means there may be a loss during conversions (e.g. for very large integers)
//
// TODO: Normally, a JSON MAX_SAFE_INTEGER check would ensure conversion remains loss-free
func MaximumNativeType(path, in string, val interface{}, max float64, exclusive bool) *errors.Validation {
	kind := reflect.ValueOf(val).Type().Kind()
	switch kind {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		value := valueHelp.asInt64(val)
		return MaximumInt(path, in, value, int64(max), exclusive)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		value := valueHelp.asUint64(val)
		if max < 0 {
			return errors.ExceedsMaximum(path, in, max, exclusive, val)
		}
		return MaximumUint(path, in, value, uint64(max), exclusive)
	case reflect.Float32, reflect.Float64:
		fallthrough
	default:
		value := valueHelp.asFloat64(val)
		return Maximum(path, in, value, max, exclusive)
	}
}

// MinimumNativeType provides native type constraint validation as a facade
// to various numeric types versions of Minimum constraint check.
//
// Assumes that any possible loss conversion during conversion has been
// checked beforehand.
//
// NOTE: currently, the min value is marshalled as a float64, no matter what,
// which means there may be a loss during conversions (e.g. for very large integers)
//
// TODO: Normally, a JSON MAX_SAFE_INTEGER check would ensure conversion remains loss-free
func MinimumNativeType(path, in string, val interface{}, min float64, exclusive bool) *errors.Validation {
	kind := reflect.ValueOf(val).Type().Kind()
	switch kind {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		value := valueHelp.asInt64(val)
		return MinimumInt(path, in, value, int64(min), exclusive)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		value := valueHelp.asUint64(val)
		if min < 0 {
			return nil
		}
		return MinimumUint(path, in, value, uint64(min), exclusive)
	case reflect.Float32, reflect.Float64:
		fallthrough
	default:
		value := valueHelp.asFloat64(val)
		return Minimum(path, in, value, min, exclusive)
	}
}

// MultipleOfNativeType provides native type constraint validation as a facade
// to various numeric types version of MultipleOf constraint check.
//
// Assumes that any possible loss conversion during conversion has been
// checked beforehand.
//
// NOTE: currently, the multipleOf factor is marshalled as a float64, no matter what,
// which means there may be a loss during conversions (e.g. for very large integers)
//
// TODO: Normally, a JSON MAX_SAFE_INTEGER check would ensure conversion remains loss-free
func MultipleOfNativeType(path, in string, val interface{}, multipleOf float64) *errors.Validation {
	kind := reflect.ValueOf(val).Type().Kind()
	switch kind {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		value := valueHelp.asInt64(val)
		return MultipleOfInt(path, in, value, int64(multipleOf))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		value := valueHelp.asUint64(val)
		return MultipleOfUint(path, in, value, uint64(multipleOf))
	case reflect.Float32, reflect.Float64:
		fallthrough
	default:
		value := valueHelp.asFloat64(val)
		return MultipleOf(path, in, value, multipleOf)
	}
}

// IsValueValidAgainstRange checks that a numeric value is compatible with
// the range defined by Type and Format, that is, may be converted without loss.
//
// NOTE: this check is about type capacity and not formal verification such as: 1.0 != 1L
func IsValueValidAgainstRange(val interface{}, typeName, format, prefix, path string) error {
	kind := reflect.ValueOf(val).Type().Kind()

	// What is the string representation of val
	stringRep := ""
	switch kind {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		stringRep = swag.FormatUint64(valueHelp.asUint64(val))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		stringRep = swag.FormatInt64(valueHelp.asInt64(val))
	case reflect.Float32, reflect.Float64:
		stringRep = swag.FormatFloat64(valueHelp.asFloat64(val))
	default:
		return fmt.Errorf("%s value number range checking called with invalid (non numeric) val type in %s", prefix, path)
	}

	var errVal error

	switch typeName {
	case integerType:
		switch format {
		case integerFormatInt32:
			_, errVal = swag.ConvertInt32(stringRep)
		case integerFormatUInt32:
			_, errVal = swag.ConvertUint32(stringRep)
		case integerFormatUInt64:
			_, errVal = swag.ConvertUint64(stringRep)
		case integerFormatInt64:
			fallthrough
		default:
			_, errVal = swag.ConvertInt64(stringRep)
		}
	case numberType:
		fallthrough
	default:
		switch format {
		case numberFormatFloat, numberFormatFloat32:
			_, errVal = swag.ConvertFloat32(stringRep)
		case numberFormatDouble, numberFormatFloat64:
			fallthrough
		default:
			// No check can be performed here since
			// no number beyond float64 is supported
		}
	}
	if errVal != nil { // We don't report the actual errVal from strconv
		if format != "" {
			errVal = fmt.Errorf("%s value must be of type %s with format %s in %s", prefix, typeName, format, path)
		} else {
			errVal = fmt.Errorf("%s value must be of type %s (default format) in %s", prefix, typeName, path)
		}
	}
	return errVal
}
