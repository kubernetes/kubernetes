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

package errors

import (
	"fmt"
	"strings"
)

const (
	invalidType               = "%s is an invalid type name"
	typeFail                  = "%s in %s must be of type %s"
	typeFailWithData          = "%s in %s must be of type %s: %q"
	typeFailWithError         = "%s in %s must be of type %s, because: %s"
	requiredFail              = "%s in %s is required"
	tooLongMessage            = "%s in %s should be at most %d chars long"
	tooShortMessage           = "%s in %s should be at least %d chars long"
	patternFail               = "%s in %s should match '%s'"
	enumFail                  = "%s in %s should be one of %v"
	multipleOfFail            = "%s in %s should be a multiple of %v"
	maxIncFail                = "%s in %s should be less than or equal to %v"
	maxExcFail                = "%s in %s should be less than %v"
	minIncFail                = "%s in %s should be greater than or equal to %v"
	minExcFail                = "%s in %s should be greater than %v"
	uniqueFail                = "%s in %s shouldn't contain duplicates"
	maxItemsFail              = "%s in %s should have at most %d items"
	minItemsFail              = "%s in %s should have at least %d items"
	typeFailNoIn              = "%s must be of type %s"
	typeFailWithDataNoIn      = "%s must be of type %s: %q"
	typeFailWithErrorNoIn     = "%s must be of type %s, because: %s"
	requiredFailNoIn          = "%s is required"
	tooLongMessageNoIn        = "%s should be at most %d chars long"
	tooShortMessageNoIn       = "%s should be at least %d chars long"
	patternFailNoIn           = "%s should match '%s'"
	enumFailNoIn              = "%s should be one of %v"
	multipleOfFailNoIn        = "%s should be a multiple of %v"
	maxIncFailNoIn            = "%s should be less than or equal to %v"
	maxExcFailNoIn            = "%s should be less than %v"
	minIncFailNoIn            = "%s should be greater than or equal to %v"
	minExcFailNoIn            = "%s should be greater than %v"
	uniqueFailNoIn            = "%s shouldn't contain duplicates"
	maxItemsFailNoIn          = "%s should have at most %d items"
	minItemsFailNoIn          = "%s should have at least %d items"
	noAdditionalItems         = "%s in %s can't have additional items"
	noAdditionalItemsNoIn     = "%s can't have additional items"
	tooFewProperties          = "%s in %s should have at least %d properties"
	tooFewPropertiesNoIn      = "%s should have at least %d properties"
	tooManyProperties         = "%s in %s should have at most %d properties"
	tooManyPropertiesNoIn     = "%s should have at most %d properties"
	unallowedProperty         = "%s.%s in %s is a forbidden property"
	unallowedPropertyNoIn     = "%s.%s is a forbidden property"
	failedAllPatternProps     = "%s.%s in %s failed all pattern properties"
	failedAllPatternPropsNoIn = "%s.%s failed all pattern properties"
	multipleOfMustBePositive  = "factor MultipleOf declared for %s must be positive: %v"
)

// All code responses can be used to differentiate errors for different handling
// by the consuming program
const (
	// CompositeErrorCode remains 422 for backwards-compatibility
	// and to separate it from validation errors with cause
	CompositeErrorCode = 422
	// InvalidTypeCode is used for any subclass of invalid types
	InvalidTypeCode = 600 + iota
	RequiredFailCode
	TooLongFailCode
	TooShortFailCode
	PatternFailCode
	EnumFailCode
	MultipleOfFailCode
	MaxFailCode
	MinFailCode
	UniqueFailCode
	MaxItemsFailCode
	MinItemsFailCode
	NoAdditionalItemsCode
	TooFewPropertiesCode
	TooManyPropertiesCode
	UnallowedPropertyCode
	FailedAllPatternPropsCode
	MultipleOfMustBePositiveCode
)

// CompositeError is an error that groups several errors together
type CompositeError struct {
	Errors  []error
	code    int32
	message string
}

// Code for this error
func (c *CompositeError) Code() int32 {
	return c.code
}

func (c *CompositeError) Error() string {
	if len(c.Errors) > 0 {
		msgs := []string{c.message + ":"}
		for _, e := range c.Errors {
			msgs = append(msgs, e.Error())
		}
		return strings.Join(msgs, "\n")
	}
	return c.message
}

// CompositeValidationError an error to wrap a bunch of other errors
func CompositeValidationError(errors ...error) *CompositeError {
	return &CompositeError{
		code:    CompositeErrorCode,
		Errors:  append([]error{}, errors...),
		message: "validation failure list",
	}
}

// FailedAllPatternProperties an error for when the property doesn't match a pattern
func FailedAllPatternProperties(name, in, key string) *Validation {
	msg := fmt.Sprintf(failedAllPatternProps, name, key, in)
	if in == "" {
		msg = fmt.Sprintf(failedAllPatternPropsNoIn, name, key)
	}
	return &Validation{
		code:    FailedAllPatternPropsCode,
		Name:    name,
		In:      in,
		Value:   key,
		message: msg,
	}
}

// PropertyNotAllowed an error for when the property doesn't match a pattern
func PropertyNotAllowed(name, in, key string) *Validation {
	msg := fmt.Sprintf(unallowedProperty, name, key, in)
	if in == "" {
		msg = fmt.Sprintf(unallowedPropertyNoIn, name, key)
	}
	return &Validation{
		code:    UnallowedPropertyCode,
		Name:    name,
		In:      in,
		Value:   key,
		message: msg,
	}
}

// TooFewProperties an error for an object with too few properties
func TooFewProperties(name, in string, n int64) *Validation {
	msg := fmt.Sprintf(tooFewProperties, name, in, n)
	if in == "" {
		msg = fmt.Sprintf(tooFewPropertiesNoIn, name, n)
	}
	return &Validation{
		code:    TooFewPropertiesCode,
		Name:    name,
		In:      in,
		Value:   n,
		message: msg,
	}
}

// TooManyProperties an error for an object with too many properties
func TooManyProperties(name, in string, n int64) *Validation {
	msg := fmt.Sprintf(tooManyProperties, name, in, n)
	if in == "" {
		msg = fmt.Sprintf(tooManyPropertiesNoIn, name, n)
	}
	return &Validation{
		code:    TooManyPropertiesCode,
		Name:    name,
		In:      in,
		Value:   n,
		message: msg,
	}
}

// AdditionalItemsNotAllowed an error for invalid additional items
func AdditionalItemsNotAllowed(name, in string) *Validation {
	msg := fmt.Sprintf(noAdditionalItems, name, in)
	if in == "" {
		msg = fmt.Sprintf(noAdditionalItemsNoIn, name)
	}
	return &Validation{
		code:    NoAdditionalItemsCode,
		Name:    name,
		In:      in,
		message: msg,
	}
}

// InvalidCollectionFormat another flavor of invalid type error
func InvalidCollectionFormat(name, in, format string) *Validation {
	return &Validation{
		code:    InvalidTypeCode,
		Name:    name,
		In:      in,
		Value:   format,
		message: fmt.Sprintf("the collection format %q is not supported for the %s param %q", format, in, name),
	}
}

// InvalidTypeName an error for when the type is invalid
func InvalidTypeName(typeName string) *Validation {
	return &Validation{
		code:    InvalidTypeCode,
		Value:   typeName,
		message: fmt.Sprintf(invalidType, typeName),
	}
}

// InvalidType creates an error for when the type is invalid
func InvalidType(name, in, typeName string, value interface{}) *Validation {
	var message string

	if in != "" {
		switch value.(type) {
		case string:
			message = fmt.Sprintf(typeFailWithData, name, in, typeName, value)
		case error:
			message = fmt.Sprintf(typeFailWithError, name, in, typeName, value)
		default:
			message = fmt.Sprintf(typeFail, name, in, typeName)
		}
	} else {
		switch value.(type) {
		case string:
			message = fmt.Sprintf(typeFailWithDataNoIn, name, typeName, value)
		case error:
			message = fmt.Sprintf(typeFailWithErrorNoIn, name, typeName, value)
		default:
			message = fmt.Sprintf(typeFailNoIn, name, typeName)
		}
	}

	return &Validation{
		code:    InvalidTypeCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: message,
	}

}

// DuplicateItems error for when an array contains duplicates
func DuplicateItems(name, in string) *Validation {
	msg := fmt.Sprintf(uniqueFail, name, in)
	if in == "" {
		msg = fmt.Sprintf(uniqueFailNoIn, name)
	}
	return &Validation{
		code:    UniqueFailCode,
		Name:    name,
		In:      in,
		message: msg,
	}
}

// TooManyItems error for when an array contains too many items
func TooManyItems(name, in string, max int64, value interface{}) *Validation {
	msg := fmt.Sprintf(maxItemsFail, name, in, max)
	if in == "" {
		msg = fmt.Sprintf(maxItemsFailNoIn, name, max)
	}

	return &Validation{
		code:    MaxItemsFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: msg,
	}
}

// TooFewItems error for when an array contains too few items
func TooFewItems(name, in string, min int64, value interface{}) *Validation {
	msg := fmt.Sprintf(minItemsFail, name, in, min)
	if in == "" {
		msg = fmt.Sprintf(minItemsFailNoIn, name, min)
	}
	return &Validation{
		code:    MinItemsFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: msg,
	}
}

// ExceedsMaximumInt error for when maximum validation fails
func ExceedsMaximumInt(name, in string, max int64, exclusive bool, value interface{}) *Validation {
	var message string
	if in == "" {
		m := maxIncFailNoIn
		if exclusive {
			m = maxExcFailNoIn
		}
		message = fmt.Sprintf(m, name, max)
	} else {
		m := maxIncFail
		if exclusive {
			m = maxExcFail
		}
		message = fmt.Sprintf(m, name, in, max)
	}
	return &Validation{
		code:    MaxFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: message,
	}
}

// ExceedsMaximumUint error for when maximum validation fails
func ExceedsMaximumUint(name, in string, max uint64, exclusive bool, value interface{}) *Validation {
	var message string
	if in == "" {
		m := maxIncFailNoIn
		if exclusive {
			m = maxExcFailNoIn
		}
		message = fmt.Sprintf(m, name, max)
	} else {
		m := maxIncFail
		if exclusive {
			m = maxExcFail
		}
		message = fmt.Sprintf(m, name, in, max)
	}
	return &Validation{
		code:    MaxFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: message,
	}
}

// ExceedsMaximum error for when maximum validation fails
func ExceedsMaximum(name, in string, max float64, exclusive bool, value interface{}) *Validation {
	var message string
	if in == "" {
		m := maxIncFailNoIn
		if exclusive {
			m = maxExcFailNoIn
		}
		message = fmt.Sprintf(m, name, max)
	} else {
		m := maxIncFail
		if exclusive {
			m = maxExcFail
		}
		message = fmt.Sprintf(m, name, in, max)
	}
	return &Validation{
		code:    MaxFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: message,
	}
}

// ExceedsMinimumInt error for when minimum validation fails
func ExceedsMinimumInt(name, in string, min int64, exclusive bool, value interface{}) *Validation {
	var message string
	if in == "" {
		m := minIncFailNoIn
		if exclusive {
			m = minExcFailNoIn
		}
		message = fmt.Sprintf(m, name, min)
	} else {
		m := minIncFail
		if exclusive {
			m = minExcFail
		}
		message = fmt.Sprintf(m, name, in, min)
	}
	return &Validation{
		code:    MinFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: message,
	}
}

// ExceedsMinimumUint error for when minimum validation fails
func ExceedsMinimumUint(name, in string, min uint64, exclusive bool, value interface{}) *Validation {
	var message string
	if in == "" {
		m := minIncFailNoIn
		if exclusive {
			m = minExcFailNoIn
		}
		message = fmt.Sprintf(m, name, min)
	} else {
		m := minIncFail
		if exclusive {
			m = minExcFail
		}
		message = fmt.Sprintf(m, name, in, min)
	}
	return &Validation{
		code:    MinFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: message,
	}
}

// ExceedsMinimum error for when minimum validation fails
func ExceedsMinimum(name, in string, min float64, exclusive bool, value interface{}) *Validation {
	var message string
	if in == "" {
		m := minIncFailNoIn
		if exclusive {
			m = minExcFailNoIn
		}
		message = fmt.Sprintf(m, name, min)
	} else {
		m := minIncFail
		if exclusive {
			m = minExcFail
		}
		message = fmt.Sprintf(m, name, in, min)
	}
	return &Validation{
		code:    MinFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: message,
	}
}

// NotMultipleOf error for when multiple of validation fails
func NotMultipleOf(name, in string, multiple, value interface{}) *Validation {
	var msg string
	if in == "" {
		msg = fmt.Sprintf(multipleOfFailNoIn, name, multiple)
	} else {
		msg = fmt.Sprintf(multipleOfFail, name, in, multiple)
	}
	return &Validation{
		code:    MultipleOfFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: msg,
	}
}

// EnumFail error for when an enum validation fails
func EnumFail(name, in string, value interface{}, values []interface{}) *Validation {
	var msg string
	if in == "" {
		msg = fmt.Sprintf(enumFailNoIn, name, values)
	} else {
		msg = fmt.Sprintf(enumFail, name, in, values)
	}

	return &Validation{
		code:    EnumFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		Values:  values,
		message: msg,
	}
}

// Required error for when a value is missing
func Required(name, in string, value interface{}) *Validation {
	var msg string
	if in == "" {
		msg = fmt.Sprintf(requiredFailNoIn, name)
	} else {
		msg = fmt.Sprintf(requiredFail, name, in)
	}
	return &Validation{
		code:    RequiredFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: msg,
	}
}

// TooLong error for when a string is too long
func TooLong(name, in string, max int64, value interface{}) *Validation {
	var msg string
	if in == "" {
		msg = fmt.Sprintf(tooLongMessageNoIn, name, max)
	} else {
		msg = fmt.Sprintf(tooLongMessage, name, in, max)
	}
	return &Validation{
		code:    TooLongFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: msg,
	}
}

// TooShort error for when a string is too short
func TooShort(name, in string, min int64, value interface{}) *Validation {
	var msg string
	if in == "" {
		msg = fmt.Sprintf(tooShortMessageNoIn, name, min)
	} else {
		msg = fmt.Sprintf(tooShortMessage, name, in, min)
	}

	return &Validation{
		code:    TooShortFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: msg,
	}
}

// FailedPattern error for when a string fails a regex pattern match
// the pattern that is returned is the ECMA syntax version of the pattern not the golang version.
func FailedPattern(name, in, pattern string, value interface{}) *Validation {
	var msg string
	if in == "" {
		msg = fmt.Sprintf(patternFailNoIn, name, pattern)
	} else {
		msg = fmt.Sprintf(patternFail, name, in, pattern)
	}

	return &Validation{
		code:    PatternFailCode,
		Name:    name,
		In:      in,
		Value:   value,
		message: msg,
	}
}

// MultipleOfMustBePositive error for when a
// multipleOf factor is negative
func MultipleOfMustBePositive(name, in string, factor interface{}) *Validation {
	return &Validation{
		code:    MultipleOfMustBePositiveCode,
		Name:    name,
		In:      in,
		Value:   factor,
		message: fmt.Sprintf(multipleOfMustBePositive, name, factor),
	}
}
