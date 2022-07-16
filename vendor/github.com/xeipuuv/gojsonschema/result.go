// Copyright 2015 xeipuuv ( https://github.com/xeipuuv )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author           xeipuuv
// author-github    https://github.com/xeipuuv
// author-mail      xeipuuv@gmail.com
//
// repository-name  gojsonschema
// repository-desc  An implementation of JSON Schema, based on IETF's draft v4 - Go language.
//
// description      Result and ResultError implementations.
//
// created          01-01-2015

package gojsonschema

import (
	"fmt"
	"strings"
)

type (
	// ErrorDetails is a map of details specific to each error.
	// While the values will vary, every error will contain a "field" value
	ErrorDetails map[string]interface{}

	// ResultError is the interface that library errors must implement
	ResultError interface {
		// Field returns the field name without the root context
		// i.e. firstName or person.firstName instead of (root).firstName or (root).person.firstName
		Field() string
		// SetType sets the error-type
		SetType(string)
		// Type returns the error-type
		Type() string
		// SetContext sets the JSON-context for the error
		SetContext(*JsonContext)
		// Context returns the JSON-context of the error
		Context() *JsonContext
		// SetDescription sets a description for the error
		SetDescription(string)
		// Description returns the description of the error
		Description() string
		// SetDescriptionFormat sets the format for the description in the default text/template format
		SetDescriptionFormat(string)
		// DescriptionFormat returns the format for the description in the default text/template format
		DescriptionFormat() string
		// SetValue sets the value related to the error
		SetValue(interface{})
		// Value returns the value related to the error
		Value() interface{}
		// SetDetails sets the details specific to the error
		SetDetails(ErrorDetails)
		// Details returns details about the error
		Details() ErrorDetails
		// String returns a string representation of the error
		String() string
	}

	// ResultErrorFields holds the fields for each ResultError implementation.
	// ResultErrorFields implements the ResultError interface, so custom errors
	// can be defined by just embedding this type
	ResultErrorFields struct {
		errorType         string       // A string with the type of error (i.e. invalid_type)
		context           *JsonContext // Tree like notation of the part that failed the validation. ex (root).a.b ...
		description       string       // A human readable error message
		descriptionFormat string       // A format for human readable error message
		value             interface{}  // Value given by the JSON file that is the source of the error
		details           ErrorDetails
	}

	// Result holds the result of a validation
	Result struct {
		errors []ResultError
		// Scores how well the validation matched. Useful in generating
		// better error messages for anyOf and oneOf.
		score int
	}
)

// Field returns the field name without the root context
// i.e. firstName or person.firstName instead of (root).firstName or (root).person.firstName
func (v *ResultErrorFields) Field() string {
	return strings.TrimPrefix(v.context.String(), STRING_ROOT_SCHEMA_PROPERTY+".")
}

// SetType sets the error-type
func (v *ResultErrorFields) SetType(errorType string) {
	v.errorType = errorType
}

// Type returns the error-type
func (v *ResultErrorFields) Type() string {
	return v.errorType
}

// SetContext sets the JSON-context for the error
func (v *ResultErrorFields) SetContext(context *JsonContext) {
	v.context = context
}

// Context returns the JSON-context of the error
func (v *ResultErrorFields) Context() *JsonContext {
	return v.context
}

// SetDescription sets a description for the error
func (v *ResultErrorFields) SetDescription(description string) {
	v.description = description
}

// Description returns the description of the error
func (v *ResultErrorFields) Description() string {
	return v.description
}

// SetDescriptionFormat sets the format for the description in the default text/template format
func (v *ResultErrorFields) SetDescriptionFormat(descriptionFormat string) {
	v.descriptionFormat = descriptionFormat
}

// DescriptionFormat returns the format for the description in the default text/template format
func (v *ResultErrorFields) DescriptionFormat() string {
	return v.descriptionFormat
}

// SetValue sets the value related to the error
func (v *ResultErrorFields) SetValue(value interface{}) {
	v.value = value
}

// Value returns the value related to the error
func (v *ResultErrorFields) Value() interface{} {
	return v.value
}

// SetDetails sets the details specific to the error
func (v *ResultErrorFields) SetDetails(details ErrorDetails) {
	v.details = details
}

// Details returns details about the error
func (v *ResultErrorFields) Details() ErrorDetails {
	return v.details
}

// String returns a string representation of the error
func (v ResultErrorFields) String() string {
	// as a fallback, the value is displayed go style
	valueString := fmt.Sprintf("%v", v.value)

	// marshal the go value value to json
	if v.value == nil {
		valueString = TYPE_NULL
	} else {
		if vs, err := marshalToJSONString(v.value); err == nil {
			if vs == nil {
				valueString = TYPE_NULL
			} else {
				valueString = *vs
			}
		}
	}

	return formatErrorDescription(Locale.ErrorFormat(), ErrorDetails{
		"context":     v.context.String(),
		"description": v.description,
		"value":       valueString,
		"field":       v.Field(),
	})
}

// Valid indicates if no errors were found
func (v *Result) Valid() bool {
	return len(v.errors) == 0
}

// Errors returns the errors that were found
func (v *Result) Errors() []ResultError {
	return v.errors
}

// AddError appends a fully filled error to the error set
// SetDescription() will be called with the result of the parsed err.DescriptionFormat()
func (v *Result) AddError(err ResultError, details ErrorDetails) {
	if _, exists := details["context"]; !exists && err.Context() != nil {
		details["context"] = err.Context().String()
	}

	err.SetDescription(formatErrorDescription(err.DescriptionFormat(), details))

	v.errors = append(v.errors, err)
}

func (v *Result) addInternalError(err ResultError, context *JsonContext, value interface{}, details ErrorDetails) {
	newError(err, context, value, Locale, details)
	v.errors = append(v.errors, err)
	v.score -= 2 // results in a net -1 when added to the +1 we get at the end of the validation function
}

// Used to copy errors from a sub-schema to the main one
func (v *Result) mergeErrors(otherResult *Result) {
	v.errors = append(v.errors, otherResult.Errors()...)
	v.score += otherResult.score
}

func (v *Result) incrementScore() {
	v.score++
}
