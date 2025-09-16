/*
Copyright 2014 The Kubernetes Authors.

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

package field

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
)

// Error is an implementation of the 'error' interface, which represents a
// field-level validation error.
type Error struct {
	Type     ErrorType
	Field    string
	BadValue interface{}
	Detail   string

	// Origin uniquely identifies where this error was generated from. It is used in testing to
	// compare expected errors against actual errors without relying on exact detail string matching.
	// This allows tests to verify the correct validation logic triggered the error
	// regardless of how the error message might be formatted or localized.
	//
	// The value should be either:
	// - A simple camelCase identifier (e.g., "maximum", "maxItems")
	// - A structured format using "format=<dash-style-identifier>" for validation errors related to specific formats
	//   (e.g., "format=dns-label", "format=qualified-name")
	//
	// If the Origin corresponds to an existing declarative validation tag or JSON Schema keyword,
	// use that same name for consistency.
	//
	// Origin should be set in the most deeply nested validation function that
	// can still identify the unique source of the error.
	Origin string

	// CoveredByDeclarative is true when this error is covered by declarative
	// validation. This field is to identify errors from imperative validation
	// that should also be caught by declarative validation.
	CoveredByDeclarative bool
}

var _ error = &Error{}

// Error implements the error interface.
func (e *Error) Error() string {
	return fmt.Sprintf("%s: %s", e.Field, e.ErrorBody())
}

type OmitValueType struct{}

var omitValue = OmitValueType{}

// ErrorBody returns the error message without the field name.  This is useful
// for building nice-looking higher-level error reporting.
func (e *Error) ErrorBody() string {
	var s string
	switch e.Type {
	case ErrorTypeRequired, ErrorTypeForbidden, ErrorTypeTooLong, ErrorTypeInternal:
		s = e.Type.String()
	case ErrorTypeInvalid, ErrorTypeTypeInvalid, ErrorTypeNotSupported,
		ErrorTypeNotFound, ErrorTypeDuplicate, ErrorTypeTooMany:
		if e.BadValue == omitValue {
			s = e.Type.String()
			break
		}
		switch t := e.BadValue.(type) {
		case int64, int32, float64, float32, bool:
			// use simple printer for simple types
			s = fmt.Sprintf("%s: %v", e.Type, t)
		case string:
			s = fmt.Sprintf("%s: %q", e.Type, t)
		default:
			// use more complex techniques to render more complex types
			valstr := ""
			jb, err := json.Marshal(e.BadValue)
			if err == nil {
				// best case
				valstr = string(jb)
			} else if stringer, ok := e.BadValue.(fmt.Stringer); ok {
				// anything that defines String() is better than raw struct
				valstr = stringer.String()
			} else {
				// worst case - fallback to raw struct
				// TODO: internal types have panic guards against json.Marshalling to prevent
				// accidental use of internal types in external serialized form.  For now, use
				// %#v, although it would be better to show a more expressive output in the future
				valstr = fmt.Sprintf("%#v", e.BadValue)
			}
			s = fmt.Sprintf("%s: %s", e.Type, valstr)
		}
	default:
		internal := InternalError(nil, fmt.Errorf("unhandled error code: %s: please report this", e.Type))
		s = internal.ErrorBody()
	}
	if len(e.Detail) != 0 {
		s += fmt.Sprintf(": %s", e.Detail)
	}
	return s
}

// WithOrigin adds origin information to the FieldError
func (e *Error) WithOrigin(o string) *Error {
	e.Origin = o
	return e
}

// MarkCoveredByDeclarative marks the error as covered by declarative validation.
func (e *Error) MarkCoveredByDeclarative() *Error {
	e.CoveredByDeclarative = true
	return e
}

// ErrorType is a machine readable value providing more detail about why
// a field is invalid.  These values are expected to match 1-1 with
// CauseType in api/types.go.
type ErrorType string

// TODO: These values are duplicated in api/types.go, but there's a circular dep.  Fix it.
const (
	// ErrorTypeNotFound is used to report failure to find a requested value
	// (e.g. looking up an ID).  See NotFound().
	ErrorTypeNotFound ErrorType = "FieldValueNotFound"
	// ErrorTypeRequired is used to report required values that are not
	// provided (e.g. empty strings, null values, or empty arrays).  See
	// Required().
	ErrorTypeRequired ErrorType = "FieldValueRequired"
	// ErrorTypeDuplicate is used to report collisions of values that must be
	// unique (e.g. unique IDs).  See Duplicate().
	ErrorTypeDuplicate ErrorType = "FieldValueDuplicate"
	// ErrorTypeInvalid is used to report malformed values (e.g. failed regex
	// match, too long, out of bounds).  See Invalid().
	ErrorTypeInvalid ErrorType = "FieldValueInvalid"
	// ErrorTypeNotSupported is used to report unknown values for enumerated
	// fields (e.g. a list of valid values).  See NotSupported().
	ErrorTypeNotSupported ErrorType = "FieldValueNotSupported"
	// ErrorTypeForbidden is used to report valid (as per formatting rules)
	// values which would be accepted under some conditions, but which are not
	// permitted by the current conditions (such as security policy).  See
	// Forbidden().
	ErrorTypeForbidden ErrorType = "FieldValueForbidden"
	// ErrorTypeTooLong is used to report that the given value is too long.
	// This is similar to ErrorTypeInvalid, but the error will not include the
	// too-long value.  See TooLong().
	ErrorTypeTooLong ErrorType = "FieldValueTooLong"
	// ErrorTypeTooMany is used to report "too many". This is used to
	// report that a given list has too many items. This is similar to FieldValueTooLong,
	// but the error indicates quantity instead of length.
	ErrorTypeTooMany ErrorType = "FieldValueTooMany"
	// ErrorTypeInternal is used to report other errors that are not related
	// to user input.  See InternalError().
	ErrorTypeInternal ErrorType = "InternalError"
	// ErrorTypeTypeInvalid is for the value did not match the schema type for that field
	ErrorTypeTypeInvalid ErrorType = "FieldValueTypeInvalid"
)

// String converts a ErrorType into its corresponding canonical error message.
func (t ErrorType) String() string {
	switch t {
	case ErrorTypeNotFound:
		return "Not found"
	case ErrorTypeRequired:
		return "Required value"
	case ErrorTypeDuplicate:
		return "Duplicate value"
	case ErrorTypeInvalid:
		return "Invalid value"
	case ErrorTypeNotSupported:
		return "Unsupported value"
	case ErrorTypeForbidden:
		return "Forbidden"
	case ErrorTypeTooLong:
		return "Too long"
	case ErrorTypeTooMany:
		return "Too many"
	case ErrorTypeInternal:
		return "Internal error"
	case ErrorTypeTypeInvalid:
		return "Invalid value"
	default:
		return fmt.Sprintf("<unknown error %q>", string(t))
	}
}

// TypeInvalid returns a *Error indicating "type is invalid"
func TypeInvalid(field *Path, value interface{}, detail string) *Error {
	return &Error{ErrorTypeTypeInvalid, field.String(), value, detail, "", false}
}

// NotFound returns a *Error indicating "value not found".  This is
// used to report failure to find a requested value (e.g. looking up an ID).
func NotFound(field *Path, value interface{}) *Error {
	return &Error{ErrorTypeNotFound, field.String(), value, "", "", false}
}

// Required returns a *Error indicating "value required".  This is used
// to report required values that are not provided (e.g. empty strings, null
// values, or empty arrays).
func Required(field *Path, detail string) *Error {
	return &Error{ErrorTypeRequired, field.String(), "", detail, "", false}
}

// Duplicate returns a *Error indicating "duplicate value".  This is
// used to report collisions of values that must be unique (e.g. names or IDs).
func Duplicate(field *Path, value interface{}) *Error {
	return &Error{ErrorTypeDuplicate, field.String(), value, "", "", false}
}

// Invalid returns a *Error indicating "invalid value".  This is used
// to report malformed values (e.g. failed regex match, too long, out of bounds).
func Invalid(field *Path, value interface{}, detail string) *Error {
	return &Error{ErrorTypeInvalid, field.String(), value, detail, "", false}
}

// NotSupported returns a *Error indicating "unsupported value".
// This is used to report unknown values for enumerated fields (e.g. a list of
// valid values).
func NotSupported[T ~string](field *Path, value interface{}, validValues []T) *Error {
	detail := ""
	if len(validValues) > 0 {
		quotedValues := make([]string, len(validValues))
		for i, v := range validValues {
			quotedValues[i] = strconv.Quote(fmt.Sprint(v))
		}
		detail = "supported values: " + strings.Join(quotedValues, ", ")
	}
	return &Error{ErrorTypeNotSupported, field.String(), value, detail, "", false}
}

// Forbidden returns a *Error indicating "forbidden".  This is used to
// report valid (as per formatting rules) values which would be accepted under
// some conditions, but which are not permitted by current conditions (e.g.
// security policy).
func Forbidden(field *Path, detail string) *Error {
	return &Error{ErrorTypeForbidden, field.String(), "", detail, "", false}
}

// TooLong returns a *Error indicating "too long".  This is used to report that
// the given value is too long.  This is similar to Invalid, but the returned
// error will not include the too-long value. If maxLength is negative, it will
// be included in the message.  The value argument is not used.
func TooLong(field *Path, _ interface{}, maxLength int) *Error {
	var msg string
	if maxLength >= 0 {
		bs := "bytes"
		if maxLength == 1 {
			bs = "byte"
		}
		msg = fmt.Sprintf("may not be more than %d %s", maxLength, bs)
	} else {
		msg = "value is too long"
	}
	return &Error{ErrorTypeTooLong, field.String(), "<value omitted>", msg, "", false}
}

// TooLongMaxLength returns a *Error indicating "too long".
// Deprecated: Use TooLong instead.
func TooLongMaxLength(field *Path, value interface{}, maxLength int) *Error {
	return TooLong(field, "", maxLength)
}

// TooMany returns a *Error indicating "too many". This is used to
// report that a given list has too many items. This is similar to TooLong,
// but the returned error indicates quantity instead of length.
func TooMany(field *Path, actualQuantity, maxQuantity int) *Error {
	var msg string

	if maxQuantity >= 0 {
		is := "items"
		if maxQuantity == 1 {
			is = "item"
		}
		msg = fmt.Sprintf("must have at most %d %s", maxQuantity, is)
	} else {
		msg = "has too many items"
	}

	var actual interface{}
	if actualQuantity >= 0 {
		actual = actualQuantity
	} else {
		actual = omitValue
	}

	return &Error{ErrorTypeTooMany, field.String(), actual, msg, "", false}
}

// InternalError returns a *Error indicating "internal error".  This is used
// to signal that an error was found that was not directly related to user
// input.  The err argument must be non-nil.
func InternalError(field *Path, err error) *Error {
	return &Error{ErrorTypeInternal, field.String(), nil, err.Error(), "", false}
}

// ErrorList holds a set of Errors.  It is plausible that we might one day have
// non-field errors in this same umbrella package, but for now we don't, so
// we can keep it simple and leave ErrorList here.
type ErrorList []*Error

// NewErrorTypeMatcher returns an errors.Matcher that returns true
// if the provided error is a Error and has the provided ErrorType.
func NewErrorTypeMatcher(t ErrorType) utilerrors.Matcher {
	return func(err error) bool {
		if e, ok := err.(*Error); ok {
			return e.Type == t
		}
		return false
	}
}

// WithOrigin sets the origin for all errors in the list and returns the updated list.
func (list ErrorList) WithOrigin(origin string) ErrorList {
	for _, err := range list {
		err.Origin = origin
	}
	return list
}

// MarkCoveredByDeclarative marks all errors in the list as covered by declarative validation.
func (list ErrorList) MarkCoveredByDeclarative() ErrorList {
	for _, err := range list {
		err.CoveredByDeclarative = true
	}
	return list
}

// ToAggregate converts the ErrorList into an errors.Aggregate.
func (list ErrorList) ToAggregate() utilerrors.Aggregate {
	if len(list) == 0 {
		return nil
	}
	errs := make([]error, 0, len(list))
	errorMsgs := sets.NewString()
	for _, err := range list {
		msg := fmt.Sprintf("%v", err)
		if errorMsgs.Has(msg) {
			continue
		}
		errorMsgs.Insert(msg)
		errs = append(errs, err)
	}
	return utilerrors.NewAggregate(errs)
}

func fromAggregate(agg utilerrors.Aggregate) ErrorList {
	errs := agg.Errors()
	list := make(ErrorList, len(errs))
	for i := range errs {
		list[i] = errs[i].(*Error)
	}
	return list
}

// Filter removes items from the ErrorList that match the provided fns.
func (list ErrorList) Filter(fns ...utilerrors.Matcher) ErrorList {
	err := utilerrors.FilterOut(list.ToAggregate(), fns...)
	if err == nil {
		return nil
	}
	// FilterOut takes an Aggregate and returns an Aggregate
	return fromAggregate(err.(utilerrors.Aggregate))
}

// ExtractCoveredByDeclarative returns a new ErrorList containing only the errors that should be covered by declarative validation.
func (list ErrorList) ExtractCoveredByDeclarative() ErrorList {
	newList := ErrorList{}
	for _, err := range list {
		if err.CoveredByDeclarative {
			newList = append(newList, err)
		}
	}
	return newList
}

// RemoveCoveredByDeclarative returns a new ErrorList containing only the errors that should not be covered by declarative validation.
func (list ErrorList) RemoveCoveredByDeclarative() ErrorList {
	newList := ErrorList{}
	for _, err := range list {
		if !err.CoveredByDeclarative {
			newList = append(newList, err)
		}
	}
	return newList
}
