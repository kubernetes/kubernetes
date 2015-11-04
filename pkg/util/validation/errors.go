/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"fmt"
	"strings"

	utilerrors "k8s.io/kubernetes/pkg/util/errors"

	"github.com/davecgh/go-spew/spew"
)

// Error is an implementation of the 'error' interface, which represents a
// validation error.
type Error struct {
	Type     ErrorType
	Field    string
	BadValue interface{}
	Detail   string
}

var _ error = &Error{}

// Error implements the error interface.
func (v *Error) Error() string {
	return fmt.Sprintf("%s: %s", v.Field, v.ErrorBody())
}

// ErrorBody returns the error message without the field name.  This is useful
// for building nice-looking higher-level error reporting.
func (v *Error) ErrorBody() string {
	var s string
	switch v.Type {
	case ErrorTypeRequired, ErrorTypeTooLong, ErrorTypeInternal:
		s = spew.Sprintf("%s", v.Type)
	default:
		s = spew.Sprintf("%s '%+v'", v.Type, v.BadValue)
	}
	if len(v.Detail) != 0 {
		s += fmt.Sprintf(", Details: %s", v.Detail)
	}
	return s
}

// ErrorType is a machine readable value providing more detail about why
// a field is invalid.  These values are expected to match 1-1 with
// CauseType in api/types.go.
type ErrorType string

// TODO: These values are duplicated in api/types.go, but there's a circular dep.  Fix it.
const (
	// ErrorTypeNotFound is used to report failure to find a requested value
	// (e.g. looking up an ID).  See NewNotFoundError.
	ErrorTypeNotFound ErrorType = "FieldValueNotFound"
	// ErrorTypeRequired is used to report required values that are not
	// provided (e.g. empty strings, null values, or empty arrays).  See
	// NewRequiredError.
	ErrorTypeRequired ErrorType = "FieldValueRequired"
	// ErrorTypeDuplicate is used to report collisions of values that must be
	// unique (e.g. unique IDs).  See NewDuplicateError.
	ErrorTypeDuplicate ErrorType = "FieldValueDuplicate"
	// ErrorTypeInvalid is used to report malformed values (e.g. failed regex
	// match, too long, out of bounds).  See NewInvalidError.
	ErrorTypeInvalid ErrorType = "FieldValueInvalid"
	// ErrorTypeNotSupported is used to report unknown values for enumerated
	// fields (e.g. a list of valid values).  See NewNotSupportedError.
	ErrorTypeNotSupported ErrorType = "FieldValueNotSupported"
	// ErrorTypeForbidden is used to report valid (as per formatting rules)
	// values which would be accepted under some conditions, but which are not
	// permitted by the current conditions (such as security policy).  See
	// NewForbiddenError.
	ErrorTypeForbidden ErrorType = "FieldValueForbidden"
	// ErrorTypeTooLong is used to report that the given value is too long.
	// This is similar to ErrorTypeInvalid, but the error will not include the
	// too-long value.  See NewTooLongError.
	ErrorTypeTooLong ErrorType = "FieldValueTooLong"
	// ErrorTypeInternal is used to report other errors that are not related
	// to user input.
	ErrorTypeInternal ErrorType = "InternalError"
)

// String converts a ErrorType into its corresponding canonical error message.
func (t ErrorType) String() string {
	switch t {
	case ErrorTypeNotFound:
		return "not found"
	case ErrorTypeRequired:
		return "required value"
	case ErrorTypeDuplicate:
		return "duplicate value"
	case ErrorTypeInvalid:
		return "invalid value"
	case ErrorTypeNotSupported:
		return "unsupported value"
	case ErrorTypeForbidden:
		return "forbidden"
	case ErrorTypeTooLong:
		return "too long"
	case ErrorTypeInternal:
		return "internal error"
	default:
		panic(fmt.Sprintf("unrecognized validation error: %q", t))
		return ""
	}
}

// NewNotFoundError returns a *Error indicating "value not found".  This is
// used to report failure to find a requested value (e.g. looking up an ID).
func NewNotFoundError(field string, value interface{}) *Error {
	return &Error{ErrorTypeNotFound, field, value, ""}
}

// NewRequiredError returns a *Error indicating "value required".  This is used
// to report required values that are not provided (e.g. empty strings, null
// values, or empty arrays).
func NewRequiredError(field string) *Error {
	return &Error{ErrorTypeRequired, field, "", ""}
}

// NewDuplicateError returns a *Error indicating "duplicate value".  This is
// used to report collisions of values that must be unique (e.g. names or IDs).
func NewDuplicateError(field string, value interface{}) *Error {
	return &Error{ErrorTypeDuplicate, field, value, ""}
}

// NewInvalidError returns a *Error indicating "invalid value".  This is used
// to report malformed values (e.g. failed regex match, too long, out of bounds).
func NewInvalidError(field string, value interface{}, detail string) *Error {
	return &Error{ErrorTypeInvalid, field, value, detail}
}

// NewNotSupportedError returns a *Error indicating "unsupported value".
// This is used to report unknown values for enumerated fields (e.g. a list of
// valid values).
func NewNotSupportedError(field string, value interface{}, validValues []string) *Error {
	detail := ""
	if validValues != nil && len(validValues) > 0 {
		detail = "supported values: " + strings.Join(validValues, ", ")
	}
	return &Error{ErrorTypeNotSupported, field, value, detail}
}

// NewForbiddenError returns a *Error indicating "forbidden".  This is used to
// report valid (as per formatting rules) values which would be accepted under
// some conditions, but which are not permitted by current conditions (e.g.
// security policy).
func NewForbiddenError(field string, value interface{}) *Error {
	return &Error{ErrorTypeForbidden, field, value, ""}
}

// NewTooLongError returns a *Error indicating "too long".  This is used to
// report that the given value is too long.  This is similar to
// NewInvalidError, but the returned error will not include the too-long
// value.
func NewTooLongError(field string, value interface{}, maxLength int) *Error {
	return &Error{ErrorTypeTooLong, field, value, fmt.Sprintf("must have at most %d characters", maxLength)}
}

// NewInternalError returns a *Error indicating "internal error".  This is used
// to signal that an error was found that was not directly related to user
// input.  The err argument must be non-nil.
func NewInternalError(field string, err error) *Error {
	return &Error{ErrorTypeInternal, field, nil, err.Error()}
}

// ErrorList holds a set of errors.
type ErrorList []*Error

// Prefix adds a prefix to the Field of every Error in the list.
// Returns the list for convenience.
func (list ErrorList) Prefix(prefix string) ErrorList {
	for i := range list {
		err := list[i]
		if strings.HasPrefix(err.Field, "[") {
			err.Field = prefix + err.Field
		} else if len(err.Field) != 0 {
			err.Field = prefix + "." + err.Field
		} else {
			err.Field = prefix
		}
	}
	return list
}

// PrefixIndex adds an index to the Field of every Error in the list.
// Returns the list for convenience.
func (list ErrorList) PrefixIndex(index int) ErrorList {
	return list.Prefix(fmt.Sprintf("[%d]", index))
}

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

// ToAggregate converts the ErrorList into an errors.Aggregate.
func (list ErrorList) ToAggregate() utilerrors.Aggregate {
	errs := make([]error, len(list))
	for i := range list {
		errs[i] = list[i]
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
