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

// ErrorType is a machine readable value providing more detail about why
// a field is invalid.  These values are expected to match 1-1 with
// CauseType in api/types.go.
type ErrorType string

// TODO: These values are duplicated in api/types.go, but there's a circular dep.  Fix it.
const (
	// ErrorType is used to report failure to find a requested value
	// (e.g. looking up an ID).
	ErrorTypeNotFound ErrorType = "FieldValueNotFound"
	// ErrorTypeRequired is used to report required values that are not
	// provided (e.g. empty strings, null values, or empty arrays).
	ErrorTypeRequired ErrorType = "FieldValueRequired"
	// ErrorTypeDuplicate is used to report collisions of values that must be
	// unique (e.g. unique IDs).
	ErrorTypeDuplicate ErrorType = "FieldValueDuplicate"
	// ErrorTypeInvalid is used to report malformed values (e.g. failed regex
	// match).
	ErrorTypeInvalid ErrorType = "FieldValueInvalid"
	// ErrorTypeNotSupported is used to report valid (as per formatting rules)
	// values that can not be handled (e.g. an enumerated string).
	ErrorTypeNotSupported ErrorType = "FieldValueNotSupported"
	// ErrorTypeForbidden is used to report valid (as per formatting rules)
	// values which would be accepted by some api instances, but which would invoke behavior
	// not permitted by this api instance (such as due to stricter security policy).
	ErrorTypeForbidden ErrorType = "FieldValueForbidden"
	// ErrorTypeTooLong is used to report that given value is too long.
	ErrorTypeTooLong ErrorType = "FieldValueTooLong"
)

// String converts a ErrorType into its corresponding error message.
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
	default:
		panic(fmt.Sprintf("unrecognized validation type: %#v", t))
		return ""
	}
}

// Error is an implementation of the 'error' interface, which represents an error of validation.
type Error struct {
	Type     ErrorType
	Field    string
	BadValue interface{}
	Detail   string
}

var _ error = &Error{}

func (v *Error) Error() string {
	return fmt.Sprintf("%s: %s", v.Field, v.ErrorBody())
}

func (v *Error) ErrorBody() string {
	var s string
	switch v.Type {
	case ErrorTypeRequired, ErrorTypeTooLong:
		s = spew.Sprintf("%s", v.Type)
	default:
		s = spew.Sprintf("%s '%+v'", v.Type, v.BadValue)
	}
	if len(v.Detail) != 0 {
		s += fmt.Sprintf(", Details: %s", v.Detail)
	}
	return s
}

// NewFieldRequired returns a *Error indicating "value required"
func NewFieldRequired(field string) *Error {
	return &Error{ErrorTypeRequired, field, "", ""}
}

// NewFieldInvalid returns a *Error indicating "invalid value"
func NewFieldInvalid(field string, value interface{}, detail string) *Error {
	return &Error{ErrorTypeInvalid, field, value, detail}
}

// NewFieldValueNotSupported returns a *Error indicating "unsupported value"
func NewFieldValueNotSupported(field string, value interface{}, validValues []string) *Error {
	detail := ""
	if validValues != nil && len(validValues) > 0 {
		detail = "supported values: " + strings.Join(validValues, ", ")
	}
	return &Error{ErrorTypeNotSupported, field, value, detail}
}

// NewFieldForbidden returns a *Error indicating "forbidden"
func NewFieldForbidden(field string, value interface{}) *Error {
	return &Error{ErrorTypeForbidden, field, value, ""}
}

// NewFieldDuplicate returns a *Error indicating "duplicate value"
func NewFieldDuplicate(field string, value interface{}) *Error {
	return &Error{ErrorTypeDuplicate, field, value, ""}
}

// NewFieldNotFound returns a *Error indicating "value not found"
func NewFieldNotFound(field string, value interface{}) *Error {
	return &Error{ErrorTypeNotFound, field, value, ""}
}

func NewFieldTooLong(field string, value interface{}, maxLength int) *Error {
	return &Error{ErrorTypeTooLong, field, value, fmt.Sprintf("must have at most %d characters", maxLength)}
}

type ErrorList []error

// Prefix adds a prefix to the Field of every Error in the list.
// Returns the list for convenience.
func (list ErrorList) Prefix(prefix string) ErrorList {
	for i := range list {
		if err, ok := list[i].(*Error); ok {
			if strings.HasPrefix(err.Field, "[") {
				err.Field = prefix + err.Field
			} else if len(err.Field) != 0 {
				err.Field = prefix + "." + err.Field
			} else {
				err.Field = prefix
			}
			list[i] = err
		} else {
			panic(fmt.Sprintf("Programmer error: ErrorList holds non-Error: %#v", list[i]))
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

// Filter removes items from the ErrorList that match the provided fns.
func (list ErrorList) Filter(fns ...utilerrors.Matcher) ErrorList {
	err := utilerrors.FilterOut(utilerrors.NewAggregate(list), fns...)
	if err == nil {
		return nil
	}
	// FilterOut that takes an Aggregate returns an Aggregate
	agg := err.(utilerrors.Aggregate)
	return ErrorList(agg.Errors())
}
