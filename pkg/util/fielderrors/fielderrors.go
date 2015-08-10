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

package fielderrors

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/util/errors"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"
)

// ValidationErrorType is a machine readable value providing more detail about why
// a field is invalid.  These values are expected to match 1-1 with
// CauseType in api/types.go.
type ValidationErrorType string

// TODO: These values are duplicated in api/types.go, but there's a circular dep.  Fix it.
const (
	// ValidationErrorTypeNotFound is used to report failure to find a requested value
	// (e.g. looking up an ID).
	ValidationErrorTypeNotFound ValidationErrorType = "FieldValueNotFound"
	// ValidationErrorTypeRequired is used to report required values that are not
	// provided (e.g. empty strings, null values, or empty arrays).
	ValidationErrorTypeRequired ValidationErrorType = "FieldValueRequired"
	// ValidationErrorTypeDuplicate is used to report collisions of values that must be
	// unique (e.g. unique IDs).
	ValidationErrorTypeDuplicate ValidationErrorType = "FieldValueDuplicate"
	// ValidationErrorTypeInvalid is used to report malformed values (e.g. failed regex
	// match).
	ValidationErrorTypeInvalid ValidationErrorType = "FieldValueInvalid"
	// ValidationErrorTypeNotSupported is used to report valid (as per formatting rules)
	// values that can not be handled (e.g. an enumerated string).
	ValidationErrorTypeNotSupported ValidationErrorType = "FieldValueNotSupported"
	// ValidationErrorTypeForbidden is used to report valid (as per formatting rules)
	// values which would be accepted by some api instances, but which would invoke behavior
	// not permitted by this api instance (such as due to stricter security policy).
	ValidationErrorTypeForbidden ValidationErrorType = "FieldValueForbidden"
	// ValidationErrorTypeTooLong is used to report that given value is too long.
	ValidationErrorTypeTooLong ValidationErrorType = "FieldValueTooLong"
)

// String converts a ValidationErrorType into its corresponding error message.
func (t ValidationErrorType) String() string {
	switch t {
	case ValidationErrorTypeNotFound:
		return "not found"
	case ValidationErrorTypeRequired:
		return "required value"
	case ValidationErrorTypeDuplicate:
		return "duplicate value"
	case ValidationErrorTypeInvalid:
		return "invalid value"
	case ValidationErrorTypeNotSupported:
		return "unsupported value"
	case ValidationErrorTypeForbidden:
		return "forbidden"
	case ValidationErrorTypeTooLong:
		return "too long"
	default:
		glog.Errorf("unrecognized validation type: %#v", t)
		return ""
	}
}

// ValidationError is an implementation of the 'error' interface, which represents an error of validation.
type ValidationError struct {
	Type     ValidationErrorType
	Field    string
	BadValue interface{}
	Detail   string
}

var _ error = &ValidationError{}

func (v *ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", v.Field, v.ErrorBody())
}

func (v *ValidationError) ErrorBody() string {
	var s string
	switch v.Type {
	case ValidationErrorTypeRequired, ValidationErrorTypeTooLong:
		s = spew.Sprintf("%s", v.Type)
	default:
		s = spew.Sprintf("%s '%+v'", v.Type, v.BadValue)
	}
	if len(v.Detail) != 0 {
		s += fmt.Sprintf(", Details: %s", v.Detail)
	}
	return s
}

// NewFieldRequired returns a *ValidationError indicating "value required"
func NewFieldRequired(field string) *ValidationError {
	return &ValidationError{ValidationErrorTypeRequired, field, "", ""}
}

// NewFieldInvalid returns a *ValidationError indicating "invalid value"
func NewFieldInvalid(field string, value interface{}, detail string) *ValidationError {
	return &ValidationError{ValidationErrorTypeInvalid, field, value, detail}
}

// NewFieldValueNotSupported returns a *ValidationError indicating "unsupported value"
func NewFieldValueNotSupported(field string, value interface{}, validValues []string) *ValidationError {
	detail := ""
	if validValues != nil && len(validValues) > 0 {
		detail = "supported values: " + strings.Join(validValues, ", ")
	}
	return &ValidationError{ValidationErrorTypeNotSupported, field, value, detail}
}

// NewFieldForbidden returns a *ValidationError indicating "forbidden"
func NewFieldForbidden(field string, value interface{}) *ValidationError {
	return &ValidationError{ValidationErrorTypeForbidden, field, value, ""}
}

// NewFieldDuplicate returns a *ValidationError indicating "duplicate value"
func NewFieldDuplicate(field string, value interface{}) *ValidationError {
	return &ValidationError{ValidationErrorTypeDuplicate, field, value, ""}
}

// NewFieldNotFound returns a *ValidationError indicating "value not found"
func NewFieldNotFound(field string, value interface{}) *ValidationError {
	return &ValidationError{ValidationErrorTypeNotFound, field, value, ""}
}

func NewFieldTooLong(field string, value interface{}, maxLength int) *ValidationError {
	return &ValidationError{ValidationErrorTypeTooLong, field, value, fmt.Sprintf("must have at most %d characters", maxLength)}
}

type ValidationErrorList []error

// Prefix adds a prefix to the Field of every ValidationError in the list.
// Also adds prefixes to multiple fields if you send an or separator.
// Returns the list for convenience.
func (list ValidationErrorList) Prefix(prefix string) ValidationErrorList {
	for i := range list {
		if err, ok := list[i].(*ValidationError); ok {
			if strings.HasPrefix(err.Field, "[") {
				err.Field = prefix + err.Field
			} else if len(err.Field) != 0 {
				fields := strings.SplitAfter(err.Field, " or ")
				err.Field = ""
				for j := range fields {
					err.Field += prefix + "." + fields[j]
				}
			} else {
				err.Field = prefix
			}
			list[i] = err
		} else {
			glog.Warningf("Programmer error: ValidationErrorList holds non-ValidationError: %#v", list[i])
		}
	}
	return list
}

// PrefixIndex adds an index to the Field of every ValidationError in the list.
// Returns the list for convenience.
func (list ValidationErrorList) PrefixIndex(index int) ValidationErrorList {
	return list.Prefix(fmt.Sprintf("[%d]", index))
}

// NewValidationErrorFieldPrefixMatcher returns an errors.Matcher that returns true
// if the provided error is a ValidationError and has the provided ValidationErrorType.
func NewValidationErrorTypeMatcher(t ValidationErrorType) errors.Matcher {
	return func(err error) bool {
		if e, ok := err.(*ValidationError); ok {
			return e.Type == t
		}
		return false
	}
}

// NewValidationErrorFieldPrefixMatcher returns an errors.Matcher that returns true
// if the provided error is a ValidationError and has a field with the provided
// prefix.
func NewValidationErrorFieldPrefixMatcher(prefix string) errors.Matcher {
	return func(err error) bool {
		if e, ok := err.(*ValidationError); ok {
			return strings.HasPrefix(e.Field, prefix)
		}
		return false
	}
}

// Filter removes items from the ValidationErrorList that match the provided fns.
func (list ValidationErrorList) Filter(fns ...errors.Matcher) ValidationErrorList {
	err := errors.FilterOut(errors.NewAggregate(list), fns...)
	if err == nil {
		return nil
	}
	// FilterOut that takes an Aggregate returns an Aggregate
	agg := err.(errors.Aggregate)
	return ValidationErrorList(agg.Errors())
}
