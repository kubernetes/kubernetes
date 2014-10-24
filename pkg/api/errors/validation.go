/*
Copyright 2014 Google Inc. All rights reserved.

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

package errors

import (
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
)

func ValueOf(t ValidationErrorType) string {
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
}

func (v ValidationError) Error() string {
	return fmt.Sprintf("%s: %v '%v'", v.Field, ValueOf(v.Type), v.BadValue)
}

// NewFieldRequired returns a ValidationError indicating "value required"
func NewFieldRequired(field string, value interface{}) ValidationError {
	return ValidationError{ValidationErrorTypeRequired, field, value}
}

// NewFieldInvalid returns a ValidationError indicating "invalid value"
func NewFieldInvalid(field string, value interface{}) ValidationError {
	return ValidationError{ValidationErrorTypeInvalid, field, value}
}

// NewFieldNotSupported returns a ValidationError indicating "unsupported value"
func NewFieldNotSupported(field string, value interface{}) ValidationError {
	return ValidationError{ValidationErrorTypeNotSupported, field, value}
}

// NewFieldForbidden returns a ValidationError indicating "forbidden"
func NewFieldForbidden(field string, value interface{}) ValidationError {
	return ValidationError{ValidationErrorTypeForbidden, field, value}
}

// NewFieldDuplicate returns a ValidationError indicating "duplicate value"
func NewFieldDuplicate(field string, value interface{}) ValidationError {
	return ValidationError{ValidationErrorTypeDuplicate, field, value}
}

// NewFieldNotFound returns a ValidationError indicating "value not found"
func NewFieldNotFound(field string, value interface{}) ValidationError {
	return ValidationError{ValidationErrorTypeNotFound, field, value}
}

// ValidationErrorList is a collection of ValidationErrors.  This does not
// implement the error interface to avoid confusion where an empty
// ValidationErrorList would still be an error (non-nil).  To produce a single
// error instance from a ValidationErrorList, use the ToError() method, which
// will return nil for an empty ValidationErrorList.
type ValidationErrorList util.ErrorList

// ToError converts a ValidationErrorList into a "normal" error, or nil if the
// list is empty.
func (list ValidationErrorList) ToError() error {
	return util.ErrorList(list).ToError()
}

// Prefix adds a prefix to the Field of every ValidationError in the list.
// Returns the list for convenience.
func (list ValidationErrorList) Prefix(prefix string) ValidationErrorList {
	for i := range list {
		if err, ok := list[i].(ValidationError); ok {
			if strings.HasPrefix(err.Field, "[") {
				err.Field = prefix + err.Field
			} else if len(err.Field) != 0 {
				err.Field = prefix + "." + err.Field
			} else {
				err.Field = prefix
			}
			list[i] = err
		} else {
			glog.Warningf("ValidationErrorList holds non-ValidationError: %T", list)
		}
	}
	return list
}

// PrefixIndex adds an index to the Field of every ValidationError in the list.
// Returns the list for convenience.
func (list ValidationErrorList) PrefixIndex(index int) ValidationErrorList {
	return list.Prefix(fmt.Sprintf("[%d]", index))
}
