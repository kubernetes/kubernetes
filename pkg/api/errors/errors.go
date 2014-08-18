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
)

// ValidationErrorEnum is a type of validation error.
type ValidationErrorEnum string

// These are known errors of validation.
const (
	Invalid      ValidationErrorEnum = "invalid value"
	NotSupported ValidationErrorEnum = "unsupported value"
	Duplicate    ValidationErrorEnum = "duplicate value"
	NotFound     ValidationErrorEnum = "not found"
)

// ValidationError is an implementation of the 'error' interface, which represents an error of validation.
type ValidationError struct {
	Type     ValidationErrorEnum
	Field    string
	BadValue interface{}
}

func (v ValidationError) Error() string {
	return fmt.Sprintf("%s: %v '%v'", v.Field, v.Type, v.BadValue)
}

// NewInvalid returns a ValidationError indicating "invalid value".  Use this to
// report malformed values (e.g. failed regex match) or missing "required" fields.
func NewInvalid(field string, value interface{}) ValidationError {
	return ValidationError{Invalid, field, value}
}

// NewNotSupported returns a ValidationError indicating "unsuported value".  Use
// this to report valid (as per formatting rules) values that can not be handled
// (e.g. an enumerated string).
func NewNotSupported(field string, value interface{}) ValidationError {
	return ValidationError{NotSupported, field, value}
}

// NewDuplicate returns a ValidationError indicating "duplicate value".  Use this
// to report collisions of values that must be unique (e.g. unique IDs).
func NewDuplicate(field string, value interface{}) ValidationError {
	return ValidationError{Duplicate, field, value}
}

// NewNotFound returns a ValidationError indicating "value not found".  Use this
// to report failure to find a requested value (e.g. looking up an ID).
func NewNotFound(field string, value interface{}) ValidationError {
	return ValidationError{NotFound, field, value}
}

// ErrorList is a collection of errors.  This does not implement the error
// interface to avoid confusion where an empty ErrorList would still be an
// error (non-nil).  To produce a single error instance from an ErrorList, use
// the ToError() method, which will return nil for an empty ErrorList.
type ErrorList []error

// This helper implements the error interface for ErrorList, but must prevents
// accidental conversion of ErrorList to error.
type errorListInternal ErrorList

// Error is part of the error interface.
func (list errorListInternal) Error() string {
	if len(list) == 0 {
		return ""
	}
	sl := make([]string, len(list))
	for i := range list {
		sl[i] = list[i].Error()
	}
	return strings.Join(sl, "; ")
}

// ToError converts an ErrorList into a "normal" error, or nil if the list is empty.
func (list ErrorList) ToError() error {
	if len(list) == 0 {
		return nil
	}
	return errorListInternal(list)
}
