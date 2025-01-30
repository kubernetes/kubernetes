/*
Copyright 2021 The Kubernetes Authors.

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

package cel

import (
	"fmt"

	"github.com/google/cel-go/cel"
)

// ErrInternal the basic error that occurs when the expression fails to evaluate
// due to internal reasons. Any Error that has the Type of
// ErrorInternal is considered equal to ErrInternal
var ErrInternal = fmt.Errorf("internal")

// ErrInvalid is the basic error that occurs when the expression fails to
// evaluate but not due to internal reasons. Any Error that has the Type of
// ErrorInvalid is considered equal to ErrInvalid.
var ErrInvalid = fmt.Errorf("invalid")

// ErrRequired is the basic error that occurs when the expression is required
// but absent.
// Any Error that has the Type of ErrorRequired is considered equal
// to ErrRequired.
var ErrRequired = fmt.Errorf("required")

// ErrCompilation is the basic error that occurs when the expression fails to
// compile. Any CompilationError wraps ErrCompilation.
// ErrCompilation wraps ErrInvalid
var ErrCompilation = fmt.Errorf("%w: compilation error", ErrInvalid)

// ErrOutOfBudget is the basic error that occurs when the expression fails due to
// exceeding budget.
var ErrOutOfBudget = fmt.Errorf("out of budget")

// Error is an implementation of the 'error' interface, which represents a
// XValidation error.
type Error struct {
	Type   ErrorType
	Detail string

	// Cause is an optional wrapped errors that can be useful to
	// programmatically retrieve detailed errors.
	Cause error
}

var _ error = &Error{}

// Error implements the error interface.
func (v *Error) Error() string {
	return v.Detail
}

func (v *Error) Is(err error) bool {
	switch v.Type {
	case ErrorTypeRequired:
		return err == ErrRequired
	case ErrorTypeInvalid:
		return err == ErrInvalid
	case ErrorTypeInternal:
		return err == ErrInternal
	}
	return false
}

// Unwrap returns the wrapped Cause.
func (v *Error) Unwrap() error {
	return v.Cause
}

// ErrorType is a machine-readable value providing more detail about why
// a XValidation is invalid.
type ErrorType string

const (
	// ErrorTypeRequired is used to report withNullable values that are not
	// provided (e.g. empty strings, null values, or empty arrays).  See
	// Required().
	ErrorTypeRequired ErrorType = "RuleRequired"
	// ErrorTypeInvalid is used to report malformed values
	ErrorTypeInvalid ErrorType = "RuleInvalid"
	// ErrorTypeInternal is used to report other errors that are not related
	// to user input.  See InternalError().
	ErrorTypeInternal ErrorType = "InternalError"
)

// CompilationError indicates an error during expression compilation.
// It wraps ErrCompilation.
type CompilationError struct {
	err    *Error
	Issues *cel.Issues
}

// NewCompilationError wraps a cel.Issues to indicate a compilation failure.
func NewCompilationError(issues *cel.Issues) *CompilationError {
	return &CompilationError{
		Issues: issues,
		err: &Error{
			Type:   ErrorTypeInvalid,
			Detail: fmt.Sprintf("compilation error: %s", issues),
		}}
}

func (e *CompilationError) Error() string {
	return e.err.Error()
}

func (e *CompilationError) Unwrap() []error {
	return []error{e.err, ErrCompilation}
}
