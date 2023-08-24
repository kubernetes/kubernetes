// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types/ref"
)

// Error interface which allows types types.Err values to be treated as error values.
type Error interface {
	error
	ref.Val
}

// Err type which extends the built-in go error and implements ref.Val.
type Err struct {
	error
}

var (
	// ErrType singleton.
	ErrType = NewOpaqueType("error")

	// errDivideByZero is an error indicating a division by zero of an integer value.
	errDivideByZero = errors.New("division by zero")
	// errModulusByZero is an error indicating a modulus by zero of an integer value.
	errModulusByZero = errors.New("modulus by zero")
	// errIntOverflow is an error representing integer overflow.
	errIntOverflow = errors.New("integer overflow")
	// errUintOverflow is an error representing unsigned integer overflow.
	errUintOverflow = errors.New("unsigned integer overflow")
	// errDurationOverflow is an error representing duration overflow.
	errDurationOverflow = errors.New("duration overflow")
	// errTimestampOverflow is an error representing timestamp overflow.
	errTimestampOverflow    = errors.New("timestamp overflow")
	celErrTimestampOverflow = &Err{error: errTimestampOverflow}

	// celErrNoSuchOverload indicates that the call arguments did not match a supported method signature.
	celErrNoSuchOverload = NewErr("no such overload")
)

// NewErr creates a new Err described by the format string and args.
// TODO: Audit the use of this function and standardize the error messages and codes.
func NewErr(format string, args ...any) ref.Val {
	return &Err{fmt.Errorf(format, args...)}
}

// NoSuchOverloadErr returns a new types.Err instance with a no such overload message.
func NoSuchOverloadErr() ref.Val {
	return celErrNoSuchOverload
}

// UnsupportedRefValConversionErr returns a types.NewErr instance with a no such conversion
// message that indicates that the native value could not be converted to a CEL ref.Val.
func UnsupportedRefValConversionErr(val any) ref.Val {
	return NewErr("unsupported conversion to ref.Val: (%T)%v", val, val)
}

// MaybeNoSuchOverloadErr returns the error or unknown if the input ref.Val is one of these types,
// else a new no such overload error.
func MaybeNoSuchOverloadErr(val ref.Val) ref.Val {
	return ValOrErr(val, "no such overload")
}

// ValOrErr either returns the existing error or creates a new one.
// TODO: Audit the use of this function and standardize the error messages and codes.
func ValOrErr(val ref.Val, format string, args ...any) ref.Val {
	if val == nil || !IsUnknownOrError(val) {
		return NewErr(format, args...)
	}
	return val
}

// WrapErr wraps an existing Go error value into a CEL Err value.
func WrapErr(err error) ref.Val {
	return &Err{error: err}
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (e *Err) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, e.error
}

// ConvertToType implements ref.Val.ConvertToType.
func (e *Err) ConvertToType(typeVal ref.Type) ref.Val {
	// Errors are not convertible to other representations.
	return e
}

// Equal implements ref.Val.Equal.
func (e *Err) Equal(other ref.Val) ref.Val {
	// An error cannot be equal to any other value, so it returns itself.
	return e
}

// String implements fmt.Stringer.
func (e *Err) String() string {
	return e.error.Error()
}

// Type implements ref.Val.Type.
func (e *Err) Type() ref.Type {
	return ErrType
}

// Value implements ref.Val.Value.
func (e *Err) Value() any {
	return e.error
}

// Is implements errors.Is.
func (e *Err) Is(target error) bool {
	return e.error.Error() == target.Error()
}

// Unwrap implements errors.Unwrap.
func (e *Err) Unwrap() error {
	return e.error
}

// IsError returns whether the input element ref.Type or ref.Val is equal to
// the ErrType singleton.
func IsError(val ref.Val) bool {
	switch val.(type) {
	case *Err:
		return true
	default:
		return false
	}
}
