/*
Copyright 2016 The Kubernetes Authors.

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

package volume

import (
	"fmt"
)

const (
	// ErrCodeNotSupported code for NotSupported Errors.
	ErrCodeNotSupported int = iota + 1

	// ErrCodeNoPathDefined code for NoPathDefined Errors.
	ErrCodeNoPathDefined

	// ErrCodeFsInfoFailed code for FsInfoFailed Errors.
	ErrCodeFsInfoFailed
)

// NewNotSupportedError creates a new MetricsError with code NotSupported.
func NewNotSupportedError() *MetricsError {
	return &MetricsError{
		Code: ErrCodeNotSupported,
		Msg:  "metrics are not supported for MetricsNil Volumes",
	}
}

// NewNotImplementedError creates a new MetricsError with code NotSupported.
func NewNotImplementedError(reason string) *MetricsError {
	return &MetricsError{
		Code: ErrCodeNotSupported,
		Msg:  fmt.Sprintf("metrics support is not implemented: %s", reason),
	}
}

// NewNotSupportedErrorWithDriverName creates a new MetricsError with code NotSupported.
// driver name is added to the error message.
func NewNotSupportedErrorWithDriverName(name string) *MetricsError {
	return &MetricsError{
		Code: ErrCodeNotSupported,
		Msg:  fmt.Sprintf("metrics are not supported for %s volumes", name),
	}
}

// NewNoPathDefinedError creates a new MetricsError with code NoPathDefined.
func NewNoPathDefinedError() *MetricsError {
	return &MetricsError{
		Code: ErrCodeNoPathDefined,
		Msg:  "no path defined for disk usage metrics.",
	}
}

// NewFsInfoFailedError creates a new MetricsError with code FsInfoFailed.
func NewFsInfoFailedError(err error) *MetricsError {
	return &MetricsError{
		Code: ErrCodeFsInfoFailed,
		Msg:  fmt.Sprintf("failed to get FsInfo due to error %v", err),
	}
}

// MetricsError to distinguish different Metrics Errors.
type MetricsError struct {
	Code int
	Msg  string
}

func (e *MetricsError) Error() string {
	return e.Msg
}

// IsNotSupported returns true if and only if err is "key" not found error.
func IsNotSupported(err error) bool {
	return isErrCode(err, ErrCodeNotSupported)
}

func isErrCode(err error, code int) bool {
	if err == nil {
		return false
	}
	if e, ok := err.(*MetricsError); ok {
		return e.Code == code
	}
	return false
}
