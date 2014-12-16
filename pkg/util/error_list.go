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

package util

import (
	"fmt"
)

// []error is a collection of errors. This does not implement the error
// interface to avoid confusion where an empty []error would still be an
// error (non-nil).  To produce a single error instance from an []error, use
// the SliceToError() method, which will return nil for an empty []error.

// This helper implements the error interface for []error, but prevents
// accidental conversion of []error to error.
type errorList []error

// Error is part of the error interface.
func (list errorList) Error() string {
	if len(list) == 0 {
		return ""
	}
	if len(list) == 1 {
		return list[0].Error()
	}
	result := fmt.Sprintf("[%s", list[0].Error())
	for i := 1; i < len(list); i++ {
		result += fmt.Sprintf(", %s", list[i].Error())
	}
	result += "]"
	return result
}

// SliceToError converts an []error into a "normal" error, or nil if the slice is empty.
func SliceToError(errs []error) error {
	if len(errs) == 0 {
		return nil
	}
	return errorList(errs)
}
