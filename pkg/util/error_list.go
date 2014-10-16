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

// ErrorList is a collection of errors.  This does not implement the error
// interface to avoid confusion where an empty ErrorList would still be an
// error (non-nil).  To produce a single error instance from an ErrorList, use
// the ToError() method, which will return nil for an empty ErrorList.
type ErrorList []error

// This helper implements the error interface for ErrorList, but prevents
// accidental conversion of ErrorList to error.
type errorListInternal ErrorList

// Error is part of the error interface.
func (list errorListInternal) Error() string {
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

// ToError converts an ErrorList into a "normal" error, or nil if the list is empty.
func (list ErrorList) ToError() error {
	if len(list) == 0 {
		return nil
	}
	return errorListInternal(list)
}
