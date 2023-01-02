/*
Copyright (c) 2020 VMware, Inc. All Rights Reserved.

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

package tags

import (
	"fmt"
)

const (
	errFormat = "[error: %d type: %s reason: %s]"
	separator = "," // concat multiple error strings
)

// BatchError is an error returned for a single item which failed in a batch
// operation
type BatchError struct {
	Type    string `json:"id"`
	Message string `json:"default_message"`
}

// BatchErrors contains all errors which occurred in a batch operation
type BatchErrors []BatchError

func (b BatchErrors) Error() string {
	if len(b) == 0 {
		return ""
	}

	var errString string
	for i := range b {
		errType := b[i].Type
		reason := b[i].Message
		errString += fmt.Sprintf(errFormat, i, errType, reason)

		// no separator after last item
		if i+1 < len(b) {
			errString += separator
		}
	}
	return errString
}
