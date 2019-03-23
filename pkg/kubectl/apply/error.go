/*
Copyright 2017 The Kubernetes Authors.

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

package apply

import "fmt"

// ConflictError represents a conflict error occurred during the merge operation.
type ConflictError struct {
	element Element
}

// NewConflictError returns a ConflictError with detailed conflict information in element
func NewConflictError(e PrimitiveElement) *ConflictError {
	return &ConflictError{
		element: e,
	}
}

// Error implements error
func (c *ConflictError) Error() string {
	return fmt.Sprintf("conflict detected, recorded value (%+v) and remote value (%+v)",
		c.element.GetRecorded(), c.element.GetRemote())
}
