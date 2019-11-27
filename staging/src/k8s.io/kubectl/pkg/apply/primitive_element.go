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

import "reflect"

// PrimitiveElement contains the recorded, local and remote values for a field
// of type primitive
type PrimitiveElement struct {
	// FieldMetaImpl contains metadata about the field from openapi
	FieldMetaImpl

	// RawElementData contains the values the field was set to
	RawElementData
}

// Merge implements Element.Merge
func (e PrimitiveElement) Merge(v Strategy) (Result, error) {
	return v.MergePrimitive(e)
}

var _ Element = &PrimitiveElement{}

// HasConflict returns ConflictError if primitive element has conflict field.
// Conflicts happen when either of the following conditions:
// 1. A field is specified in both recorded and remote values, but does not match.
// 2. A field is specified in recorded values, but missing in remote values.
func (e PrimitiveElement) HasConflict() error {
	if e.HasRecorded() && e.HasRemote() {
		if !reflect.DeepEqual(e.GetRecorded(), e.GetRemote()) {
			return NewConflictError(e)
		}
	}
	if e.HasRecorded() && !e.HasRemote() {
		return NewConflictError(e)
	}
	return nil
}

var _ ConflictDetector = &PrimitiveElement{}
