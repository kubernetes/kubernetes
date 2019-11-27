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

// TypeElement contains the recorded, local and remote values for a field
// that is a complex type
type TypeElement struct {
	// FieldMetaImpl contains metadata about the field from openapi
	FieldMetaImpl

	MapElementData

	// Values contains the combined recorded-local-remote value of each field in the type
	// Values contains the values in mapElement.  Element must contain
	// a Name matching its key in Values
	Values map[string]Element
}

// Merge implements Element.Merge
func (e TypeElement) Merge(v Strategy) (Result, error) {
	return v.MergeType(e)
}

// GetValues implements Element.GetValues
func (e TypeElement) GetValues() map[string]Element {
	return e.Values
}

// HasConflict returns ConflictError if some elements in type conflict.
func (e TypeElement) HasConflict() error {
	for _, item := range e.GetValues() {
		if item, ok := item.(ConflictDetector); ok {
			if err := item.HasConflict(); err != nil {
				return err
			}
		}
	}
	return nil
}

var _ Element = &TypeElement{}
var _ ConflictDetector = &TypeElement{}
