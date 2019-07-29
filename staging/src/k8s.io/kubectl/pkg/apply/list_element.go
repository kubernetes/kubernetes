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

// ListElement contains the recorded, local and remote values for a field
// of type list
type ListElement struct {
	// FieldMetaImpl contains metadata about the field from openapi
	FieldMetaImpl

	ListElementData

	// Values contains the combined recorded-local-remote value of each item in the list
	// Present for lists that can be merged only.  Contains the items
	// from each of the 3 lists merged into single Elements using
	// the merge-key.
	Values []Element
}

// Merge implements Element.Merge
func (e ListElement) Merge(v Strategy) (Result, error) {
	return v.MergeList(e)
}

var _ Element = &ListElement{}

// ListElementData contains the recorded, local and remote data for a list
type ListElementData struct {
	RawElementData
}

// GetRecordedList returns the Recorded value as a list
func (e ListElementData) GetRecordedList() []interface{} {
	return sliceCast(e.recorded)
}

// GetLocalList returns the Local value as a list
func (e ListElementData) GetLocalList() []interface{} {
	return sliceCast(e.local)
}

// GetRemoteList returns the Remote value as a list
func (e ListElementData) GetRemoteList() []interface{} {
	return sliceCast(e.remote)
}

// sliceCast casts i to a slice if it is non-nil, otherwise returns nil
func sliceCast(i interface{}) []interface{} {
	if i == nil {
		return nil
	}
	return i.([]interface{})
}

// HasConflict returns ConflictError if fields in recorded and remote of ListElement conflict
func (e ListElement) HasConflict() error {
	for _, item := range e.Values {
		if item, ok := item.(ConflictDetector); ok {
			if err := item.HasConflict(); err != nil {
				return err
			}
		}
	}
	return nil
}

var _ ConflictDetector = &ListElement{}
