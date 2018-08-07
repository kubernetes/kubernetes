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

// MapElement contains the recorded, local and remote values for a field
// of type map
type MapElement struct {
	// FieldMetaImpl contains metadata about the field from openapi
	FieldMetaImpl

	// MapElementData contains the value a field was set to
	MapElementData

	// Values contains the combined recorded-local-remote value of each item in the map
	// Values contains the values in mapElement.  Element must contain
	// a Name matching its key in Values
	Values map[string]Element
}

// Merge implements Element.Merge
func (e MapElement) Merge(v Strategy) (Result, error) {
	return v.MergeMap(e)
}

// GetValues implements Element.GetValues
func (e MapElement) GetValues() map[string]Element {
	return e.Values
}

var _ Element = &MapElement{}

// MapElementData contains the recorded, local and remote data for a map or type
type MapElementData struct {
	RawElementData
}

// GetRecordedMap returns the Recorded value as a map
func (e MapElementData) GetRecordedMap() map[string]interface{} {
	return mapCast(e.recorded)
}

// GetLocalMap returns the Local value as a map
func (e MapElementData) GetLocalMap() map[string]interface{} {
	return mapCast(e.local)
}

// GetRemoteMap returns the Remote value as a map
func (e MapElementData) GetRemoteMap() map[string]interface{} {
	return mapCast(e.remote)
}

// mapCast casts i to a map if it is non-nil, otherwise returns nil
func mapCast(i interface{}) map[string]interface{} {
	if i == nil {
		return nil
	}
	return i.(map[string]interface{})
}

// HasConflict returns ConflictError if some elements in map conflict.
func (e MapElement) HasConflict() error {
	for _, item := range e.GetValues() {
		if item, ok := item.(ConflictDetector); ok {
			if err := item.HasConflict(); err != nil {
				return err
			}
		}
	}
	return nil
}

var _ ConflictDetector = &MapElement{}
