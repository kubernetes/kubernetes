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

	// HasElementData contains whether the field was set
	HasElementData

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
	// recorded contains the value of the field from the recorded object
	Recorded map[string]interface{}

	// Local contains the value of the field from the recorded object
	Local map[string]interface{}

	// Remote contains the value of the field from the recorded object
	Remote map[string]interface{}
}

// GetRecorded implements Element.GetRecorded
func (e MapElementData) GetRecorded() interface{} {
	// https://golang.org/doc/faq#nil_error
	if e.Recorded == nil {
		return nil
	}
	return e.Recorded
}

// GetLocal implements Element.GetLocal
func (e MapElementData) GetLocal() interface{} {
	// https://golang.org/doc/faq#nil_error
	if e.Local == nil {
		return nil
	}
	return e.Local
}

// GetRemote implements Element.GetRemote
func (e MapElementData) GetRemote() interface{} {
	// https://golang.org/doc/faq#nil_error
	if e.Remote == nil {
		return nil
	}
	return e.Remote
}
