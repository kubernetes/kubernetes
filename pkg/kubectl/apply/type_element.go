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

	// Name contains of the field
	Name string

	// RecordedSet is true if the field was found in the recorded object
	RecordedSet bool

	// LocalSet is true if the field was found in the loca object
	LocalSet bool

	// RemoteSet is true if the field was found in the remote object
	RemoteSet bool

	// Recorded contains the value of the field from the recorded object
	Recorded map[string]interface{}

	// Local contains the value of the field from the recorded object
	Local map[string]interface{}

	// Remote contains the value of the field from the recorded object
	Remote map[string]interface{}

	// Values contains the combined recorded-local-remote value of each field in the type
	// Values contains the values in mapElement.  Element must contain
	// a Name matching its key in Values
	Values map[string]Element
}

// Accept implements Element.Accept
func (e TypeElement) Accept(v Visitor) (Result, error) {
	return v.VisitType(e)
}

// GetRecorded implements Element.GetRecorded
func (e TypeElement) GetRecorded() interface{} {
	// https://golang.org/doc/faq#nil_error
	if e.Recorded == nil {
		return nil
	}
	return e.Recorded
}

// HasRecorded implements Element.HasRecorded
func (e TypeElement) HasRecorded() bool {
	return e.RecordedSet
}

// GetLocal implements Element.GetLocal
func (e TypeElement) GetLocal() interface{} {
	// https://golang.org/doc/faq#nil_error
	if e.Local == nil {
		return nil
	}
	return e.Local
}

// GetRemote implements Element.GetRemote
func (e TypeElement) GetRemote() interface{} {
	// https://golang.org/doc/faq#nil_error
	if e.Remote == nil {
		return nil
	}
	return e.Remote
}

// GetValues implements Element.GetValues
func (e TypeElement) GetValues() map[string]Element {
	return e.Values
}

// HasLocal implements Element.HasLocal
func (e TypeElement) HasLocal() bool {
	return e.LocalSet
}

// HasRemote implements Element.HasRemote
func (e TypeElement) HasRemote() bool {
	return e.RemoteSet
}

var _ Element = &TypeElement{}
