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

import (
	"fmt"
)

// PrimitiveElement contains the recorded, local and remote values for a field
// of type primitive
type PrimitiveElement struct {
	// FieldMetaImpl contains metadata about the field from openapi
	FieldMetaImpl

	// HasElementData contains whether the field was set
	HasElementData

	// RawElementData contains the values the field was set to
	RawElementData
}

// Accept implements Element.Accept
func (e PrimitiveElement) Accept(v Visitor) (Result, error) {
	return v.VisitPrimitive(e)
}

// String returns a string representation of the PrimitiveElement
func (e PrimitiveElement) String() string {
	return fmt.Sprintf("name: %s recorded: %v local: %v remote: %v", e.Name, e.Recorded, e.Local, e.Remote)
}

var _ Element = &PrimitiveElement{}

// RawElementData contains the recorded, local and remote data for a primitive
type RawElementData struct {
	// recorded contains the value of the field from the recorded object
	Recorded interface{}

	// Local contains the value of the field from the recorded object
	Local interface{}

	// Remote contains the value of the field from the recorded object
	Remote interface{}
}

// GetRecorded implements Element.GetRecorded
func (e RawElementData) GetRecorded() interface{} {
	return e.Recorded
}

// GetLocal implements Element.GetLocal
func (e RawElementData) GetLocal() interface{} {
	return e.Local
}

// GetRemote implements Element.GetRemote
func (e RawElementData) GetRemote() interface{} {
	return e.Remote
}
