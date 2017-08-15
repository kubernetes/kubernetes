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

	// Name contains of the field
	Name string

	// RecordedSet is true if the field was found in the recorded object
	RecordedSet bool

	// LocalSet is true if the field was found in the local object
	LocalSet bool

	// RemoteSet is true if the field was found in the remote object
	RemoteSet bool

	// Recorded contains the value of the field from the recorded object
	Recorded interface{}

	// Local contains the value of the field from the recorded object
	Local interface{}

	// Remote contains the value of the field from the recorded object
	Remote interface{}
}

// Accept implements Element.Accept
func (e PrimitiveElement) Accept(v Visitor) (Result, error) {
	return v.VisitPrimitive(e)
}

// String returns a string representation of the PrimitiveElement
func (e PrimitiveElement) String() string {
	return fmt.Sprintf("name: %s recorded: %v local: %v remote: %v", e.Name, e.Recorded, e.Local, e.Remote)
}

// GetRecorded implements Element.GetRecorded
func (e PrimitiveElement) GetRecorded() interface{} {
	return e.Recorded
}

// GetLocal implements Element.GetLocal
func (e PrimitiveElement) GetLocal() interface{} {
	return e.Local
}

// GetRemote implements Element.GetRemote
func (e PrimitiveElement) GetRemote() interface{} {
	return e.Remote
}

// HasRecorded implements Element.HasRecorded
func (e PrimitiveElement) HasRecorded() bool {
	return e.RecordedSet
}

// HasLocal implements Element.HasLocal
func (e PrimitiveElement) HasLocal() bool {
	return e.LocalSet
}

// HasRemote implements Element.HasRemote
func (e PrimitiveElement) HasRemote() bool {
	return e.RemoteSet
}

var _ Element = &PrimitiveElement{}
