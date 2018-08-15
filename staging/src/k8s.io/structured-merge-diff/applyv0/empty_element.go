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

// EmptyElement is a placeholder for when no value is set for a field so its type is unknown
type EmptyElement struct {
	// FieldMetaImpl contains metadata about the field from openapi
	FieldMetaImpl
}

// Merge implements Element.Merge
func (e EmptyElement) Merge(v Strategy) (Result, error) {
	return v.MergeEmpty(e)
}

// IsAdd implements Element.IsAdd
func (e EmptyElement) IsAdd() bool {
	return false
}

// IsDelete implements Element.IsDelete
func (e EmptyElement) IsDelete() bool {
	return false
}

// GetRecorded implements Element.GetRecorded
func (e EmptyElement) GetRecorded() interface{} {
	return nil
}

// GetLocal implements Element.GetLocal
func (e EmptyElement) GetLocal() interface{} {
	return nil
}

// GetRemote implements Element.GetRemote
func (e EmptyElement) GetRemote() interface{} {
	return nil
}

// HasRecorded implements Element.HasRecorded
func (e EmptyElement) HasRecorded() bool {
	return false
}

// HasLocal implements Element.HasLocal
func (e EmptyElement) HasLocal() bool {
	return false
}

// HasRemote implements Element.IsAdd
func (e EmptyElement) HasRemote() bool {
	return false
}

var _ Element = &EmptyElement{}
