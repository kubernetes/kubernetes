//go:build fieldsv1string

/*
Copyright The Kubernetes Authors.

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

package v1

import (
	"strings"
	"unique"
)

// FieldsV1 stores a set of fields in a data structure like a Trie, in JSON format.
//
// Each key is either a '.' representing the field itself, and will always map to an empty set,
// or a string representing a sub-field or item. The string will follow one of these four formats:
// 'f:<name>', where <name> is the name of a field in a struct, or key in a map
// 'v:<value>', where <value> is the exact json formatted value of a list item
// 'i:<index>', where <index> is position of a item in a list
// 'k:<keys>', where <keys> is a map of  a list item's key fields to their unique values
// If a key maps to an empty Fields value, the field that key represents is part of the set.
//
// The exact format is defined in sigs.k8s.io/structured-merge-diff
// +k8s:deepcopy-gen=false
// +protobuf.options.marshal=false
// +protobuf.options.(gogoproto.goproto_stringer)=false
type FieldsV1 struct {
	// The zero value of a unique.Handle[string] has an uninitialized underlying pointer.
	// Calling .Value() on it panics. We must explicitly check for this uninitialized
	// state (f.handle == unique.Handle[string]{}) across accessors to safely support
	// uninitialized metav1.FieldsV1{} objects.
	// See ongoing golang discussion related to this here: https://github.com/golang/go/issues/73344
	handle unique.Handle[string]
}

func (f FieldsV1) String() string {
	if f.handle == (unique.Handle[string]{}) {
		return ""
	}
	return f.handle.Value()
}

func (f FieldsV1) Equal(f2 FieldsV1) bool {
	if f.handle == f2.handle {
		return true
	}
	// An uninitialized FieldsV1 compared to an explicitly empty
	// FieldsV1 (unique.Make("") will fail the handle check above.
	// Evaluate string contents directly as well to maintain parity with legacy
	// bytes.Equal(nil, []byte{}) == true behavior.
	return f.GetRawString() == f2.GetRawString()
}

func (f *FieldsV1) GetRawReader() FieldsV1Reader {
	if f == nil || f.handle == (unique.Handle[string]{}) {
		return strings.NewReader("")
	}
	return strings.NewReader(f.handle.Value())
}

// GetRawBytes returns the raw bytes.
// These may or may not be a copy of the underlying bytes.
// If mutating the underlying bytes is desired, the returned bytes may be mutated and then passed to SetRawBytes().
// If mutating the underlying bytes is not desired, make a copy of the returned bytes.
func (f *FieldsV1) GetRawBytes() []byte {
	if f == nil || f.handle == (unique.Handle[string]{}) {
		return nil
	}
	return []byte(f.handle.Value())
}

// GetRawString returns the raw data as a string.
func (f *FieldsV1) GetRawString() string {
	if f == nil || f.handle == (unique.Handle[string]{}) {
		return ""
	}
	return f.handle.Value()
}

// SetRawBytes sets the raw bytes. It does not retain the passed-in byte slice.
func (f *FieldsV1) SetRawBytes(b []byte) {
	if f != nil {
		f.handle = unique.Make(string(b))
	}
}

// SetRawString sets the raw data from a string.
func (f *FieldsV1) SetRawString(s string) {
	if f != nil {
		f.handle = unique.Make(s)
	}
}

func NewFieldsV1(raw string) *FieldsV1 {
	return &FieldsV1{handle: unique.Make(raw)}
}

func (f *FieldsV1) DeepCopyInto(out *FieldsV1) {
	*out = *f
}
