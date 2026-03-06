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
	"bytes"
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
	// Raw is the underlying serialization of this object.
	//
	// Deprecated: Direct access to this field is deprecated. Use GetRawBytes, GetRawString, SetRawBytes, SetRawString, GetRawReader, NewFieldsV1 instead.
	Raw []byte `json:"-" protobuf:"bytes,1,opt,name=Raw"`
}

func (f FieldsV1) String() string {
	return string(f.Raw)
}

func (f FieldsV1) Equal(f2 FieldsV1) bool {
	return bytes.Equal(f.Raw, f2.Raw)
}

func (f *FieldsV1) GetRawReader() FieldsV1Reader {
	if f == nil || len(f.Raw) == 0 {
		return bytes.NewReader(nil)
	}
	return bytes.NewReader(f.Raw)
}

// GetRawBytes returns the raw bytes.
// These may or may not be a copy of the underlying bytes.
// If mutating the underlying bytes is desired, the returned bytes may be mutated and then passed to SetRawBytes().
// If mutating the underlying bytes is not desired, make a copy of the returned bytes.
func (f *FieldsV1) GetRawBytes() []byte {
	if f == nil {
		return nil
	}
	return f.Raw
}

// GetRawString returns the raw data as a string.
func (f *FieldsV1) GetRawString() string {
	if f == nil {
		return ""
	}
	return string(f.Raw)
}

// SetRawBytes sets the raw bytes. It does not retain the passed-in byte slice.
func (f *FieldsV1) SetRawBytes(b []byte) {
	if f != nil {
		f.Raw = bytes.Clone(b)
	}
}

// SetRawString sets the raw data from a string.
func (f *FieldsV1) SetRawString(s string) {
	if f != nil {
		f.Raw = []byte(s)
	}
}

func NewFieldsV1(raw string) *FieldsV1 {
	return &FieldsV1{Raw: []byte(raw)}
}

func (f *FieldsV1) DeepCopyInto(out *FieldsV1) {
	*out = *f
	if f.Raw != nil {
		in, out := &f.Raw, &out.Raw
		*out = make([]byte, len(*in))
		copy(*out, *in)
	}
}
