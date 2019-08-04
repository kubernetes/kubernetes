/*
Copyright 2019 The Kubernetes Authors.

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
	"encoding/json"
)

// Fields is declared in types.go

// ProtoFields is a struct that is equivalent to Fields, but intended for
// protobuf marshalling/unmarshalling. It is generated into a serialization
// that matches Fields. Do not use in Go structs.
type ProtoFields struct {
	// Map is the representation used in the alpha version of this API
	Map map[string]Fields `json:"-" protobuf:"bytes,1,rep,name=map"`

	// Raw is the underlying serialization of this object.
	Raw []byte `json:"-" protobuf:"bytes,2,opt,name=raw"`
}

// ProtoFields returns the Fields as a new ProtoFields value.
func (m *Fields) ProtoFields() *ProtoFields {
	if m == nil {
		return &ProtoFields{}
	}
	return &ProtoFields{
		Raw: m.Raw,
	}
}

// Size implements the protobuf marshalling interface.
func (m *Fields) Size() (n int) {
	return m.ProtoFields().Size()
}

// Unmarshal implements the protobuf marshalling interface.
func (m *Fields) Unmarshal(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	p := ProtoFields{}
	if err := p.Unmarshal(data); err != nil {
		return err
	}
	if len(p.Map) == 0 {
		return json.Unmarshal(p.Raw, &m)
	}
	b, err := json.Marshal(&p.Map)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, &m)
}

// Marshal implements the protobuf marshaling interface.
func (m *Fields) Marshal() (data []byte, err error) {
	return m.ProtoFields().Marshal()
}

// MarshalTo implements the protobuf marshaling interface.
func (m *Fields) MarshalTo(data []byte) (int, error) {
	return m.ProtoFields().MarshalTo(data)
}

// MarshalToSizedBuffer implements the protobuf reverse marshaling interface.
func (m *Fields) MarshalToSizedBuffer(data []byte) (int, error) {
	return m.ProtoFields().MarshalToSizedBuffer(data)
}

// String implements the protobuf goproto_stringer interface.
func (m *Fields) String() string {
	return m.ProtoFields().String()
}
