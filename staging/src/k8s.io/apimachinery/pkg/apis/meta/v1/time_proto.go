/*
Copyright 2015 The Kubernetes Authors.

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
	"time"
)

// Timestamp is a struct that is equivalent to Time, but intended for
// protobuf marshalling/unmarshalling. It is generated into a serialization
// that matches Time. Do not use in Go structs.
type Timestamp struct {
	// Represents seconds of UTC time since Unix epoch
	// 1970-01-01T00:00:00Z. Must be from 0001-01-01T00:00:00Z to
	// 9999-12-31T23:59:59Z inclusive.
	Seconds int64 `json:"seconds" protobuf:"varint,1,opt,name=seconds"`
	// Non-negative fractions of a second at nanosecond resolution. Negative
	// second values with fractions must still have non-negative nanos values
	// that count forward in time. Must be from 0 to 999,999,999
	// inclusive. This field may be limited in precision depending on context.
	Nanos int32 `json:"nanos" protobuf:"varint,2,opt,name=nanos"`
}

// Timestamp returns the Time as a new Timestamp value.
func (m *Time) ProtoTime() *Timestamp {
	if m == nil {
		return &Timestamp{}
	}
	return &Timestamp{
		Seconds: m.Time.Unix(),
		Nanos:   int32(m.Time.Nanosecond()),
	}
}

// Size implements the protobuf marshalling interface.
func (m *Time) Size() (n int) {
	if m == nil || m.Time.IsZero() {
		return 0
	}
	return m.ProtoTime().Size()
}

// Reset implements the protobuf marshalling interface.
func (m *Time) Unmarshal(data []byte) error {
	if len(data) == 0 {
		m.Time = time.Time{}
		return nil
	}
	p := Timestamp{}
	if err := p.Unmarshal(data); err != nil {
		return err
	}
	m.Time = time.Unix(p.Seconds, int64(p.Nanos)).Local()
	return nil
}

// Marshal implements the protobuf marshalling interface.
func (m *Time) Marshal() (data []byte, err error) {
	if m == nil || m.Time.IsZero() {
		return nil, nil
	}
	return m.ProtoTime().Marshal()
}

// MarshalTo implements the protobuf marshalling interface.
func (m *Time) MarshalTo(data []byte) (int, error) {
	if m == nil || m.Time.IsZero() {
		return 0, nil
	}
	return m.ProtoTime().MarshalTo(data)
}
