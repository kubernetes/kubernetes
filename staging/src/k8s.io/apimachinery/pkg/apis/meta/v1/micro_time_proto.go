/*
Copyright 2016 The Kubernetes Authors.

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

// Timestamp is declared in time_proto.go

// Timestamp returns the Time as a new Timestamp value.
func (m *MicroTime) ProtoMicroTime() *Timestamp {
	if m == nil {
		return &Timestamp{}
	}

	// truncate precision to microseconds to match JSON marshaling/unmarshaling
	truncatedNanoseconds := time.Duration(m.Time.Nanosecond()).Truncate(time.Microsecond)
	return &Timestamp{
		Seconds: m.Time.Unix(),
		Nanos:   int32(truncatedNanoseconds),
	}
}

// Size implements the protobuf marshalling interface.
func (m *MicroTime) Size() (n int) {
	if m == nil || m.Time.IsZero() {
		return 0
	}
	return m.ProtoMicroTime().Size()
}

// Reset implements the protobuf marshalling interface.
func (m *MicroTime) Unmarshal(data []byte) error {
	if len(data) == 0 {
		m.Time = time.Time{}
		return nil
	}
	p := Timestamp{}
	if err := p.Unmarshal(data); err != nil {
		return err
	}

	// truncate precision to microseconds to match JSON marshaling/unmarshaling
	truncatedNanoseconds := time.Duration(p.Nanos).Truncate(time.Microsecond)
	m.Time = time.Unix(p.Seconds, int64(truncatedNanoseconds)).Local()
	return nil
}

// Marshal implements the protobuf marshalling interface.
func (m *MicroTime) Marshal() (data []byte, err error) {
	if m == nil || m.Time.IsZero() {
		return nil, nil
	}
	return m.ProtoMicroTime().Marshal()
}

// MarshalTo implements the protobuf marshalling interface.
func (m *MicroTime) MarshalTo(data []byte) (int, error) {
	if m == nil || m.Time.IsZero() {
		return 0, nil
	}
	return m.ProtoMicroTime().MarshalTo(data)
}

// MarshalToSizedBuffer implements the protobuf marshalling interface.
func (m *MicroTime) MarshalToSizedBuffer(data []byte) (int, error) {
	if m == nil || m.Time.IsZero() {
		return 0, nil
	}
	return m.ProtoMicroTime().MarshalToSizedBuffer(data)
}
