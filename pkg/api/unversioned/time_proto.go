// +build proto

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"time"
)

// ProtoTime is a struct that is equivalent to Time, but intended for
// protobuf marshalling/unmarshalling. It is generated into a serialization
// that matches Time. Do not use in Go structs.
type ProtoTime struct {
	// Represents the time of an event.
	Timestamp Timestamp `json:"timestamp"`
}

// Timestamp is a protobuf Timestamp compatible representation of time.Time
type Timestamp struct {
	// Represents seconds of UTC time since Unix epoch
	// 1970-01-01T00:00:00Z. Must be from from 0001-01-01T00:00:00Z to
	// 9999-12-31T23:59:59Z inclusive.
	Seconds int64 `json:"seconds"`
	// Non-negative fractions of a second at nanosecond resolution. Negative
	// second values with fractions must still have non-negative nanos values
	// that count forward in time. Must be from 0 to 999,999,999
	// inclusive.
	Nanos int32 `json:"nanos"`
}

// ProtoTime returns the Time as a new ProtoTime value.
func (m *Time) ProtoTime() *ProtoTime {
	if m == nil {
		return &ProtoTime{}
	}
	return &ProtoTime{
		Timestamp: Timestamp{
			Seconds: m.Time.Unix(),
			Nanos:   int32(m.Time.Nanosecond()),
		},
	}
}

// Size implements the protobuf marshalling interface.
func (m *Time) Size() (n int) { return m.ProtoTime().Size() }

// Reset implements the protobuf marshalling interface.
func (m *Time) Unmarshal(data []byte) error {
	p := ProtoTime{}
	if err := p.Unmarshal(data); err != nil {
		return err
	}
	m.Time = time.Unix(p.Timestamp.Seconds, int64(p.Timestamp.Nanos))
	return nil
}

// Marshal implements the protobuf marshalling interface.
func (m *Time) Marshal() (data []byte, err error) {
	return m.ProtoTime().Marshal()
}

// MarshalTo implements the protobuf marshalling interface.
func (m *Time) MarshalTo(data []byte) (int, error) {
	return m.ProtoTime().MarshalTo(data)
}
