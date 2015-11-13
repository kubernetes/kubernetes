/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"time"

	"github.com/google/gofuzz"
)

// Time is a wrapper around time.Time which supports correct
// marshaling to YAML and JSON.  Wrappers are provided for many
// of the factory methods that the time package offers.
//
// +genprotoidl.options.marshal=false
type Time struct {
	time.Time `protobuf:"Timestamp,1,req,name=time"`
}

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

// NewTime returns a wrapped instance of the provided time
func NewTime(time time.Time) Time {
	return Time{time}
}

// Date returns the Time corresponding to the supplied parameters
// by wrapping time.Date.
func Date(year int, month time.Month, day, hour, min, sec, nsec int, loc *time.Location) Time {
	return Time{time.Date(year, month, day, hour, min, sec, nsec, loc)}
}

// Now returns the current local time.
func Now() Time {
	return Time{time.Now()}
}

// IsZero returns true if the value is nil or time is zero.
func (t *Time) IsZero() bool {
	if t == nil {
		return true
	}
	return t.Time.IsZero()
}

// Before reports whether the time instant t is before u.
func (t Time) Before(u Time) bool {
	return t.Time.Before(u.Time)
}

// Equal reports whether the time instant t is equal to u.
func (t Time) Equal(u Time) bool {
	return t.Time.Equal(u.Time)
}

// Unix returns the local time corresponding to the given Unix time
// by wrapping time.Unix.
func Unix(sec int64, nsec int64) Time {
	return Time{time.Unix(sec, nsec)}
}

// Rfc3339Copy returns a copy of the Time at second-level precision.
func (t Time) Rfc3339Copy() Time {
	copied, _ := time.Parse(time.RFC3339, t.Format(time.RFC3339))
	return Time{copied}
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (t *Time) UnmarshalJSON(b []byte) error {
	if len(b) == 4 && string(b) == "null" {
		t.Time = time.Time{}
		return nil
	}

	var str string
	json.Unmarshal(b, &str)

	pt, err := time.Parse(time.RFC3339, str)
	if err != nil {
		return err
	}

	t.Time = pt.Local()
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (t Time) MarshalJSON() ([]byte, error) {
	if t.IsZero() {
		// Encode unset/nil objects as JSON's "null".
		return []byte("null"), nil
	}

	return json.Marshal(t.UTC().Format(time.RFC3339))
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

// Fuzz satisfies fuzz.Interface.
func (t *Time) Fuzz(c fuzz.Continue) {
	if t == nil {
		return
	}
	// Allow for about 1000 years of randomness.  Leave off nanoseconds
	// because JSON doesn't represent them so they can't round-trip
	// properly.
	t.Time = time.Unix(c.Rand.Int63n(1000*365*24*60*60), 0)
}

var _ fuzz.Interface = &Time{}
