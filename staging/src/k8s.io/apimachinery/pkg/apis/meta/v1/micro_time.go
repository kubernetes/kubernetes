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
	"encoding/json"
	"time"

	"k8s.io/apimachinery/pkg/openapi"

	"github.com/go-openapi/spec"
	"github.com/google/gofuzz"
)

const RFC3339Micro = "2006-01-02T15:04:05.000000Z07:00"

// MicroTime is version of Time with microsecond level precision.
//
// +protobuf.options.marshal=false
// +protobuf.as=Timestamp
// +protobuf.options.(gogoproto.goproto_stringer)=false
type MicroTime struct {
	time.Time `protobuf:"-"`
}

// DeepCopy returns a deep-copy of the MicroTime value.  The underlying time.Time
// type is effectively immutable in the time API, so it is safe to
// copy-by-assign, despite the presence of (unexported) Pointer fields.
func (t MicroTime) DeepCopy() MicroTime {
	return t
}

// String returns the representation of the time.
func (t MicroTime) String() string {
	return t.Time.String()
}

// NewMicroTime returns a wrapped instance of the provided time
func NewMicroTime(time time.Time) MicroTime {
	return MicroTime{time}
}

// DateMicro returns the MicroTime corresponding to the supplied parameters
// by wrapping time.Date.
func DateMicro(year int, month time.Month, day, hour, min, sec, nsec int, loc *time.Location) MicroTime {
	return MicroTime{time.Date(year, month, day, hour, min, sec, nsec, loc)}
}

// NowMicro returns the current local time.
func NowMicro() MicroTime {
	return MicroTime{time.Now()}
}

// IsZero returns true if the value is nil or time is zero.
func (t *MicroTime) IsZero() bool {
	if t == nil {
		return true
	}
	return t.Time.IsZero()
}

// Before reports whether the time instant t is before u.
func (t MicroTime) Before(u MicroTime) bool {
	return t.Time.Before(u.Time)
}

// Equal reports whether the time instant t is equal to u.
func (t MicroTime) Equal(u MicroTime) bool {
	return t.Time.Equal(u.Time)
}

// BeforeTime reports whether the time instant t is before second-lever precision u.
func (t MicroTime) BeforeTime(u Time) bool {
	return t.Time.Before(u.Time)
}

// EqualTime reports whether the time instant t is equal to second-lever precision u.
func (t MicroTime) EqualTime(u Time) bool {
	return t.Time.Equal(u.Time)
}

// UnixMicro returns the local time corresponding to the given Unix time
// by wrapping time.Unix.
func UnixMicro(sec int64, nsec int64) MicroTime {
	return MicroTime{time.Unix(sec, nsec)}
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (t *MicroTime) UnmarshalJSON(b []byte) error {
	if len(b) == 4 && string(b) == "null" {
		t.Time = time.Time{}
		return nil
	}

	var str string
	json.Unmarshal(b, &str)

	pt, err := time.Parse(RFC3339Micro, str)
	if err != nil {
		return err
	}

	t.Time = pt.Local()
	return nil
}

// UnmarshalQueryParameter converts from a URL query parameter value to an object
func (t *MicroTime) UnmarshalQueryParameter(str string) error {
	if len(str) == 0 {
		t.Time = time.Time{}
		return nil
	}
	// Tolerate requests from older clients that used JSON serialization to build query params
	if len(str) == 4 && str == "null" {
		t.Time = time.Time{}
		return nil
	}

	pt, err := time.Parse(RFC3339Micro, str)
	if err != nil {
		return err
	}

	t.Time = pt.Local()
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (t MicroTime) MarshalJSON() ([]byte, error) {
	if t.IsZero() {
		// Encode unset/nil objects as JSON's "null".
		return []byte("null"), nil
	}

	return json.Marshal(t.UTC().Format(RFC3339Micro))
}

func (_ MicroTime) OpenAPIDefinition() openapi.OpenAPIDefinition {
	return openapi.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type:   []string{"string"},
				Format: "date-time",
			},
		},
	}
}

// MarshalQueryParameter converts to a URL query parameter value
func (t MicroTime) MarshalQueryParameter() (string, error) {
	if t.IsZero() {
		// Encode unset/nil objects as an empty string
		return "", nil
	}

	return t.UTC().Format(RFC3339Micro), nil
}

// Fuzz satisfies fuzz.Interface.
func (t *MicroTime) Fuzz(c fuzz.Continue) {
	if t == nil {
		return
	}
	// Allow for about 1000 years of randomness.  Leave off nanoseconds
	// because JSON doesn't represent them so they can't round-trip
	// properly.
	t.Time = time.Unix(c.Rand.Int63n(1000*365*24*60*60*1000*1000), 0)
}

var _ fuzz.Interface = &MicroTime{}
