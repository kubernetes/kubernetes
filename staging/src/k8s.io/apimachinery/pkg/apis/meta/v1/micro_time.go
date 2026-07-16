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

	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
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
func (t *MicroTime) DeepCopyInto(out *MicroTime) {
	*out = *t
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
func (t *MicroTime) Before(u *MicroTime) bool {
	if t != nil && u != nil {
		return t.Time.Before(u.Time)
	}
	return false
}

// Equal reports whether the time instant t is equal to u.
func (t *MicroTime) Equal(u *MicroTime) bool {
	if t == nil && u == nil {
		return true
	}
	if t != nil && u != nil {
		return t.Time.Equal(u.Time)
	}
	return false
}

// BeforeTime reports whether the time instant t is before second-lever precision u.
func (t *MicroTime) BeforeTime(u *Time) bool {
	if t != nil && u != nil {
		return t.Time.Before(u.Time)
	}
	return false
}

// EqualTime reports whether the time instant t is equal to second-lever precision u.
func (t *MicroTime) EqualTime(u *Time) bool {
	if t == nil && u == nil {
		return true
	}
	if t != nil && u != nil {
		return t.Time.Equal(u.Time)
	}
	return false
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
	err := json.Unmarshal(b, &str)
	if err != nil {
		return err
	}

	pt, err := time.Parse(RFC3339Micro, str)
	if err != nil {
		return err
	}

	t.Time = pt.Local()
	return nil
}

func (t *MicroTime) UnmarshalCBOR(b []byte) error {
	var s *string
	if err := cbor.Unmarshal(b, &s); err != nil {
		return err
	}
	if s == nil {
		t.Time = time.Time{}
		return nil
	}

	parsed, err := time.Parse(RFC3339Micro, *s)
	if err != nil {
		return err
	}

	t.Time = parsed.Local()
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

func (t MicroTime) MarshalCBOR() ([]byte, error) {
	if t.IsZero() {
		return cbor.Marshal(nil)
	}
	return cbor.Marshal(t.UTC().Format(RFC3339Micro))
}

// OpenAPISchemaType is used by the kube-openapi generator when constructing
// the OpenAPI spec of this type.
//
// See: https://github.com/kubernetes/kube-openapi/tree/master/pkg/generators
func (_ MicroTime) OpenAPISchemaType() []string { return []string{"string"} }

// OpenAPISchemaFormat is used by the kube-openapi generator when constructing
// the OpenAPI spec of this type.
func (_ MicroTime) OpenAPISchemaFormat() string { return "date-time" }

// MarshalQueryParameter converts to a URL query parameter value
func (t MicroTime) MarshalQueryParameter() (string, error) {
	if t.IsZero() {
		// Encode unset/nil objects as an empty string
		return "", nil
	}

	return t.UTC().Format(RFC3339Micro), nil
}
