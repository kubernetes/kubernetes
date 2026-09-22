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

package json

import (
	"bytes"
	gojson "encoding/json"
	"math"
	"reflect"
	"testing"
	"time"
)

type pointerMarshaler int

func (*pointerMarshaler) MarshalJSON() ([]byte, error) {
	return []byte(`"custom"`), nil
}

// TestMarshalLegacyCompatibility compares v2 legacy-mode output and error types
// for collection states, tags, deterministic values, raw and custom marshalers,
// addressability-sensitive methods, and invalid inputs.
func TestMarshalLegacyCompatibility(t *testing.T) {
	p := pointerMarshaler(1)
	for _, tc := range []struct {
		name  string
		value any
	}{
		{name: "nil", value: nil},
		{name: "nil collections", value: struct {
			Slice []string
			Map   map[string]string
			Bytes []byte
		}{}},
		{name: "empty collections", value: []any{[]string{}, map[string]string{}, []byte{}}},
		{name: "omission", value: struct {
			Bool      bool           `json:"bool,omitempty"`
			Number    int            `json:"number,omitempty"`
			Struct    struct{}       `json:"struct,omitempty"`
			Zero      struct{}       `json:"zero,omitzero"`
			Pointer   *int           `json:"pointer,omitempty"`
			Interface any            `json:"interface,omitempty"`
			Slice     []int          `json:"slice,omitempty"`
			Map       map[string]int `json:"map,omitempty"`
		}{Pointer: new(int), Interface: false, Slice: []int{}, Map: map[string]int{}}},
		{name: "string tags", value: struct {
			Bool   bool   `json:",string"`
			String string `json:",string"`
			Int    int64  `json:",string"`
		}{Bool: true, String: "text", Int: math.MaxInt64}},
		{name: "sorted maps and precise integers", value: map[string]any{
			"z": uint64(math.MaxUint64), "a": int64(math.MaxInt64), "m": gojson.Number("9007199254740993"),
		}},
		{name: "escaping", value: "<>&\u2028\u2029"},
		{name: "invalid utf8", value: "a\xffb"},
		{name: "byte array", value: [3]byte{1, 2, 3}},
		{name: "duration", value: time.Second},
		{name: "raw duplicate names", value: gojson.RawMessage(`{"a":1,"a":2}`)},
		{name: "pointer marshaler", value: &p},
		{name: "unaddressable marshaler", value: p},
		{name: "map element marshaler", value: map[string]pointerMarshaler{"p": p}},
		{name: "slice element marshaler", value: []pointerMarshaler{p}},
		{name: "unsupported type", value: make(chan int)},
		{name: "nonfinite number", value: math.Inf(1)},
		{name: "invalid raw message", value: gojson.RawMessage(`{"a":`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, wantErr := gojson.Marshal(tc.value)
			got, err := Marshal(tc.value)
			if reflect.TypeOf(err) != reflect.TypeOf(wantErr) {
				t.Fatalf("error type = %T (%v), want %T (%v)", err, err, wantErr, wantErr)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("encoding = %q, want %q", got, want)
			}
		})
	}
}

// TestNewEncoderCompatibility verifies the concrete return type and output for
// sequential values across indentation and HTML-escaping settings.
func TestNewEncoderCompatibility(t *testing.T) {
	for _, indent := range []bool{false, true} {
		for _, escapeHTML := range []bool{false, true} {
			var got, want bytes.Buffer
			// Preserve compatibility with callers that name the concrete return type.
			var encoder *gojson.Encoder = NewEncoder(&got)
			reference := gojson.NewEncoder(&want)
			encoder.SetEscapeHTML(escapeHTML)
			reference.SetEscapeHTML(escapeHTML)
			if indent {
				encoder.SetIndent("prefix", "  ")
				reference.SetIndent("prefix", "  ")
			}
			for _, value := range []any{map[string]any{"html": "<>&", "nil": []string(nil)}, true} {
				if err := encoder.Encode(value); err != nil {
					t.Fatal(err)
				}
				if err := reference.Encode(value); err != nil {
					t.Fatal(err)
				}
			}
			if !bytes.Equal(got.Bytes(), want.Bytes()) {
				t.Errorf("indent=%t escapeHTML=%t: encoding = %q, want %q", indent, escapeHTML, got.Bytes(), want.Bytes())
			}
		}
	}
}
