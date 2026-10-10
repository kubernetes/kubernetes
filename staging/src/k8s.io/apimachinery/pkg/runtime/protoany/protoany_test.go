/*
Copyright 2026 The Kubernetes Authors.

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

package protoany

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

// rawMessage encodes as its own bytes, which keeps expected wire output readable.
type rawMessage struct{ data []byte }

func (r *rawMessage) Size() int { return len(r.data) }
func (r *rawMessage) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	return copy(dAtA[len(dAtA)-len(r.data):], r.data), nil
}
func (r *rawMessage) Unmarshal(dAtA []byte) error {
	r.data = append([]byte(nil), dAtA...)
	return nil
}
func (r *rawMessage) Kind() string { return "raw" }

type failingMessage struct{ rawMessage }

func (f *failingMessage) Unmarshal([]byte) error { return errors.New("boom") }

type unregistered struct{ rawMessage }

type kinded interface{ Kind() string }

const (
	rawURL     = "example.com/raw"
	failingURL = "example.com/failing"
)

func init() {
	Register(rawURL, func() Message { return &rawMessage{} })
	Register(failingURL, func() Message { return &failingMessage{} })
}

func marshal(t *testing.T, v any) []byte {
	t.Helper()
	buf := make([]byte, Size(v))
	n, err := MarshalToSizedBuffer(v, buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != len(buf) {
		t.Fatalf("wrote %d bytes, Size reported %d", n, len(buf))
	}
	return buf
}

func TestMarshal(t *testing.T) {
	for name, tc := range map[string]struct {
		in       any
		expected []byte
	}{
		"with value": {
			in:       &rawMessage{data: []byte("ab")},
			expected: append([]byte{0x0a, byte(len(rawURL))}, append([]byte(rawURL), 0x12, 0x02, 'a', 'b')...),
		},
		"empty value is omitted": {
			in:       &rawMessage{},
			expected: append([]byte{0x0a, byte(len(rawURL))}, []byte(rawURL)...),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := marshal(t, tc.in); !bytes.Equal(got, tc.expected) {
				t.Errorf("expected %q, got %q", tc.expected, got)
			}
		})
	}
}

func TestMarshalWritesToTail(t *testing.T) {
	in := &rawMessage{data: []byte("ab")}
	buf := make([]byte, Size(in)+3)
	n, err := MarshalToSizedBuffer(in, buf)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf[:3], []byte{0, 0, 0}) || !bytes.Equal(buf[3:], marshal(t, in)) || n != len(buf)-3 {
		t.Errorf("expected encoding at the tail of the buffer, got %q (n=%d)", buf, n)
	}
}

func TestMarshalErrors(t *testing.T) {
	for name, tc := range map[string]struct {
		in       any
		expected string
	}{
		"nil interface": {in: nil, expected: "must not contain nil elements"},
		"typed nil":     {in: (*rawMessage)(nil), expected: "cannot marshal a nil *protoany.rawMessage"},
		"unregistered":  {in: &unregistered{}, expected: "type *protoany.unregistered has no registered type URL"},
	} {
		t.Run(name, func(t *testing.T) {
			if s := Size(tc.in); s != 0 {
				t.Errorf("expected size 0, got %d", s)
			}
			_, err := MarshalToSizedBuffer(tc.in, make([]byte, 64))
			if err == nil || !strings.Contains(err.Error(), tc.expected) {
				t.Errorf("expected error containing %q, got %v", tc.expected, err)
			}
		})
	}
}

func TestUnmarshalRoundTrip(t *testing.T) {
	var field kinded
	if err := Unmarshal(marshal(t, &rawMessage{data: []byte("ab")}), &field); err != nil {
		t.Fatal(err)
	}
	if got, ok := field.(*rawMessage); !ok || string(got.data) != "ab" {
		t.Errorf("unexpected result %#v", field)
	}

	var empty any
	if err := Unmarshal(marshal(t, &rawMessage{}), &empty); err != nil {
		t.Fatal(err)
	}
	if got, ok := empty.(*rawMessage); !ok || len(got.data) != 0 {
		t.Errorf("unexpected result %#v", empty)
	}
}

func TestUnmarshalSkipsUnknownFields(t *testing.T) {
	data := []byte{
		0x18, 0x96, 0x01, // field 3 varint
		0x21, 1, 2, 3, 4, 5, 6, 7, 8, // field 4 fixed64
		0x2a, 0x01, 'z', // field 5 bytes
		0x35, 1, 2, 3, 4, // field 6 fixed32
		0x3b, 0x08, 0x01, 0x3c, // field 7 group containing a varint
	}
	data = append(data, marshal(t, &rawMessage{data: []byte("ab")})...)
	var field any
	if err := Unmarshal(data, &field); err != nil {
		t.Fatal(err)
	}
	if got, ok := field.(*rawMessage); !ok || string(got.data) != "ab" {
		t.Errorf("unexpected result %#v", field)
	}
}

func TestUnmarshalErrors(t *testing.T) {
	withURL := func(url string, rest ...byte) []byte {
		return append(append([]byte{0x0a, byte(len(url))}, url...), rest...)
	}
	for name, tc := range map[string]struct {
		data     []byte
		into     func(data []byte) error
		expected string
	}{
		"empty url":        {data: []byte{0x12, 0x00}, expected: "empty type URL"},
		"unregistered url": {data: withURL("example.com/nope"), expected: `no type registered for type URL "example.com/nope"`},
		"wrong wire type":  {data: []byte{0x08, 0x01}, expected: "wrong wire type 0 for google.protobuf.Any field 1"},
		"unclosed group":   {data: []byte{0x1b}, expected: "skipping unknown google.protobuf.Any field 3"},
		"field zero":       {data: []byte{0x02, 0x00}, expected: "invalid field number"},
		"truncated":        {data: []byte{0x0a, 0x05, 'a'}, expected: io.ErrUnexpectedEOF.Error()},
		"truncated fixed":  {data: []byte{0x21, 1, 2}, expected: io.ErrUnexpectedEOF.Error()},
		"overflow":         {data: bytes.Repeat([]byte{0xff}, 11), expected: "integer overflow"},
		"decode failure":   {data: withURL(failingURL, 0x12, 0x01, 'x'), expected: `decoding "example.com/failing": boom`},
		"not assignable": {
			data: withURL(rawURL),
			into: func(data []byte) error {
				var s interface{ NotImplemented() }
				return Unmarshal(data, &s)
			},
			expected: "is not assignable to field of type interface { NotImplemented() }",
		},
	} {
		t.Run(name, func(t *testing.T) {
			into := tc.into
			if into == nil {
				into = func(data []byte) error {
					var v any
					return Unmarshal(data, &v)
				}
			}
			if err := into(tc.data); err == nil || !strings.Contains(err.Error(), tc.expected) {
				t.Errorf("expected error containing %q, got %v", tc.expected, err)
			}
		})
	}
}

func TestRegisterPanics(t *testing.T) {
	type other struct{ rawMessage }
	for name, register := range map[string]func(){
		"empty url":       func() { Register("", func() Message { return &other{} }) },
		"nil constructor": func() { Register("example.com/nil", nil) },
		"nil result":      func() { Register("example.com/nilresult", func() Message { return nil }) },
		"duplicate url":   func() { Register(rawURL, func() Message { return &other{} }) },
		"duplicate type":  func() { Register("example.com/raw2", func() Message { return &rawMessage{} }) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("expected panic")
				}
			}()
			register()
		})
	}
}

func TestTypeURL(t *testing.T) {
	if url, ok := TypeURL(&rawMessage{}); !ok || url != rawURL {
		t.Errorf("expected %q, got %q (%v)", rawURL, url, ok)
	}
	if _, ok := TypeURL(&unregistered{}); ok {
		t.Error("expected unregistered type to have no URL")
	}
}
