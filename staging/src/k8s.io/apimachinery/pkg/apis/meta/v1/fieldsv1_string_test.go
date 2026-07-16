//go:build fieldsv1string

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

package v1_test

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
	"unsafe"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// requireInterned fails the test if the two strings do not point to the exact same memory address.
func requireInterned(t *testing.T, a, b string) {
	t.Helper()
	ptrA := unsafe.StringData(a)
	ptrB := unsafe.StringData(b)
	if ptrA != ptrB {
		t.Fatalf("Expected strings to be interned (same memory address) but pointers differ: %p != %p", ptrA, ptrB)
	}
}

func TestFieldsV1_String(t *testing.T) {
	for _, tc := range []struct {
		name     string
		f        metav1.FieldsV1
		expected string
	}{
		{
			name:     "zero value handle",
			f:        metav1.FieldsV1{},
			expected: "",
		},
		{
			name:     "initialized empty handle",
			f:        *metav1.NewFieldsV1(""),
			expected: "",
		},
		{
			name:     "valid payload",
			f:        *metav1.NewFieldsV1(`{"f:app":{}}`),
			expected: `{"f:app":{}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.f.String(); got != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, got)
			}
		})
	}
}

func TestFieldsV1_Equal(t *testing.T) {
	valid := *metav1.NewFieldsV1(`{"f:app":{}}`)
	validClone := *metav1.NewFieldsV1(`{"f:app":{}}`)
	different := *metav1.NewFieldsV1(`{"f:other":{}}`)

	for _, tc := range []struct {
		name     string
		a        metav1.FieldsV1
		b        metav1.FieldsV1
		expected bool
	}{
		{"both zero value", metav1.FieldsV1{}, metav1.FieldsV1{}, true},
		{"zero value and initialized empty handle", metav1.FieldsV1{}, *metav1.NewFieldsV1(""), true},
		{"identical valid payloads", valid, validClone, true},
		{"different payloads", valid, different, false},
		{"zero value and valid payload", metav1.FieldsV1{}, valid, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.a.Equal(tc.b); got != tc.expected {
				t.Errorf("expected Equal() to be %v, got %v", tc.expected, got)
			}
			// Commutative property
			if got := tc.b.Equal(tc.a); got != tc.expected {
				t.Errorf("expected Equal() commutative to be %v, got %v", tc.expected, got)
			}
		})
	}
}

func TestFieldsV1_GetRawReader(t *testing.T) {
	for _, tc := range []struct {
		name     string
		f        *metav1.FieldsV1
		expected []byte
	}{
		{"nil receiver", nil, []byte("")},
		{"zero value handle", &metav1.FieldsV1{}, []byte("")},
		{"initialized empty handle", metav1.NewFieldsV1(""), []byte("")},
		{"valid payload", metav1.NewFieldsV1(`{"f:app":{}}`), []byte(`{"f:app":{}}`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reader := tc.f.GetRawReader()
			got, err := io.ReadAll(reader)
			if err != nil {
				t.Fatalf("unexpected error reading from RawReader: %v", err)
			}
			if !bytes.Equal(got, tc.expected) {
				t.Errorf("expected %q, got %q", tc.expected, got)
			}
		})
	}
}

func TestFieldsV1_GetRawBytes(t *testing.T) {
	for _, tc := range []struct {
		name     string
		f        *metav1.FieldsV1
		expected []byte
	}{
		{"nil receiver", nil, nil},
		{"zero value handle", &metav1.FieldsV1{}, nil},
		{"initialized empty handle", metav1.NewFieldsV1(""), []byte{}},
		{"valid payload", metav1.NewFieldsV1(`{"f:app":{}}`), []byte(`{"f:app":{}}`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.f.GetRawBytes()
			if !bytes.Equal(got, tc.expected) {
				t.Errorf("expected %v, got %v", tc.expected, got)
			}
			// Explicitly check for nil
			if tc.expected == nil && got != nil {
				t.Errorf("expected strict nil, got %v", got)
			}
		})
	}

	t.Run("mutation safety", func(t *testing.T) {
		f := metav1.NewFieldsV1(`{"f:app":{}}`)
		b := f.GetRawBytes()

		// Mutate the returned bytes
		b[2] = 'X'

		// The original interned string must remain unchanged
		if got := f.GetRawString(); got != `{"f:app":{}}` {
			t.Errorf("GetRawBytes returned an unisolated slice! Mutating it corrupted the handle. Got: %s", got)
		}
	})
}

func TestFieldsV1_GetRawString(t *testing.T) {
	for _, tc := range []struct {
		name     string
		f        *metav1.FieldsV1
		expected string
	}{
		{"nil receiver", nil, ""},
		{"zero value handle", &metav1.FieldsV1{}, ""},
		{"initialized empty handle", metav1.NewFieldsV1(""), ""},
		{"valid payload", metav1.NewFieldsV1(`{"f:app":{}}`), `{"f:app":{}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.f.GetRawString(); got != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, got)
			}
		})
	}
}

func TestFieldsV1_SetRawBytes(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var f *metav1.FieldsV1        // nil
		f.SetRawBytes([]byte("test")) // Should not panic
	})

	t.Run("empty slice input", func(t *testing.T) {
		f := &metav1.FieldsV1{}
		f.SetRawBytes([]byte{})
		if f.GetRawString() != "" {
			t.Errorf("Expected empty string, got %q", f.GetRawString())
		}
	})

	t.Run("nil slice input", func(t *testing.T) {
		f := &metav1.FieldsV1{}
		f.SetRawBytes(nil)
		if f.GetRawString() != "" {
			t.Errorf("Expected empty string, got %q", f.GetRawString())
		}
	})

	t.Run("interning behavior", func(t *testing.T) {
		payload := []byte(`{"f:app":{}}`)

		// Two distinct slices with identical content
		b1 := append([]byte(nil), payload...)
		b2 := append([]byte(nil), payload...)

		f1 := &metav1.FieldsV1{}
		f1.SetRawBytes(b1)

		f2 := &metav1.FieldsV1{}
		f2.SetRawBytes(b2)

		if f1.GetRawString() != string(payload) {
			t.Errorf("Expected GetRawString to match payload, got %q", f1.GetRawString())
		}

		requireInterned(t, f1.GetRawString(), f2.GetRawString())
	})
}

func TestFieldsV1_SetRawString(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var f *metav1.FieldsV1 // nil
		f.SetRawString("test") // Should not panic
	})

	t.Run("interning behavior", func(t *testing.T) {
		payload := `{"f:app":{}}`

		// Force string copy to avoid compiler static string optimization if possible
		s1 := string(append([]byte(nil), payload...))
		s2 := string(append([]byte(nil), payload...))

		f1 := &metav1.FieldsV1{}
		f1.SetRawString(s1)

		f2 := &metav1.FieldsV1{}
		f2.SetRawString(s2)

		requireInterned(t, f1.GetRawString(), f2.GetRawString())
	})
}

func TestFieldsV1_NewFieldsV1(t *testing.T) {
	t.Run("interning behavior", func(t *testing.T) {
		payload := `{"f:app":{}}`
		s1 := string(append([]byte(nil), payload...))
		s2 := string(append([]byte(nil), payload...))

		f1 := metav1.NewFieldsV1(s1)
		f2 := metav1.NewFieldsV1(s2)

		requireInterned(t, f1.GetRawString(), f2.GetRawString())
	})
}

func TestFieldsV1_DeepCopyInto(t *testing.T) {
	t.Run("preserves interning", func(t *testing.T) {
		orig := metav1.NewFieldsV1(`{"f:app":{}}`)
		var clone metav1.FieldsV1
		orig.DeepCopyInto(&clone)

		if orig.GetRawString() != clone.GetRawString() {
			t.Fatalf("Expected strings to match after DeepCopyInto, got %q and %q", orig.GetRawString(), clone.GetRawString())
		}

		requireInterned(t, orig.GetRawString(), clone.GetRawString())
	})

	t.Run("zero value deep copy", func(t *testing.T) {
		var orig metav1.FieldsV1
		var clone metav1.FieldsV1
		orig.DeepCopyInto(&clone)

		if clone.GetRawString() != "" {
			t.Errorf("Expected empty string from cloned zero-value, got %q", clone.GetRawString())
		}
		if !orig.Equal(clone) {
			t.Errorf("Expected original zero-value and its clone to be equal")
		}
	})

	t.Run("independence after deep copy", func(t *testing.T) {
		orig := metav1.NewFieldsV1(`{"f:app":{}}`)
		var clone metav1.FieldsV1
		orig.DeepCopyInto(&clone)

		// Modify the clone
		clone.SetRawString(`{"f:new":{}}`)

		// The original should be completely untouched
		if got := orig.GetRawString(); got != `{"f:app":{}}` {
			t.Errorf("DeepCopyInto failed to isolate clone! Modifying clone corrupted original. Got: %s", got)
		}
	})
}
func TestFieldsV1_UnmarshalInterning(t *testing.T) {
	jsonPayload := []byte(`{"f:metadata":{"f:labels":{"f:app":{}}}}`)
	orig := metav1.NewFieldsV1(string(jsonPayload))

	protoPayload, err := orig.Marshal()
	if err != nil {
		t.Fatalf("Failed to marshal protobuf payload during setup: %v", err)
	}

	cborPayload, err := orig.MarshalCBOR()
	if err != nil {
		t.Fatalf("Failed to marshal CBOR payload during setup: %v", err)
	}

	tests := []struct {
		name      string
		payload   []byte
		unmarshal func(t *testing.T, f *metav1.FieldsV1, payload []byte)
	}{
		{
			name:    "json unmarshal",
			payload: jsonPayload,
			unmarshal: func(t *testing.T, f *metav1.FieldsV1, payload []byte) {
				if err := json.Unmarshal(payload, f); err != nil {
					t.Fatalf("JSON Unmarshal failed: %v", err)
				}
			},
		},
		{
			name:    "protobuf unmarshal",
			payload: protoPayload,
			unmarshal: func(t *testing.T, f *metav1.FieldsV1, payload []byte) {
				if err := f.Unmarshal(payload); err != nil {
					t.Fatalf("Protobuf Unmarshal failed: %v", err)
				}
			},
		},
		{
			name:    "cbor unmarshal",
			payload: cborPayload,
			unmarshal: func(t *testing.T, f *metav1.FieldsV1, payload []byte) {
				if err := f.UnmarshalCBOR(payload); err != nil {
					t.Fatalf("CBOR Unmarshal failed: %v", err)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Ensure independent slice memory
			b1 := append([]byte(nil), tc.payload...)
			b2 := append([]byte(nil), tc.payload...)

			var f1 metav1.FieldsV1
			var f2 metav1.FieldsV1

			tc.unmarshal(t, &f1, b1)
			tc.unmarshal(t, &f2, b2)

			requireInterned(t, f1.GetRawString(), f2.GetRawString())
		})
	}
}
