/*
Copyright 2014 The Kubernetes Authors.

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

package runtime_test

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"

	"github.com/google/go-cmp/cmp"
)

func TestEmbeddedRawExtensionMarshal(t *testing.T) {
	type test struct {
		Ext runtime.RawExtension
	}

	extension := test{Ext: runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)}}
	data, err := json.Marshal(extension)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(data) != `{"Ext":{"foo":"bar"}}` {
		t.Errorf("unexpected data: %s", string(data))
	}
}
func TestEmbeddedRawExtensionUnmarshal(t *testing.T) {
	type test struct {
		Ext runtime.RawExtension
	}

	testCases := map[string]struct {
		orig test
	}{
		"non-empty object": {
			orig: test{Ext: runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)}},
		},
		"empty object": {
			orig: test{Ext: runtime.RawExtension{}},
		},
	}

	for k, tc := range testCases {
		new := test{}
		data, _ := json.Marshal(tc.orig)
		if err := json.Unmarshal(data, &new); err != nil {
			t.Errorf("%s: umarshal error: %v", k, err)
		}
		if !reflect.DeepEqual(tc.orig, new) {
			t.Errorf("%s: unmarshaled struct differs from original: %v %v", k, tc.orig, new)
		}
	}
}

func TestEmbeddedRawExtensionRoundTrip(t *testing.T) {
	type test struct {
		Ext runtime.RawExtension
	}

	testCases := map[string]struct {
		orig test
	}{
		"non-empty object": {
			orig: test{Ext: runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)}},
		},
		"empty object": {
			orig: test{Ext: runtime.RawExtension{}},
		},
	}

	for k, tc := range testCases {
		new1 := test{}
		new2 := test{}
		data, err := json.Marshal(tc.orig)
		if err != nil {
			t.Errorf("1st marshal error: %v", err)
		}
		if err = json.Unmarshal(data, &new1); err != nil {
			t.Errorf("1st unmarshal error: %v", err)
		}
		newData, err := json.Marshal(new1)
		if err != nil {
			t.Errorf("2st marshal error: %v", err)
		}
		if err = json.Unmarshal(newData, &new2); err != nil {
			t.Errorf("2nd unmarshal error: %v", err)
		}
		if !bytes.Equal(data, newData) {
			t.Errorf("%s: re-marshaled data differs from original: %v %v", k, data, newData)
		}
		if !reflect.DeepEqual(tc.orig, new1) {
			t.Errorf("%s: unmarshaled struct differs from original: %v %v", k, tc.orig, new1)
		}
		if !reflect.DeepEqual(new1, new2) {
			t.Errorf("%s: re-unmarshaled struct differs from original: %v %v", k, new1, new2)
		}
	}
}

func TestRawExtensionMarshalUnstructured(t *testing.T) {
	for _, tc := range []struct {
		Name              string
		In                runtime.RawExtension
		WantCBOR          []byte
		ExpectedErrorCBOR string
		WantJSON          string
		ExpectedErrorJSON string
	}{
		{
			Name:     "nil bytes and nil object",
			In:       runtime.RawExtension{},
			WantCBOR: []byte{0xf6},
			WantJSON: "null",
		},
		{
			Name:     "nil bytes and non-nil object",
			In:       runtime.RawExtension{Object: &runtimetesting.ExternalSimple{TestString: "foo"}},
			WantCBOR: []byte("\xa1\x4atestString\x43foo"),
			WantJSON: `{"testString":"foo"}`,
		},
		{
			Name:              "cbor bytes not enclosed in self-described tag",
			In:                runtime.RawExtension{Raw: []byte{0x43, 'f', 'o', 'o'}}, // 'foo'
			ExpectedErrorCBOR: "cannot convert RawExtension with unrecognized content type to unstructured",
			ExpectedErrorJSON: "cannot convert RawExtension with unrecognized content type to unstructured",
		},
		{
			Name:     "cbor bytes enclosed in self-described tag",
			In:       runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0x43, 'f', 'o', 'o'}}, // 55799('foo')
			WantCBOR: []byte{0xd9, 0xd9, 0xf7, 0x43, 'f', 'o', 'o'},                            // 55799('foo')
			WantJSON: `"foo"`,
		},
		{
			Name:     "json bytes",
			In:       runtime.RawExtension{Raw: []byte(`"foo"`)},
			WantCBOR: []byte{0x43, 'f', 'o', 'o'},
			WantJSON: `"foo"`,
		},
		{
			Name:     "ambiguous bytes not enclosed in self-described cbor tag",
			In:       runtime.RawExtension{Raw: []byte{'0'}}, // CBOR -17 / JSON 0
			WantCBOR: []byte{0x00},
			WantJSON: `0`,
		},
		{
			Name:     "ambiguous bytes enclosed in self-described cbor tag",
			In:       runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, '0'}}, // 55799(-17)
			WantCBOR: []byte{0xd9, 0xd9, 0xf7, '0'},
			WantJSON: `-17`,
		},
		{
			Name:              "unrecognized bytes",
			In:                runtime.RawExtension{Raw: []byte{0xff}},
			ExpectedErrorCBOR: "cannot convert RawExtension with unrecognized content type to unstructured",
			ExpectedErrorJSON: "cannot convert RawExtension with unrecognized content type to unstructured",
		},
		{
			Name:              "invalid cbor with self-described cbor prefix",
			In:                runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0xff}},
			WantCBOR:          []byte{0xd9, 0xd9, 0xf7, 0xff}, // verbatim
			ExpectedErrorJSON: `failed to parse RawExtension bytes as CBOR: cbor: unexpected "break" code`,
		},
		{
			Name:              "invalid json with json prefix",
			In:                runtime.RawExtension{Raw: []byte(`{{`)},
			ExpectedErrorCBOR: `failed to parse RawExtension bytes as JSON: invalid character '{' looking for beginning of object key string`,
			WantJSON:          `{{`, // verbatim
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			t.Run("CBOR", func(t *testing.T) {
				got, err := tc.In.MarshalCBOR()
				if err != nil {
					if tc.ExpectedErrorCBOR == "" {
						t.Fatalf("unexpected error: %v", err)
					}
					if msg := err.Error(); msg != tc.ExpectedErrorCBOR {
						t.Fatalf("expected error %q but got %q", tc.ExpectedErrorCBOR, msg)
					}
				}

				if diff := cmp.Diff(tc.WantCBOR, got); diff != "" {
					t.Errorf("unexpected diff:\n%s", diff)
				}
			})

			t.Run("JSON", func(t *testing.T) {
				got, err := tc.In.MarshalJSON()
				if err != nil {
					if tc.ExpectedErrorJSON == "" {
						t.Fatalf("unexpected error: %v", err)
					}
					if msg := err.Error(); msg != tc.ExpectedErrorJSON {
						t.Fatalf("expected error %q but got %q", tc.ExpectedErrorJSON, msg)
					}
				}

				if diff := cmp.Diff(tc.WantJSON, string(got)); diff != "" {
					t.Errorf("unexpected diff:\n%s", diff)
				}
			})
		})
	}
}

func TestRawExtensionUnmarshalCBOR(t *testing.T) {
	for _, tc := range []struct {
		Name string
		In   []byte
		Want runtime.RawExtension
	}{
		{
			// From json.Unmarshaler: By convention, to approximate the behavior of
			// Unmarshal itself, Unmarshalers implement UnmarshalJSON([]byte("null")) as
			// a no-op.
			Name: "no-op on null",
			In:   []byte{0xf6},
			Want: runtime.RawExtension{},
		},
		{
			Name: "input copied verbatim",
			In:   []byte{0xd9, 0xd9, 0xf7, 0x5f, 0x41, 'f', 0x42, 'o', 'o', 0xff}, // 55799(_ 'f' 'oo')
			Want: runtime.RawExtension{
				Raw: []byte{0xd9, 0xd9, 0xf7, 0x5f, 0x41, 'f', 0x42, 'o', 'o', 0xff}, // 55799(_ 'f' 'oo')
			},
		},
		{
			Name: "input enclosed in self-described tag if absent",
			In:   []byte{0x5f, 0x41, 'f', 0x42, 'o', 'o', 0xff}, // (_ 'f' 'oo')
			Want: runtime.RawExtension{
				Raw: []byte{0xd9, 0xd9, 0xf7, 0x5f, 0x41, 'f', 0x42, 'o', 'o', 0xff}, // 55799(_ 'f' 'oo')
			},
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			var got runtime.RawExtension
			if err := got.UnmarshalCBOR(tc.In); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(tc.Want, got); diff != "" {
				t.Errorf("unexpected diff:\n%s", diff)
			}
		})
	}
}
