/*
Copyright 2024 The Kubernetes Authors.

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

package modes_test

import (
	"encoding/base64"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/fxamacker/cbor/v2"
	"github.com/google/go-cmp/cmp"
)

func nilPointerFor[T interface{}]() *T {
	return nil
}

// TestRoundtrip roundtrips object serialization to interface{} and back via CBOR.
func TestRoundtrip(t *testing.T) {
	type modePair struct {
		enc cbor.EncMode
		dec cbor.DecMode
	}

	for _, tc := range []struct {
		name      string
		modePairs []modePair
		obj       interface{}
	}{
		{
			name: "nil slice",
			obj:  []interface{}(nil),
		},
		{
			name: "byte array",
			obj:  [3]byte{0x01, 0x02, 0x03},
		},
		{
			name: "nil map",
			obj:  map[string]interface{}(nil),
		},
		{
			name: "empty slice",
			obj:  []interface{}{},
		},
		{
			name: "empty map",
			obj:  map[string]interface{}{},
		},
		{
			name: "nil pointer to slice",
			obj:  nilPointerFor[[]interface{}](),
		},
		{
			name: "nil pointer to map",
			obj:  nilPointerFor[map[string]interface{}](),
		},
		{
			name: "nonempty string",
			obj:  "hello world",
		},
		{
			name: "empty string",
			obj:  "",
		},
		{
			name: "string containing invalid UTF-8 sequence",
			obj:  "\x80", // first byte is a continuation byte
		},
		{
			name: "true",
			obj:  true,
		},
		{
			name: "false",
			obj:  false,
		},
		{
			name: "int64",
			obj:  int64(5),
		},
		{
			name: "int64 max",
			obj:  int64(math.MaxInt64),
		},
		{
			name: "int64 min",
			obj:  int64(math.MinInt64),
		},
		{
			name: "int64 zero",
			obj:  int64(math.MinInt64),
		},
		{
			name: "uint64 zero",
			obj:  uint64(0),
		},
		{
			name: "int32 max",
			obj:  int32(math.MaxInt32),
		},
		{
			name: "int32 min",
			obj:  int32(math.MinInt32),
		},
		{
			name: "int32 zero",
			obj:  int32(math.MinInt32),
		},
		{
			name: "uint32 max",
			obj:  uint32(math.MaxUint32),
		},
		{
			name: "uint32 zero",
			obj:  uint32(0),
		},
		{
			name: "int16 max",
			obj:  int16(math.MaxInt16),
		},
		{
			name: "int16 min",
			obj:  int16(math.MinInt16),
		},
		{
			name: "int16 zero",
			obj:  int16(math.MinInt16),
		},
		{
			name: "uint16 max",
			obj:  uint16(math.MaxUint16),
		},
		{
			name: "uint16 zero",
			obj:  uint16(0),
		},
		{
			name: "int8 max",
			obj:  int8(math.MaxInt8),
		},
		{
			name: "int8 min",
			obj:  int8(math.MinInt8),
		},
		{
			name: "int8 zero",
			obj:  int8(math.MinInt8),
		},
		{
			name: "uint8 max",
			obj:  uint8(math.MaxUint8),
		},
		{
			name: "uint8 zero",
			obj:  uint8(0),
		},
		{
			name: "float64",
			obj:  float64(2.71),
		},
		{
			name: "float64 max",
			obj:  float64(math.MaxFloat64),
		},
		{
			name: "float64 smallest nonzero",
			obj:  float64(math.SmallestNonzeroFloat64),
		},
		{
			name: "float64 no fractional component",
			obj:  float64(5),
		},
		{
			name: "float32",
			obj:  float32(2.71),
		},
		{
			name: "float32 max",
			obj:  float32(math.MaxFloat32),
		},
		{
			name: "float32 smallest nonzero",
			obj:  float32(math.SmallestNonzeroFloat32),
		},
		{
			name: "float32 no fractional component",
			obj:  float32(5),
		},
		{
			name: "time.Time",
			obj:  time.Date(2222, time.May, 4, 12, 13, 14, 123, time.UTC),
		},
		{
			name: "int64 omitempty",
			obj: struct {
				V int64 `json:"v,omitempty"`
			}{},
		},
		{
			name: "float64 omitempty",
			obj: struct {
				V float64 `json:"v,omitempty"`
			}{},
		},
		{
			name: "string omitempty",
			obj: struct {
				V string `json:"v,omitempty"`
			}{},
		},
		{
			name: "bool omitempty",
			obj: struct {
				V bool `json:"v,omitempty"`
			}{},
		},
		{
			name: "nil pointer omitempty",
			obj: struct {
				V *struct{} `json:"v,omitempty"`
			}{},
		},
		{
			name: "nil pointer to slice as struct field",
			obj: struct {
				V *[]interface{} `json:"v"`
			}{},
		},
		{
			name: "nil pointer to slice as struct field with omitempty",
			obj: struct {
				V *[]interface{} `json:"v,omitempty"`
			}{},
		},
		{
			name: "nil pointer to map as struct field",
			obj: struct {
				V *map[string]interface{} `json:"v"`
			}{},
		},
		{
			name: "nil pointer to map as struct field with omitempty",
			obj: struct {
				V *map[string]interface{} `json:"v,omitempty"`
			}{},
		},
	} {
		modePairs := tc.modePairs
		if len(modePairs) == 0 {
			// Default is all modes to all modes.
			modePairs = []modePair{}
			for _, encMode := range allEncModes {
				for _, decMode := range allDecModes {
					modePairs = append(modePairs, modePair{enc: encMode, dec: decMode})
				}
			}
		}

		for _, modePair := range modePairs {
			encModeName, ok := encModeNames[modePair.enc]
			if !ok {
				t.Fatal("test case configured to run against unrecognized encode mode")
			}

			decModeName, ok := decModeNames[modePair.dec]
			if !ok {
				t.Fatal("test case configured to run against unrecognized decode mode")
			}

			t.Run(fmt.Sprintf("enc=%s/dec=%s/%s", encModeName, decModeName, tc.name), func(t *testing.T) {
				original := tc.obj

				cborFromOriginal, err := modePair.enc.Marshal(original)
				if err != nil {
					t.Fatalf("unexpected error from Marshal of original: %v", err)
				}

				var iface interface{}
				if err := modePair.dec.Unmarshal(cborFromOriginal, &iface); err != nil {
					t.Fatalf("unexpected error from Unmarshal into %T: %v", &iface, err)
				}

				cborFromIface, err := modePair.enc.Marshal(iface)
				if err != nil {
					t.Fatalf("unexpected error from Marshal of iface: %v", err)
				}

				{
					// interface{} to interface{}
					var iface2 interface{}
					if err := modePair.dec.Unmarshal(cborFromIface, &iface2); err != nil {
						t.Fatalf("unexpected error from Unmarshal into %T: %v", &iface2, err)
					}
					if diff := cmp.Diff(iface, iface2); diff != "" {
						t.Errorf("unexpected difference on roundtrip from interface{} to interface{}:\n%s", diff)
					}
				}

				{
					// original to original
					final := reflect.New(reflect.TypeOf(original))
					err = modePair.dec.Unmarshal(cborFromOriginal, final.Interface())
					if err != nil {
						t.Fatalf("unexpected error from Unmarshal into %T: %v", final.Interface(), err)
					}
					if diff := cmp.Diff(original, final.Elem().Interface()); diff != "" {
						t.Errorf("unexpected difference on roundtrip from original to original:\n%s", diff)
					}
				}

				{
					// original to interface{} to original
					finalViaIface := reflect.New(reflect.TypeOf(original))
					err = modePair.dec.Unmarshal(cborFromIface, finalViaIface.Interface())
					if err != nil {
						t.Fatalf("unexpected error from Unmarshal into %T: %v", finalViaIface.Interface(), err)
					}
					if diff := cmp.Diff(original, finalViaIface.Elem().Interface()); diff != "" {
						t.Errorf("unexpected difference on roundtrip from original to interface{} to original:\n%s", diff)
					}
				}
			})
		}
	}
}

// TestRoundtripTextEncoding exercises roundtrips between []byte and string.
func TestRoundtripTextEncoding(t *testing.T) {
	for _, encMode := range allEncModes {
		for _, decMode := range allDecModes {
			t.Run(fmt.Sprintf("enc=%s/dec=%s/byte slice", encModeNames[encMode], decModeNames[decMode]), func(t *testing.T) {
				original := []byte("foo")

				c, err := encMode.Marshal(original)
				if err != nil {
					t.Fatal(err)
				}

				var unstructured interface{}
				if err := decMode.Unmarshal(c, &unstructured); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(base64.StdEncoding.EncodeToString(original), unstructured); diff != "" {
					t.Errorf("[]byte to interface{}: unexpected diff:\n%s", diff)
				}

				var s string
				if err := decMode.Unmarshal(c, &s); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(base64.StdEncoding.EncodeToString(original), s); diff != "" {
					t.Errorf("[]byte to string: unexpected diff:\n%s", diff)
				}

				var final []byte
				if err := decMode.Unmarshal(c, &final); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(original, final); diff != "" {
					t.Errorf("[]byte to []byte: unexpected diff:\n%s", diff)
				}
			})

			t.Run(fmt.Sprintf("enc=%s/dec=%s/string", encModeNames[encMode], decModeNames[decMode]), func(t *testing.T) {
				decoded := "foo"
				original := base64.StdEncoding.EncodeToString([]byte(decoded)) // "Zm9v"

				c, err := encMode.Marshal(original)
				if err != nil {
					t.Fatal(err)
				}

				var unstructured interface{}
				if err := decMode.Unmarshal(c, &unstructured); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(original, unstructured); diff != "" {
					t.Errorf("string to interface{}: unexpected diff:\n%s", diff)
				}

				var b []byte
				if err := decMode.Unmarshal(c, &b); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff([]byte(decoded), b); diff != "" {
					t.Errorf("string to []byte: unexpected diff:\n%s", diff)
				}

				var final string
				if err := decMode.Unmarshal(c, &final); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(original, final); diff != "" {
					t.Errorf("string to string: unexpected diff:\n%s", diff)
				}
			})
		}
	}
}
