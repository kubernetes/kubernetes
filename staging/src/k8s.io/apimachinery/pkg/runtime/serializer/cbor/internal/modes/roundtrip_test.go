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
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/fxamacker/cbor/v2"
)

func nilPointerFor[T interface{}]() *T {
	return nil
}

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
			name: "uint64 max",
			obj:  uint64(math.MaxUint64),
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
		mps := tc.modePairs
		if len(mps) == 0 {
			// Default is all modes to all modes.
			mps = []modePair{}
			for _, em := range allEncModes {
				for _, dm := range allDecModes {
					mps = append(mps, modePair{enc: em, dec: dm})
				}
			}
		}

		for _, mp := range mps {
			encModeName, ok := encModeNames[mp.enc]
			if !ok {
				t.Fatal("test case configured to run against unrecognized encode mode")
			}

			decModeName, ok := decModeNames[mp.dec]
			if !ok {
				t.Fatal("test case configured to run against unrecognized decode mode")
			}

			t.Run(fmt.Sprintf("enc=%s/dec=%s/%s", encModeName, decModeName, tc.name), func(t *testing.T) {
				original := tc.obj

				b, err := mp.enc.Marshal(original)
				if err != nil {
					t.Fatalf("unexpected error from Marshal: %v", err)
				}

				final := reflect.New(reflect.TypeOf(original))
				err = mp.dec.Unmarshal(b, final.Interface())
				if err != nil {
					t.Fatalf("unexpected error from Unmarshal: %v", err)
				}
				if !reflect.DeepEqual(original, final.Elem().Interface()) {
					t.Errorf("roundtrip difference:\nwant: %#v\ngot: %#v", original, final)
				}
			})
		}
	}
}
