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
	"math/big"
	"reflect"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/google/go-cmp/cmp"
)

func TestEncode(t *testing.T) {
	for _, tc := range []struct {
		name          string
		modes         []cbor.EncMode
		in            interface{}
		want          []byte
		assertOnError func(t *testing.T, e error)
	}{
		{
			name: "all duplicate fields are ignored", // Matches behavior of JSON serializer.
			in: struct {
				A1 int `json:"a"`
				A2 int `json:"a"` //nolint:govet // This is intentional to test that the encoder will not encode two map entries with the same key.
			}{},
			want:          []byte{0xa0}, // {}
			assertOnError: assertNilError,
		},
		{
			name: "only tagged field is considered if any are tagged", // Matches behavior of JSON serializer.
			in: struct {
				A       int
				TaggedA int `json:"A"`
			}{
				A:       1,
				TaggedA: 2,
			},
			want:          []byte{0xa1, 0x41, 0x41, 0x02}, // {"A": 2}
			assertOnError: assertNilError,
		},
		{
			name: "math/big.Int values are rejected",
			in:   big.NewInt(1),
			assertOnError: assertOnConcreteError(func(t *testing.T, got *cbor.UnsupportedTypeError) {
				if want := (&cbor.UnsupportedTypeError{Type: reflect.TypeFor[big.Int]()}); *want != *got {
					t.Errorf("unexpected error, got %#v (%q), want %#v (%q)", got, got.Error(), want, want.Error())
				}
			}),
		},
		{
			name:          "byte array encodes to array of integers",
			in:            [3]byte{0x01, 0x02, 0x03},
			want:          []byte{0x83, 0x01, 0x02, 0x03}, // [1, 2, 3]
			assertOnError: assertNilError,
		},
		{
			name:          "string marshalled to byte string",
			in:            "hello",
			want:          []byte{0x45, 'h', 'e', 'l', 'l', 'o'},
			assertOnError: assertNilError,
		},
		{
			name:          "[]byte marshalled to byte string in expected base64 encoding tag",
			in:            []byte("hello"),
			want:          []byte{0xd6, 0x45, 'h', 'e', 'l', 'l', 'o'},
			assertOnError: assertNilError,
		},
	} {
		encModes := tc.modes
		if len(encModes) == 0 {
			encModes = allEncModes
		}

		for _, encMode := range encModes {
			modeName, ok := encModeNames[encMode]
			if !ok {
				t.Fatal("test case configured to run against unrecognized mode")
			}

			t.Run(fmt.Sprintf("mode=%s/%s", modeName, tc.name), func(t *testing.T) {
				out, err := encMode.Marshal(tc.in)
				tc.assertOnError(t, err)
				if diff := cmp.Diff(tc.want, out, cmp.Comparer(func(a, b reflect.Type) bool { return a == b })); diff != "" {
					t.Errorf("unexpected output:\n%s", diff)
				}
			})
		}
	}
}
