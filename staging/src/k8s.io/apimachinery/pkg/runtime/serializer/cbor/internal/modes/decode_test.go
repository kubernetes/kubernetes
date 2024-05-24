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
	"encoding/hex"
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"

	"github.com/fxamacker/cbor/v2"
	"github.com/google/go-cmp/cmp"
)

func TestDecode(t *testing.T) {
	hex := func(h string) []byte {
		b, err := hex.DecodeString(h)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}

	for _, tc := range []struct {
		name          string
		modes         []cbor.DecMode
		in            []byte
		into          interface{} // prototype for concrete destination type. if nil, decode into empty interface value.
		want          interface{}
		assertOnError func(t *testing.T, e error)
	}{
		{
			name:          "reject duplicate negative int keys into struct",
			modes:         []cbor.DecMode{modes.DecodeLax},
			in:            hex("a220012002"), // {-1: 1, -1: 2}
			into:          struct{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(-1), Index: 1}),
		},
		{
			name:          "reject duplicate negative int keys into map",
			in:            hex("a220012002"), // {-1: 1, -1: 2}
			into:          map[int64]interface{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(-1), Index: 1}),
		},
		{
			name:          "reject duplicate positive int keys into struct",
			modes:         []cbor.DecMode{modes.DecodeLax},
			in:            hex("a201010102"), // {1: 1, 1: 2}
			into:          struct{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(1), Index: 1}),
		},
		{
			name:          "reject duplicate positive int keys into map",
			in:            hex("a201010102"), // {1: 1, 1: 2}
			into:          map[int64]interface{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(1), Index: 1}),
		},
		{
			name: "reject duplicate text string keys into struct",
			in:   hex("a2614101614102"), // {"A": 1, "A": 2}
			into: struct {
				A int `json:"A"`
			}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name:          "reject duplicate text string keys into map",
			in:            hex("a2614101614102"), // {"A": 1, "A": 2}
			into:          map[string]interface{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name:          "reject duplicate byte string keys into map",
			in:            hex("a2414101414102"), // {'A': 1, 'A': 2}
			into:          map[string]interface{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name: "reject duplicate byte string keys into struct",
			in:   hex("a2414101414102"), // {'A': 1, 'A': 2}
			into: struct {
				A int `json:"A"`
			}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name:          "reject duplicate byte string and text string keys into map",
			in:            hex("a2414101614102"), // {'A': 1, "A": 2}
			into:          map[string]interface{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name: "reject duplicate byte string and text string keys into struct",
			in:   hex("a2414101614102"), // {'A': 1, "A": 2}
			into: struct {
				A int `json:"A"`
			}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name: "reject two identical indefinite-length byte string keys split into chunks differently into struct",
			in:   hex("a25f426865436c6c6fff015f416844656c6c6fff02"), // {(_ 'he', 'llo'): 1, (_ 'h', 'ello'): 2}
			into: struct {
				Hello int `json:"hello"`
			}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name:          "reject two identical indefinite-length byte string keys split into chunks differently into map",
			in:            hex("a25f426865436c6c6fff015f416844656c6c6fff02"), // {(_ 'he', 'llo'): 1, (_ 'h', 'ello'): 2}
			into:          map[string]interface{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name: "reject two identical indefinite-length text string keys split into chunks differently into struct",
			in:   hex("a27f626865636c6c6fff017f616864656c6c6fff02"), // {(_ "he", "llo"): 1, (_ "h", "ello"): 2}
			into: struct {
				Hello int `json:"hello"`
			}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name:          "reject two identical indefinite-length text string keys split into chunks differently into map",
			modes:         []cbor.DecMode{modes.DecodeLax},
			in:            hex("a27f626865636c6c6fff017f616864656c6c6fff02"), // {(_ "he", "llo"): 1, (_ "h", "ello"): 2}
			into:          map[string]interface{}{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name:  "case-insensitive match treated as unknown field",
			modes: []cbor.DecMode{modes.Decode},
			in:    hex("a1614101"), // {"A": 1}
			into: struct {
				A int `json:"a"`
			}{},
			assertOnError: assertIdenticalError(&cbor.UnknownFieldError{Index: 0}),
		},
		{
			name:  "case-insensitive match ignored in lax mode",
			modes: []cbor.DecMode{modes.DecodeLax},
			in:    hex("a1614101"), // {"A": 1}
			into: struct {
				A int `json:"a"`
			}{},
			want: struct {
				A int `json:"a"`
			}{
				A: 0,
			},
			assertOnError: assertNilError,
		},
		{
			name:  "case-insensitive match after exact match treated as unknown field",
			modes: []cbor.DecMode{modes.Decode},
			in:    hex("a2616101614102"), // {"a": 1, "A": 2}
			into: struct {
				A int `json:"a"`
			}{},
			want: struct {
				A int `json:"a"`
			}{
				A: 1,
			},
			assertOnError: assertIdenticalError(&cbor.UnknownFieldError{Index: 1}),
		},
		{
			name:  "case-insensitive match after exact match ignored in lax mode",
			modes: []cbor.DecMode{modes.DecodeLax},
			in:    hex("a2616101614102"), // {"a": 1, "A": 2}
			into: struct {
				A int `json:"a"`
			}{},
			want: struct {
				A int `json:"a"`
			}{
				A: 1,
			},
			assertOnError: assertNilError,
		},
		{
			name:  "case-insensitive match before exact match treated as unknown field",
			modes: []cbor.DecMode{modes.Decode},
			in:    hex("a2614101616102"), // {"A": 1, "a": 2}
			into: struct {
				A int `json:"a"`
			}{},
			assertOnError: assertIdenticalError(&cbor.UnknownFieldError{Index: 0}),
		},
		{
			name:  "case-insensitive match before exact match ignored in lax mode",
			modes: []cbor.DecMode{modes.DecodeLax},
			in:    hex("a2614101616102"), // {"A": 1, "a": 2}
			into: struct {
				A int `json:"a"`
			}{},
			want: struct {
				A int `json:"a"`
			}{
				A: 2,
			},
			assertOnError: assertNilError,
		},
		{
			name: "reject text string containing invalid utf-8 sequence",
			in:   hex("6180"), // text string beginning with continuation byte 0x80
			assertOnError: assertOnConcreteError(func(t *testing.T, e *cbor.SemanticError) {
				const expected = "cbor: invalid UTF-8 string"
				if msg := e.Error(); msg != expected {
					t.Errorf("expected %v, got %v", expected, msg)
				}
			}),
		},
		{
			name:          "unsigned integer decodes to interface{} as int64",
			in:            hex("0a"), // 10
			want:          int64(10),
			assertOnError: assertNilError,
		},
		{
			name:  "unknown field error",
			modes: []cbor.DecMode{modes.Decode},
			in:    hex("a1616101"), // {"a": 1}
			into:  struct{}{},
			assertOnError: assertOnConcreteError(func(t *testing.T, e *cbor.UnknownFieldError) {
				if e.Index != 0 {
					t.Errorf("expected %#v, got %#v", &cbor.UnknownFieldError{Index: 0}, e)
				}
			}),
		},
		{
			name:          "no unknown field error in lax mode",
			modes:         []cbor.DecMode{modes.DecodeLax},
			in:            hex("a1616101"), // {"a": 1}
			into:          struct{}{},
			want:          struct{}{},
			assertOnError: assertNilError,
		},
		{
			name:          "indefinite-length text string",
			in:            hex("7f616161626163ff"), // (_ "a", "b", "c")
			want:          "abc",
			assertOnError: assertNilError,
		},
		{
			name: "nested indefinite-length array",
			in:   hex("9f9f8080ff9f8080ffff"), // [_ [_ [] []] [_ [][]]]
			want: []interface{}{
				[]interface{}{[]interface{}{}, []interface{}{}},
				[]interface{}{[]interface{}{}, []interface{}{}},
			},
			assertOnError: assertNilError,
		},
		{
			name: "nested indefinite-length map",
			in:   hex("bf6141bf616101616202ff6142bf616901616a02ffff"), // {_ "A": {_ "a": 1, "b": 2}, "B": {_ "i": 1, "j": 2}}
			want: map[string]interface{}{
				"A": map[string]interface{}{"a": int64(1), "b": int64(2)},
				"B": map[string]interface{}{"i": int64(1), "j": int64(2)},
			},
			assertOnError: assertNilError,
		},
	} {
		decModes := tc.modes
		if len(decModes) == 0 {
			decModes = allDecModes
		}

		for _, decMode := range decModes {
			modeName, ok := decModeNames[decMode]
			if !ok {
				t.Fatal("test case configured to run against unrecognized mode")
			}

			t.Run(fmt.Sprintf("mode=%s/%s", modeName, tc.name), func(t *testing.T) {
				var dst reflect.Value
				if tc.into == nil {
					var i interface{}
					dst = reflect.ValueOf(&i)
				} else {
					dst = reflect.New(reflect.TypeOf(tc.into))
				}
				err := decMode.Unmarshal(tc.in, dst.Interface())
				tc.assertOnError(t, err)
				if tc.want != nil {
					if diff := cmp.Diff(tc.want, dst.Elem().Interface()); diff != "" {
						t.Errorf("unexpected output:\n%s", diff)
					}
				}
			})
		}
	}
}
