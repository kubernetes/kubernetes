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
		ms := tc.modes
		if len(ms) == 0 {
			ms = allDecModes
		}

		for _, dm := range ms {
			modeName, ok := decModeNames[dm]
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
				err := dm.Unmarshal(tc.in, dst.Interface())
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
