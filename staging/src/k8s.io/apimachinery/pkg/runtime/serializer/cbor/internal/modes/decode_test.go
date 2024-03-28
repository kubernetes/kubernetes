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
	"math"
	"reflect"
	"strings"
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

	// Testcases that intend only to assert an error can use this type in the want field of the testcase struct
	// Comparison of the decoded output against the want field will be skipped if a value of this type is present in the testcase
	type UncheckedValue struct{}

	type testcase struct {
		name          string
		modes         []cbor.DecMode
		in            []byte
		into          interface{} // prototype for concrete destination type. if nil, decode into empty interface value.
		want          interface{}
		assertOnError func(t *testing.T, e error)
		failureReason string
	}

	testcases := []testcase{
		{
			name:          "reject duplicate negative int keys into struct",
			modes:         []cbor.DecMode{modes.DecodeLax},
			in:            hex("a220012002"), // {-1: 1, -1: 2}
			into:          struct{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(-1), Index: 1}),
		},
		{
			name:          "reject duplicate negative int keys into map",
			in:            hex("a220012002"), // {-1: 1, -1: 2}
			into:          map[int64]interface{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(-1), Index: 1}),
		},
		{
			name:          "reject duplicate positive int keys into struct",
			modes:         []cbor.DecMode{modes.DecodeLax},
			in:            hex("a201010102"), // {1: 1, 1: 2}
			into:          struct{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(1), Index: 1}),
		},
		{
			name:          "reject duplicate positive int keys into map",
			in:            hex("a201010102"), // {1: 1, 1: 2}
			into:          map[int64]interface{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: int64(1), Index: 1}),
		},
		{
			name: "reject duplicate text string keys into struct",
			in:   hex("a2614101614102"), // {"A": 1, "A": 2}
			into: struct {
				A int `json:"A"`
			}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name:          "reject duplicate text string keys into map",
			in:            hex("a2614101614102"), // {"A": 1, "A": 2}
			into:          map[string]interface{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name:          "reject duplicate byte string keys into map",
			in:            hex("a2414101414102"), // {'A': 1, 'A': 2}
			into:          map[string]interface{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name: "reject duplicate byte string keys into struct",
			in:   hex("a2414101414102"), // {'A': 1, 'A': 2}
			into: struct {
				A int `json:"A"`
			}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name:          "reject duplicate byte string and text string keys into map",
			in:            hex("a2414101614102"), // {'A': 1, "A": 2}
			into:          map[string]interface{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name: "reject duplicate byte string and text string keys into struct",
			in:   hex("a2414101614102"), // {'A': 1, "A": 2}
			into: struct {
				A int `json:"A"`
			}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("A"), Index: 1}),
		},
		{
			name: "reject two identical indefinite-length byte string keys split into chunks differently into struct",
			in:   hex("a25f426865436c6c6fff015f416844656c6c6fff02"), // {(_ 'he', 'llo'): 1, (_ 'h', 'ello'): 2}
			into: struct {
				Hello int `json:"hello"`
			}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name:          "reject two identical indefinite-length byte string keys split into chunks differently into map",
			in:            hex("a25f426865436c6c6fff015f416844656c6c6fff02"), // {(_ 'he', 'llo'): 1, (_ 'h', 'ello'): 2}
			into:          map[string]interface{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name: "reject two identical indefinite-length text string keys split into chunks differently into struct",
			in:   hex("a27f626865636c6c6fff017f616864656c6c6fff02"), // {(_ "he", "llo"): 1, (_ "h", "ello"): 2}
			into: struct {
				Hello int `json:"hello"`
			}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name:          "reject two identical indefinite-length text string keys split into chunks differently into map",
			modes:         []cbor.DecMode{modes.DecodeLax},
			in:            hex("a27f626865636c6c6fff017f616864656c6c6fff02"), // {(_ "he", "llo"): 1, (_ "h", "ello"): 2}
			into:          map[string]interface{}{},
			want:          UncheckedValue{},
			assertOnError: assertIdenticalError(&cbor.DupMapKeyError{Key: string("hello"), Index: 1}),
		},
		{
			name:  "case-insensitive match treated as unknown field",
			modes: []cbor.DecMode{modes.Decode},
			in:    hex("a1614101"), // {"A": 1}
			into: struct {
				A int `json:"a"`
			}{},
			want:          UncheckedValue{},
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
			want:          UncheckedValue{},
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
			want:  struct{}{},
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
		{
			name:          "int64 minimum positive value",
			in:            hex("00"), // 0
			want:          int64(0),
			assertOnError: assertNilError,
		},
		{
			name:          "int64 max positive value",
			in:            hex("1b7fffffffffffffff"), // 9223372036854775807
			want:          int64(9223372036854775807),
			assertOnError: assertNilError,
		},
		{
			name:          "int64 max negative value",
			in:            hex("20"), // -1
			want:          int64(-1),
			assertOnError: assertNilError,
		},
		{
			name:          "int64 min negative value",
			in:            hex("3b7fffffffffffffff"), // -9223372036854775808
			want:          int64(-9223372036854775808),
			assertOnError: assertNilError,
		},
		{
			name:          "empty byte string",
			in:            hex("40"), // ''
			want:          "",
			assertOnError: assertNilError,
		},
		{
			name:          "empty text string",
			in:            hex("60"), // ""
			want:          "",
			assertOnError: assertNilError,
		},
		{
			name: "map with different key types",
			in:   hex("a201616161626132"), // { 1: "a", "b": "2"}
			assertOnError: assertOnConcreteError(func(t *testing.T, e *cbor.UnmarshalTypeError) {
				if e.CBORType != "positive integer" || e.GoType != "string" {
					t.Errorf("Expected types: positive integer,string. Got: %#v,%#v", e.CBORType, e.GoType)
				}
			}),
		},
		{
			name: "map with non string key types",
			in:   hex("a1fb40091eb851eb851f63706965"), // { 3.14: "pie"}
			assertOnError: assertOnConcreteError(func(t *testing.T, e *cbor.UnmarshalTypeError) {
				if e.CBORType != "primitives" || e.GoType != "string" {
					t.Errorf("Expected types: primitives,string. Got: %#v,%#v", e.CBORType, e.GoType)
				}
			}),
		},
		{
			name:          "map with byte string key type",
			in:            hex("a143abcdef187b"), // {h'ABCDEF': 123}
			want:          map[string]interface{}{"\xab\xcd\xef": int64(123)},
			assertOnError: assertNilError,
		},
		{
			name:  "max positive integer value supported by cbor: 2^64 - 1",
			modes: []cbor.DecMode{modes.Decode},
			in:    hex("1bffffffffffffffff"), // 18446744073709551615
			assertOnError: assertOnConcreteError(func(t *testing.T, e *cbor.UnmarshalTypeError) {
				if e == nil {
					t.Error("Expected non-nil error. ")
				} else if !strings.Contains(e.Error(), "cbor: cannot unmarshal positive integer into Go value of type int64 (18446744073709551615 overflows Go's int64)") {
					t.Errorf("Expected overflow error. Got: %s", e.Error())
				}
			}),
		},
		{
			name: "min negative integer value supported by cbor: -2^64",
			in:   hex("3bffffffffffffffff"), // -18446744073709551616
			assertOnError: assertOnConcreteError(func(t *testing.T, e *cbor.UnmarshalTypeError) {
				if e == nil {
					t.Error("Expected non-nil error. ")
				} else if !strings.Contains(e.Error(), "cbor: cannot unmarshal negative integer into Go value of type int64 (-18446744073709551616 overflows Go's int64)") {
					t.Errorf("Expected overflow error. Got: %s", e.Error())
				}
			}),
		},
		{
			name:          "tag 0 timestamp string",
			in:            hex("c074323030362d30312d30325431353a30343a30355a"), // 0("2006-01-02T15:04:05Z")
			want:          "2006-01-02T15:04:05Z",
			failureReason: "decoding cbor data tagged with 0 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name: "tag 0 with byte string",
			in:   hex("c043414243"), // 0('ABC')
			assertOnError: func(t *testing.T, e error) {
				if e == nil {
					t.Error("Expected non-nil error. ")
				} else if !strings.Contains(e.Error(), "cbor: tag number 0 must be followed by text string, got byte string") {
					t.Errorf("Expected error: %s, got: %s \n", "cbor: tag number 0 must be followed by text string, got byte string", e.Error())
				}
			},
			want: UncheckedValue{},
		},
		{
			name: "tag 0 with invalid text string",
			in:   hex("c06474657874"), // 0("text")
			assertOnError: func(t *testing.T, e error) {
				if e == nil {
					t.Error("Expected non-nil error. ")
				} else if !strings.Contains(e.Error(), "cbor: cannot set text for time.Time: parsing time") {
					t.Errorf("Expected error: %s, got: %s \n", "cbor: cannot set text for time.Time: parsing time", e.Error())
				}
			},
			want: UncheckedValue{},
		},
		{
			name:          "tag 1 timestamp unsigned integer",
			in:            hex("c11a43b940e5"), // 1(1136214245)
			want:          "2006-01-02T15:04:05Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with float16 value",
			in:            hex("c1f93c00"), // 1(1.0_1)
			want:          "1970-01-01T00:00:01Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with float32 value",
			in:            hex("c1fa3f800000"), // 1(1.0_2)
			want:          "1970-01-01T00:00:01Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with float64 value",
			in:            hex("c1fb3ff0000000000000"), // 1(1.0_3)
			want:          "1970-01-01T00:00:01Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with a four digit year",
			in:            hex("c11b0000003afff44181"), // 1(253402300801)
			want:          "10000-01-01T00:00:01Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with a negative integer value",
			in:            hex("c120"), // 1(-1)
			want:          "1969-12-31T23:59:59Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with a negative float16 value",
			in:            hex("c1f9bc00"), // 1(-1.0_1)
			want:          "1969-12-31T23:59:59Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with a negative float32 value",
			in:            hex("c1fabf800000"), // 1(-1.0_2)
			want:          "1969-12-31T23:59:59Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with a negative float64 value",
			in:            hex("c1fbbff0000000000000"), // 1(-1.0_3)
			want:          "1969-12-31T23:59:59Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with a positive infinity",
			in:            hex("c1f97c00"), // 1(Infinity)
			want:          "0001-01-01T00:00:00Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with a negative infinity",
			in:            hex("c1f9fc00"), // 1(-Infinity)
			want:          "0001-01-01T00:00:00Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 1 with NaN",
			in:            hex("c1f9fc00"), // 1(NaN)
			want:          "0001-01-01T00:00:00Z",
			failureReason: "decoding cbor data tagged with 1 produces time.Time instead of RFC3339 timestamp string",
			assertOnError: assertNilError,
		},
		{
			name: "tag 2 positive big integer",
			// TODO: decide how to handle arbitrary precision numeric tags (integer and floating-point)
			in:            hex("c249010000000000000000"), // 2(18446744073709551616)
			failureReason: "decoding cbor data tagged with 2 produces big.Int instead of []byte",
			assertOnError: assertNilError,
		},
		{
			name: "tag 3 negative big integer",
			// TODO: decide how to handle arbitrary precision numeric tags (integer and floating-point)
			in:            hex("c349010000000000000000"), // 3(-18446744073709551617)
			failureReason: "decoding cbor data tagged with 3 produces big.Int instead of []byte",
			assertOnError: assertNilError,
		},
		{
			name:          "tag 4 decimal fraction",
			in:            hex("c48221196ab3"), // 4([-2, 27315])
			want:          []interface{}{int64(-2), int64(27315)},
			assertOnError: assertNilError,
		},
		{
			name:          "tag 5 Big floats",
			in:            hex("c5822003"), // 5([-1, 3])
			want:          []interface{}{int64(-1), int64(3)},
			assertOnError: assertNilError,
		},
		{
			name:          "simple value 20",
			in:            hex("f4"), // false
			want:          false,
			assertOnError: assertNilError,
		},
		{
			name:          "simple value 21",
			in:            hex("f5"), // true
			want:          true,
			assertOnError: assertNilError,
		},
		{
			name:          "simple value 22",
			in:            hex("f6"), // nil
			want:          nil,
			assertOnError: assertNilError,
		},
		{
			name: "simple value 23",
			in:   hex("f7"), // undefined
			assertOnError: func(t *testing.T, e error) {
				t.Error("todo: test must be updated once it becomes possible to configure this behavior")
			},
			failureReason: "cbor data with simple value 23 should throw an error.",
		},
		{
			name:          "infinity",
			in:            hex("f97c00"),
			want:          math.Inf(1),
			assertOnError: assertNilError,
		},
		{
			name:          "negative infinity",
			in:            hex("f9fc00"),
			want:          math.Inf(-1),
			assertOnError: assertNilError,
		},
		{
			name:          "smallest nonzero float64",
			in:            hex("fb0000000000000001"),
			want:          math.SmallestNonzeroFloat64,
			assertOnError: assertNilError,
		},
		{
			name:          "max float64 value",
			in:            hex("fb7fefffffffffffff"),
			want:          math.MaxFloat64,
			assertOnError: assertNilError,
		},
	}

	// Generate test cases for all simple values (0 to 255) because the number of possible simple values is fixed and small.
	for i := 0; i <= 255; i++ {
		each := testcase{
			name: fmt.Sprintf("simple value %d", i),
			want: nil,
		}
		if i <= 23 {
			each.in = []byte{byte(0xe0) | byte(i)}
		} else {
			// larger simple values encode to two bytes
			each.in = []byte{byte(0xe0) | byte(24), byte(i)}
		}
		switch i {
		case 20, 21, 22, 23: // recognized values with explicit cases
			continue
		case 24, 25, 26, 27, 28, 29, 30, 31: // reserved
			// these are considered not well-formed
			each.assertOnError = assertOnConcreteError(func(t *testing.T, e *cbor.SyntaxError) {
				if e == nil {
					t.Error("Expected non-nil error.")
				} else if !strings.Contains(e.Error(), "cbor: invalid simple value") {
					t.Errorf("Expected error: cbor: invalid simple value, got: %s \n", e.Error())
				}
			})
		default:
			// don't accept unrecognized values
			each.assertOnError = func(t *testing.T, e error) {
				// TODO: Once implemented, this can be a stronger assertion.
				if e == nil {
					t.Error("expected non-nil error")
				}
			}
			each.failureReason = "unrecognized simple values should be rejected"
		}
		testcases = append(testcases, each)
	}

	for _, tc := range testcases {
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

				// testcases that contain tc.failureReason are skipped because these are known failures
				// TODO remove this block when all the known failures are fixed
				if tc.failureReason != "" {
					t.Skip(tc.failureReason)
				}

				var dst reflect.Value
				if tc.into == nil {
					var i interface{}
					dst = reflect.ValueOf(&i)
				} else {
					dst = reflect.New(reflect.TypeOf(tc.into))
				}

				err := decMode.Unmarshal(tc.in, dst.Interface())

				tc.assertOnError(t, err)

				if _, ok := tc.want.(UncheckedValue); !ok {
					if diff := cmp.Diff(tc.want, dst.Elem().Interface()); diff != "" {
						t.Errorf("unexpected output:\n%s", diff)
					}
				}
			})
		}
	}
}
