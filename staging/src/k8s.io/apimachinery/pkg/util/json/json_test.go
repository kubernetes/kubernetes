//go:build go1.8
// +build go1.8

/*
Copyright 2015 The Kubernetes Authors.

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
	gojson "encoding/json"

	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestEvaluateTypes(t *testing.T) {
	testCases := []struct {
		In   string
		Data interface{}
		Out  string
		Err  bool
	}{
		// Invalid syntaxes
		{
			In:  `x`,
			Err: true,
		},
		{
			In:  ``,
			Err: true,
		},

		// Null
		{
			In:   `null`,
			Data: nil,
			Out:  `null`,
		},
		// Booleans
		{
			In:   `true`,
			Data: true,
			Out:  `true`,
		},
		{
			In:   `false`,
			Data: false,
			Out:  `false`,
		},

		// Integers
		{
			In:   `0`,
			Data: int64(0),
			Out:  `0`,
		},
		{
			In:   `-0`,
			Data: int64(-0),
			Out:  `0`,
		},
		{
			In:   `1`,
			Data: int64(1),
			Out:  `1`,
		},
		{
			In:   `2147483647`,
			Data: int64(math.MaxInt32),
			Out:  `2147483647`,
		},
		{
			In:   `-2147483648`,
			Data: int64(math.MinInt32),
			Out:  `-2147483648`,
		},
		{
			In:   `9223372036854775807`,
			Data: int64(math.MaxInt64),
			Out:  `9223372036854775807`,
		},
		{
			In:   `-9223372036854775808`,
			Data: int64(math.MinInt64),
			Out:  `-9223372036854775808`,
		},

		// Int overflow
		{
			In:   `9223372036854775808`, // MaxInt64 + 1
			Data: float64(9223372036854775808),
			Out:  `9223372036854776000`,
		},
		{
			In:   `-9223372036854775809`, // MinInt64 - 1
			Data: float64(math.MinInt64),
			Out:  `-9223372036854776000`,
		},

		// Floats
		{
			In:   `0.0`,
			Data: float64(0),
			Out:  `0`,
		},
		{
			In:   `-0.0`,
			Data: float64(-0.0), //nolint:staticcheck // SA4026: in Go, the floating-point literal '-0.0' is the same as '0.0'
			Out:  `-0`,
		},
		{
			In:   `0.5`,
			Data: float64(0.5),
			Out:  `0.5`,
		},
		{
			In:   `1e3`,
			Data: float64(1e3),
			Out:  `1000`,
		},
		{
			In:   `1.5`,
			Data: float64(1.5),
			Out:  `1.5`,
		},
		{
			In:   `-0.3`,
			Data: float64(-.3),
			Out:  `-0.3`,
		},
		{
			// Largest representable float32
			In:   `3.40282346638528859811704183484516925440e+38`,
			Data: float64(math.MaxFloat32),
			Out:  strconv.FormatFloat(math.MaxFloat32, 'g', -1, 64),
		},
		{
			// Smallest float32 without losing precision
			In:   `1.175494351e-38`,
			Data: float64(1.175494351e-38),
			Out:  `1.175494351e-38`,
		},
		{
			// float32 closest to zero
			In:   `1.401298464324817070923729583289916131280e-45`,
			Data: float64(math.SmallestNonzeroFloat32),
			Out:  strconv.FormatFloat(math.SmallestNonzeroFloat32, 'g', -1, 64),
		},
		{
			// Largest representable float64
			In:   `1.797693134862315708145274237317043567981e+308`,
			Data: float64(math.MaxFloat64),
			Out:  strconv.FormatFloat(math.MaxFloat64, 'g', -1, 64),
		},
		{
			// Closest to zero without losing precision
			In:   `2.2250738585072014e-308`,
			Data: float64(2.2250738585072014e-308),
			Out:  `2.2250738585072014e-308`,
		},

		{
			// float64 closest to zero
			In:   `4.940656458412465441765687928682213723651e-324`,
			Data: float64(math.SmallestNonzeroFloat64),
			Out:  strconv.FormatFloat(math.SmallestNonzeroFloat64, 'g', -1, 64),
		},

		{
			// math.MaxFloat64 + 2 overflow
			In:  `1.7976931348623159e+308`,
			Err: true,
		},

		// Strings
		{
			In:   `""`,
			Data: string(""),
			Out:  `""`,
		},
		{
			In:   `"0"`,
			Data: string("0"),
			Out:  `"0"`,
		},
		{
			In:   `"A"`,
			Data: string("A"),
			Out:  `"A"`,
		},
		{
			In:   `"Iñtërnâtiônàlizætiøn"`,
			Data: string("Iñtërnâtiônàlizætiøn"),
			Out:  `"Iñtërnâtiônàlizætiøn"`,
		},

		// Arrays
		{
			In:   `[]`,
			Data: []interface{}{},
			Out:  `[]`,
		},
		{
			In: `[` + strings.Join([]string{
				`null`,
				`true`,
				`false`,
				`0`,
				`9223372036854775807`,
				`0.0`,
				`0.5`,
				`1.0`,
				`1.797693134862315708145274237317043567981e+308`,
				`"0"`,
				`"A"`,
				`"Iñtërnâtiônàlizætiøn"`,
				`[null,true,1,1.0,1.5]`,
				`{"boolkey":true,"floatkey":1.0,"intkey":1,"nullkey":null}`,
			}, ",") + `]`,
			Data: []interface{}{
				nil,
				true,
				false,
				int64(0),
				int64(math.MaxInt64),
				float64(0.0),
				float64(0.5),
				float64(1.0),
				float64(math.MaxFloat64),
				string("0"),
				string("A"),
				string("Iñtërnâtiônàlizætiøn"),
				[]interface{}{nil, true, int64(1), float64(1.0), float64(1.5)},
				map[string]interface{}{"nullkey": nil, "boolkey": true, "intkey": int64(1), "floatkey": float64(1.0)},
			},
			Out: `[` + strings.Join([]string{
				`null`,
				`true`,
				`false`,
				`0`,
				`9223372036854775807`,
				`0`,
				`0.5`,
				`1`,
				strconv.FormatFloat(math.MaxFloat64, 'g', -1, 64),
				`"0"`,
				`"A"`,
				`"Iñtërnâtiônàlizætiøn"`,
				`[null,true,1,1,1.5]`,
				`{"boolkey":true,"floatkey":1,"intkey":1,"nullkey":null}`, // gets alphabetized by Marshal
			}, ",") + `]`,
		},

		// Maps
		{
			In:   `{}`,
			Data: map[string]interface{}{},
			Out:  `{}`,
		},
		{
			In:   `{"boolkey":true,"floatkey":1.0,"intkey":1,"nullkey":null}`,
			Data: map[string]interface{}{"nullkey": nil, "boolkey": true, "intkey": int64(1), "floatkey": float64(1.0)},
			Out:  `{"boolkey":true,"floatkey":1,"intkey":1,"nullkey":null}`, // gets alphabetized by Marshal
		},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("%d_map", i), func(t *testing.T) {
			// decode the input as a map item
			inputJSON := fmt.Sprintf(`{"data":%s}`, tc.In)
			expectedJSON := fmt.Sprintf(`{"data":%s}`, tc.Out)
			m := map[string]interface{}{}
			err := Unmarshal([]byte(inputJSON), &m)
			if tc.Err && err != nil {
				// Expected error
				return
			}
			if err != nil {
				t.Fatalf("%s: error decoding: %v", tc.In, err)
			}
			if tc.Err {
				t.Fatalf("%s: expected error, got none", tc.In)
			}
			data, ok := m["data"]
			if !ok {
				t.Fatalf("%s: decoded object missing data key: %#v", tc.In, m)
			}
			if !reflect.DeepEqual(tc.Data, data) {
				t.Fatalf("%s: expected\n\t%#v (%v), got\n\t%#v (%v)", tc.In, tc.Data, reflect.TypeOf(tc.Data), data, reflect.TypeOf(data))
			}

			outputJSON, err := Marshal(m)
			if err != nil {
				t.Fatalf("%s: error encoding: %v", tc.In, err)
			}

			if expectedJSON != string(outputJSON) {
				t.Fatalf("%s: expected\n\t%s, got\n\t%s", tc.In, expectedJSON, string(outputJSON))
			}
		})

		t.Run(fmt.Sprintf("%d_slice", i), func(t *testing.T) {
			// decode the input as an array item
			inputJSON := fmt.Sprintf(`[0,%s]`, tc.In)
			expectedJSON := fmt.Sprintf(`[0,%s]`, tc.Out)
			m := []interface{}{}
			err := Unmarshal([]byte(inputJSON), &m)
			if tc.Err && err != nil {
				// Expected error
				return
			}
			if err != nil {
				t.Fatalf("%s: error decoding: %v", tc.In, err)
			}
			if tc.Err {
				t.Fatalf("%s: expected error, got none", tc.In)
			}
			if len(m) != 2 {
				t.Fatalf("%s: decoded object wasn't the right length: %#v", tc.In, m)
			}
			data := m[1]
			if !reflect.DeepEqual(tc.Data, data) {
				t.Fatalf("%s: expected\n\t%#v (%v), got\n\t%#v (%v)", tc.In, tc.Data, reflect.TypeOf(tc.Data), data, reflect.TypeOf(data))
			}

			outputJSON, err := Marshal(m)
			if err != nil {
				t.Fatalf("%s: error encoding: %v", tc.In, err)
			}

			if expectedJSON != string(outputJSON) {
				t.Fatalf("%s: expected\n\t%s, got\n\t%s", tc.In, expectedJSON, string(outputJSON))
			}
		})

		t.Run(fmt.Sprintf("%d_raw", i), func(t *testing.T) {
			// decode the input as a standalone object
			inputJSON := fmt.Sprintf(`%s`, tc.In)
			expectedJSON := fmt.Sprintf(`%s`, tc.Out)
			var m interface{}
			err := Unmarshal([]byte(inputJSON), &m)
			if tc.Err && err != nil {
				// Expected error
				return
			}
			if err != nil {
				t.Fatalf("%s: error decoding: %v", tc.In, err)
			}
			if tc.Err {
				t.Fatalf("%s: expected error, got none", tc.In)
			}
			data := m
			if !reflect.DeepEqual(tc.Data, data) {
				t.Fatalf("%s: expected\n\t%#v (%v), got\n\t%#v (%v)", tc.In, tc.Data, reflect.TypeOf(tc.Data), data, reflect.TypeOf(data))
			}

			outputJSON, err := Marshal(m)
			if err != nil {
				t.Fatalf("%s: error encoding: %v", tc.In, err)
			}

			if expectedJSON != string(outputJSON) {
				t.Fatalf("%s: expected\n\t%s, got\n\t%s", tc.In, expectedJSON, string(outputJSON))
			}
		})
	}
}

func TestUnmarshalNil(t *testing.T) {
	{
		var v *interface{}
		err := Unmarshal([]byte(`0`), v)
		goerr := gojson.Unmarshal([]byte(`0`), v)
		if err == nil || goerr == nil || err.Error() != goerr.Error() {
			t.Fatalf("expected error matching stdlib, got %v, %v", err, goerr)
		} else {
			t.Log(err)
		}
	}

	{
		var v *[]interface{}
		err := Unmarshal([]byte(`[]`), v)
		goerr := gojson.Unmarshal([]byte(`[]`), v)
		if err == nil || goerr == nil || err.Error() != goerr.Error() {
			t.Fatalf("expected error matching stdlib, got %v, %v", err, goerr)
		} else {
			t.Log(err)
		}
	}

	{
		var v *map[string]interface{}
		err := Unmarshal([]byte(`{}`), v)
		goerr := gojson.Unmarshal([]byte(`{}`), v)
		if err == nil || goerr == nil || err.Error() != goerr.Error() {
			t.Fatalf("expected error matching stdlib, got %v, %v", err, goerr)
		} else {
			t.Log(err)
		}
	}
}
