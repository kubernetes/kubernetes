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

package json

import (
	jsonv2 "encoding/json/v2"
	"reflect"
	"strings"
	"testing"
)

// Keep the options in sync with Unmarshal: both must decode every value alike.
func TestCaseSensitivePreserveIntsV2Options(t *testing.T) {
	for _, tc := range []struct{ name, in string }{
		{"null", `null`},
		{"scalars", `[true, "s", null]`},
		{"integers", `[0, -0, 1, -1, 9223372036854775807, -9223372036854775808]`},
		{"integers out of int64 range", `[9223372036854775808, -9223372036854775809, 18446744073709551616]`},
		{"fractions and exponents", `[1.5, -1.5, 0.1, 1.0, 1e3, 1E3, 1e-3, 100e-2, 1e308, 5e-324]`},
		{"nested", `{"a": 1, "b": 2.5, "c": [1, 2, {"d": 9223372036854775807}], "e": null}`},
		{"keys differing in case", `{"a": 1, "A": 2}`},
		{"repeated key", `{"a": {"b": 1}, "a": {"c": 2}}`},
		{"escapes", `{"s": "\u00e9\ud83d\ude00\n\"\\"}`},
		{"invalid UTF-8", "\"a\xffb\""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var want, got interface{}
			if err := Unmarshal([]byte(tc.in), &want); err != nil {
				t.Fatal(err)
			}
			if err := jsonv2.Unmarshal([]byte(tc.in), &got, CaseSensitivePreserveIntsV2Options()); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("got  %#v\nwant %#v", got, want)
			}
		})
	}
}

func TestCaseSensitivePreserveIntsV2OptionsStructFields(t *testing.T) {
	var got struct {
		Name string `json:"name"`
	}
	if err := jsonv2.Unmarshal([]byte(`{"Name": "x", "NAME": "y"}`), &got, CaseSensitivePreserveIntsV2Options()); err != nil {
		t.Fatal(err)
	}
	if got.Name != "" {
		t.Errorf("matched a key that differs in case: %q", got.Name)
	}
}

func TestCaseSensitivePreserveIntsV2OptionsErrors(t *testing.T) {
	for _, tc := range []struct{ name, in string }{
		{"number out of float64 range", `[1e400]`},
		{"negative number out of float64 range", `{"a": -1e400}`},
		{"depth limit", strings.Repeat("[", 20000) + strings.Repeat("]", 20000)},
		{"syntax", `{"a": }`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var v interface{}
			if err := Unmarshal([]byte(tc.in), &v); err == nil {
				t.Fatal("Unmarshal: expected an error")
			}
			if err := jsonv2.Unmarshal([]byte(tc.in), &v, CaseSensitivePreserveIntsV2Options()); err == nil {
				t.Error("expected an error")
			}
		})
	}

	// Deep nesting under the limit decodes.
	deep := strings.Repeat("[", 9000) + strings.Repeat("]", 9000)
	var v interface{}
	if err := jsonv2.Unmarshal([]byte(deep), &v, CaseSensitivePreserveIntsV2Options()); err != nil {
		t.Errorf("9000 levels: %v", err)
	}
}
