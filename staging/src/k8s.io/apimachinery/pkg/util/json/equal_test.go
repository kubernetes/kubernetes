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
	"bytes"
	"testing"
)

func TestEqualWhenNormalized(t *testing.T) {
	for _, tc := range []struct {
		name, a, b string
		equal      bool
	}{
		{"identical invalid", `{"a":`, `{"a":`, true},
		{"identical duplicate", `{"a":1,"a":2}`, `{"a":1,"a":2}`, true},
		{"empty", "", "", true},
		{"empty and null", "", "null", false},
		{"escaped string", `{"a":"\u0062"}`, `{"a":"b"}`, true},
		{"escaped key", `{"\u0061":1}`, `{"a":1}`, true},
		{"identical", `{"a":1}`, `{"a":1}`, true},
		{"whitespace", `{"a": 1}`, `{"a":1}`, true},
		{"newlines and tabs", "{\n\t\"a\": 1,\r\n\t\"b\": 2\n}", `{"b":2,"a":1}`, true},
		{"object order", `{"z":1,"a":"b"}`, `{"a":"b","z":1}`, true},
		{"nested order", `{"a":[{"z":1,"b":2}]}`, `{ "a": [ {"b":2,"z":1} ] }`, true},
		{"array order", `{"a":[1,2]}`, `{"a":[2,1]}`, false},
		{"value", `{"a":1}`, `{"a":2}`, false},
		{"type", `{"a":"1"}`, `{"a":1}`, false},
		{"missing and null", `{}`, `{"a":null}`, false},
		{"large integer equal", `{"a": 18446744073709551615}`, `{"a":18446744073709551615}`, true},
		{"large integer changed", `{"a":18446744073709551615}`, `{"a":18446744073709551614}`, false},
		{"precise decimal changed", `{"a":0.12345678901234567891}`, `{"a":0.12345678901234567892}`, false},
		{"number spelling", `{"a":1}`, `{"a":1.0}`, false},
		{"exponent spelling", `{"a":1e0}`, `{"a":1}`, false},
		{"negative zero", `{"a":-0}`, `{"a":0}`, false},
		{"invalid old", `{"a":1`, `{"a":1}`, false},
		{"invalid new", `{"a":1}`, `{"a":1`, false},
		{"trailing value", `{"a":1}`, `{"a":1} {}`, false},
		{"duplicate whitespace", `{"a": 1, "a": 2}`, `{"a":1,"a":2}`, true},
		{"duplicate object order", `{"b":3,"a":1,"a":2}`, `{"a":1,"a":2,"b":3}`, true},
		{"duplicate value order", `{"a":1,"a":2}`, `{"a":2,"a":1}`, true},
		{"duplicate nested whitespace", `{"a":{"b": 1, "b": 2}}`, `{"a":{"b":1,"b":2}}`, true},
		{"escaped duplicate names", `{"a":1,"\u0061":2}`, `{"a":1,"a":2}`, true},
		{"duplicate nested value order", `{"a":{"b":1,"b":2}}`, `{"a":{"b":2,"b":1}}`, true},
		{"duplicate object values", `{"a":{"z":1,"b":2},"a":3}`, `{"a":{"b":2,"z":1},"a":3}`, true},
		{"duplicate array values", `[{"a":1,"a":2},{"z":1,"b":2}]`, `[{"a":1,"a":2},{"b":2,"z":1}]`, true},
		{"duplicate array order", `{"a":[1,2],"a":3}`, `{"a":[2,1],"a":3}`, false},
		{"duplicate number spelling", `{"a":1,"a":2}`, `{"a":1.0,"a":2}`, false},
		{"duplicate malformed", `{"a":1,"a":2,}`, `{"a":1,"a":2}`, false},
		{"duplicate trailing value", `{"a":1,"a":2} {}`, `{"a":1,"a":2}`, false},
		{"duplicate escaped keys", `{"<":1,"<":2,"b":3}`, `{"b":3,"\u003c":1,"<":2}`, true},
		{"duplicate old", `{"a":1,"a":2}`, `{"a":2}`, false},
		{"duplicate new", `{"a":2}`, `{"a":1,"a":2}`, false},
		{"duplicate nested", `{"a":{"b":1,"b":2}}`, `{"a":{"b":2}}`, false},
		{"escaped duplicate", `{"a":1,"\u0061":2}`, `{"a":2}`, false},
		{"invalid utf8", "{\"a\":\"\xff\"}", `{"a":"�"}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b := []byte(tc.a), []byte(tc.b)
			if got := EqualWhenNormalized(a, b); got != tc.equal {
				t.Errorf("EqualWhenNormalized(%q, %q) = %t, want %t", a, b, got, tc.equal)
			}
			if got := EqualWhenNormalized(b, a); got != tc.equal {
				t.Errorf("EqualWhenNormalized(%q, %q) = %t, want %t", b, a, got, tc.equal)
			}
			if !bytes.Equal(a, []byte(tc.a)) || !bytes.Equal(b, []byte(tc.b)) {
				t.Fatal("EqualWhenNormalized modified its inputs")
			}
		})
	}
}

func FuzzEqualWhenNormalized(f *testing.F) {
	f.Add(`{"a":1,"a":2}`, `{"a":2,"a":1}`)
	f.Add(`{"a":{"b":1,"b":2}}`, `{"a":{"b": 1, "b": 2}}`)
	f.Add("{\"a\":\"\xff\"}", `{"a":"�"}`)
	f.Fuzz(func(t *testing.T, a, b string) {
		x, y := []byte(a), []byte(b)
		if EqualWhenNormalized(x, y) != EqualWhenNormalized(y, x) {
			t.Fatal("comparison is not symmetric")
		}
		if !EqualWhenNormalized(x, x) {
			t.Fatal("byte-identical inputs compare unequal")
		}
		if string(x) != a || string(y) != b {
			t.Fatal("comparison modified its inputs")
		}
	})
}
