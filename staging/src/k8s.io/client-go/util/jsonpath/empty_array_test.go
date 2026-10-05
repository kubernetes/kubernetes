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

package jsonpath

import (
	"bytes"
	"testing"
)

func TestEmptyArrayDoesNotStopFollowingInputs(t *testing.T) {
	cases := []struct {
		name       string
		data       [][]string
		expression string
		want       string
	}{
		{"empty first", [][]string{{}, {"second"}}, "{[*][*]}", "second"},
		{"empty middle", [][]string{{"first"}, {}, {"third"}}, "{[*][*]}", "first third"},
		{"empty last", [][]string{{"first"}, {}}, "{[*][:]}", "first"},
		{"all empty", [][]string{{}, {}}, "{[*][*]}", ""},
		{"empty negative slice", [][]string{{"first"}, {"second", "third"}}, "{[*][0:-1]}", "second"},
		{"no empty slices", [][]string{{"first"}, {"second"}}, "{[*][*]}", "first second"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			j := New(tc.name)
			if err := j.Parse(tc.expression); err != nil {
				t.Fatal(err)
			}
			var out bytes.Buffer
			if err := j.Execute(&out, tc.data); err != nil {
				t.Fatal(err)
			}
			if got := out.String(); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}
