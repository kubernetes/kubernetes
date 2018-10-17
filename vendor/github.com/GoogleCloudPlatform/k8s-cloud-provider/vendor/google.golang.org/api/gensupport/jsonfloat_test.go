// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gensupport

import (
	"encoding/json"
	"math"
	"testing"
)

func TestJSONFloat(t *testing.T) {
	for _, test := range []struct {
		in   string
		want float64
	}{
		{"0", 0},
		{"-10", -10},
		{"1e23", 1e23},
		{`"Infinity"`, math.Inf(1)},
		{`"-Infinity"`, math.Inf(-1)},
		{`"NaN"`, math.NaN()},
	} {
		var f64 JSONFloat64
		if err := json.Unmarshal([]byte(test.in), &f64); err != nil {
			t.Fatal(err)
		}
		got := float64(f64)
		if got != test.want && math.IsNaN(got) != math.IsNaN(test.want) {
			t.Errorf("%s: got %f, want %f", test.in, got, test.want)
		}
	}
}

func TestJSONFloatErrors(t *testing.T) {
	var f64 JSONFloat64
	for _, in := range []string{"", "a", `"Inf"`, `"-Inf"`, `"nan"`, `"nana"`} {
		if err := json.Unmarshal([]byte(in), &f64); err == nil {
			t.Errorf("%q: got nil, want error", in)
		}
	}
}
