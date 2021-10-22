// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
