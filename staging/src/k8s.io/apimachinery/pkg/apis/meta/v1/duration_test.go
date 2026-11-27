/*
Copyright 2014 The Kubernetes Authors.

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

package v1

import (
	"math"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/internal/encodingtest"
)

func TestDurationMarshalYAML(t *testing.T) {
	//nolint:govet // Unkeyed fields are fine for test cases.
	cases := []encodingtest.MarshalCase[Duration]{
		{Duration{5 * time.Second}, "5s\n"},
		{Duration{2 * time.Minute}, "2m0s\n"},
		{Duration{time.Hour + 3*time.Millisecond}, "1h0m0.003s\n"},
	}

	encodingtest.MarshalYAML(t, cases)
}

func TestDurationUnmarshalYAML(t *testing.T) {
	//nolint:govet // Unkeyed fields are fine for test cases.
	cases := []encodingtest.UnmarshalCase[Duration]{
		{"0s\n", Duration{}},
		{"5s\n", Duration{5 * time.Second}},
		{"2m0s\n", Duration{2 * time.Minute}},
		{"1h0m0.003s\n", Duration{time.Hour + 3*time.Millisecond}},

		// Units with zero values can optionally be dropped
		{"2m\n", Duration{2 * time.Minute}},
		{"1h0.003s\n", Duration{time.Hour + 3*time.Millisecond}},
	}

	encodingtest.UnmarshalYAML(t, cases)
}

func durationMarshalJSONCases() []encodingtest.MarshalCase[Duration] {
	//nolint:govet // Unkeyed fields are fine for test cases.
	return []encodingtest.MarshalCase[Duration]{
		{Duration{5 * time.Second}, `"5s"`},
		{Duration{2 * time.Minute}, `"2m0s"`},
		{Duration{time.Hour + 3*time.Millisecond}, `"1h0m0.003s"`},
		{Duration{}, `"0s"`},
		{Duration{-90 * time.Second}, `"-1m30s"`},
		{Duration{1}, `"1ns"`},
		{Duration{math.MaxInt64}, `"2562047h47m16.854775807s"`},
		{Duration{math.MinInt64}, `"-2562047h47m16.854775808s"`},
	}
}

func TestDurationMarshalJSON(t *testing.T) {
	encodingtest.MarshalJSON(t, durationMarshalJSONCases())
}

func durationUnmarshalJSONCases() []encodingtest.UnmarshalCase[Duration] {
	//nolint:govet // Unkeyed fields are fine for test cases.
	return []encodingtest.UnmarshalCase[Duration]{
		{`"0s"`, Duration{}},
		{`"5s"`, Duration{5 * time.Second}},
		{`"2m0s"`, Duration{2 * time.Minute}},
		{`"1h0m0.003s"`, Duration{time.Hour + 3*time.Millisecond}},

		// Units with zero values can optionally be dropped
		{`"2m"`, Duration{2 * time.Minute}},
		{`"1h0.003s"`, Duration{time.Hour + 3*time.Millisecond}},
		{`"-1m30s"`, Duration{-90 * time.Second}},
		{`"1.5h"`, Duration{90 * time.Minute}},
	}
}

func TestDurationUnmarshalJSON(t *testing.T) {
	encodingtest.UnmarshalJSON(t, durationUnmarshalJSONCases())
	encodingtest.UnmarshalJSONErrors[Duration](t, []string{
		`null`,
		`5`,
		`true`,
		`{}`,
		`""`,
		`"5"`,
		`"1d"`,
		`"2562048h"`,
	})
}

func BenchmarkDurationUnmarshalJSON(b *testing.B) {
	encodingtest.BenchmarkUnmarshalJSON(b, durationUnmarshalJSONCases())
}

func BenchmarkDurationMarshalJSON(b *testing.B) {
	encodingtest.BenchmarkMarshalJSON(b, durationMarshalJSONCases())
}

func TestDurationMarshalJSONUnmarshalYAML(t *testing.T) {
	inputs := []Duration{
		{},
		{5 * time.Second},
		{2 * time.Minute},
		{time.Hour + 3*time.Millisecond},
	}

	encodingtest.JSONToYAML(t, inputs, nil)
}
