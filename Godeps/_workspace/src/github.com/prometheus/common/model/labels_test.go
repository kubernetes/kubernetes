// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"sort"
	"testing"
)

func testLabelNames(t testing.TB) {
	var scenarios = []struct {
		in  LabelNames
		out LabelNames
	}{
		{
			in:  LabelNames{"ZZZ", "zzz"},
			out: LabelNames{"ZZZ", "zzz"},
		},
		{
			in:  LabelNames{"aaa", "AAA"},
			out: LabelNames{"AAA", "aaa"},
		},
	}

	for i, scenario := range scenarios {
		sort.Sort(scenario.in)

		for j, expected := range scenario.out {
			if expected != scenario.in[j] {
				t.Errorf("%d.%d expected %s, got %s", i, j, expected, scenario.in[j])
			}
		}
	}
}

func TestLabelNames(t *testing.T) {
	testLabelNames(t)
}

func BenchmarkLabelNames(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testLabelNames(b)
	}
}

func testLabelValues(t testing.TB) {
	var scenarios = []struct {
		in  LabelValues
		out LabelValues
	}{
		{
			in:  LabelValues{"ZZZ", "zzz"},
			out: LabelValues{"ZZZ", "zzz"},
		},
		{
			in:  LabelValues{"aaa", "AAA"},
			out: LabelValues{"AAA", "aaa"},
		},
	}

	for i, scenario := range scenarios {
		sort.Sort(scenario.in)

		for j, expected := range scenario.out {
			if expected != scenario.in[j] {
				t.Errorf("%d.%d expected %s, got %s", i, j, expected, scenario.in[j])
			}
		}
	}
}

func TestLabelValues(t *testing.T) {
	testLabelValues(t)
}

func BenchmarkLabelValues(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testLabelValues(b)
	}
}
