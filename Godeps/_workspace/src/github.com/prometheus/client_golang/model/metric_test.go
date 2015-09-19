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

import "testing"

func testMetric(t testing.TB) {
	var scenarios = []struct {
		input       Metric
		fingerprint Fingerprint
	}{
		{
			input:       Metric{},
			fingerprint: 14695981039346656037,
		},
		{
			input: Metric{
				"first_name":   "electro",
				"occupation":   "robot",
				"manufacturer": "westinghouse",
			},
			fingerprint: 11310079640881077873,
		},
		{
			input: Metric{
				"x": "y",
			},
			fingerprint: 13948396922932177635,
		},
		{
			input: Metric{
				"a": "bb",
				"b": "c",
			},
			fingerprint: 3198632812309449502,
		},
		{
			input: Metric{
				"a":  "b",
				"bb": "c",
			},
			fingerprint: 5774953389407657638,
		},
	}

	for i, scenario := range scenarios {
		if scenario.fingerprint != scenario.input.Fingerprint() {
			t.Errorf("%d. expected %d, got %d", i, scenario.fingerprint, scenario.input.Fingerprint())
		}
	}
}

func TestMetric(t *testing.T) {
	testMetric(t)
}

func BenchmarkMetric(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testMetric(b)
	}
}

func TestCOWMetric(t *testing.T) {
	testMetric := Metric{
		"to_delete": "test1",
		"to_change": "test2",
	}

	scenarios := []struct {
		fn  func(*COWMetric)
		out Metric
	}{
		{
			fn: func(cm *COWMetric) {
				cm.Delete("to_delete")
			},
			out: Metric{
				"to_change": "test2",
			},
		},
		{
			fn: func(cm *COWMetric) {
				cm.Set("to_change", "changed")
			},
			out: Metric{
				"to_delete": "test1",
				"to_change": "changed",
			},
		},
	}

	for i, s := range scenarios {
		orig := testMetric.Clone()
		cm := &COWMetric{
			Metric: orig,
		}

		s.fn(cm)

		// Test that the original metric was not modified.
		if !orig.Equal(testMetric) {
			t.Fatalf("%d. original metric changed; expected %v, got %v", i, testMetric, orig)
		}

		// Test that the new metric has the right changes.
		if !cm.Metric.Equal(s.out) {
			t.Fatalf("%d. copied metric doesn't contain expected changes; expected %v, got %v", i, s.out, cm.Metric)
		}
	}
}
