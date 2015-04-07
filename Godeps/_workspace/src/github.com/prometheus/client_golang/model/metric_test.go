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
			fingerprint: 2676020557754725067,
		},
		{
			input: Metric{
				"first_name":   "electro",
				"occupation":   "robot",
				"manufacturer": "westinghouse",
			},
			fingerprint: 13260944541294022935,
		},
		{
			input: Metric{
				"x": "y",
			},
			fingerprint: 1470933794305433534,
		},
		// The following two demonstrate a bug in fingerprinting. They
		// should not have the same fingerprint with a sane
		// fingerprinting function. See
		// https://github.com/prometheus/client_golang/issues/74 .
		{
			input: Metric{
				"a": "bb",
				"b": "c",
			},
			fingerprint: 3734646176939799877,
		},
		{
			input: Metric{
				"a":  "b",
				"bb": "c",
			},
			fingerprint: 3734646176939799877,
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
