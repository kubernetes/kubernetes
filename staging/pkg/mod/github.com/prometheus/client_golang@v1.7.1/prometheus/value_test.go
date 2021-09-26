// Copyright 2018 The Prometheus Authors
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

package prometheus

import (
	"fmt"
	"testing"
)

func TestNewConstMetricInvalidLabelValues(t *testing.T) {
	testCases := []struct {
		desc   string
		labels Labels
	}{
		{
			desc:   "non utf8 label value",
			labels: Labels{"a": "\xFF"},
		},
		{
			desc:   "not enough label values",
			labels: Labels{},
		},
		{
			desc:   "too many label values",
			labels: Labels{"a": "1", "b": "2"},
		},
	}

	for _, test := range testCases {
		metricDesc := NewDesc(
			"sample_value",
			"sample value",
			[]string{"a"},
			Labels{},
		)

		expectPanic(t, func() {
			MustNewConstMetric(metricDesc, CounterValue, 0.3, "\xFF")
		}, fmt.Sprintf("WithLabelValues: expected panic because: %s", test.desc))

		if _, err := NewConstMetric(metricDesc, CounterValue, 0.3, "\xFF"); err == nil {
			t.Errorf("NewConstMetric: expected error because: %s", test.desc)
		}
	}
}
