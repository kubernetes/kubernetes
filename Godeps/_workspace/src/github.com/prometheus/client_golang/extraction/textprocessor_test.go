// Copyright 2014 The Prometheus Authors
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

package extraction

import (
	"sort"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/model"
)

var (
	ts = model.Now()
	in = `
# Only a quite simple scenario with two metric families.
# More complicated tests of the parser itself can be found in the text package.
# TYPE mf2 counter
mf2 3
mf1{label="value1"} -3.14 123456
mf1{label="value2"} 42
mf2 4
`
	out = map[model.LabelValue]model.Samples{
		"mf1": model.Samples{
			&model.Sample{
				Metric:    model.Metric{model.MetricNameLabel: "mf1", "label": "value1"},
				Value:     -3.14,
				Timestamp: 123456,
			},
			&model.Sample{
				Metric:    model.Metric{model.MetricNameLabel: "mf1", "label": "value2"},
				Value:     42,
				Timestamp: ts,
			},
		},
		"mf2": model.Samples{
			&model.Sample{
				Metric:    model.Metric{model.MetricNameLabel: "mf2"},
				Value:     3,
				Timestamp: ts,
			},
			&model.Sample{
				Metric:    model.Metric{model.MetricNameLabel: "mf2"},
				Value:     4,
				Timestamp: ts,
			},
		},
	}
)

type testIngester struct {
	results []model.Samples
}

func (i *testIngester) Ingest(s model.Samples) error {
	i.results = append(i.results, s)
	return nil
}

func TestTextProcessor(t *testing.T) {
	var ingester testIngester
	i := strings.NewReader(in)
	o := &ProcessOptions{
		Timestamp: ts,
	}

	err := Processor004.ProcessSingle(i, &ingester, o)
	if err != nil {
		t.Fatal(err)
	}
	if expected, got := len(out), len(ingester.results); expected != got {
		t.Fatalf("Expected length %d, got %d", expected, got)
	}
	for _, r := range ingester.results {
		expected, ok := out[r[0].Metric[model.MetricNameLabel]]
		if !ok {
			t.Fatalf(
				"Unexpected metric name %q",
				r[0].Metric[model.MetricNameLabel],
			)
		}
		sort.Sort(expected)
		sort.Sort(r)
		if !expected.Equal(r) {
			t.Errorf("expected %s, got %s", expected, r)
		}
	}
}
