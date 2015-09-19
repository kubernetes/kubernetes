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

package extraction

import (
	"sort"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/model"
)

var testTime = model.Now()

type metricFamilyProcessorScenario struct {
	in               string
	expected, actual []model.Samples
}

func (s *metricFamilyProcessorScenario) Ingest(samples model.Samples) error {
	s.actual = append(s.actual, samples)
	return nil
}

func (s *metricFamilyProcessorScenario) test(t *testing.T, set int) {
	i := strings.NewReader(s.in)

	o := &ProcessOptions{
		Timestamp: testTime,
	}

	err := MetricFamilyProcessor.ProcessSingle(i, s, o)
	if err != nil {
		t.Fatalf("%d. got error: %s", set, err)
	}

	if len(s.expected) != len(s.actual) {
		t.Fatalf("%d. expected length %d, got %d", set, len(s.expected), len(s.actual))
	}

	for i, expected := range s.expected {
		sort.Sort(s.actual[i])
		sort.Sort(expected)

		if !expected.Equal(s.actual[i]) {
			t.Errorf("%d.%d. expected %s, got %s", set, i, expected, s.actual[i])
		}
	}
}

func TestMetricFamilyProcessor(t *testing.T) {
	scenarios := []metricFamilyProcessorScenario{
		{
			in: "",
		},
		{
			in: "\x8f\x01\n\rrequest_count\x12\x12Number of requests\x18\x00\"0\n#\n\x0fsome_label_name\x12\x10some_label_value\x1a\t\t\x00\x00\x00\x00\x00\x00E\xc0\"6\n)\n\x12another_label_name\x12\x13another_label_value\x1a\t\t\x00\x00\x00\x00\x00\x00U@",
			expected: []model.Samples{
				model.Samples{
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_count", "some_label_name": "some_label_value"},
						Value:     -42,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_count", "another_label_name": "another_label_value"},
						Value:     84,
						Timestamp: testTime,
					},
				},
			},
		},
		{
			in: "\xb9\x01\n\rrequest_count\x12\x12Number of requests\x18\x02\"O\n#\n\x0fsome_label_name\x12\x10some_label_value\"(\x1a\x12\t\xaeG\xe1z\x14\xae\xef?\x11\x00\x00\x00\x00\x00\x00E\xc0\x1a\x12\t+\x87\x16\xd9\xce\xf7\xef?\x11\x00\x00\x00\x00\x00\x00U\xc0\"A\n)\n\x12another_label_name\x12\x13another_label_value\"\x14\x1a\x12\t\x00\x00\x00\x00\x00\x00\xe0?\x11\x00\x00\x00\x00\x00\x00$@",
			expected: []model.Samples{
				model.Samples{
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_count", "some_label_name": "some_label_value", "quantile": "0.99"},
						Value:     -42,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_count", "some_label_name": "some_label_value", "quantile": "0.999"},
						Value:     -84,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_count", "another_label_name": "another_label_value", "quantile": "0.5"},
						Value:     10,
						Timestamp: testTime,
					},
				},
			},
		},
		{
			in: "\x8d\x01\n\x1drequest_duration_microseconds\x12\x15The response latency.\x18\x04\"S:Q\b\x85\x15\x11\xcd\xcc\xccL\x8f\xcb:A\x1a\v\b{\x11\x00\x00\x00\x00\x00\x00Y@\x1a\f\b\x9c\x03\x11\x00\x00\x00\x00\x00\x00^@\x1a\f\b\xd0\x04\x11\x00\x00\x00\x00\x00\x00b@\x1a\f\b\xf4\v\x11\x9a\x99\x99\x99\x99\x99e@\x1a\f\b\x85\x15\x11\x00\x00\x00\x00\x00\x00\xf0\u007f",
			expected: []model.Samples{
				model.Samples{
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_duration_microseconds_bucket", "le": "100"},
						Value:     123,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_duration_microseconds_bucket", "le": "120"},
						Value:     412,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_duration_microseconds_bucket", "le": "144"},
						Value:     592,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_duration_microseconds_bucket", "le": "172.8"},
						Value:     1524,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_duration_microseconds_bucket", "le": "+Inf"},
						Value:     2693,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_duration_microseconds_sum"},
						Value:     1756047.3,
						Timestamp: testTime,
					},
					&model.Sample{
						Metric:    model.Metric{model.MetricNameLabel: "request_duration_microseconds_count"},
						Value:     2693,
						Timestamp: testTime,
					},
				},
			},
		},
	}

	for i, scenario := range scenarios {
		scenario.test(t, i)
	}
}
