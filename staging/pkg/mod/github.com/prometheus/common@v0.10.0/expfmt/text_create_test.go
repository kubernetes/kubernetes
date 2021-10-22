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

package expfmt

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"

	dto "github.com/prometheus/client_model/go"
)

func TestCreate(t *testing.T) {
	var scenarios = []struct {
		in  *dto.MetricFamily
		out string
	}{
		// 0: Counter, NaN as value, timestamp given.
		{
			in: &dto.MetricFamily{
				Name: proto.String("name"),
				Help: proto.String("two-line\n doc  str\\ing"),
				Type: dto.MetricType_COUNTER.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Label: []*dto.LabelPair{
							&dto.LabelPair{
								Name:  proto.String("labelname"),
								Value: proto.String("val1"),
							},
							&dto.LabelPair{
								Name:  proto.String("basename"),
								Value: proto.String("basevalue"),
							},
						},
						Counter: &dto.Counter{
							Value: proto.Float64(math.NaN()),
						},
					},
					&dto.Metric{
						Label: []*dto.LabelPair{
							&dto.LabelPair{
								Name:  proto.String("labelname"),
								Value: proto.String("val2"),
							},
							&dto.LabelPair{
								Name:  proto.String("basename"),
								Value: proto.String("basevalue"),
							},
						},
						Counter: &dto.Counter{
							Value: proto.Float64(.23),
						},
						TimestampMs: proto.Int64(1234567890),
					},
				},
			},
			out: `# HELP name two-line\n doc  str\\ing
# TYPE name counter
name{labelname="val1",basename="basevalue"} NaN
name{labelname="val2",basename="basevalue"} 0.23 1234567890
`,
		},
		// 1: Gauge, some escaping required, +Inf as value, multi-byte characters in label values.
		{
			in: &dto.MetricFamily{
				Name: proto.String("gauge_name"),
				Help: proto.String("gauge\ndoc\nstr\"ing"),
				Type: dto.MetricType_GAUGE.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Label: []*dto.LabelPair{
							&dto.LabelPair{
								Name:  proto.String("name_1"),
								Value: proto.String("val with\nnew line"),
							},
							&dto.LabelPair{
								Name:  proto.String("name_2"),
								Value: proto.String("val with \\backslash and \"quotes\""),
							},
						},
						Gauge: &dto.Gauge{
							Value: proto.Float64(math.Inf(+1)),
						},
					},
					&dto.Metric{
						Label: []*dto.LabelPair{
							&dto.LabelPair{
								Name:  proto.String("name_1"),
								Value: proto.String("Björn"),
							},
							&dto.LabelPair{
								Name:  proto.String("name_2"),
								Value: proto.String("佖佥"),
							},
						},
						Gauge: &dto.Gauge{
							Value: proto.Float64(3.14e42),
						},
					},
				},
			},
			out: `# HELP gauge_name gauge\ndoc\nstr"ing
# TYPE gauge_name gauge
gauge_name{name_1="val with\nnew line",name_2="val with \\backslash and \"quotes\""} +Inf
gauge_name{name_1="Björn",name_2="佖佥"} 3.14e+42
`,
		},
		// 2: Untyped, no help, one sample with no labels and -Inf as value, another sample with one label.
		{
			in: &dto.MetricFamily{
				Name: proto.String("untyped_name"),
				Type: dto.MetricType_UNTYPED.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Untyped: &dto.Untyped{
							Value: proto.Float64(math.Inf(-1)),
						},
					},
					&dto.Metric{
						Label: []*dto.LabelPair{
							&dto.LabelPair{
								Name:  proto.String("name_1"),
								Value: proto.String("value 1"),
							},
						},
						Untyped: &dto.Untyped{
							Value: proto.Float64(-1.23e-45),
						},
					},
				},
			},
			out: `# TYPE untyped_name untyped
untyped_name -Inf
untyped_name{name_1="value 1"} -1.23e-45
`,
		},
		// 3: Summary.
		{
			in: &dto.MetricFamily{
				Name: proto.String("summary_name"),
				Help: proto.String("summary docstring"),
				Type: dto.MetricType_SUMMARY.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Summary: &dto.Summary{
							SampleCount: proto.Uint64(42),
							SampleSum:   proto.Float64(-3.4567),
							Quantile: []*dto.Quantile{
								&dto.Quantile{
									Quantile: proto.Float64(0.5),
									Value:    proto.Float64(-1.23),
								},
								&dto.Quantile{
									Quantile: proto.Float64(0.9),
									Value:    proto.Float64(.2342354),
								},
								&dto.Quantile{
									Quantile: proto.Float64(0.99),
									Value:    proto.Float64(0),
								},
							},
						},
					},
					&dto.Metric{
						Label: []*dto.LabelPair{
							&dto.LabelPair{
								Name:  proto.String("name_1"),
								Value: proto.String("value 1"),
							},
							&dto.LabelPair{
								Name:  proto.String("name_2"),
								Value: proto.String("value 2"),
							},
						},
						Summary: &dto.Summary{
							SampleCount: proto.Uint64(4711),
							SampleSum:   proto.Float64(2010.1971),
							Quantile: []*dto.Quantile{
								&dto.Quantile{
									Quantile: proto.Float64(0.5),
									Value:    proto.Float64(1),
								},
								&dto.Quantile{
									Quantile: proto.Float64(0.9),
									Value:    proto.Float64(2),
								},
								&dto.Quantile{
									Quantile: proto.Float64(0.99),
									Value:    proto.Float64(3),
								},
							},
						},
					},
				},
			},
			out: `# HELP summary_name summary docstring
# TYPE summary_name summary
summary_name{quantile="0.5"} -1.23
summary_name{quantile="0.9"} 0.2342354
summary_name{quantile="0.99"} 0
summary_name_sum -3.4567
summary_name_count 42
summary_name{name_1="value 1",name_2="value 2",quantile="0.5"} 1
summary_name{name_1="value 1",name_2="value 2",quantile="0.9"} 2
summary_name{name_1="value 1",name_2="value 2",quantile="0.99"} 3
summary_name_sum{name_1="value 1",name_2="value 2"} 2010.1971
summary_name_count{name_1="value 1",name_2="value 2"} 4711
`,
		},
		// 4: Histogram
		{
			in: &dto.MetricFamily{
				Name: proto.String("request_duration_microseconds"),
				Help: proto.String("The response latency."),
				Type: dto.MetricType_HISTOGRAM.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Histogram: &dto.Histogram{
							SampleCount: proto.Uint64(2693),
							SampleSum:   proto.Float64(1756047.3),
							Bucket: []*dto.Bucket{
								&dto.Bucket{
									UpperBound:      proto.Float64(100),
									CumulativeCount: proto.Uint64(123),
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(120),
									CumulativeCount: proto.Uint64(412),
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(144),
									CumulativeCount: proto.Uint64(592),
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(172.8),
									CumulativeCount: proto.Uint64(1524),
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(math.Inf(+1)),
									CumulativeCount: proto.Uint64(2693),
								},
							},
						},
					},
				},
			},
			out: `# HELP request_duration_microseconds The response latency.
# TYPE request_duration_microseconds histogram
request_duration_microseconds_bucket{le="100"} 123
request_duration_microseconds_bucket{le="120"} 412
request_duration_microseconds_bucket{le="144"} 592
request_duration_microseconds_bucket{le="172.8"} 1524
request_duration_microseconds_bucket{le="+Inf"} 2693
request_duration_microseconds_sum 1.7560473e+06
request_duration_microseconds_count 2693
`,
		},
		// 5: Histogram with missing +Inf bucket.
		{
			in: &dto.MetricFamily{
				Name: proto.String("request_duration_microseconds"),
				Help: proto.String("The response latency."),
				Type: dto.MetricType_HISTOGRAM.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Histogram: &dto.Histogram{
							SampleCount: proto.Uint64(2693),
							SampleSum:   proto.Float64(1756047.3),
							Bucket: []*dto.Bucket{
								&dto.Bucket{
									UpperBound:      proto.Float64(100),
									CumulativeCount: proto.Uint64(123),
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(120),
									CumulativeCount: proto.Uint64(412),
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(144),
									CumulativeCount: proto.Uint64(592),
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(172.8),
									CumulativeCount: proto.Uint64(1524),
								},
							},
						},
					},
				},
			},
			out: `# HELP request_duration_microseconds The response latency.
# TYPE request_duration_microseconds histogram
request_duration_microseconds_bucket{le="100"} 123
request_duration_microseconds_bucket{le="120"} 412
request_duration_microseconds_bucket{le="144"} 592
request_duration_microseconds_bucket{le="172.8"} 1524
request_duration_microseconds_bucket{le="+Inf"} 2693
request_duration_microseconds_sum 1.7560473e+06
request_duration_microseconds_count 2693
`,
		},
		// 6: No metric type, should result in default type Counter.
		{
			in: &dto.MetricFamily{
				Name: proto.String("name"),
				Help: proto.String("doc string"),
				Metric: []*dto.Metric{
					&dto.Metric{
						Counter: &dto.Counter{
							Value: proto.Float64(math.Inf(-1)),
						},
					},
				},
			},
			out: `# HELP name doc string
# TYPE name counter
name -Inf
`,
		},
	}

	for i, scenario := range scenarios {
		out := bytes.NewBuffer(make([]byte, 0, len(scenario.out)))
		n, err := MetricFamilyToText(out, scenario.in)
		if err != nil {
			t.Errorf("%d. error: %s", i, err)
			continue
		}
		if expected, got := len(scenario.out), n; expected != got {
			t.Errorf(
				"%d. expected %d bytes written, got %d",
				i, expected, got,
			)
		}
		if expected, got := scenario.out, out.String(); expected != got {
			t.Errorf(
				"%d. expected out=%q, got %q",
				i, expected, got,
			)
		}
	}

}

func BenchmarkCreate(b *testing.B) {
	mf := &dto.MetricFamily{
		Name: proto.String("request_duration_microseconds"),
		Help: proto.String("The response latency."),
		Type: dto.MetricType_HISTOGRAM.Enum(),
		Metric: []*dto.Metric{
			&dto.Metric{
				Label: []*dto.LabelPair{
					&dto.LabelPair{
						Name:  proto.String("name_1"),
						Value: proto.String("val with\nnew line"),
					},
					&dto.LabelPair{
						Name:  proto.String("name_2"),
						Value: proto.String("val with \\backslash and \"quotes\""),
					},
					&dto.LabelPair{
						Name:  proto.String("name_3"),
						Value: proto.String("Just a quite long label value to test performance."),
					},
				},
				Histogram: &dto.Histogram{
					SampleCount: proto.Uint64(2693),
					SampleSum:   proto.Float64(1756047.3),
					Bucket: []*dto.Bucket{
						&dto.Bucket{
							UpperBound:      proto.Float64(100),
							CumulativeCount: proto.Uint64(123),
						},
						&dto.Bucket{
							UpperBound:      proto.Float64(120),
							CumulativeCount: proto.Uint64(412),
						},
						&dto.Bucket{
							UpperBound:      proto.Float64(144),
							CumulativeCount: proto.Uint64(592),
						},
						&dto.Bucket{
							UpperBound:      proto.Float64(172.8),
							CumulativeCount: proto.Uint64(1524),
						},
						&dto.Bucket{
							UpperBound:      proto.Float64(math.Inf(+1)),
							CumulativeCount: proto.Uint64(2693),
						},
					},
				},
			},
			&dto.Metric{
				Label: []*dto.LabelPair{
					&dto.LabelPair{
						Name:  proto.String("name_1"),
						Value: proto.String("Björn"),
					},
					&dto.LabelPair{
						Name:  proto.String("name_2"),
						Value: proto.String("佖佥"),
					},
					&dto.LabelPair{
						Name:  proto.String("name_3"),
						Value: proto.String("Just a quite long label value to test performance."),
					},
				},
				Histogram: &dto.Histogram{
					SampleCount: proto.Uint64(5699),
					SampleSum:   proto.Float64(49484343543.4343),
					Bucket: []*dto.Bucket{
						&dto.Bucket{
							UpperBound:      proto.Float64(100),
							CumulativeCount: proto.Uint64(120),
						},
						&dto.Bucket{
							UpperBound:      proto.Float64(120),
							CumulativeCount: proto.Uint64(412),
						},
						&dto.Bucket{
							UpperBound:      proto.Float64(144),
							CumulativeCount: proto.Uint64(596),
						},
						&dto.Bucket{
							UpperBound:      proto.Float64(172.8),
							CumulativeCount: proto.Uint64(1535),
						},
					},
				},
				TimestampMs: proto.Int64(1234567890),
			},
		},
	}
	out := bytes.NewBuffer(make([]byte, 0, 1024))

	for i := 0; i < b.N; i++ {
		_, err := MetricFamilyToText(out, mf)
		if err != nil {
			b.Fatal(err)
		}
		out.Reset()
	}
}

func BenchmarkCreateBuildInfo(b *testing.B) {
	mf := &dto.MetricFamily{
		Name: proto.String("benchmark_build_info"),
		Help: proto.String("Test the creation of constant 1-value build_info metric."),
		Type: dto.MetricType_GAUGE.Enum(),
		Metric: []*dto.Metric{
			&dto.Metric{
				Label: []*dto.LabelPair{
					&dto.LabelPair{
						Name:  proto.String("version"),
						Value: proto.String("1.2.3"),
					},
					&dto.LabelPair{
						Name:  proto.String("revision"),
						Value: proto.String("2e84f5e4eacdffb574035810305191ff390360fe"),
					},
					&dto.LabelPair{
						Name:  proto.String("go_version"),
						Value: proto.String("1.11.1"),
					},
				},
				Gauge: &dto.Gauge{
					Value: proto.Float64(1),
				},
			},
		},
	}
	out := bytes.NewBuffer(make([]byte, 0, 1024))

	for i := 0; i < b.N; i++ {
		_, err := MetricFamilyToText(out, mf)
		if err != nil {
			b.Fatal(err)
		}
		out.Reset()
	}
}

func TestCreateError(t *testing.T) {
	var scenarios = []struct {
		in  *dto.MetricFamily
		err string
	}{
		// 0: No metric.
		{
			in: &dto.MetricFamily{
				Name:   proto.String("name"),
				Help:   proto.String("doc string"),
				Type:   dto.MetricType_COUNTER.Enum(),
				Metric: []*dto.Metric{},
			},
			err: "MetricFamily has no metrics",
		},
		// 1: No metric name.
		{
			in: &dto.MetricFamily{
				Help: proto.String("doc string"),
				Type: dto.MetricType_UNTYPED.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Untyped: &dto.Untyped{
							Value: proto.Float64(math.Inf(-1)),
						},
					},
				},
			},
			err: "MetricFamily has no name",
		},
		// 2: Wrong type.
		{
			in: &dto.MetricFamily{
				Name: proto.String("name"),
				Help: proto.String("doc string"),
				Type: dto.MetricType_COUNTER.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Untyped: &dto.Untyped{
							Value: proto.Float64(math.Inf(-1)),
						},
					},
				},
			},
			err: "expected counter in metric",
		},
	}

	for i, scenario := range scenarios {
		var out bytes.Buffer
		_, err := MetricFamilyToText(&out, scenario.in)
		if err == nil {
			t.Errorf("%d. expected error, got nil", i)
			continue
		}
		if expected, got := scenario.err, err.Error(); strings.Index(got, expected) != 0 {
			t.Errorf(
				"%d. expected error starting with %q, got %q",
				i, expected, got,
			)
		}
	}

}
