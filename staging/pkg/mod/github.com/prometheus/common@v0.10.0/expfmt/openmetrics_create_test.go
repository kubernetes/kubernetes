// Copyright 2020 The Prometheus Authors
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
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"

	dto "github.com/prometheus/client_model/go"
)

func TestCreateOpenMetrics(t *testing.T) {
	openMetricsTimestamp, err := ptypes.TimestampProto(time.Unix(12345, 600000000))
	if err != nil {
		t.Error(err)
	}

	var scenarios = []struct {
		in  *dto.MetricFamily
		out string
	}{
		// 0: Counter, timestamp given, no _total suffix.
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
							Value: proto.Float64(42),
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
# TYPE name unknown
name{labelname="val1",basename="basevalue"} 42.0
name{labelname="val2",basename="basevalue"} 0.23 1.23456789e+06
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
			out: `# HELP gauge_name gauge\ndoc\nstr\"ing
# TYPE gauge_name gauge
gauge_name{name_1="val with\nnew line",name_2="val with \\backslash and \"quotes\""} +Inf
gauge_name{name_1="Björn",name_2="佖佥"} 3.14e+42
`,
		},
		// 2: Unknown, no help, one sample with no labels and -Inf as value, another sample with one label.
		{
			in: &dto.MetricFamily{
				Name: proto.String("unknown_name"),
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
			out: `# TYPE unknown_name unknown
unknown_name -Inf
unknown_name{name_1="value 1"} -1.23e-45
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
summary_name{quantile="0.99"} 0.0
summary_name_sum -3.4567
summary_name_count 42
summary_name{name_1="value 1",name_2="value 2",quantile="0.5"} 1.0
summary_name{name_1="value 1",name_2="value 2",quantile="0.9"} 2.0
summary_name{name_1="value 1",name_2="value 2",quantile="0.99"} 3.0
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
request_duration_microseconds_bucket{le="100.0"} 123
request_duration_microseconds_bucket{le="120.0"} 412
request_duration_microseconds_bucket{le="144.0"} 592
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
request_duration_microseconds_bucket{le="100.0"} 123
request_duration_microseconds_bucket{le="120.0"} 412
request_duration_microseconds_bucket{le="144.0"} 592
request_duration_microseconds_bucket{le="172.8"} 1524
request_duration_microseconds_bucket{le="+Inf"} 2693
request_duration_microseconds_sum 1.7560473e+06
request_duration_microseconds_count 2693
`,
		},
		// 6: Histogram with missing +Inf bucket but with different exemplars.
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
									Exemplar: &dto.Exemplar{
										Label: []*dto.LabelPair{
											&dto.LabelPair{
												Name:  proto.String("foo"),
												Value: proto.String("bar"),
											},
										},
										Value:     proto.Float64(119.9),
										Timestamp: openMetricsTimestamp,
									},
								},
								&dto.Bucket{
									UpperBound:      proto.Float64(144),
									CumulativeCount: proto.Uint64(592),
									Exemplar: &dto.Exemplar{
										Label: []*dto.LabelPair{
											&dto.LabelPair{
												Name:  proto.String("foo"),
												Value: proto.String("baz"),
											},
											&dto.LabelPair{
												Name:  proto.String("dings"),
												Value: proto.String("bums"),
											},
										},
										Value: proto.Float64(140.14),
									},
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
request_duration_microseconds_bucket{le="100.0"} 123
request_duration_microseconds_bucket{le="120.0"} 412 # {foo="bar"} 119.9 12345.6
request_duration_microseconds_bucket{le="144.0"} 592 # {foo="baz",dings="bums"} 140.14
request_duration_microseconds_bucket{le="172.8"} 1524
request_duration_microseconds_bucket{le="+Inf"} 2693
request_duration_microseconds_sum 1.7560473e+06
request_duration_microseconds_count 2693
`,
		},
		// 7: Simple Counter.
		{
			in: &dto.MetricFamily{
				Name: proto.String("foos_total"),
				Help: proto.String("Number of foos."),
				Type: dto.MetricType_COUNTER.Enum(),
				Metric: []*dto.Metric{
					&dto.Metric{
						Counter: &dto.Counter{
							Value: proto.Float64(42),
						},
					},
				},
			},
			out: `# HELP foos Number of foos.
# TYPE foos counter
foos_total 42.0
`,
		},
		// 8: No metric.
		{
			in: &dto.MetricFamily{
				Name:   proto.String("name_total"),
				Help:   proto.String("doc string"),
				Type:   dto.MetricType_COUNTER.Enum(),
				Metric: []*dto.Metric{},
			},
			out: `# HELP name doc string
# TYPE name counter
`,
		},
	}

	for i, scenario := range scenarios {
		out := bytes.NewBuffer(make([]byte, 0, len(scenario.out)))
		n, err := MetricFamilyToOpenMetrics(out, scenario.in)
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

func BenchmarkOpenMetricsCreate(b *testing.B) {
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
		_, err := MetricFamilyToOpenMetrics(out, mf)
		if err != nil {
			b.Fatal(err)
		}
		out.Reset()
	}
}

func TestOpenMetricsCreateError(t *testing.T) {
	var scenarios = []struct {
		in  *dto.MetricFamily
		err string
	}{
		// 0: No metric name.
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
		// 1: Wrong type.
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
		_, err := MetricFamilyToOpenMetrics(&out, scenario.in)
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
