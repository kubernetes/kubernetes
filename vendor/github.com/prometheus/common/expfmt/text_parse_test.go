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
	"math"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"
	dto "github.com/prometheus/client_model/go"
)

func testTextParse(t testing.TB) {
	var scenarios = []struct {
		in  string
		out []*dto.MetricFamily
	}{
		// 0: Empty lines as input.
		{
			in: `

`,
			out: []*dto.MetricFamily{},
		},
		// 1: Minimal case.
		{
			in: `
minimal_metric 1.234
another_metric -3e3 103948
# Even that:
no_labels{} 3
# HELP line for non-existing metric will be ignored.
`,
			out: []*dto.MetricFamily{
				&dto.MetricFamily{
					Name: proto.String("minimal_metric"),
					Type: dto.MetricType_UNTYPED.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Untyped: &dto.Untyped{
								Value: proto.Float64(1.234),
							},
						},
					},
				},
				&dto.MetricFamily{
					Name: proto.String("another_metric"),
					Type: dto.MetricType_UNTYPED.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Untyped: &dto.Untyped{
								Value: proto.Float64(-3e3),
							},
							TimestampMs: proto.Int64(103948),
						},
					},
				},
				&dto.MetricFamily{
					Name: proto.String("no_labels"),
					Type: dto.MetricType_UNTYPED.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Untyped: &dto.Untyped{
								Value: proto.Float64(3),
							},
						},
					},
				},
			},
		},
		// 2: Counters & gauges, docstrings, various whitespace, escape sequences.
		{
			in: `
# A normal comment.
#
# TYPE name counter
name{labelname="val1",basename="basevalue"} NaN
name {labelname="val2",basename="base\"v\\al\nue"} 0.23 1234567890
# HELP name two-line\n doc  str\\ing

 # HELP  name2  	doc str"ing 2
  #    TYPE    name2 gauge
name2{labelname="val2"	,basename   =   "basevalue2"		} +Inf 54321
name2{ labelname = "val1" , }-Inf
`,
			out: []*dto.MetricFamily{
				&dto.MetricFamily{
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
									Value: proto.String("base\"v\\al\nue"),
								},
							},
							Counter: &dto.Counter{
								Value: proto.Float64(.23),
							},
							TimestampMs: proto.Int64(1234567890),
						},
					},
				},
				&dto.MetricFamily{
					Name: proto.String("name2"),
					Help: proto.String("doc str\"ing 2"),
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("labelname"),
									Value: proto.String("val2"),
								},
								&dto.LabelPair{
									Name:  proto.String("basename"),
									Value: proto.String("basevalue2"),
								},
							},
							Gauge: &dto.Gauge{
								Value: proto.Float64(math.Inf(+1)),
							},
							TimestampMs: proto.Int64(54321),
						},
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("labelname"),
									Value: proto.String("val1"),
								},
							},
							Gauge: &dto.Gauge{
								Value: proto.Float64(math.Inf(-1)),
							},
						},
					},
				},
			},
		},
		// 3: The evil summary, mixed with other types and funny comments.
		{
			in: `
# TYPE my_summary summary
my_summary{n1="val1",quantile="0.5"} 110
decoy -1 -2
my_summary{n1="val1",quantile="0.9"} 140 1
my_summary_count{n1="val1"} 42
# Latest timestamp wins in case of a summary.
my_summary_sum{n1="val1"} 4711 2
fake_sum{n1="val1"} 2001
# TYPE another_summary summary
another_summary_count{n2="val2",n1="val1"} 20
my_summary_count{n2="val2",n1="val1"} 5 5
another_summary{n1="val1",n2="val2",quantile=".3"} -1.2
my_summary_sum{n1="val2"} 08 15
my_summary{n1="val3", quantile="0.2"} 4711
  my_summary{n1="val1",n2="val2",quantile="-12.34",} NaN
# some
# funny comments
# HELP 
# HELP
# HELP my_summary
# HELP my_summary 
`,
			out: []*dto.MetricFamily{
				&dto.MetricFamily{
					Name: proto.String("fake_sum"),
					Type: dto.MetricType_UNTYPED.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("n1"),
									Value: proto.String("val1"),
								},
							},
							Untyped: &dto.Untyped{
								Value: proto.Float64(2001),
							},
						},
					},
				},
				&dto.MetricFamily{
					Name: proto.String("decoy"),
					Type: dto.MetricType_UNTYPED.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Untyped: &dto.Untyped{
								Value: proto.Float64(-1),
							},
							TimestampMs: proto.Int64(-2),
						},
					},
				},
				&dto.MetricFamily{
					Name: proto.String("my_summary"),
					Type: dto.MetricType_SUMMARY.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("n1"),
									Value: proto.String("val1"),
								},
							},
							Summary: &dto.Summary{
								SampleCount: proto.Uint64(42),
								SampleSum:   proto.Float64(4711),
								Quantile: []*dto.Quantile{
									&dto.Quantile{
										Quantile: proto.Float64(0.5),
										Value:    proto.Float64(110),
									},
									&dto.Quantile{
										Quantile: proto.Float64(0.9),
										Value:    proto.Float64(140),
									},
								},
							},
							TimestampMs: proto.Int64(2),
						},
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("n2"),
									Value: proto.String("val2"),
								},
								&dto.LabelPair{
									Name:  proto.String("n1"),
									Value: proto.String("val1"),
								},
							},
							Summary: &dto.Summary{
								SampleCount: proto.Uint64(5),
								Quantile: []*dto.Quantile{
									&dto.Quantile{
										Quantile: proto.Float64(-12.34),
										Value:    proto.Float64(math.NaN()),
									},
								},
							},
							TimestampMs: proto.Int64(5),
						},
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("n1"),
									Value: proto.String("val2"),
								},
							},
							Summary: &dto.Summary{
								SampleSum: proto.Float64(8),
							},
							TimestampMs: proto.Int64(15),
						},
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("n1"),
									Value: proto.String("val3"),
								},
							},
							Summary: &dto.Summary{
								Quantile: []*dto.Quantile{
									&dto.Quantile{
										Quantile: proto.Float64(0.2),
										Value:    proto.Float64(4711),
									},
								},
							},
						},
					},
				},
				&dto.MetricFamily{
					Name: proto.String("another_summary"),
					Type: dto.MetricType_SUMMARY.Enum(),
					Metric: []*dto.Metric{
						&dto.Metric{
							Label: []*dto.LabelPair{
								&dto.LabelPair{
									Name:  proto.String("n2"),
									Value: proto.String("val2"),
								},
								&dto.LabelPair{
									Name:  proto.String("n1"),
									Value: proto.String("val1"),
								},
							},
							Summary: &dto.Summary{
								SampleCount: proto.Uint64(20),
								Quantile: []*dto.Quantile{
									&dto.Quantile{
										Quantile: proto.Float64(0.3),
										Value:    proto.Float64(-1.2),
									},
								},
							},
						},
					},
				},
			},
		},
		// 4: The histogram.
		{
			in: `
# HELP request_duration_microseconds The response latency.
# TYPE request_duration_microseconds histogram
request_duration_microseconds_bucket{le="100"} 123
request_duration_microseconds_bucket{le="120"} 412
request_duration_microseconds_bucket{le="144"} 592
request_duration_microseconds_bucket{le="172.8"} 1524
request_duration_microseconds_bucket{le="+Inf"} 2693
request_duration_microseconds_sum 1.7560473e+06
request_duration_microseconds_count 2693
`,
			out: []*dto.MetricFamily{
				{
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
			},
		},
	}

	for i, scenario := range scenarios {
		out, err := parser.TextToMetricFamilies(strings.NewReader(scenario.in))
		if err != nil {
			t.Errorf("%d. error: %s", i, err)
			continue
		}
		if expected, got := len(scenario.out), len(out); expected != got {
			t.Errorf(
				"%d. expected %d MetricFamilies, got %d",
				i, expected, got,
			)
		}
		for _, expected := range scenario.out {
			got, ok := out[expected.GetName()]
			if !ok {
				t.Errorf(
					"%d. expected MetricFamily %q, found none",
					i, expected.GetName(),
				)
				continue
			}
			if expected.String() != got.String() {
				t.Errorf(
					"%d. expected MetricFamily %s, got %s",
					i, expected, got,
				)
			}
		}
	}
}

func TestTextParse(t *testing.T) {
	testTextParse(t)
}

func BenchmarkTextParse(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testTextParse(b)
	}
}

func testTextParseError(t testing.TB) {
	var scenarios = []struct {
		in  string
		err string
	}{
		// 0: No new-line at end of input.
		{
			in: `
bla 3.14
blubber 42`,
			err: "text format parsing error in line 3: unexpected end of input stream",
		},
		// 1: Invalid escape sequence in label value.
		{
			in:  `metric{label="\t"} 3.14`,
			err: "text format parsing error in line 1: invalid escape sequence",
		},
		// 2: Newline in label value.
		{
			in: `
metric{label="new
line"} 3.14
`,
			err: `text format parsing error in line 2: label value "new" contains unescaped new-line`,
		},
		// 3:
		{
			in:  `metric{@="bla"} 3.14`,
			err: "text format parsing error in line 1: invalid label name for metric",
		},
		// 4:
		{
			in:  `metric{__name__="bla"} 3.14`,
			err: `text format parsing error in line 1: label name "__name__" is reserved`,
		},
		// 5:
		{
			in:  `metric{label+="bla"} 3.14`,
			err: "text format parsing error in line 1: expected '=' after label name",
		},
		// 6:
		{
			in:  `metric{label=bla} 3.14`,
			err: "text format parsing error in line 1: expected '\"' at start of label value",
		},
		// 7:
		{
			in: `
# TYPE metric summary
metric{quantile="bla"} 3.14
`,
			err: "text format parsing error in line 3: expected float as value for 'quantile' label",
		},
		// 8:
		{
			in:  `metric{label="bla"+} 3.14`,
			err: "text format parsing error in line 1: unexpected end of label value",
		},
		// 9:
		{
			in: `metric{label="bla"} 3.14 2.72
`,
			err: "text format parsing error in line 1: expected integer as timestamp",
		},
		// 10:
		{
			in: `metric{label="bla"} 3.14 2 3
`,
			err: "text format parsing error in line 1: spurious string after timestamp",
		},
		// 11:
		{
			in: `metric{label="bla"} blubb
`,
			err: "text format parsing error in line 1: expected float as value",
		},
		// 12:
		{
			in: `
# HELP metric one
# HELP metric two
`,
			err: "text format parsing error in line 3: second HELP line for metric name",
		},
		// 13:
		{
			in: `
# TYPE metric counter
# TYPE metric untyped
`,
			err: `text format parsing error in line 3: second TYPE line for metric name "metric", or TYPE reported after samples`,
		},
		// 14:
		{
			in: `
metric 4.12
# TYPE metric counter
`,
			err: `text format parsing error in line 3: second TYPE line for metric name "metric", or TYPE reported after samples`,
		},
		// 14:
		{
			in: `
# TYPE metric bla
`,
			err: "text format parsing error in line 2: unknown metric type",
		},
		// 15:
		{
			in: `
# TYPE met-ric
`,
			err: "text format parsing error in line 2: invalid metric name in comment",
		},
		// 16:
		{
			in:  `@invalidmetric{label="bla"} 3.14 2`,
			err: "text format parsing error in line 1: invalid metric name",
		},
		// 17:
		{
			in:  `{label="bla"} 3.14 2`,
			err: "text format parsing error in line 1: invalid metric name",
		},
		// 18:
		{
			in: `
# TYPE metric histogram
metric_bucket{le="bla"} 3.14
`,
			err: "text format parsing error in line 3: expected float as value for 'le' label",
		},
	}

	for i, scenario := range scenarios {
		_, err := parser.TextToMetricFamilies(strings.NewReader(scenario.in))
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

func TestTextParseError(t *testing.T) {
	testTextParseError(t)
}

func BenchmarkParseError(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testTextParseError(b)
	}
}
