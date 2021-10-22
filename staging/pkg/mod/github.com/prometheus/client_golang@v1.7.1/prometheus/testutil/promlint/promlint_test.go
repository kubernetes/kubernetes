// Copyright 2020 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package promlint_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil/promlint"
)

type test struct {
	name     string
	in       string
	problems []promlint.Problem
}

func TestLintNoHelpText(t *testing.T) {
	const msg = "no help text"

	tests := []test{
		{
			name: "no help",
			in: `
# TYPE go_goroutines gauge
go_goroutines 24
`,
			problems: []promlint.Problem{{
				Metric: "go_goroutines",
				Text:   msg,
			}},
		},
		{
			name: "empty help",
			in: `
# HELP go_goroutines
# TYPE go_goroutines gauge
go_goroutines 24
`,
			problems: []promlint.Problem{{
				Metric: "go_goroutines",
				Text:   msg,
			}},
		},
		{
			name: "no help and empty help",
			in: `
# HELP go_goroutines
# TYPE go_goroutines gauge
go_goroutines 24
# TYPE go_threads gauge
go_threads 10
`,
			problems: []promlint.Problem{
				{
					Metric: "go_goroutines",
					Text:   msg,
				},
				{
					Metric: "go_threads",
					Text:   msg,
				},
			},
		},
		{
			name: "OK",
			in: `
# HELP go_goroutines Number of goroutines that currently exist.
# TYPE go_goroutines gauge
go_goroutines 24
`,
		},
	}
	runTests(t, tests)
}

func TestLintMetricUnits(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		problems []promlint.Problem
	}{
		// good cases.
		{
			name: "amperes",
			in: `
# HELP x_amperes Test metric.
# TYPE x_amperes untyped
x_amperes 10
`,
		},
		{
			name: "bytes",
			in: `
# HELP x_bytes Test metric.
# TYPE x_bytes untyped
x_bytes 10
`,
		},
		{
			name: "grams",
			in: `
# HELP x_grams Test metric.
# TYPE x_grams untyped
x_grams 10
`,
		},
		{
			name: "celsius",
			in: `
# HELP x_celsius Test metric.
# TYPE x_celsius untyped
x_celsius 10
`,
		},
		{
			name: "meters",
			in: `
# HELP x_meters Test metric.
# TYPE x_meters untyped
x_meters 10
`,
		},
		{
			name: "metres",
			in: `
# HELP x_metres Test metric.
# TYPE x_metres untyped
x_metres 10
`,
		},
		{
			name: "moles",
			in: `
# HELP x_moles Test metric.
# TYPE x_moles untyped
x_moles 10
`,
		},
		{
			name: "seconds",
			in: `
# HELP x_seconds Test metric.
# TYPE x_seconds untyped
x_seconds 10
`,
		},
		{
			name: "joules",
			in: `
# HELP x_joules Test metric.
# TYPE x_joules untyped
x_joules 10
`,
		},
		{
			name: "kelvin",
			in: `
# HELP x_kelvin Test metric.
# TYPE x_kelvin untyped
x_kelvin 10
`,
		},
		// bad cases.
		{
			name: "milliamperes",
			in: `
# HELP x_milliamperes Test metric.
# TYPE x_milliamperes untyped
x_milliamperes 10
`,
			problems: []promlint.Problem{{
				Metric: "x_milliamperes",
				Text:   `use base unit "amperes" instead of "milliamperes"`,
			}},
		},
		{
			name: "gigabytes",
			in: `
# HELP x_gigabytes Test metric.
# TYPE x_gigabytes untyped
x_gigabytes 10
`,
			problems: []promlint.Problem{{
				Metric: "x_gigabytes",
				Text:   `use base unit "bytes" instead of "gigabytes"`,
			}},
		},
		{
			name: "kilograms",
			in: `
# HELP x_kilograms Test metric.
# TYPE x_kilograms untyped
x_kilograms 10
`,
			problems: []promlint.Problem{{
				Metric: "x_kilograms",
				Text:   `use base unit "grams" instead of "kilograms"`,
			}},
		},
		{
			name: "nanocelsius",
			in: `
# HELP x_nanocelsius Test metric.
# TYPE x_nanocelsius untyped
x_nanocelsius 10
`,
			problems: []promlint.Problem{{
				Metric: "x_nanocelsius",
				Text:   `use base unit "celsius" instead of "nanocelsius"`,
			}},
		},
		{
			name: "kilometers",
			in: `
# HELP x_kilometers Test metric.
# TYPE x_kilometers untyped
x_kilometers 10
`,
			problems: []promlint.Problem{{
				Metric: "x_kilometers",
				Text:   `use base unit "meters" instead of "kilometers"`,
			}},
		},
		{
			name: "picometers",
			in: `
# HELP x_picometers Test metric.
# TYPE x_picometers untyped
x_picometers 10
`,
			problems: []promlint.Problem{{
				Metric: "x_picometers",
				Text:   `use base unit "meters" instead of "picometers"`,
			}},
		},
		{
			name: "microseconds",
			in: `
# HELP x_microseconds Test metric.
# TYPE x_microseconds untyped
x_microseconds 10
`,
			problems: []promlint.Problem{{
				Metric: "x_microseconds",
				Text:   `use base unit "seconds" instead of "microseconds"`,
			}},
		},
		{
			name: "minutes",
			in: `
# HELP x_minutes Test metric.
# TYPE x_minutes untyped
x_minutes 10
`,
			problems: []promlint.Problem{{
				Metric: "x_minutes",
				Text:   `use base unit "seconds" instead of "minutes"`,
			}},
		},
		{
			name: "hours",
			in: `
# HELP x_hours Test metric.
# TYPE x_hours untyped
x_hours 10
`,
			problems: []promlint.Problem{{
				Metric: "x_hours",
				Text:   `use base unit "seconds" instead of "hours"`,
			}},
		},
		{
			name: "days",
			in: `
# HELP x_days Test metric.
# TYPE x_days untyped
x_days 10
`,
			problems: []promlint.Problem{{
				Metric: "x_days",
				Text:   `use base unit "seconds" instead of "days"`,
			}},
		},
		{
			name: "kelvins",
			in: `
# HELP x_kelvins Test metric.
# TYPE x_kelvins untyped
x_kelvins 10
`,
			problems: []promlint.Problem{{
				Metric: "x_kelvins",
				Text:   `use base unit "kelvin" instead of "kelvins"`,
			}},
		},
		{
			name: "fahrenheit",
			in: `
# HELP thermometers_fahrenheit Test metric.
# TYPE thermometers_fahrenheit untyped
thermometers_fahrenheit 10
`,
			problems: []promlint.Problem{{
				Metric: "thermometers_fahrenheit",
				Text:   `use base unit "celsius" instead of "fahrenheit"`,
			}},
		},
		{
			name: "rankine",
			in: `
# HELP thermometers_rankine Test metric.
# TYPE thermometers_rankine untyped
thermometers_rankine 10
`,
			problems: []promlint.Problem{{
				Metric: "thermometers_rankine",
				Text:   `use base unit "celsius" instead of "rankine"`,
			}},
		}, {
			name: "inches",
			in: `
# HELP x_inches Test metric.
# TYPE x_inches untyped
x_inches 10
`,
			problems: []promlint.Problem{{
				Metric: "x_inches",
				Text:   `use base unit "meters" instead of "inches"`,
			}},
		}, {
			name: "yards",
			in: `
# HELP x_yards Test metric.
# TYPE x_yards untyped
x_yards 10
`,
			problems: []promlint.Problem{{
				Metric: "x_yards",
				Text:   `use base unit "meters" instead of "yards"`,
			}},
		}, {
			name: "miles",
			in: `
# HELP x_miles Test metric.
# TYPE x_miles untyped
x_miles 10
`,
			problems: []promlint.Problem{{
				Metric: "x_miles",
				Text:   `use base unit "meters" instead of "miles"`,
			}},
		}, {
			name: "bits",
			in: `
# HELP x_bits Test metric.
# TYPE x_bits untyped
x_bits 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bits",
				Text:   `use base unit "bytes" instead of "bits"`,
			}},
		},
		{
			name: "calories",
			in: `
# HELP x_calories Test metric.
# TYPE x_calories untyped
x_calories 10
`,
			problems: []promlint.Problem{{
				Metric: "x_calories",
				Text:   `use base unit "joules" instead of "calories"`,
			}},
		},
		{
			name: "pounds",
			in: `
# HELP x_pounds Test metric.
# TYPE x_pounds untyped
x_pounds 10
`,
			problems: []promlint.Problem{{
				Metric: "x_pounds",
				Text:   `use base unit "grams" instead of "pounds"`,
			}},
		},
		{
			name: "ounces",
			in: `
# HELP x_ounces Test metric.
# TYPE x_ounces untyped
x_ounces 10
`,
			problems: []promlint.Problem{{
				Metric: "x_ounces",
				Text:   `use base unit "grams" instead of "ounces"`,
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := promlint.New(strings.NewReader(tt.in))

			problems, err := l.Lint()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if want, got := tt.problems, problems; !reflect.DeepEqual(want, got) {
				t.Fatalf("unexpected problems:\n- want: %v\n-  got: %v",
					want, got)
			}
		})
	}
}

func TestLintCounter(t *testing.T) {
	tests := []test{
		{
			name: "counter without _total suffix",
			in: `
# HELP x_bytes Test metric.
# TYPE x_bytes counter
x_bytes 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bytes",
				Text:   `counter metrics should have "_total" suffix`,
			}},
		},
		{
			name: "gauge with _total suffix",
			in: `
# HELP x_bytes_total Test metric.
# TYPE x_bytes_total gauge
x_bytes_total 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bytes_total",
				Text:   `non-counter metrics should not have "_total" suffix`,
			}},
		},
		{
			name: "counter with _total suffix",
			in: `
# HELP x_bytes_total Test metric.
# TYPE x_bytes_total counter
x_bytes_total 10
`,
		},
		{
			name: "gauge without _total suffix",
			in: `
# HELP x_bytes Test metric.
# TYPE x_bytes gauge
x_bytes 10
`,
		},
		{
			name: "untyped with _total suffix",
			in: `
# HELP x_bytes_total Test metric.
# TYPE x_bytes_total untyped
x_bytes_total 10
`,
		},
		{
			name: "untyped without _total suffix",
			in: `
# HELP x_bytes Test metric.
# TYPE x_bytes untyped
x_bytes 10
`,
		},
	}

	runTests(t, tests)
}

func TestLintHistogramSummaryReserved(t *testing.T) {
	tests := []test{
		{
			name: "gauge with _bucket suffix",
			in: `
# HELP x_bytes_bucket Test metric.
# TYPE x_bytes_bucket gauge
x_bytes_bucket 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bytes_bucket",
				Text:   `non-histogram metrics should not have "_bucket" suffix`,
			}},
		},
		{
			name: "gauge with _count suffix",
			in: `
# HELP x_bytes_count Test metric.
# TYPE x_bytes_count gauge
x_bytes_count 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bytes_count",
				Text:   `non-histogram and non-summary metrics should not have "_count" suffix`,
			}},
		},
		{
			name: "gauge with _sum suffix",
			in: `
# HELP x_bytes_sum Test metric.
# TYPE x_bytes_sum gauge
x_bytes_sum 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bytes_sum",
				Text:   `non-histogram and non-summary metrics should not have "_sum" suffix`,
			}},
		},
		{
			name: "gauge with le label",
			in: `
# HELP x_bytes Test metric.
# TYPE x_bytes gauge
x_bytes{le="1"} 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bytes",
				Text:   `non-histogram metrics should not have "le" label`,
			}},
		},
		{
			name: "gauge with quantile label",
			in: `
# HELP x_bytes Test metric.
# TYPE x_bytes gauge
x_bytes{quantile="1"} 10
`,
			problems: []promlint.Problem{{
				Metric: "x_bytes",
				Text:   `non-summary metrics should not have "quantile" label`,
			}},
		},
		{
			name: "histogram with quantile label",
			in: `
# HELP tsdb_compaction_duration Duration of compaction runs.
# TYPE tsdb_compaction_duration histogram
tsdb_compaction_duration_bucket{le="0.005",quantile="0.01"} 0
tsdb_compaction_duration_bucket{le="0.01",quantile="0.01"} 0
tsdb_compaction_duration_bucket{le="0.025",quantile="0.01"} 0
tsdb_compaction_duration_bucket{le="0.05",quantile="0.01"} 0
tsdb_compaction_duration_bucket{le="0.1",quantile="0.01"} 0
tsdb_compaction_duration_bucket{le="0.25",quantile="0.01"} 0
tsdb_compaction_duration_bucket{le="0.5",quantile="0.01"} 57
tsdb_compaction_duration_bucket{le="1",quantile="0.01"} 68
tsdb_compaction_duration_bucket{le="2.5",quantile="0.01"} 69
tsdb_compaction_duration_bucket{le="5",quantile="0.01"} 69
tsdb_compaction_duration_bucket{le="10",quantile="0.01"} 69
tsdb_compaction_duration_bucket{le="+Inf",quantile="0.01"} 69
tsdb_compaction_duration_sum 28.740810936000006
tsdb_compaction_duration_count 69
`,
			problems: []promlint.Problem{{
				Metric: "tsdb_compaction_duration",
				Text:   `non-summary metrics should not have "quantile" label`,
			}},
		},
		{
			name: "summary with le label",
			in: `
# HELP go_gc_duration_seconds A summary of the GC invocation durations.
# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0",le="0.01"} 4.2365e-05
go_gc_duration_seconds{quantile="0.25",le="0.01"} 8.1492e-05
go_gc_duration_seconds{quantile="0.5",le="0.01"} 0.000100656
go_gc_duration_seconds{quantile="0.75",le="0.01"} 0.000113913
go_gc_duration_seconds{quantile="1",le="0.01"} 0.021754305
go_gc_duration_seconds_sum 1.769429004
go_gc_duration_seconds_count 5962
`,
			problems: []promlint.Problem{{
				Metric: "go_gc_duration_seconds",
				Text:   `non-histogram metrics should not have "le" label`,
			}},
		},
		{
			name: "histogram OK",
			in: `
# HELP tsdb_compaction_duration Duration of compaction runs.
# TYPE tsdb_compaction_duration histogram
tsdb_compaction_duration_bucket{le="0.005"} 0
tsdb_compaction_duration_bucket{le="0.01"} 0
tsdb_compaction_duration_bucket{le="0.025"} 0
tsdb_compaction_duration_bucket{le="0.05"} 0
tsdb_compaction_duration_bucket{le="0.1"} 0
tsdb_compaction_duration_bucket{le="0.25"} 0
tsdb_compaction_duration_bucket{le="0.5"} 57
tsdb_compaction_duration_bucket{le="1"} 68
tsdb_compaction_duration_bucket{le="2.5"} 69
tsdb_compaction_duration_bucket{le="5"} 69
tsdb_compaction_duration_bucket{le="10"} 69
tsdb_compaction_duration_bucket{le="+Inf"} 69
tsdb_compaction_duration_sum 28.740810936000006
tsdb_compaction_duration_count 69
`,
		},
		{
			name: "summary OK",
			in: `
# HELP go_gc_duration_seconds A summary of the GC invocation durations.
# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0"} 4.2365e-05
go_gc_duration_seconds{quantile="0.25"} 8.1492e-05
go_gc_duration_seconds{quantile="0.5"} 0.000100656
go_gc_duration_seconds{quantile="0.75"} 0.000113913
go_gc_duration_seconds{quantile="1"} 0.021754305
go_gc_duration_seconds_sum 1.769429004
go_gc_duration_seconds_count 5962
`,
		},
	}
	runTests(t, tests)
}

func TestLintMetricTypeInName(t *testing.T) {
	genTest := func(n, t, err string, problems ...promlint.Problem) test {
		return test{
			name: fmt.Sprintf("%s with _%s suffix", t, t),
			in: fmt.Sprintf(`
# HELP %s Test metric.
# TYPE %s %s
%s 10
`, n, n, t, n),
			problems: append(problems, promlint.Problem{
				Metric: n,
				Text:   fmt.Sprintf(`metric name should not include type '%s'`, err),
			}),
		}
	}

	twoProbTest := genTest("http_requests_counter", "counter", "counter", promlint.Problem{
		Metric: "http_requests_counter",
		Text:   `counter metrics should have "_total" suffix`,
	})

	tests := []test{
		twoProbTest,
		genTest("instance_memory_limit_bytes_gauge", "gauge", "gauge"),
		genTest("request_duration_seconds_summary", "summary", "summary"),
		genTest("request_duration_seconds_summary", "histogram", "summary"),
		genTest("request_duration_seconds_histogram", "histogram", "histogram"),
		genTest("request_duration_seconds_HISTOGRAM", "histogram", "histogram"),

		genTest("instance_memory_limit_gauge_bytes", "gauge", "gauge"),
	}
	runTests(t, tests)
}

func TestLintReservedChars(t *testing.T) {
	tests := []test{
		{
			name: "request_duration::_seconds",
			in: `
# HELP request_duration::_seconds Test metric.
# TYPE request_duration::_seconds histogram
request_duration::_seconds 10
`,
			problems: []promlint.Problem{
				{
					Metric: "request_duration::_seconds",
					Text:   "metric names should not contain ':'",
				},
			},
		},
	}
	runTests(t, tests)
}

func TestLintCamelCase(t *testing.T) {
	tests := []test{
		{
			name: "requestDuration_seconds",
			in: `
# HELP requestDuration_seconds Test metric.
# TYPE requestDuration_seconds histogram
requestDuration_seconds 10
`,
			problems: []promlint.Problem{
				{
					Metric: "requestDuration_seconds",
					Text:   "metric names should be written in 'snake_case' not 'camelCase'",
				},
			},
		},
		{
			name: "request_duration_seconds",
			in: `
# HELP request_duration_seconds Test metric.
# TYPE request_duration_seconds histogram
request_duration_seconds{httpService="foo"} 10
`,
			problems: []promlint.Problem{
				{
					Metric: "request_duration_seconds",
					Text:   "label names should be written in 'snake_case' not 'camelCase'",
				},
			},
		},
	}
	runTests(t, tests)
}

func TestLintUnitAbbreviations(t *testing.T) {
	genTest := func(n string) test {
		return test{
			name: fmt.Sprintf("%s with abbreviated unit", n),
			in: fmt.Sprintf(`
# HELP %s Test metric.
# TYPE %s gauge
%s 10
`, n, n, n),
			problems: []promlint.Problem{
				{
					Metric: n,
					Text:   "metric names should not contain abbreviated units",
				},
			},
		}
	}
	tests := []test{
		genTest("instance_memory_limit_b"),
		genTest("instance_memory_limit_kb"),
		genTest("instance_memory_limit_mb"),
		genTest("instance_memory_limit_MB"),
		genTest("instance_memory_limit_gb"),
		genTest("instance_memory_limit_tb"),
		genTest("instance_memory_limit_pb"),

		genTest("request_duration_s"),
		genTest("request_duration_ms"),
		genTest("request_duration_us"),
		genTest("request_duration_ns"),
		genTest("request_duration_sec"),
		genTest("request_sec_duration"),
		genTest("request_duration_m"),
		genTest("request_duration_h"),
		genTest("request_duration_d"),
	}
	runTests(t, tests)
}

func runTests(t *testing.T, tests []test) {
	t.Helper()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := promlint.New(strings.NewReader(tt.in))

			problems, err := l.Lint()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if want, got := tt.problems, problems; !reflect.DeepEqual(want, got) {
				t.Fatalf("unexpected problems:\n- want: %v\n-  got: %v",
					want, got)
			}
		})
	}
}
