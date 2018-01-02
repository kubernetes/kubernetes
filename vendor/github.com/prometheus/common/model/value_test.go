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
	"encoding/json"
	"math"
	"reflect"
	"sort"
	"testing"
)

func TestEqualValues(t *testing.T) {
	tests := map[string]struct {
		in1, in2 SampleValue
		want     bool
	}{
		"equal floats": {
			in1:  3.14,
			in2:  3.14,
			want: true,
		},
		"unequal floats": {
			in1:  3.14,
			in2:  3.1415,
			want: false,
		},
		"positive inifinities": {
			in1:  SampleValue(math.Inf(+1)),
			in2:  SampleValue(math.Inf(+1)),
			want: true,
		},
		"negative inifinities": {
			in1:  SampleValue(math.Inf(-1)),
			in2:  SampleValue(math.Inf(-1)),
			want: true,
		},
		"different inifinities": {
			in1:  SampleValue(math.Inf(+1)),
			in2:  SampleValue(math.Inf(-1)),
			want: false,
		},
		"number and infinity": {
			in1:  42,
			in2:  SampleValue(math.Inf(+1)),
			want: false,
		},
		"number and NaN": {
			in1:  42,
			in2:  SampleValue(math.NaN()),
			want: false,
		},
		"NaNs": {
			in1:  SampleValue(math.NaN()),
			in2:  SampleValue(math.NaN()),
			want: true, // !!!
		},
	}

	for name, test := range tests {
		got := test.in1.Equal(test.in2)
		if got != test.want {
			t.Errorf("Comparing %s, %f and %f: got %t, want %t", name, test.in1, test.in2, got, test.want)
		}
	}
}

func TestEqualSamples(t *testing.T) {
	testSample := &Sample{}

	tests := map[string]struct {
		in1, in2 *Sample
		want     bool
	}{
		"equal pointers": {
			in1:  testSample,
			in2:  testSample,
			want: true,
		},
		"different metrics": {
			in1:  &Sample{Metric: Metric{"foo": "bar"}},
			in2:  &Sample{Metric: Metric{"foo": "biz"}},
			want: false,
		},
		"different timestamp": {
			in1:  &Sample{Timestamp: 0},
			in2:  &Sample{Timestamp: 1},
			want: false,
		},
		"different value": {
			in1:  &Sample{Value: 0},
			in2:  &Sample{Value: 1},
			want: false,
		},
		"equal samples": {
			in1: &Sample{
				Metric:    Metric{"foo": "bar"},
				Timestamp: 0,
				Value:     1,
			},
			in2: &Sample{
				Metric:    Metric{"foo": "bar"},
				Timestamp: 0,
				Value:     1,
			},
			want: true,
		},
	}

	for name, test := range tests {
		got := test.in1.Equal(test.in2)
		if got != test.want {
			t.Errorf("Comparing %s, %v and %v: got %t, want %t", name, test.in1, test.in2, got, test.want)
		}
	}

}

func TestSamplePairJSON(t *testing.T) {
	input := []struct {
		plain string
		value SamplePair
	}{
		{
			plain: `[1234.567,"123.1"]`,
			value: SamplePair{
				Value:     123.1,
				Timestamp: 1234567,
			},
		},
	}

	for _, test := range input {
		b, err := json.Marshal(test.value)
		if err != nil {
			t.Error(err)
			continue
		}

		if string(b) != test.plain {
			t.Errorf("encoding error: expected %q, got %q", test.plain, b)
			continue
		}

		var sp SamplePair
		err = json.Unmarshal(b, &sp)
		if err != nil {
			t.Error(err)
			continue
		}

		if sp != test.value {
			t.Errorf("decoding error: expected %v, got %v", test.value, sp)
		}
	}
}

func TestSampleJSON(t *testing.T) {
	input := []struct {
		plain string
		value Sample
	}{
		{
			plain: `{"metric":{"__name__":"test_metric"},"value":[1234.567,"123.1"]}`,
			value: Sample{
				Metric: Metric{
					MetricNameLabel: "test_metric",
				},
				Value:     123.1,
				Timestamp: 1234567,
			},
		},
	}

	for _, test := range input {
		b, err := json.Marshal(test.value)
		if err != nil {
			t.Error(err)
			continue
		}

		if string(b) != test.plain {
			t.Errorf("encoding error: expected %q, got %q", test.plain, b)
			continue
		}

		var sv Sample
		err = json.Unmarshal(b, &sv)
		if err != nil {
			t.Error(err)
			continue
		}

		if !reflect.DeepEqual(sv, test.value) {
			t.Errorf("decoding error: expected %v, got %v", test.value, sv)
		}
	}
}

func TestVectorJSON(t *testing.T) {
	input := []struct {
		plain string
		value Vector
	}{
		{
			plain: `[]`,
			value: Vector{},
		},
		{
			plain: `[{"metric":{"__name__":"test_metric"},"value":[1234.567,"123.1"]}]`,
			value: Vector{&Sample{
				Metric: Metric{
					MetricNameLabel: "test_metric",
				},
				Value:     123.1,
				Timestamp: 1234567,
			}},
		},
		{
			plain: `[{"metric":{"__name__":"test_metric"},"value":[1234.567,"123.1"]},{"metric":{"foo":"bar"},"value":[1.234,"+Inf"]}]`,
			value: Vector{
				&Sample{
					Metric: Metric{
						MetricNameLabel: "test_metric",
					},
					Value:     123.1,
					Timestamp: 1234567,
				},
				&Sample{
					Metric: Metric{
						"foo": "bar",
					},
					Value:     SampleValue(math.Inf(1)),
					Timestamp: 1234,
				},
			},
		},
	}

	for _, test := range input {
		b, err := json.Marshal(test.value)
		if err != nil {
			t.Error(err)
			continue
		}

		if string(b) != test.plain {
			t.Errorf("encoding error: expected %q, got %q", test.plain, b)
			continue
		}

		var vec Vector
		err = json.Unmarshal(b, &vec)
		if err != nil {
			t.Error(err)
			continue
		}

		if !reflect.DeepEqual(vec, test.value) {
			t.Errorf("decoding error: expected %v, got %v", test.value, vec)
		}
	}
}

func TestScalarJSON(t *testing.T) {
	input := []struct {
		plain string
		value Scalar
	}{
		{
			plain: `[123.456,"456"]`,
			value: Scalar{
				Timestamp: 123456,
				Value:     456,
			},
		},
		{
			plain: `[123123.456,"+Inf"]`,
			value: Scalar{
				Timestamp: 123123456,
				Value:     SampleValue(math.Inf(1)),
			},
		},
		{
			plain: `[123123.456,"-Inf"]`,
			value: Scalar{
				Timestamp: 123123456,
				Value:     SampleValue(math.Inf(-1)),
			},
		},
	}

	for _, test := range input {
		b, err := json.Marshal(test.value)
		if err != nil {
			t.Error(err)
			continue
		}

		if string(b) != test.plain {
			t.Errorf("encoding error: expected %q, got %q", test.plain, b)
			continue
		}

		var sv Scalar
		err = json.Unmarshal(b, &sv)
		if err != nil {
			t.Error(err)
			continue
		}

		if sv != test.value {
			t.Errorf("decoding error: expected %v, got %v", test.value, sv)
		}
	}
}

func TestStringJSON(t *testing.T) {
	input := []struct {
		plain string
		value String
	}{
		{
			plain: `[123.456,"test"]`,
			value: String{
				Timestamp: 123456,
				Value:     "test",
			},
		},
		{
			plain: `[123123.456,"台北"]`,
			value: String{
				Timestamp: 123123456,
				Value:     "台北",
			},
		},
	}

	for _, test := range input {
		b, err := json.Marshal(test.value)
		if err != nil {
			t.Error(err)
			continue
		}

		if string(b) != test.plain {
			t.Errorf("encoding error: expected %q, got %q", test.plain, b)
			continue
		}

		var sv String
		err = json.Unmarshal(b, &sv)
		if err != nil {
			t.Error(err)
			continue
		}

		if sv != test.value {
			t.Errorf("decoding error: expected %v, got %v", test.value, sv)
		}
	}
}

func TestVectorSort(t *testing.T) {
	input := Vector{
		&Sample{
			Metric: Metric{
				MetricNameLabel: "A",
			},
			Timestamp: 1,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "A",
			},
			Timestamp: 2,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "C",
			},
			Timestamp: 1,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "C",
			},
			Timestamp: 2,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "B",
			},
			Timestamp: 1,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "B",
			},
			Timestamp: 2,
		},
	}

	expected := Vector{
		&Sample{
			Metric: Metric{
				MetricNameLabel: "A",
			},
			Timestamp: 1,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "A",
			},
			Timestamp: 2,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "B",
			},
			Timestamp: 1,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "B",
			},
			Timestamp: 2,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "C",
			},
			Timestamp: 1,
		},
		&Sample{
			Metric: Metric{
				MetricNameLabel: "C",
			},
			Timestamp: 2,
		},
	}

	sort.Sort(input)

	for i, actual := range input {
		actualFp := actual.Metric.Fingerprint()
		expectedFp := expected[i].Metric.Fingerprint()

		if actualFp != expectedFp {
			t.Fatalf("%d. Incorrect fingerprint. Got %s; want %s", i, actualFp.String(), expectedFp.String())
		}

		if actual.Timestamp != expected[i].Timestamp {
			t.Fatalf("%d. Incorrect timestamp. Got %s; want %s", i, actual.Timestamp, expected[i].Timestamp)
		}
	}
}
