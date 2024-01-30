/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package testutil

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	dto "github.com/prometheus/client_model/go"
	"k8s.io/component-base/metrics"
	"k8s.io/utils/pointer"
)

func samples2Histogram(samples []float64, upperBounds []float64) Histogram {
	histogram := dto.Histogram{
		SampleCount: uint64Ptr(0),
		SampleSum:   pointer.Float64Ptr(0.0),
	}

	for _, ub := range upperBounds {
		histogram.Bucket = append(histogram.Bucket, &dto.Bucket{
			CumulativeCount: uint64Ptr(0),
			UpperBound:      pointer.Float64Ptr(ub),
		})
	}

	for _, sample := range samples {
		for i, bucket := range histogram.Bucket {
			if sample < *bucket.UpperBound {
				*histogram.Bucket[i].CumulativeCount++
			}
		}
		*histogram.SampleCount++
		*histogram.SampleSum += sample
	}
	return Histogram{
		&histogram,
	}
}

func TestHistogramQuantile(t *testing.T) {
	tests := []struct {
		name    string
		samples []float64
		bounds  []float64
		q50     float64
		q90     float64
		q99     float64
	}{
		{
			name:    "Repeating numbers",
			samples: []float64{0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 6, 6, 6, 6},
			bounds:  []float64{1, 2, 4, 8},
			q50:     2,
			q90:     6.4,
			q99:     7.84,
		},
		{
			name:    "Random numbers",
			samples: []float64{11, 67, 61, 21, 40, 36, 52, 63, 8, 3, 67, 35, 61, 1, 36, 58},
			bounds:  []float64{10, 20, 40, 80},
			q50:     40,
			q90:     72,
			q99:     79.2,
		},
		{
			name:    "The last bucket is empty",
			samples: []float64{6, 34, 30, 10, 20, 18, 26, 31, 4, 2, 33, 17, 30, 1, 18, 29},
			bounds:  []float64{10, 20, 40, 80},
			q50:     20,
			q90:     36,
			q99:     39.6,
		},
		{
			name:    "The last bucket has positive infinity upper bound",
			samples: []float64{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 500},
			bounds:  []float64{10, 20, 40, math.Inf(1)},
			q50:     5.3125,
			q90:     9.5625,
			q99:     40,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			h := samples2Histogram(test.samples, test.bounds)
			q50 := h.Quantile(0.5)
			q90 := h.Quantile(0.9)
			q99 := h.Quantile(0.99)
			q999999 := h.Quantile(0.999999)

			if q50 != test.q50 {
				t.Errorf("Expected q50 to be %v, got %v instead", test.q50, q50)
			}
			if q90 != test.q90 {
				t.Errorf("Expected q90 to be %v, got %v instead", test.q90, q90)
			}
			if q99 != test.q99 {
				t.Errorf("Expected q99 to be %v, got %v instead", test.q99, q99)
			}
			lastUpperBound := test.bounds[len(test.bounds)-1]
			if !(q999999 < lastUpperBound) {
				t.Errorf("Expected q999999 to be less than %v, got %v instead", lastUpperBound, q999999)
			}
		})
	}
}

func TestHistogramValidate(t *testing.T) {
	tests := []struct {
		name string
		h    Histogram
		err  error
	}{
		{
			name: "nil SampleCount",
			h: Histogram{
				&dto.Histogram{},
			},
			err: fmt.Errorf("nil or empty histogram SampleCount"),
		},
		{
			name: "empty SampleCount",
			h: Histogram{
				&dto.Histogram{
					SampleCount: uint64Ptr(0),
				},
			},
			err: fmt.Errorf("nil or empty histogram SampleCount"),
		},
		{
			name: "nil SampleSum",
			h: Histogram{
				&dto.Histogram{
					SampleCount: uint64Ptr(1),
				},
			},
			err: fmt.Errorf("nil or empty histogram SampleSum"),
		},
		{
			name: "empty SampleSum",
			h: Histogram{
				&dto.Histogram{
					SampleCount: uint64Ptr(1),
					SampleSum:   pointer.Float64Ptr(0.0),
				},
			},
			err: fmt.Errorf("nil or empty histogram SampleSum"),
		},
		{
			name: "nil bucket",
			h: Histogram{
				&dto.Histogram{
					SampleCount: uint64Ptr(1),
					SampleSum:   pointer.Float64Ptr(1.0),
					Bucket: []*dto.Bucket{
						nil,
					},
				},
			},
			err: fmt.Errorf("empty histogram bucket"),
		},
		{
			name: "nil bucket UpperBound",
			h: Histogram{
				&dto.Histogram{
					SampleCount: uint64Ptr(1),
					SampleSum:   pointer.Float64Ptr(1.0),
					Bucket: []*dto.Bucket{
						{},
					},
				},
			},
			err: fmt.Errorf("nil or negative histogram bucket UpperBound"),
		},
		{
			name: "negative bucket UpperBound",
			h: Histogram{
				&dto.Histogram{
					SampleCount: uint64Ptr(1),
					SampleSum:   pointer.Float64Ptr(1.0),
					Bucket: []*dto.Bucket{
						{UpperBound: pointer.Float64Ptr(-1.0)},
					},
				},
			},
			err: fmt.Errorf("nil or negative histogram bucket UpperBound"),
		},
		{
			name: "valid histogram",
			h: samples2Histogram(
				[]float64{0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 6, 6, 6, 6},
				[]float64{1, 2, 4, 8},
			),
		},
	}

	for _, test := range tests {
		err := test.h.Validate()
		if test.err != nil {
			if err == nil || err.Error() != test.err.Error() {
				t.Errorf("Expected %q error, got %q instead", test.err, err)
			}
		} else {
			if err != nil {
				t.Errorf("Expected error to be nil, got %q instead", err)
			}
		}
	}
}

func TestLabelsMatch(t *testing.T) {
	cases := []struct {
		name          string
		metric        *dto.Metric
		labelFilter   map[string]string
		expectedMatch bool
	}{
		{name: "metric labels and labelFilter have the same labels and values", metric: &dto.Metric{
			Label: []*dto.LabelPair{
				{Name: pointer.StringPtr("a"), Value: pointer.StringPtr("1")},
				{Name: pointer.StringPtr("b"), Value: pointer.StringPtr("2")},
				{Name: pointer.StringPtr("c"), Value: pointer.StringPtr("3")},
			}}, labelFilter: map[string]string{
			"a": "1",
			"b": "2",
			"c": "3",
		}, expectedMatch: true},
		{name: "metric labels contain all labelFilter labels, and labelFilter is a subset of metric labels", metric: &dto.Metric{
			Label: []*dto.LabelPair{
				{Name: pointer.StringPtr("a"), Value: pointer.StringPtr("1")},
				{Name: pointer.StringPtr("b"), Value: pointer.StringPtr("2")},
				{Name: pointer.StringPtr("c"), Value: pointer.StringPtr("3")},
			}}, labelFilter: map[string]string{
			"a": "1",
			"b": "2",
		}, expectedMatch: true},
		{name: "metric labels don't have all labelFilter labels and value", metric: &dto.Metric{
			Label: []*dto.LabelPair{
				{Name: pointer.StringPtr("a"), Value: pointer.StringPtr("1")},
				{Name: pointer.StringPtr("b"), Value: pointer.StringPtr("2")},
			}}, labelFilter: map[string]string{
			"a": "1",
			"b": "2",
			"c": "3",
		}, expectedMatch: false},
		{name: "The intersection of metric labels and labelFilter labels is empty", metric: &dto.Metric{
			Label: []*dto.LabelPair{
				{Name: pointer.StringPtr("aa"), Value: pointer.StringPtr("11")},
				{Name: pointer.StringPtr("bb"), Value: pointer.StringPtr("22")},
				{Name: pointer.StringPtr("cc"), Value: pointer.StringPtr("33")},
			}}, labelFilter: map[string]string{
			"a": "1",
			"b": "2",
			"c": "3",
		}, expectedMatch: false},
		{name: "metric labels have the same labels names but different values with labelFilter labels and value", metric: &dto.Metric{
			Label: []*dto.LabelPair{
				{Name: pointer.StringPtr("a"), Value: pointer.StringPtr("1")},
				{Name: pointer.StringPtr("b"), Value: pointer.StringPtr("2")},
				{Name: pointer.StringPtr("c"), Value: pointer.StringPtr("3")},
			}}, labelFilter: map[string]string{
			"a": "11",
			"b": "2",
			"c": "3",
		}, expectedMatch: false},
		{name: "metric labels contain label name but different values with labelFilter labels and value", metric: &dto.Metric{
			Label: []*dto.LabelPair{
				{Name: pointer.StringPtr("a"), Value: pointer.StringPtr("1")},
				{Name: pointer.StringPtr("b"), Value: pointer.StringPtr("2")},
				{Name: pointer.StringPtr("c"), Value: pointer.StringPtr("33")},
				{Name: pointer.StringPtr("d"), Value: pointer.StringPtr("4")},
			}}, labelFilter: map[string]string{
			"a": "1",
			"b": "2",
			"c": "3",
		}, expectedMatch: false},
		{name: "metric labels is empty and labelFilter is not empty", metric: &dto.Metric{
			Label: []*dto.LabelPair{}}, labelFilter: map[string]string{
			"a": "1",
			"b": "2",
			"c": "3",
		}, expectedMatch: false},
		{name: "metric labels is not empty and labelFilter is empty", metric: &dto.Metric{
			Label: []*dto.LabelPair{
				{Name: pointer.StringPtr("a"), Value: pointer.StringPtr("1")},
				{Name: pointer.StringPtr("b"), Value: pointer.StringPtr("2")},
			}}, labelFilter: map[string]string{}, expectedMatch: true},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := LabelsMatch(tt.metric, tt.labelFilter)
			if got != tt.expectedMatch {
				t.Errorf("Expected %v, got %v instead", tt.expectedMatch, got)
			}
		})
	}
}

func TestHistogramVec_GetAggregatedSampleCount(t *testing.T) {
	tests := []struct {
		name string
		vec  HistogramVec
		want uint64
	}{
		{
			name: "nil case",
			want: 0,
		},
		{
			name: "zero case",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(0), SampleSum: pointer.Float64Ptr(0.0)}},
			},
			want: 0,
		},
		{
			name: "standard case",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(1), SampleSum: pointer.Float64Ptr(2.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(2), SampleSum: pointer.Float64Ptr(4.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(4), SampleSum: pointer.Float64Ptr(8.0)}},
			},
			want: 7,
		},
		{
			name: "mixed case",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(1), SampleSum: pointer.Float64Ptr(2.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(0), SampleSum: pointer.Float64Ptr(0.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(2), SampleSum: pointer.Float64Ptr(4.0)}},
			},
			want: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.vec.GetAggregatedSampleCount(); got != tt.want {
				t.Errorf("GetAggregatedSampleCount() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHistogramVec_GetAggregatedSampleSum(t *testing.T) {
	tests := []struct {
		name string
		vec  HistogramVec
		want float64
	}{
		{
			name: "nil case",
			want: 0.0,
		},
		{
			name: "zero case",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(0), SampleSum: pointer.Float64Ptr(0.0)}},
			},
			want: 0.0,
		},
		{
			name: "standard case",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(1), SampleSum: pointer.Float64Ptr(2.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(2), SampleSum: pointer.Float64Ptr(4.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(4), SampleSum: pointer.Float64Ptr(8.0)}},
			},
			want: 14.0,
		},
		{
			name: "mixed case",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(1), SampleSum: pointer.Float64Ptr(2.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(0), SampleSum: pointer.Float64Ptr(0.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(2), SampleSum: pointer.Float64Ptr(4.0)}},
			},
			want: 6.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.vec.GetAggregatedSampleSum(); got != tt.want {
				t.Errorf("GetAggregatedSampleSum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHistogramVec_Quantile(t *testing.T) {
	tests := []struct {
		name     string
		samples  [][]float64
		bounds   []float64
		quantile float64
		want     []float64
	}{
		{
			name: "duplicated histograms",
			samples: [][]float64{
				{0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 6, 6, 6, 6},
				{0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 6, 6, 6, 6},
				{0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 6, 6, 6, 6},
			},
			bounds: []float64{1, 2, 4, 8},
			want:   []float64{2, 6.4, 7.2, 7.84},
		},
		{
			name: "random numbers",
			samples: [][]float64{
				{8, 35, 47, 61, 56, 69, 66, 74, 35, 69, 5, 38, 58, 40, 36, 12},
				{79, 44, 57, 46, 11, 8, 53, 77, 13, 35, 38, 47, 73, 16, 26, 29},
				{51, 76, 22, 55, 20, 63, 59, 66, 34, 58, 64, 16, 79, 7, 58, 28},
			},
			bounds: []float64{10, 20, 40, 80},
			want:   []float64{44.44, 72.89, 76.44, 79.29},
		},
		{
			name: "single histogram",
			samples: [][]float64{
				{6, 34, 30, 10, 20, 18, 26, 31, 4, 2, 33, 17, 30, 1, 18, 29},
			},
			bounds: []float64{10, 20, 40, 80},
			want:   []float64{20, 36, 38, 39.6},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var vec HistogramVec
			for _, sample := range tt.samples {
				histogram := samples2Histogram(sample, tt.bounds)
				vec = append(vec, &histogram)
			}
			var got []float64
			for _, q := range []float64{0.5, 0.9, 0.95, 0.99} {
				got = append(got, math.Round(vec.Quantile(q)*100)/100)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Quantile() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHistogramVec_Validate(t *testing.T) {
	tests := []struct {
		name string
		vec  HistogramVec
		want error
	}{
		{
			name: "nil SampleCount",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(1), SampleSum: pointer.Float64Ptr(1.0)}},
				&Histogram{&dto.Histogram{SampleSum: pointer.Float64Ptr(2.0)}},
			},
			want: fmt.Errorf("nil or empty histogram SampleCount"),
		},
		{
			name: "valid HistogramVec",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(1), SampleSum: pointer.Float64Ptr(1.0)}},
				&Histogram{&dto.Histogram{SampleCount: uint64Ptr(2), SampleSum: pointer.Float64Ptr(2.0)}},
			},
		},
		{
			name: "different bucket size",
			vec: HistogramVec{
				&Histogram{&dto.Histogram{
					SampleCount: uint64Ptr(4),
					SampleSum:   pointer.Float64Ptr(10.0),
					Bucket: []*dto.Bucket{
						{CumulativeCount: uint64Ptr(1), UpperBound: pointer.Float64Ptr(1)},
						{CumulativeCount: uint64Ptr(2), UpperBound: pointer.Float64Ptr(2)},
						{CumulativeCount: uint64Ptr(5), UpperBound: pointer.Float64Ptr(4)},
					},
				}},
				&Histogram{&dto.Histogram{
					SampleCount: uint64Ptr(3),
					SampleSum:   pointer.Float64Ptr(8.0),
					Bucket: []*dto.Bucket{
						{CumulativeCount: uint64Ptr(1), UpperBound: pointer.Float64Ptr(2)},
						{CumulativeCount: uint64Ptr(3), UpperBound: pointer.Float64Ptr(4)},
					},
				}},
			},
			want: fmt.Errorf("found different bucket size: expect 3, but got 2 at index 1"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.vec.Validate(); fmt.Sprintf("%v", got) != fmt.Sprintf("%v", tt.want) {
				t.Errorf("Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetHistogramVecFromGatherer(t *testing.T) {
	tests := []struct {
		name    string
		lvMap   map[string]string
		wantVec HistogramVec
	}{
		{
			name:  "filter with one label",
			lvMap: map[string]string{"label1": "value1-0"},
			wantVec: HistogramVec{
				&Histogram{&dto.Histogram{
					SampleCount: uint64Ptr(1),
					SampleSum:   pointer.Float64Ptr(1.5),
					Bucket: []*dto.Bucket{
						{CumulativeCount: uint64Ptr(0), UpperBound: pointer.Float64Ptr(0.5)},
						{CumulativeCount: uint64Ptr(1), UpperBound: pointer.Float64Ptr(2.0)},
						{CumulativeCount: uint64Ptr(1), UpperBound: pointer.Float64Ptr(5.0)},
					},
				}},
				&Histogram{&dto.Histogram{
					SampleCount: uint64Ptr(1),
					SampleSum:   pointer.Float64Ptr(2.5),
					Bucket: []*dto.Bucket{
						{CumulativeCount: uint64Ptr(0), UpperBound: pointer.Float64Ptr(0.5)},
						{CumulativeCount: uint64Ptr(0), UpperBound: pointer.Float64Ptr(2.0)},
						{CumulativeCount: uint64Ptr(1), UpperBound: pointer.Float64Ptr(5.0)},
					},
				}},
			},
		},
		{
			name:  "filter with two labels",
			lvMap: map[string]string{"label1": "value1-0", "label2": "value2-1"},
			wantVec: HistogramVec{
				&Histogram{&dto.Histogram{
					SampleCount: uint64Ptr(1),
					SampleSum:   pointer.Float64Ptr(2.5),
					Bucket: []*dto.Bucket{
						{CumulativeCount: uint64Ptr(0), UpperBound: pointer.Float64Ptr(0.5)},
						{CumulativeCount: uint64Ptr(0), UpperBound: pointer.Float64Ptr(2.0)},
						{CumulativeCount: uint64Ptr(1), UpperBound: pointer.Float64Ptr(5.0)},
					},
				}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buckets := []float64{.5, 2, 5}
			// HistogramVec has two labels defined.
			labels := []string{"label1", "label2"}
			HistogramOpts := &metrics.HistogramOpts{
				Namespace: "namespace",
				Name:      "metric_test_name",
				Subsystem: "subsystem",
				Help:      "histogram help message",
				Buckets:   buckets,
			}
			vec := metrics.NewHistogramVec(HistogramOpts, labels)
			// Use local registry
			var registry = metrics.NewKubeRegistry()
			var gather metrics.Gatherer = registry
			registry.MustRegister(vec)
			// Observe two metrics with same value for label1 but different value of label2.
			vec.WithLabelValues("value1-0", "value2-0").Observe(1.5)
			vec.WithLabelValues("value1-0", "value2-1").Observe(2.5)
			vec.WithLabelValues("value1-1", "value2-0").Observe(3.5)
			vec.WithLabelValues("value1-1", "value2-1").Observe(4.5)
			metricName := fmt.Sprintf("%s_%s_%s", HistogramOpts.Namespace, HistogramOpts.Subsystem, HistogramOpts.Name)
			histogramVec, _ := GetHistogramVecFromGatherer(gather, metricName, tt.lvMap)
			if diff := cmp.Diff(tt.wantVec, histogramVec, cmpopts.IgnoreFields(dto.Histogram{}, "state", "sizeCache", "unknownFields"), cmpopts.IgnoreFields(dto.Bucket{}, "state", "sizeCache", "unknownFields")); diff != "" {
				t.Errorf("Got unexpected HistogramVec (-want +got):\n%s", diff)
			}
		})
	}
}
