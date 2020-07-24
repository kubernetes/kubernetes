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
	"testing"

	"k8s.io/utils/pointer"

	dto "github.com/prometheus/client_model/go"
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

func TestHistogramClear(t *testing.T) {
	samples := []float64{0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 6, 6, 6, 6}
	bounds := []float64{1, 2, 4, 8}
	h := samples2Histogram(samples, bounds)

	if *h.SampleCount == 0 {
		t.Errorf("Expected histogram .SampleCount to be non-zero")
	}
	if *h.SampleSum == 0 {
		t.Errorf("Expected histogram .SampleSum to be non-zero")
	}

	for _, b := range h.Bucket {
		if b.CumulativeCount != nil {
			if *b.CumulativeCount == 0 {
				t.Errorf("Expected histogram bucket to have non-zero comulative count")
			}
		}
	}

	h.Clear()

	if *h.SampleCount != 0 {
		t.Errorf("Expected histogram .SampleCount to be zero, have %v instead", *h.SampleCount)
	}

	if *h.SampleSum != 0 {
		t.Errorf("Expected histogram .SampleSum to be zero, have %v instead", *h.SampleSum)
	}

	for _, b := range h.Bucket {
		if b.CumulativeCount != nil {
			if *b.CumulativeCount != 0 {
				t.Errorf("Expected histogram bucket to have zero comulative count, have %v instead", *b.CumulativeCount)
			}
		}
		if b.UpperBound != nil {
			*b.UpperBound = 0
		}
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
