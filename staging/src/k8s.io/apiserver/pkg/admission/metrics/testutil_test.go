/*
Copyright 2017 The Kubernetes Authors.

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

package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	ptype "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"k8s.io/utils/pointer"
)

func TestLabelsMatch(t *testing.T) {
	testcases := []struct {
		name          string
		metric        *ptype.Metric
		labelFilter   map[string]string
		expectedMatch bool
	}{
		{
			name: "metric's labels and labelFilter have the same labels and values",
			metric: &ptype.Metric{
				Label: []*ptype.LabelPair{
					{Name: pointer.StringPtr("labelName1"), Value: pointer.StringPtr("labelValue1")},
					{Name: pointer.StringPtr("labelName2"), Value: pointer.StringPtr("labelValue2")},
					{Name: pointer.StringPtr("labelName3"), Value: pointer.StringPtr("labelValue3")},
				},
			},
			labelFilter: map[string]string{
				"labelName1": "labelValue1",
				"labelName2": "labelValue2",
				"labelName3": "labelValue3",
			},
			expectedMatch: true,
		},
		{
			name: "metric's labels include all labelFilter labels and values",
			metric: &ptype.Metric{
				Label: []*ptype.LabelPair{
					{Name: pointer.StringPtr("labelName1"), Value: pointer.StringPtr("labelValue1")},
					{Name: pointer.StringPtr("labelName2"), Value: pointer.StringPtr("labelValue2")},
					{Name: pointer.StringPtr("labelName3"), Value: pointer.StringPtr("labelValue3")},
				},
			},
			labelFilter: map[string]string{
				"labelName1": "labelValue1",
				"labelName2": "labelValue2",
			},
			expectedMatch: true,
		},
		{
			name: "metric's labels don't have one labelFilter label and value",
			metric: &ptype.Metric{
				Label: []*ptype.LabelPair{
					{Name: pointer.StringPtr("labelName1"), Value: pointer.StringPtr("labelValue1")},
					{Name: pointer.StringPtr("labelName2"), Value: pointer.StringPtr("labelValue2")},
				},
			},
			labelFilter: map[string]string{
				"labelName1": "labelValue1",
				"labelName2": "labelValue2",
				"labelName3": "labelValue3",
			},
			expectedMatch: false,
		},
		{
			name: "metric's labels don't have any common label pair with labelFilter",
			metric: &ptype.Metric{
				Label: []*ptype.LabelPair{
					{Name: pointer.StringPtr("labelName11"), Value: pointer.StringPtr("labelValue11")},
					{Name: pointer.StringPtr("labelName22"), Value: pointer.StringPtr("labelValue22")},
					{Name: pointer.StringPtr("labelName33"), Value: pointer.StringPtr("labelValue33")},
				},
			},
			labelFilter: map[string]string{
				"labelName1": "labelValue1",
				"labelName2": "labelValue2",
				"labelName3": "labelValue3",
			},
			expectedMatch: false,
		},
		{
			name: "metric's labels have the same labels names but different values with labelFilter label and value",
			metric: &ptype.Metric{
				Label: []*ptype.LabelPair{
					{Name: pointer.StringPtr("labelName1"), Value: pointer.StringPtr("labelValue11")},
					{Name: pointer.StringPtr("labelName2"), Value: pointer.StringPtr("labelValue2")},
				},
			},
			labelFilter: map[string]string{
				"labelName1": "labelValue1",
				"labelName2": "labelValue2",
			},
			expectedMatch: false,
		},
	}
	for _, tc := range testcases {
		assert.Equal(t, tc.expectedMatch, labelsMatch(tc.metric, tc.labelFilter), tc.name)
	}
}

func labelsMatch(metric *ptype.Metric, labelFilter map[string]string) bool {
	for labelName, labelValue := range labelFilter {
		labelMatch := false
		for _, labelPair := range metric.GetLabel() {
			if labelPair.GetName() == labelName && labelPair.GetValue() == labelValue {
				labelMatch = true
				break
			}
		}
		if !labelMatch {
			return false
		}
	}
	return true
}

// expectFindMetric find a metric with the given name nad labels or reports a fatal test error.
func expectFindMetric(t *testing.T, name string, expectedLabels map[string]string) *ptype.Metric {
	metrics, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %s", err)
	}

	for _, mf := range metrics {
		if mf.GetName() == name {
			for _, metric := range mf.GetMetric() {
				if labelsMatch(metric, expectedLabels) {
					gotLabelCount := len(metric.GetLabel())
					wantLabelCount := len(expectedLabels)
					if wantLabelCount != gotLabelCount {
						t.Errorf("Got metric with %d labels, but wanted %d labels. Wanted %#+v for %s",
							gotLabelCount, wantLabelCount, expectedLabels, metric.String())
					}
					return metric
				}
			}
		}
	}
	t.Fatalf("No metric found with name %s and labels %#+v", name, expectedLabels)
	return nil
}

// expectHistogramCountTotal ensures that the sum of counts of metrics matching the labelFilter is as
// expected.
func expectHistogramCountTotal(t *testing.T, name string, labelFilter map[string]string, wantCount int) {
	metrics, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %s", err)
	}

	counterSum := 0
	for _, mf := range metrics {
		if mf.GetName() != name {
			continue // Ignore other metrics.
		}
		for _, metric := range mf.GetMetric() {
			if !labelsMatch(metric, labelFilter) {
				continue
			}
			counterSum += int(metric.GetHistogram().GetSampleCount())
		}
	}
	if wantCount != counterSum {
		t.Errorf("Wanted count %d, got %d for metric %s with labels %#+v", wantCount, counterSum, name, labelFilter)
		for _, mf := range metrics {
			if mf.GetName() == name {
				for _, metric := range mf.GetMetric() {
					t.Logf("\tnear match: %s", metric.String())
				}
			}
		}
	}
}
