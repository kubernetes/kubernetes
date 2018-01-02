// Copyright 2015 Google Inc. All Rights Reserved.
//
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

package metricsink

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/heapster/metrics/core"
)

func makeBatches(now time.Time, key, otherKey string) (core.DataBatch, core.DataBatch, core.DataBatch) {
	batch1 := core.DataBatch{
		Timestamp: now.Add(-180 * time.Second),
		MetricSets: map[string]*core.MetricSet{
			key: {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePod,
					core.LabelPodNamespace.Key:  "ns1",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   60,
					},
					"m2": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   666,
					},
				},
			},
		},
	}

	batch2 := core.DataBatch{
		Timestamp: now.Add(-60 * time.Second),
		MetricSets: map[string]*core.MetricSet{
			key: {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePod,
					core.LabelPodNamespace.Key:  "ns1",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   40,
					},
					"m2": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   444,
					},
				},
				LabeledMetrics: []core.LabeledMetric{
					{
						Name:   "somelblmetric",
						Labels: map[string]string{"lbl1": "val1.1", "lbl2": "val2.1"},
						MetricValue: core.MetricValue{
							ValueType:  core.ValueInt64,
							MetricType: core.MetricGauge,
							IntValue:   8675,
						},
					},
					{
						Name:   "otherlblmetric",
						Labels: map[string]string{"lbl1": "val1.1", "lbl2": "val2.1"},
						MetricValue: core.MetricValue{
							ValueType:  core.ValueInt64,
							MetricType: core.MetricGauge,
							IntValue:   1234,
						},
					},
				},
			},
		},
	}

	batch3 := core.DataBatch{
		Timestamp: now.Add(-20 * time.Second),
		MetricSets: map[string]*core.MetricSet{
			key: {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePod,
					core.LabelPodNamespace.Key:  "ns1",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   20,
					},
					"m2": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   222,
					},
				},
				LabeledMetrics: []core.LabeledMetric{
					{
						Name:   "somelblmetric",
						Labels: map[string]string{"lbl1": "val1.1", "lbl2": "val2.1"},
						MetricValue: core.MetricValue{
							ValueType:  core.ValueInt64,
							MetricType: core.MetricGauge,
							IntValue:   309,
						},
					},
					{
						Name:   "somelblmetric",
						Labels: map[string]string{"lbl1": "val1.2", "lbl2": "val2.1"},
						MetricValue: core.MetricValue{
							ValueType:  core.ValueInt64,
							MetricType: core.MetricGauge,
							IntValue:   5678,
						},
					},
				},
			},
			otherKey: {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePod,
					core.LabelPodNamespace.Key:  "ns1",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   123,
					},
				},
			},
		},
	}

	return batch1, batch2, batch3
}

func TestGetMetrics(t *testing.T) {
	now := time.Now()
	key := core.PodKey("ns1", "pod1")
	otherKey := core.PodKey("ns1", "other")

	batch1, batch2, batch3 := makeBatches(now, key, otherKey)

	metrics := NewMetricSink(45*time.Second, 120*time.Second, []string{"m1"})
	metrics.ExportData(&batch1)
	metrics.ExportData(&batch2)
	metrics.ExportData(&batch3)

	//batch1 is discarded by long store
	result1 := metrics.GetMetric("m1", []string{key}, now.Add(-120*time.Second), now)
	assert.Equal(t, 2, len(result1[key]))
	assert.Equal(t, 40, result1[key][0].MetricValue.IntValue)
	assert.Equal(t, 20, result1[key][1].MetricValue.IntValue)
	assert.Equal(t, 1, len(metrics.GetMetric("m1", []string{otherKey}, now.Add(-120*time.Second), now)[otherKey]))

	//batch1 is discarded by long store and batch2 doesn't belong to time window
	assert.Equal(t, 1, len(metrics.GetMetric("m1", []string{key}, now.Add(-30*time.Second), now)[key]))

	//batch1 and batch1 are discarded by short store
	assert.Equal(t, 1, len(metrics.GetMetric("m2", []string{key}, now.Add(-120*time.Second), now)[key]))

	//nothing is in time window
	assert.Equal(t, 0, len(metrics.GetMetric("m2", []string{key}, now.Add(-10*time.Second), now)[key]))

	metricNames := metrics.GetMetricNames(key)
	assert.Equal(t, 2, len(metricNames))
	assert.Contains(t, metricNames, "m1")
	assert.Contains(t, metricNames, "m2")
}

func TestGetLabeledMetrics(t *testing.T) {
	now := time.Now().UTC()
	key := core.PodKey("ns1", "pod1")
	otherKey := core.PodKey("ns1", "other")

	batch1, batch2, batch3 := makeBatches(now, key, otherKey)

	metrics := NewMetricSink(45*time.Second, 120*time.Second, []string{"m1"})
	metrics.ExportData(&batch1)
	metrics.ExportData(&batch2)
	metrics.ExportData(&batch3)

	result := metrics.GetLabeledMetric("somelblmetric", map[string]string{"lbl1": "val1.1", "lbl2": "val2.1"}, []string{key}, now.Add(-120*time.Second), now)

	assert.Equal(t, []core.TimestampedMetricValue{
		{
			Timestamp: now.Add(-20 * time.Second),
			MetricValue: core.MetricValue{
				ValueType:  core.ValueInt64,
				MetricType: core.MetricGauge,
				IntValue:   309,
			},
		},
	}, result[key])
}

func TestGetNames(t *testing.T) {
	now := time.Now()
	key := core.PodKey("ns1", "pod1")
	otherKey := core.PodKey("ns1", "other")

	batch := core.DataBatch{
		Timestamp: now.Add(-20 * time.Second),
		MetricSets: map[string]*core.MetricSet{
			key: {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePod,
					core.LabelPodNamespace.Key:  "ns1",
					core.LabelNamespaceName.Key: "ns1",
					core.LabelPodName.Key:       "pod1",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   20,
					},
					"m2": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   222,
					},
				},
			},
			otherKey: {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePod,
					core.LabelPodNamespace.Key:  "ns2",
					core.LabelNamespaceName.Key: "ns2",
					core.LabelPodName.Key:       "pod2",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   123,
					},
				},
			},
		},
	}

	metrics := NewMetricSink(45*time.Second, 120*time.Second, []string{"m1"})
	metrics.ExportData(&batch)

	assert.Contains(t, metrics.GetPods(), "ns1/pod1")
	assert.Contains(t, metrics.GetPods(), "ns2/pod2")
	assert.Contains(t, metrics.GetPodsFromNamespace("ns1"), "pod1")
	assert.NotContains(t, metrics.GetPodsFromNamespace("ns1"), "pod2")
	assert.Contains(t, metrics.GetMetricSetKeys(), key)
	assert.Contains(t, metrics.GetMetricSetKeys(), otherKey)
}
