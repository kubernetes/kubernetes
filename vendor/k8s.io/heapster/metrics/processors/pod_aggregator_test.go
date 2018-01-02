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

package processors

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/heapster/metrics/core"
)

func TestPodAggregator(t *testing.T) {
	batch := core.DataBatch{
		Timestamp: time.Now(),
		MetricSets: map[string]*core.MetricSet{
			core.PodContainerKey("ns1", "pod1", "c1"): {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePodContainer,
					core.LabelPodName.Key:       "pod1",
					core.LabelNamespaceName.Key: "ns1",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   10,
					},
					"m2": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   222,
					},
				},
			},

			core.PodContainerKey("ns1", "pod1", "c2"): {
				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePodContainer,
					core.LabelPodName.Key:       "pod1",
					core.LabelNamespaceName.Key: "ns1",
				},
				MetricValues: map[string]core.MetricValue{
					"m1": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   100,
					},
					"m3": {
						ValueType:  core.ValueInt64,
						MetricType: core.MetricGauge,
						IntValue:   30,
					},
				},
			},
		},
	}
	processor := PodAggregator{}
	result, err := processor.Process(&batch)
	assert.NoError(t, err)
	pod, found := result.MetricSets[core.PodKey("ns1", "pod1")]
	assert.True(t, found)

	m1, found := pod.MetricValues["m1"]
	assert.True(t, found)
	assert.Equal(t, 110, m1.IntValue)

	m2, found := pod.MetricValues["m2"]
	assert.True(t, found)
	assert.Equal(t, 222, m2.IntValue)

	m3, found := pod.MetricValues["m3"]
	assert.True(t, found)
	assert.Equal(t, 30, m3.IntValue)

	labelPodName, found := pod.Labels[core.LabelPodName.Key]
	assert.True(t, found)
	assert.Equal(t, "pod1", labelPodName)

	labelNsName, found := pod.Labels[core.LabelNamespaceName.Key]
	assert.True(t, found)
	assert.Equal(t, "ns1", labelNsName)
}
