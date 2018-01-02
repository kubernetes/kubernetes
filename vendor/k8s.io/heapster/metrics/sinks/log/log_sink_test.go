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

package logsink

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/heapster/metrics/core"
)

func TestSimpleWrite(t *testing.T) {
	now := time.Now()
	batch := core.DataBatch{
		Timestamp:  now,
		MetricSets: make(map[string]*core.MetricSet),
	}
	batch.MetricSets["pod1"] = &core.MetricSet{
		Labels: map[string]string{"bzium": "hocuspocus"},
		MetricValues: map[string]core.MetricValue{
			"m1": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricGauge,
				IntValue:   31415,
			},
		},
		LabeledMetrics: []core.LabeledMetric{
			{
				Name: "lm",
				MetricValue: core.MetricValue{
					MetricType: core.MetricGauge,
					ValueType:  core.ValueInt64,
					IntValue:   279,
				},
				Labels: map[string]string{
					"disk": "hard",
				},
			},
		},
	}
	log := batchToString(&batch)

	assert.True(t, strings.Contains(log, "31415"))
	assert.True(t, strings.Contains(log, "m1"))
	assert.True(t, strings.Contains(log, "bzium"))
	assert.True(t, strings.Contains(log, "hocuspocus"))
	assert.True(t, strings.Contains(log, "pod1"))
	assert.True(t, strings.Contains(log, "279"))
	assert.True(t, strings.Contains(log, "disk"))
	assert.True(t, strings.Contains(log, "hard"))
	assert.True(t, strings.Contains(log, fmt.Sprintf("%s", now)))
}

func TestSortedOutput(t *testing.T) {
	const (
		label1  = "abcLabel"
		label2  = "xyzLabel"
		pod1    = "pod1"
		pod2    = "pod2"
		metric1 = "metricA"
		metric2 = "metricB"
	)
	metricVal := core.MetricValue{
		ValueType:  core.ValueInt64,
		MetricType: core.MetricGauge,
		IntValue:   31415,
	}
	metricSet := func(pod string) *core.MetricSet {
		return &core.MetricSet{
			Labels: map[string]string{label2 + pod: pod, label1 + pod: pod},
			MetricValues: map[string]core.MetricValue{
				metric2 + pod: metricVal,
				metric1 + pod: metricVal,
			},
			LabeledMetrics: []core.LabeledMetric{},
		}
	}
	now := time.Now()
	batch := core.DataBatch{
		Timestamp: now,
		MetricSets: map[string]*core.MetricSet{
			pod2: metricSet(pod2),
			pod1: metricSet(pod1),
		},
	}
	log := batchToString(&batch)
	sorted := []string{
		pod1,
		label1 + pod1,
		label2 + pod1,
		metric1 + pod1,
		metric2 + pod1,
		pod2,
		label1 + pod2,
		label2 + pod2,
		metric1 + pod2,
		metric2 + pod2,
	}
	var (
		previous      string
		previousIndex int
	)
	for _, metric := range sorted {
		metricIndex := strings.Index(log, metric)
		assert.NotEqual(t, -1, metricIndex, "%q not found", metric)
		if previous != "" {
			assert.True(t, previousIndex < metricIndex, "%q should be before %q", previous, metric)
		}
		previous = metric
		previousIndex = metricIndex
	}
}
