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

func TestRateCalculator(t *testing.T) {
	key := core.PodContainerKey("ns1", "pod1", "c")
	now := time.Now()

	prev := &core.DataBatch{
		Timestamp: now.Add(-time.Minute),
		MetricSets: map[string]*core.MetricSet{
			key: {
				CreateTime: now.Add(-time.Hour),
				ScrapeTime: now.Add(-60 * time.Second),

				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePodContainer,
				},
				MetricValues: map[string]core.MetricValue{
					core.MetricCpuUsage.MetricDescriptor.Name: core.MetricValue{
						ValueType:  core.ValueInt64,
						MetricType: core.MetricCumulative,
						IntValue:   947130377781,
					},
					core.MetricNetworkTxErrors.MetricDescriptor.Name: core.MetricValue{
						ValueType:  core.ValueInt64,
						MetricType: core.MetricCumulative,
						IntValue:   0,
					},
				},
			},
		},
	}

	current := &core.DataBatch{
		Timestamp: now,
		MetricSets: map[string]*core.MetricSet{

			key: {
				CreateTime: now.Add(-time.Hour),
				ScrapeTime: now,

				Labels: map[string]string{
					core.LabelMetricSetType.Key: core.MetricSetTypePodContainer,
				},
				MetricValues: map[string]core.MetricValue{
					core.MetricCpuUsage.MetricDescriptor.Name: core.MetricValue{
						ValueType:  core.ValueInt64,
						MetricType: core.MetricCumulative,
						IntValue:   948071062732,
					},
					core.MetricNetworkTxErrors.MetricDescriptor.Name: core.MetricValue{
						ValueType:  core.ValueInt64,
						MetricType: core.MetricCumulative,
						IntValue:   120,
					},
				},
			},
		},
	}

	procesor := NewRateCalculator(core.RateMetricsMapping)
	procesor.Process(prev)
	procesor.Process(current)

	ms := current.MetricSets[key]
	cpuRate := ms.MetricValues[core.MetricCpuUsageRate.Name]
	txeRate := ms.MetricValues[core.MetricNetworkTxErrorsRate.Name]

	assert.InEpsilon(t, 13, cpuRate.IntValue, 2)
	assert.InEpsilon(t, 2, txeRate.FloatValue, 0.1)
}
