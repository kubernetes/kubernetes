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

package v1

import (
	"testing"
	"time"

	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/heapster/metrics/core"
	metricsink "k8s.io/heapster/metrics/sinks/metric"
)

func TestApiFactory(t *testing.T) {
	metricSink := metricsink.MetricSink{}
	api := NewApi(false, &metricSink, nil)
	as := assert.New(t)
	for _, metric := range core.StandardMetrics {
		val, exists := api.gkeMetrics[metric.Name]
		as.True(exists)
		as.Equal(val, metric.MetricDescriptor)
	}
	for _, metric := range core.LabeledMetrics {
		val, exists := api.gkeMetrics[metric.Name]
		as.True(exists)
		as.Equal(val, metric.MetricDescriptor)
	}

	for _, metric := range core.LabeledMetrics {
		val, exists := api.gkeMetrics[metric.Name]
		as.True(exists)
		as.Equal(val, metric.MetricDescriptor)
	}
	labels := append(core.CommonLabels(), core.ContainerLabels()...)
	labels = append(labels, core.PodLabels()...)
	for _, label := range labels {
		val, exists := api.gkeLabels[label.Key]
		as.True(exists)
		as.Equal(val, label)
	}
}

func TestFuzzInput(t *testing.T) {
	api := NewApi(false, nil, nil)
	data := []*core.DataBatch{}
	fuzz.New().NilChance(0).Fuzz(&data)
	_ = api.processMetricsRequest(data)
}

func generateMetricSet(objectType string, labels []core.LabelDescriptor) *core.MetricSet {
	ms := &core.MetricSet{
		CreateTime:     time.Now().Add(-time.Hour),
		ScrapeTime:     time.Now(),
		Labels:         make(map[string]string),
		MetricValues:   make(map[string]core.MetricValue),
		LabeledMetrics: make([]core.LabeledMetric, len(labels)),
	}
	// Add all necessary labels
	for _, label := range labels {
		ms.Labels[label.Key] = "test-value"
	}
	ms.Labels[core.LabelMetricSetType.Key] = objectType
	// Add all standard metrics
	for _, metric := range core.StandardMetrics {
		ms.MetricValues[metric.Name] = core.MetricValue{
			MetricType: core.MetricCumulative,
			ValueType:  core.ValueInt64,
			IntValue:   -1,
		}
	}
	// Add all labeled metrics
	for _, metric := range core.LabeledMetrics {
		lm := core.LabeledMetric{
			Name: metric.Name,
			MetricValue: core.MetricValue{
				MetricType: core.MetricCumulative,
				ValueType:  core.ValueInt64,
				IntValue:   -1,
			},
			Labels: make(map[string]string),
		}
		for _, label := range core.MetricLabels() {
			lm.Labels[label.Key] = "test-value"
		}
		ms.LabeledMetrics = append(ms.LabeledMetrics, lm)
	}
	return ms
}

func TestRealInput(t *testing.T) {
	api := NewApi(false, nil, nil)
	dataBatch := []*core.DataBatch{
		{
			Timestamp:  time.Now(),
			MetricSets: map[string]*core.MetricSet{},
		},
		{
			Timestamp:  time.Now().Add(-time.Minute),
			MetricSets: map[string]*core.MetricSet{},
		},
	}
	labels := append(core.CommonLabels(), core.ContainerLabels()...)
	labels = append(labels, core.PodLabels()...)
	for _, entry := range dataBatch {
		// Add a pod, container, node, systemcontainer
		entry.MetricSets[core.MetricSetTypePod] = generateMetricSet(core.MetricSetTypePod, labels)
		entry.MetricSets[core.MetricSetTypeNode] = generateMetricSet(core.MetricSetTypeNode, labels)
		entry.MetricSets[core.MetricSetTypePodContainer] = generateMetricSet(core.MetricSetTypePodContainer, labels)
		entry.MetricSets[core.MetricSetTypeSystemContainer] = generateMetricSet(core.MetricSetTypeSystemContainer, labels)
	}
	ts := api.processMetricsRequest(dataBatch)
	type expectation struct {
		count       int
		extraLabels bool
	}
	expectedMetrics := make(map[string]*expectation)
	for _, metric := range core.StandardMetrics {
		expectedMetrics[metric.Name] = &expectation{
			count:       4,
			extraLabels: false,
		}
	}
	for _, metric := range core.LabeledMetrics {
		expectedMetrics[metric.Name] = &expectation{
			count:       4,
			extraLabels: true,
		}
	}
	as := assert.New(t)
	for _, elem := range ts {
		// validate labels
		for _, label := range labels {
			val, exists := elem.Labels[label.Key]
			as.True(exists, "%q label does not exist", label.Key)
			if label.Key == core.LabelMetricSetType.Key {
				continue
			}
			if label.Key == core.LabelContainerName.Key && val != "machine" && val != "/pod" {
				as.Equal(val, "test-value", "%q label's value is %q, expected 'test-value'", label.Key, val)
			}
		}
		for mname, points := range elem.Metrics {
			ex := expectedMetrics[mname]
			require.NotNil(t, ex)
			as.NotEqual(ex, 0)
			ex.count--
			for _, point := range points {
				as.Equal(point.Value, -1)
				if !ex.extraLabels {
					continue
				}
				as.Equal(len(core.MetricLabels()), len(point.Labels))
				for _, label := range core.MetricLabels() {
					val, exists := point.Labels[label.Key]
					as.True(exists, "expected label %q to be found - %+v", label.Key, point.Labels)
					as.Equal(val, "test-value")
				}
			}
		}

	}
}
