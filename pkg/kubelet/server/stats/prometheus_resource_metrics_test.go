/*
Copyright 2019 The Kubernetes Authors.

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

package stats

import (
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
)

const (
	errorName = "scrape_error"
	errorHelp = "1 if there was an error while getting container metrics, 0 otherwise"
)

var (
	noError  = float64(0)
	hasError = float64(1)
)

type mockSummaryProvider struct {
	mock.Mock
}

func (m *mockSummaryProvider) Get(updateStats bool) (*statsapi.Summary, error) {
	args := m.Called(updateStats)
	return args.Get(0).(*statsapi.Summary), args.Error(1)
}

func (m *mockSummaryProvider) GetCPUAndMemoryStats() (*statsapi.Summary, error) {
	args := m.Called()
	return args.Get(0).(*statsapi.Summary), args.Error(1)
}

type collectResult struct {
	desc   *prometheus.Desc
	metric *dto.Metric
}

func TestCollectResourceMetrics(t *testing.T) {
	testTime := metav1.Now()
	for _, tc := range []struct {
		description     string
		config          ResourceMetricsConfig
		summary         *statsapi.Summary
		summaryErr      error
		expectedMetrics []collectResult
	}{
		{
			description: "error getting summary",
			config:      ResourceMetricsConfig{},
			summary:     nil,
			summaryErr:  fmt.Errorf("failed to get summary"),
			expectedMetrics: []collectResult{
				{
					desc:   prometheus.NewDesc(errorName, errorHelp, []string{}, nil),
					metric: &dto.Metric{Gauge: &dto.Gauge{Value: &hasError}},
				},
			},
		},
		{
			description: "arbitrary node metrics",
			config: ResourceMetricsConfig{
				NodeMetrics: []NodeResourceMetric{
					{
						Name:        "node_foo",
						Description: "a metric from nodestats",
						ValueFn: func(s statsapi.NodeStats) (*float64, time.Time) {
							if s.CPU == nil {
								return nil, time.Time{}
							}
							v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
							return &v, s.CPU.Time.Time
						},
					},
					{
						Name:        "node_bar",
						Description: "another metric from nodestats",
						ValueFn: func(s statsapi.NodeStats) (*float64, time.Time) {
							if s.Memory == nil {
								return nil, time.Time{}
							}
							v := float64(*s.Memory.WorkingSetBytes)
							return &v, s.Memory.Time.Time
						},
					},
				},
			},
			summary: &statsapi.Summary{
				Node: statsapi.NodeStats{
					CPU: &statsapi.CPUStats{
						Time:                 testTime,
						UsageCoreNanoSeconds: uint64Ptr(10000000000),
					},
					Memory: &statsapi.MemoryStats{
						Time:            testTime,
						WorkingSetBytes: uint64Ptr(1000),
					},
				},
			},
			summaryErr: nil,
			expectedMetrics: []collectResult{
				{
					desc:   prometheus.NewDesc("node_foo", "a metric from nodestats", []string{}, nil),
					metric: &dto.Metric{Gauge: &dto.Gauge{Value: float64Ptr(10)}},
				},
				{
					desc:   prometheus.NewDesc("node_bar", "another metric from nodestats", []string{}, nil),
					metric: &dto.Metric{Gauge: &dto.Gauge{Value: float64Ptr(1000)}},
				},
				{
					desc:   prometheus.NewDesc(errorName, errorHelp, []string{}, nil),
					metric: &dto.Metric{Gauge: &dto.Gauge{Value: &noError}},
				},
			},
		},
		{
			description: "arbitrary container metrics for different container, pods and namespaces",
			config: ResourceMetricsConfig{
				ContainerMetrics: []ContainerResourceMetric{
					{
						Name:        "container_foo",
						Description: "a metric from container stats",
						ValueFn: func(s statsapi.ContainerStats) (*float64, time.Time) {
							if s.CPU == nil {
								return nil, time.Time{}
							}
							v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
							return &v, s.CPU.Time.Time
						},
					},
					{
						Name:        "container_bar",
						Description: "another metric from container stats",
						ValueFn: func(s statsapi.ContainerStats) (*float64, time.Time) {
							if s.Memory == nil {
								return nil, time.Time{}
							}
							v := float64(*s.Memory.WorkingSetBytes)
							return &v, s.Memory.Time.Time
						},
					},
				},
			},
			summary: &statsapi.Summary{
				Pods: []statsapi.PodStats{
					{
						PodRef: statsapi.PodReference{
							Name:      "pod_a",
							Namespace: "namespace_a",
						},
						Containers: []statsapi.ContainerStats{
							{
								Name: "container_a",
								CPU: &statsapi.CPUStats{
									Time:                 testTime,
									UsageCoreNanoSeconds: uint64Ptr(10000000000),
								},
								Memory: &statsapi.MemoryStats{
									Time:            testTime,
									WorkingSetBytes: uint64Ptr(1000),
								},
							},
							{
								Name: "container_b",
								CPU: &statsapi.CPUStats{
									Time:                 testTime,
									UsageCoreNanoSeconds: uint64Ptr(10000000000),
								},
								Memory: &statsapi.MemoryStats{
									Time:            testTime,
									WorkingSetBytes: uint64Ptr(1000),
								},
							},
						},
					},
					{
						PodRef: statsapi.PodReference{
							Name:      "pod_b",
							Namespace: "namespace_b",
						},
						Containers: []statsapi.ContainerStats{
							{
								Name: "container_a",
								CPU: &statsapi.CPUStats{
									Time:                 testTime,
									UsageCoreNanoSeconds: uint64Ptr(10000000000),
								},
								Memory: &statsapi.MemoryStats{
									Time:            testTime,
									WorkingSetBytes: uint64Ptr(1000),
								},
							},
						},
					},
				},
			},
			summaryErr: nil,
			expectedMetrics: []collectResult{
				{
					desc: prometheus.NewDesc("container_foo", "a metric from container stats", []string{"container", "pod", "namespace"}, nil),
					metric: &dto.Metric{
						Gauge: &dto.Gauge{Value: float64Ptr(10)},
						Label: []*dto.LabelPair{
							{Name: stringPtr("container"), Value: stringPtr("container_a")},
							{Name: stringPtr("namespace"), Value: stringPtr("namespace_a")},
							{Name: stringPtr("pod"), Value: stringPtr("pod_a")},
						},
					},
				},
				{
					desc: prometheus.NewDesc("container_bar", "another metric from container stats", []string{"container", "pod", "namespace"}, nil),
					metric: &dto.Metric{
						Gauge: &dto.Gauge{Value: float64Ptr(1000)},
						Label: []*dto.LabelPair{
							{Name: stringPtr("container"), Value: stringPtr("container_a")},
							{Name: stringPtr("namespace"), Value: stringPtr("namespace_a")},
							{Name: stringPtr("pod"), Value: stringPtr("pod_a")},
						},
					},
				},
				{
					desc: prometheus.NewDesc("container_foo", "a metric from container stats", []string{"container", "pod", "namespace"}, nil),
					metric: &dto.Metric{
						Gauge: &dto.Gauge{Value: float64Ptr(10)},
						Label: []*dto.LabelPair{
							{Name: stringPtr("container"), Value: stringPtr("container_b")},
							{Name: stringPtr("namespace"), Value: stringPtr("namespace_a")},
							{Name: stringPtr("pod"), Value: stringPtr("pod_a")},
						},
					},
				},
				{
					desc: prometheus.NewDesc("container_bar", "another metric from container stats", []string{"container", "pod", "namespace"}, nil),
					metric: &dto.Metric{
						Gauge: &dto.Gauge{Value: float64Ptr(1000)},
						Label: []*dto.LabelPair{
							{Name: stringPtr("container"), Value: stringPtr("container_b")},
							{Name: stringPtr("namespace"), Value: stringPtr("namespace_a")},
							{Name: stringPtr("pod"), Value: stringPtr("pod_a")},
						},
					},
				},
				{
					desc: prometheus.NewDesc("container_foo", "a metric from container stats", []string{"container", "pod", "namespace"}, nil),
					metric: &dto.Metric{
						Gauge: &dto.Gauge{Value: float64Ptr(10)},
						Label: []*dto.LabelPair{
							{Name: stringPtr("container"), Value: stringPtr("container_a")},
							{Name: stringPtr("namespace"), Value: stringPtr("namespace_b")},
							{Name: stringPtr("pod"), Value: stringPtr("pod_b")},
						},
					},
				},
				{
					desc: prometheus.NewDesc("container_bar", "another metric from container stats", []string{"container", "pod", "namespace"}, nil),
					metric: &dto.Metric{
						Gauge: &dto.Gauge{Value: float64Ptr(1000)},
						Label: []*dto.LabelPair{
							{Name: stringPtr("container"), Value: stringPtr("container_a")},
							{Name: stringPtr("namespace"), Value: stringPtr("namespace_b")},
							{Name: stringPtr("pod"), Value: stringPtr("pod_b")},
						},
					},
				},
				{
					desc:   prometheus.NewDesc(errorName, errorHelp, []string{}, nil),
					metric: &dto.Metric{Gauge: &dto.Gauge{Value: &noError}},
				},
			},
		},
	} {
		t.Run(tc.description, func(t *testing.T) {
			provider := &mockSummaryProvider{}
			provider.On("GetCPUAndMemoryStats").Return(tc.summary, tc.summaryErr)
			collector := NewPrometheusResourceMetricCollector(provider, tc.config)
			metrics := collectMetrics(t, collector, len(tc.expectedMetrics))
			for i := range metrics {
				assertEqual(t, metrics[i], tc.expectedMetrics[i])
			}
		})
	}
}

// collectMetrics is a wrapper around a prometheus.Collector which returns the metrics added to the metric channel as a slice.metric
// It will block indefinitely if the collector does not collect exactly numMetrics.
func collectMetrics(t *testing.T, collector prometheus.Collector, numMetrics int) (results []collectResult) {
	metricsCh := make(chan prometheus.Metric)
	done := make(chan struct{})
	go func() {
		collector.Collect(metricsCh)
		done <- struct{}{}
	}()
	for i := 0; i < numMetrics; i++ {
		metric := <-metricsCh
		metricProto := &dto.Metric{}
		assert.NoError(t, metric.Write(metricProto))
		results = append(results, collectResult{desc: metric.Desc(), metric: metricProto})
	}
	<-done
	return
}

// assertEqual asserts for semanitic equality for fields we care about
func assertEqual(t *testing.T, expected, actual collectResult) {
	assert.Equal(t, expected.desc.String(), actual.desc.String())
	assert.Equal(t, *expected.metric.Gauge.Value, *actual.metric.Gauge.Value, "for desc: %v", expected.desc.String())
	assert.Equal(t, len(expected.metric.Label), len(actual.metric.Label))
	if len(expected.metric.Label) == len(actual.metric.Label) {
		for i := range expected.metric.Label {
			assert.Equal(t, *expected.metric.Label[i], *actual.metric.Label[i], "for desc: %v", expected.desc.String())
		}
	}
}

func stringPtr(s string) *string {
	return &s
}

func uint64Ptr(u uint64) *uint64 {
	return &u
}

func float64Ptr(f float64) *float64 {
	return &f
}
