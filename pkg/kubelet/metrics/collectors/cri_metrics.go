/*
Copyright 2022 The Kubernetes Authors.

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

package collectors

import (
	"context"
	"fmt"
	"time"

	"k8s.io/component-base/metrics"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
)

type criMetricsCollector struct {
	metrics.BaseStableCollector
	// The descriptors structure will be populated by one call to ListMetricDescriptors from the runtime.
	// They will be saved in this map, where the key is the Name and the value is the Desc.
	descriptors             map[string]*metrics.Desc
	listPodSandboxMetricsFn func(context.Context) ([]*runtimeapi.PodSandboxMetrics, error)
}

// Check if criMetricsCollector implements necessary interface
var _ metrics.StableCollector = &criMetricsCollector{}

// NewCRIMetricsCollector implements the metrics.Collector interface
func NewCRIMetricsCollector(ctx context.Context, listPodSandboxMetricsFn func(context.Context) ([]*runtimeapi.PodSandboxMetrics, error), listMetricDescriptorsFn func(context.Context) ([]*runtimeapi.MetricDescriptor, error)) metrics.StableCollector {
	descs, err := listMetricDescriptorsFn(ctx)
	if err != nil {
		logger := klog.FromContext(ctx)
		logger.Error(err, "Error reading MetricDescriptors")
		return &criMetricsCollector{
			listPodSandboxMetricsFn: listPodSandboxMetricsFn,
		}
	}
	c := &criMetricsCollector{
		listPodSandboxMetricsFn: listPodSandboxMetricsFn,
		descriptors:             make(map[string]*metrics.Desc, len(descs)),
	}

	for _, desc := range descs {
		c.descriptors[desc.Name] = criDescToProm(desc)
	}

	return c
}

// Describe implements the metrics.DescribeWithStability interface.
func (c *criMetricsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	for _, desc := range c.descriptors {
		ch <- desc
	}
}

// Collect implements the metrics.CollectWithStability interface.
// TODO(haircommander): would it be better if these were processed async?
func (c *criMetricsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	podMetrics, err := c.listPodSandboxMetricsFn(ctx)
	if err != nil {
		logger.Error(err, "Failed to get pod metrics")
		return
	}

	for _, podMetric := range podMetrics {
		for _, metric := range podMetric.GetMetrics() {
			promMetric, err := c.criMetricToProm(logger, metric)
			if err == nil {
				ch <- promMetric
			}
		}
		for _, ctrMetric := range podMetric.GetContainerMetrics() {
			for _, metric := range ctrMetric.GetMetrics() {
				promMetric, err := c.criMetricToProm(logger, metric)
				if err == nil {
					ch <- promMetric
				}
			}
		}
	}
}

func criDescToProm(m *runtimeapi.MetricDescriptor) *metrics.Desc {
	// Labels in the translation are variableLabels, as opposed to constant labels.
	// This is because the values of the labels will be different for each container.
	return metrics.NewDesc(m.Name, m.Help, m.LabelKeys, nil, metrics.INTERNAL, "")
}

func (c *criMetricsCollector) criMetricToProm(logger klog.Logger, m *runtimeapi.Metric) (metrics.Metric, error) {
	desc, ok := c.descriptors[m.Name]
	if !ok {
		err := fmt.Errorf("error converting CRI Metric to prometheus format")
		logger.V(5).Info("Descriptor not present in pre-populated list of descriptors", "name", m.Name, "err", err)
		return nil, err
	}

	typ := criTypeToProm[m.MetricType]

	pm, err := metrics.NewConstMetric(desc, typ, float64(m.GetValue().Value), m.LabelValues...)
	if err != nil {
		logger.Error(err, "Error getting CRI prometheus metric", "descriptor", desc.String())
		return nil, err
	}
	// If Timestamp is 0, then the runtime did not cache the result.
	// In this case, a cached result is a metric that was collected ahead of time,
	// as opposed to on-demand.
	// If the metric was requested as needed, then Timestamp==0.
	if m.Timestamp == 0 {
		return pm, nil
	}
	return metrics.NewLazyMetricWithTimestamp(time.Unix(0, m.Timestamp), pm), nil
}

var criTypeToProm = map[runtimeapi.MetricType]metrics.ValueType{
	runtimeapi.MetricType_COUNTER: metrics.CounterValue,
	runtimeapi.MetricType_GAUGE:   metrics.GaugeValue,
}
