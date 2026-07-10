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

// kubernetesLabelKeys are the Kubernetes-level labels appended to every CRI metric
// to match the labels produced by the cadvisor metrics path.
var kubernetesLabelKeys = []string{"namespace", "pod", "container"}

type criMetricsCollector struct {
	metrics.BaseStableCollector
	// The descriptors structure will be populated by one call to ListMetricDescriptors from the runtime.
	// They will be saved in this map, where the key is the Name and the value is the Desc.
	descriptors             map[string]*metrics.Desc
	listPodSandboxMetricsFn func(context.Context) ([]*runtimeapi.PodSandboxMetrics, error)
	listPodSandboxFn        func(context.Context, *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error)
	listContainersFn        func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error)
}

// Check if criMetricsCollector implements necessary interface
var _ metrics.StableCollector = &criMetricsCollector{}

// NewCRIMetricsCollector implements the metrics.Collector interface
func NewCRIMetricsCollector(
	ctx context.Context,
	listPodSandboxMetricsFn func(context.Context) ([]*runtimeapi.PodSandboxMetrics, error),
	listMetricDescriptorsFn func(context.Context) ([]*runtimeapi.MetricDescriptor, error),
	listPodSandboxFn func(context.Context, *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error),
	listContainersFn func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error),
) metrics.StableCollector {
	descs, err := listMetricDescriptorsFn(ctx)
	if err != nil {
		logger := klog.FromContext(ctx)
		logger.Error(err, "Error reading MetricDescriptors")
		return &criMetricsCollector{
			listPodSandboxMetricsFn: listPodSandboxMetricsFn,
			listPodSandboxFn:        listPodSandboxFn,
			listContainersFn:        listContainersFn,
		}
	}
	c := &criMetricsCollector{
		listPodSandboxMetricsFn: listPodSandboxMetricsFn,
		listPodSandboxFn:        listPodSandboxFn,
		listContainersFn:        listContainersFn,
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

	podSandboxMap, containerMap := c.getPodAndContainerMaps(ctx)

	for _, podMetric := range podMetrics {
		sandbox := podSandboxMap[podMetric.PodSandboxId]
		podName, namespace := podSandboxLabels(sandbox)

		for _, metric := range podMetric.GetMetrics() {
			promMetric, err := c.criMetricToProm(logger, metric, namespace, podName, "POD")
			if err == nil {
				ch <- promMetric
			}
		}
		for _, ctrMetric := range podMetric.GetContainerMetrics() {
			containerName := containerLabel(containerMap[ctrMetric.ContainerId])
			for _, metric := range ctrMetric.GetMetrics() {
				promMetric, err := c.criMetricToProm(logger, metric, namespace, podName, containerName)
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
	// Append Kubernetes-level labels (namespace, pod, container) that the CRI runtime
	// does not provide but that consumers expect on /metrics/cadvisor.
	labelKeys := make([]string, 0, len(m.LabelKeys)+len(kubernetesLabelKeys))
	labelKeys = append(labelKeys, m.LabelKeys...)
	labelKeys = append(labelKeys, kubernetesLabelKeys...)
	return metrics.NewDesc(m.Name, m.Help, labelKeys, nil, metrics.INTERNAL, "")
}

func (c *criMetricsCollector) criMetricToProm(logger klog.Logger, m *runtimeapi.Metric, namespace, podName, containerName string) (metrics.Metric, error) {
	desc, ok := c.descriptors[m.Name]
	if !ok {
		err := fmt.Errorf("error converting CRI Metric to prometheus format")
		logger.V(5).Info("Descriptor not present in pre-populated list of descriptors", "name", m.Name, "err", err)
		return nil, err
	}

	typ := criTypeToProm[m.MetricType]

	labelValues := make([]string, 0, len(m.LabelValues)+len(kubernetesLabelKeys))
	labelValues = append(labelValues, m.LabelValues...)
	labelValues = append(labelValues, namespace, podName, containerName)

	pm, err := metrics.NewConstMetric(desc, typ, float64(m.GetValue().Value), labelValues...)
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

// getPodAndContainerMaps builds ID-to-metadata maps by calling the CRI RPCs.
// Unlike criStatsProvider.getPodAndContainerMaps, it does not filter out
// terminated pods/containers to avoid missing metrics for recently-stopped workloads.
func (c *criMetricsCollector) getPodAndContainerMaps(ctx context.Context) (map[string]*runtimeapi.PodSandbox, map[string]*runtimeapi.Container) {
	logger := klog.FromContext(ctx)
	podSandboxMap := make(map[string]*runtimeapi.PodSandbox)
	containerMap := make(map[string]*runtimeapi.Container)

	if c.listPodSandboxFn != nil {
		podSandboxes, err := c.listPodSandboxFn(ctx, &runtimeapi.PodSandboxFilter{})
		if err != nil {
			logger.Error(err, "Failed to list pod sandboxes for label resolution")
		} else {
			for _, s := range podSandboxes {
				podSandboxMap[s.Id] = s
			}
		}
	}

	if c.listContainersFn != nil {
		containers, err := c.listContainersFn(ctx, &runtimeapi.ContainerFilter{})
		if err != nil {
			logger.Error(err, "Failed to list containers for label resolution")
		} else {
			for _, ctr := range containers {
				containerMap[ctr.Id] = ctr
			}
		}
	}

	return podSandboxMap, containerMap
}

func podSandboxLabels(sandbox *runtimeapi.PodSandbox) (podName, namespace string) {
	if sandbox != nil && sandbox.Metadata != nil {
		return sandbox.Metadata.Name, sandbox.Metadata.Namespace
	}
	return "", ""
}

func containerLabel(container *runtimeapi.Container) string {
	if container != nil && container.Metadata != nil {
		return container.Metadata.Name
	}
	return ""
}

var criTypeToProm = map[runtimeapi.MetricType]metrics.ValueType{
	runtimeapi.MetricType_COUNTER: metrics.CounterValue,
	runtimeapi.MetricType_GAUGE:   metrics.GaugeValue,
}
