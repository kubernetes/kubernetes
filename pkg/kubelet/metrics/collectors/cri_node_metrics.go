/*
Copyright The Kubernetes Authors.

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
	"time"

	cadvisorcontainer "github.com/google/cadvisor/lib/container"
	cadvisormetrics "github.com/google/cadvisor/lib/metrics"
	info "github.com/google/cadvisor/lib/model"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// nodeInfoProvider is the interface for reading container and machine info
// from cAdvisor. It mirrors the unexported cadvisormetrics.infoProvider.
type nodeInfoProvider interface {
	GetRequestedContainersInfo(string, info.RequestOptions) (map[string]*info.ContainerInfo, error)
	GetVersionInfo() (*info.VersionInfo, error)
	GetMachineInfo() (*info.MachineInfo, error)
}

// nodeMetricsGatherer reads root cgroup stats from cAdvisor via a private
// PrometheusCollector, then relabels the output to match CRI descriptors.
type nodeMetricsGatherer struct {
	logger      klog.Logger
	registry    metrics.KubeRegistry
	descriptors map[string]*metrics.Desc
	labelKeys   map[string][]string
}

func newNodeMetricsGatherer(
	ctx context.Context,
	descriptors map[string]*metrics.Desc,
	labelKeys map[string][]string,
	infoProvider nodeInfoProvider,
	containerLabelsFunc cadvisormetrics.ContainerLabelsFunc,
) *nodeMetricsGatherer {
	nodeMetrics := cadvisorcontainer.MetricSet{
		cadvisorcontainer.CpuUsageMetrics:     {},
		cadvisorcontainer.MemoryUsageMetrics:  {},
		cadvisorcontainer.NetworkUsageMetrics: {},
	}

	cadvisorCollector := cadvisormetrics.NewPrometheusCollector(
		infoProvider,
		containerLabelsFunc,
		nodeMetrics,
		clock.RealClock{},
		info.RequestOptions{
			IdType:    info.TypeName,
			Count:     1,
			Recursive: false,
		},
	)

	registry := metrics.NewKubeRegistry()
	registry.RawMustRegister(cadvisorCollector)

	return &nodeMetricsGatherer{
		logger:      klog.FromContext(ctx),
		registry:    registry,
		descriptors: descriptors,
		labelKeys:   labelKeys,
	}
}

// collectNodeMetrics gathers metrics from the private cAdvisor registry
// and relabels them to match the CRI runtime's metric descriptors.
func (g *nodeMetricsGatherer) collectNodeMetrics(ch chan<- metrics.Metric) {
	families, err := g.registry.Gather()
	if err != nil {
		g.logger.Error(err, "Error gathering node metrics from cAdvisor")
	}

	for _, family := range families {
		desc, ok := g.descriptors[family.GetName()]
		if !ok {
			continue
		}
		keys := g.labelKeys[family.GetName()]

		for _, m := range family.GetMetric() {
			cadvisorLabels := make(map[string]string, len(m.GetLabel()))
			for _, lp := range m.GetLabel() {
				cadvisorLabels[lp.GetName()] = lp.GetValue()
			}

			labelValues := make([]string, len(keys))
			for i, key := range keys {
				labelValues[i] = cadvisorLabels[key]
			}

			var valueType metrics.ValueType
			var value float64
			if c := m.GetCounter(); c != nil {
				valueType = metrics.CounterValue
				value = c.GetValue()
			} else if gauge := m.GetGauge(); gauge != nil {
				valueType = metrics.GaugeValue
				value = gauge.GetValue()
			} else {
				continue
			}

			metric, err := metrics.NewConstMetric(desc, valueType, value, labelValues...)
			if err != nil {
				g.logger.Error(err, "Error creating node metric", "name", family.GetName())
				continue
			}
			if ts := m.GetTimestampMs(); ts != 0 {
				metric = metrics.NewLazyMetricWithTimestamp(time.UnixMilli(ts), metric)
			}
			ch <- metric
		}
	}
}
