/*
Copyright 2016 The Kubernetes Authors.

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

package metricsutil

import (
	"k8s.io/kubernetes/pkg/api"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
)

type containerMetrics struct {
	name    string
	pod     *metricsapi.PodMetrics
	metrics api.ResourceList
}

type containerMetricsSort struct {
	metrics  []containerMetrics
	sortType api.ResourceName
}

func (m containerMetricsSort) Len() int {
	return len(m.metrics)
}

func (m containerMetricsSort) Swap(i, j int) {
	m.metrics[i], m.metrics[j] = m.metrics[j], m.metrics[i]
}

func (m containerMetricsSort) Less(i, j int) bool {
	switch m.sortType {
	case api.ResourceName("name"):
		return m.metrics[i].pod.Name < m.metrics[j].pod.Name
	case api.ResourceCPU:
		return m.metrics[i].metrics.Cpu().MilliValue() < m.metrics[j].metrics.Cpu().MilliValue()
	case api.ResourceMemory:
		return m.metrics[i].metrics.Memory().Value() < m.metrics[j].metrics.Memory().Value()
	}
	return false
}

type nodeMetricsSort struct {
	metrics  []metricsapi.NodeMetrics
	sortType api.ResourceName
}

func (m nodeMetricsSort) Len() int {
	return len(m.metrics)
}

func (m nodeMetricsSort) Swap(i, j int) {
	m.metrics[i], m.metrics[j] = m.metrics[j], m.metrics[i]
}

func (m nodeMetricsSort) Less(i, j int) bool {
	switch m.sortType {
	case api.ResourceName("name"):
		return m.metrics[i].Name < m.metrics[j].Name
	case api.ResourceCPU:
		return m.metrics[i].Usage.Cpu().MilliValue() < m.metrics[j].Usage.Cpu().MilliValue()
	case api.ResourceMemory:
		return m.metrics[i].Usage.Memory().Value() < m.metrics[j].Usage.Memory().Value()
	}
	return false
}

type podMetricsSort struct {
	metrics     []metricsapi.PodMetrics
	fullMetrics []api.ResourceList
	sortType    api.ResourceName
}

func (m podMetricsSort) Len() int {
	return len(m.metrics)
}

func (m podMetricsSort) Swap(i, j int) {
	m.metrics[i], m.metrics[j] = m.metrics[j], m.metrics[i]
	m.fullMetrics[i], m.fullMetrics[j] = m.fullMetrics[j], m.fullMetrics[i]
}

func (m podMetricsSort) Less(i, j int) bool {
	switch m.sortType {
	case api.ResourceName("name"):
		return m.metrics[i].Name < m.metrics[j].Name
	case api.ResourceCPU:
		return m.fullMetrics[i].Cpu().MilliValue() < m.fullMetrics[j].Cpu().MilliValue()
	case api.ResourceMemory:
		return m.fullMetrics[i].Memory().Value() < m.fullMetrics[j].Memory().Value()
	}
	return false
}
