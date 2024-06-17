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

package perf

import (
	"fmt"

	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	"k8s.io/kubernetes/test/e2e/perftype"
)

// CurrentKubeletPerfMetricsVersion is the current kubelet performance metrics
// version. This is used by multiple perf related data structures. We should
// bump up the version each time we make an incompatible change to the metrics.
const CurrentKubeletPerfMetricsVersion = "v2"

// ResourceUsageToPerfData transforms ResourceUsagePerNode to PerfData. Notice that this function
// only cares about memory usage, because cpu usage information will be extracted from NodesCPUSummary.
func ResourceUsageToPerfData(usagePerNode e2ekubelet.ResourceUsagePerNode) *perftype.PerfData {
	return ResourceUsageToPerfDataWithLabels(usagePerNode, nil)
}

// CPUUsageToPerfData transforms NodesCPUSummary to PerfData.
func CPUUsageToPerfData(usagePerNode e2ekubelet.NodesCPUSummary) *perftype.PerfData {
	return CPUUsageToPerfDataWithLabels(usagePerNode, nil)
}

// ResourceUsageToPerfDataWithLabels transforms ResourceUsagePerNode to PerfData with additional labels.
// Notice that this function only cares about memory usage, because cpu usage information will be extracted from NodesCPUSummary.
func ResourceUsageToPerfDataWithLabels(usagePerNode e2ekubelet.ResourceUsagePerNode, labels map[string]string) *perftype.PerfData {
	items := []perftype.DataItem{}
	for node, usages := range usagePerNode {
		for c, usage := range usages {
			item := perftype.DataItem{
				Data: map[string]float64{
					"memory":     float64(usage.MemoryUsageInBytes) / (1024 * 1024),
					"workingset": float64(usage.MemoryWorkingSetInBytes) / (1024 * 1024),
					"rss":        float64(usage.MemoryRSSInBytes) / (1024 * 1024),
				},
				Unit: "MB",
				Labels: map[string]string{
					"node":      node,
					"container": c,
					"datatype":  "resource",
					"resource":  "memory",
				},
			}
			items = append(items, item)
		}
	}
	return &perftype.PerfData{
		Version:   CurrentKubeletPerfMetricsVersion,
		DataItems: items,
		Labels:    labels,
	}
}

// CPUUsageToPerfDataWithLabels transforms NodesCPUSummary to PerfData with additional labels.
func CPUUsageToPerfDataWithLabels(usagePerNode e2ekubelet.NodesCPUSummary, labels map[string]string) *perftype.PerfData {
	items := []perftype.DataItem{}
	for node, usages := range usagePerNode {
		for c, usage := range usages {
			data := map[string]float64{}
			for perc, value := range usage {
				data[fmt.Sprintf("Perc%02.0f", perc*100)] = value * 1000
			}

			item := perftype.DataItem{
				Data: data,
				Unit: "mCPU",
				Labels: map[string]string{
					"node":      node,
					"container": c,
					"datatype":  "resource",
					"resource":  "cpu",
				},
			}
			items = append(items, item)
		}
	}
	return &perftype.PerfData{
		Version:   CurrentKubeletPerfMetricsVersion,
		DataItems: items,
		Labels:    labels,
	}
}
