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

package framework

import (
	"fmt"

	"k8s.io/kubernetes/test/e2e/perftype"
)

// TODO(random-liu): Change the tests to actually use PerfData from the beginning instead of
// translating one to the other here.

// currentApiCallMetricsVersion is the current apicall performance metrics version. We should
// bump up the version each time we make incompatible change to the metrics.
const currentApiCallMetricsVersion = "v1"

// ApiCallToPerfData transforms APIResponsiveness to PerfData.
func ApiCallToPerfData(apicalls APIResponsiveness) *perftype.PerfData {
	perfData := &perftype.PerfData{Version: currentApiCallMetricsVersion}
	for _, apicall := range apicalls.APICalls {
		item := perftype.DataItem{
			Data: map[string]float64{
				"Perc50": float64(apicall.Latency.Perc50) / 1000000, // us -> ms
				"Perc90": float64(apicall.Latency.Perc90) / 1000000,
				"Perc99": float64(apicall.Latency.Perc99) / 1000000,
			},
			Unit: "ms",
			Labels: map[string]string{
				"Verb":     apicall.Verb,
				"Resource": apicall.Resource,
			},
		}
		perfData.DataItems = append(perfData.DataItems, item)
	}
	return perfData
}

// currentKubeletPerfMetricsVersion is the current kubelet performance metrics version. We should
// bump up the version each time we make incompatible change to the metrics.
const currentKubeletPerfMetricsVersion = "v1"

// ResourceUsageToPerfData transforms ResourceUsagePerNode to PerfData. Notice that this function
// only cares about memory usage, because cpu usage information will be extracted from NodesCPUSummary.
func ResourceUsageToPerfData(usagePerNode ResourceUsagePerNode) *perftype.PerfData {
	return ResourceUsageToPerfDataWithLabels(usagePerNode, nil)
}

// CPUUsageToPerfData transforms NodesCPUSummary to PerfData.
func CPUUsageToPerfData(usagePerNode NodesCPUSummary) *perftype.PerfData {
	return CPUUsageToPerfDataWithLabels(usagePerNode, nil)
}

// PrintPerfData prints the perfdata in json format with PerfResultTag prefix.
// If an error occurs, nothing will be printed.
func PrintPerfData(p *perftype.PerfData) {
	// Notice that we must make sure the perftype.PerfResultEnd is in a new line.
	if str := PrettyPrintJSON(p); str != "" {
		Logf("%s %s\n%s", perftype.PerfResultTag, str, perftype.PerfResultEnd)
	}
}

// ResourceUsageToPerfDataWithLabels transforms ResourceUsagePerNode to PerfData with additional labels.
// Notice that this function only cares about memory usage, because cpu usage information will be extracted from NodesCPUSummary.
func ResourceUsageToPerfDataWithLabels(usagePerNode ResourceUsagePerNode, labels map[string]string) *perftype.PerfData {
	items := []perftype.DataItem{}
	for _, usages := range usagePerNode {
		for c, usage := range usages {
			item := perftype.DataItem{
				Data: map[string]float64{
					"memory":     float64(usage.MemoryUsageInBytes) / (1024 * 1024),
					"workingset": float64(usage.MemoryWorkingSetInBytes) / (1024 * 1024),
					"rss":        float64(usage.MemoryRSSInBytes) / (1024 * 1024),
				},
				Unit: "MB",
				Labels: map[string]string{
					"container": c,
					"datatype":  "resource",
					"resource":  "memory",
				},
			}
			items = append(items, item)
		}
	}
	return &perftype.PerfData{
		Version:   currentKubeletPerfMetricsVersion,
		DataItems: items,
		Labels:    labels,
	}
}

// CPUUsageToPerfDataWithLabels transforms NodesCPUSummary to PerfData with additional labels.
func CPUUsageToPerfDataWithLabels(usagePerNode NodesCPUSummary, labels map[string]string) *perftype.PerfData {
	items := []perftype.DataItem{}
	for _, usages := range usagePerNode {
		for c, usage := range usages {
			data := map[string]float64{}
			for perc, value := range usage {
				data[fmt.Sprintf("Perc%02.0f", perc*100)] = value * 1000
			}

			item := perftype.DataItem{
				Data: data,
				Unit: "mCPU",
				Labels: map[string]string{
					"container": c,
					"datatype":  "resource",
					"resource":  "cpu",
				},
			}
			items = append(items, item)
		}
	}
	return &perftype.PerfData{
		Version:   currentKubeletPerfMetricsVersion,
		DataItems: items,
		Labels:    labels,
	}
}
