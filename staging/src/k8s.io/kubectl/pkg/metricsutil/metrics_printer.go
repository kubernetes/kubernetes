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
	"fmt"
	"io"
	"math"
	"sort"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/cli-runtime/pkg/printers"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

var (
	MeasuredResources = []v1.ResourceName{
		v1.ResourceCPU,
		v1.ResourceMemory,
	}
	NodeColumns     = []string{"NAME", "CPU(cores)", "CPU%", "MEMORY(bytes)", "MEMORY%"}
	PodColumns      = []string{"NAME", "CPU(cores)", "MEMORY(bytes)"}
	NamespaceColumn = "NAMESPACE"
	PodColumn       = "POD"
)

type ResourceMetricsInfo struct {
	Name      string
	Metrics   v1.ResourceList
	Available v1.ResourceList
}

type TopCmdPrinter struct {
	out io.Writer
}

func NewTopCmdPrinter(out io.Writer) *TopCmdPrinter {
	return &TopCmdPrinter{out: out}
}

func (printer *TopCmdPrinter) PrintNodeMetrics(metrics []metricsapi.NodeMetrics, availableResources map[string]v1.ResourceList, noHeaders bool, sortBy string, unit bool) error {
	if len(metrics) == 0 {
		return nil
	}
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()

	sort.Sort(NewNodeMetricsSorter(metrics, sortBy))

	if !noHeaders {
		printColumnNames(w, NodeColumns)
	}
	var usage v1.ResourceList
	for _, m := range metrics {
		m.Usage.DeepCopyInto(&usage)
		printMetricsLine(w, &ResourceMetricsInfo{
			Name:      m.Name,
			Metrics:   usage,
			Available: availableResources[m.Name],
		}, unit)
		delete(availableResources, m.Name)
	}

	// print lines for nodes of which the metrics is unreachable.
	for nodeName := range availableResources {
		printMissingMetricsNodeLine(w, nodeName)
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metricsapi.PodMetrics, printContainers bool, withNamespace bool, noHeaders bool, sortBy string, sum bool, unit bool) error {
	if len(metrics) == 0 {
		return nil
	}
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()

	columnWidth := len(PodColumns)
	if !noHeaders {
		if withNamespace {
			printValue(w, NamespaceColumn)
			columnWidth++
		}
		if printContainers {
			printValue(w, PodColumn)
			columnWidth++
		}
		printColumnNames(w, PodColumns)
	}

	sort.Sort(NewPodMetricsSorter(metrics, withNamespace, sortBy))

	for _, m := range metrics {
		if printContainers {
			sort.Sort(NewContainerMetricsSorter(m.Containers, sortBy))
			printSinglePodContainerMetrics(w, &m, withNamespace, unit)
		} else {
			printSinglePodMetrics(w, &m, withNamespace, unit)
		}

	}

	if sum {
		adder := NewResourceAdder(MeasuredResources)
		for _, m := range metrics {
			adder.AddPodMetrics(&m)
		}
		printPodResourcesSum(w, adder.total, columnWidth, unit)
	}

	return nil
}

func printColumnNames(out io.Writer, names []string) {
	for _, name := range names {
		printValue(out, name)
	}
	fmt.Fprint(out, "\n")
}

func printSinglePodMetrics(out io.Writer, m *metricsapi.PodMetrics, withNamespace bool, unit bool) {
	podMetrics := getPodMetrics(m)
	if withNamespace {
		printValue(out, m.Namespace)
	}
	printMetricsLine(out, &ResourceMetricsInfo{
		Name:      m.Name,
		Metrics:   podMetrics,
		Available: v1.ResourceList{},
	}, unit)
}

func printSinglePodContainerMetrics(out io.Writer, m *metricsapi.PodMetrics, withNamespace bool, unit bool) {
	for _, c := range m.Containers {
		if withNamespace {
			printValue(out, m.Namespace)
		}
		printValue(out, m.Name)
		printMetricsLine(out, &ResourceMetricsInfo{
			Name:      c.Name,
			Metrics:   c.Usage,
			Available: v1.ResourceList{},
		}, unit)
	}
}

func getPodMetrics(m *metricsapi.PodMetrics) v1.ResourceList {
	podMetrics := make(v1.ResourceList)
	for _, res := range MeasuredResources {
		podMetrics[res], _ = resource.ParseQuantity("0")
	}

	for _, c := range m.Containers {
		for _, res := range MeasuredResources {
			quantity := podMetrics[res]
			quantity.Add(c.Usage[res])
			podMetrics[res] = quantity
		}
	}
	return podMetrics
}

func printMetricsLine(out io.Writer, metrics *ResourceMetricsInfo, unit bool) {
	printValue(out, metrics.Name)
	printAllResourceUsages(out, metrics, unit)
	fmt.Fprint(out, "\n")
}

func printMissingMetricsNodeLine(out io.Writer, nodeName string) {
	printValue(out, nodeName)
	unknownMetricsStatus := "<unknown>"
	for i := 0; i < len(MeasuredResources); i++ {
		printValue(out, unknownMetricsStatus)
		printValue(out, unknownMetricsStatus)
	}
	fmt.Fprint(out, "\n")
}

func printValue(out io.Writer, value interface{}) {
	fmt.Fprintf(out, "%v\t", value)
}

func printAllResourceUsages(out io.Writer, metrics *ResourceMetricsInfo, unit bool) {
	for _, res := range MeasuredResources {
		quantity := metrics.Metrics[res]
		printSingleResourceUsage(out, res, quantity, unit)
		fmt.Fprint(out, "\t")
		if available, found := metrics.Available[res]; found {
			fraction := float64(quantity.MilliValue()) / float64(available.MilliValue()) * 100
			fmt.Fprintf(out, "%d%%\t", int64(fraction))
		}
	}
}

func printSingleResourceUsage(out io.Writer, resourceType v1.ResourceName, quantity resource.Quantity, unit bool) {
	switch resourceType {
	case v1.ResourceCPU:
		if unit {
			v, u := quantityPrintCpuValue(quantity.MilliValue())
			fmt.Fprintf(out, "%v%s", v, u)
		} else {
			fmt.Fprintf(out, "%vm", quantity.MilliValue())
		}
	case v1.ResourceMemory:
		if unit {
			v, u := quantityPrintMemValue(quantity.Value())
			fmt.Fprintf(out, "%v%s", v, u)
		} else {
			fmt.Fprintf(out, "%vMi", quantity.Value()/(1024*1024))
		}
	default:
		fmt.Fprintf(out, "%v", quantity.Value())
	}
}

func printPodResourcesSum(out io.Writer, total v1.ResourceList, columnWidth int, unit bool) {
	for i := 0; i < columnWidth-2; i++ {
		printValue(out, "")
	}
	printValue(out, "________")
	printValue(out, "________")
	fmt.Fprintf(out, "\n")
	for i := 0; i < columnWidth-3; i++ {
		printValue(out, "")
	}
	printMetricsLine(out, &ResourceMetricsInfo{
		Name:      "",
		Metrics:   total,
		Available: v1.ResourceList{},
	}, unit)

}

func quantityPrintCpuValue(value int64) (int64, string) {
	if value >= 10e3 {
		return int64(math.Floor(float64(value)/1000 + 0.5)), ""
	} else {
		return value, "m"
	}
}

func quantityPrintMemValue(value int64) (int64, string) {
	if value >= 1024 && value < 1024*1024 {
		return int64(math.Floor(float64(value)/1024 + 0.5)), "Ki"
	} else if value >= 1024*1024 && value < 1024*1024*1024 {
		return int64(math.Floor(float64(value)/1024/1024 + 0.5)), "Mi"
	} else if value >= 1024*1024*1024 && value < 1024*1024*1024*1024 {
		return int64(math.Floor(float64(value)/1024/1024/1024 + 0.5)), "Gi"
	} else if value >= 1024*1024*1024*1024 {
		return int64(math.Floor(float64(value)/1024/1024/1024/1024 + 0.5)), "Ti"
	} else {
		return value, ""
	}
}
