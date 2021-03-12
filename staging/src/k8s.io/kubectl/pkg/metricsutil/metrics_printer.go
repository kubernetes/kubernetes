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

type NodeMetricsSorter struct {
	metrics []metricsapi.NodeMetrics
	sortBy  string
}

func (n *NodeMetricsSorter) Len() int {
	return len(n.metrics)
}

func (n *NodeMetricsSorter) Swap(i, j int) {
	n.metrics[i], n.metrics[j] = n.metrics[j], n.metrics[i]
}

func (n *NodeMetricsSorter) Less(i, j int) bool {
	switch n.sortBy {
	case "cpu":
		return n.metrics[i].Usage.Cpu().MilliValue() > n.metrics[j].Usage.Cpu().MilliValue()
	case "memory":
		return n.metrics[i].Usage.Memory().Value() > n.metrics[j].Usage.Memory().Value()
	default:
		return n.metrics[i].Name < n.metrics[j].Name
	}
}

func NewNodeMetricsSorter(metrics []metricsapi.NodeMetrics, sortBy string) *NodeMetricsSorter {
	return &NodeMetricsSorter{
		metrics: metrics,
		sortBy:  sortBy,
	}
}

type PodMetricsSorter struct {
	metrics       []metricsapi.PodMetrics
	sortBy        string
	withNamespace bool
	podMetrics    []v1.ResourceList
}

func (p *PodMetricsSorter) Len() int {
	return len(p.metrics)
}

func (p *PodMetricsSorter) Swap(i, j int) {
	p.metrics[i], p.metrics[j] = p.metrics[j], p.metrics[i]
	p.podMetrics[i], p.podMetrics[j] = p.podMetrics[j], p.podMetrics[i]
}

func (p *PodMetricsSorter) Less(i, j int) bool {
	switch p.sortBy {
	case "cpu":
		return p.podMetrics[i].Cpu().MilliValue() > p.podMetrics[j].Cpu().MilliValue()
	case "memory":
		return p.podMetrics[i].Memory().Value() > p.podMetrics[j].Memory().Value()
	default:
		if p.withNamespace && p.metrics[i].Namespace != p.metrics[j].Namespace {
			return p.metrics[i].Namespace < p.metrics[j].Namespace
		}
		return p.metrics[i].Name < p.metrics[j].Name
	}
}

func NewPodMetricsSorter(metrics []metricsapi.PodMetrics, withNamespace bool, sortBy string) *PodMetricsSorter {
	var podMetrics = make([]v1.ResourceList, len(metrics))
	if len(sortBy) > 0 {
		for i, v := range metrics {
			podMetrics[i] = getPodMetrics(&v)
		}
	}

	return &PodMetricsSorter{
		metrics:       metrics,
		sortBy:        sortBy,
		withNamespace: withNamespace,
		podMetrics:    podMetrics,
	}
}

type ContainerMetricsSorter struct {
	metrics []metricsapi.ContainerMetrics
	sortBy  string
}

func (s *ContainerMetricsSorter) Len() int {
	return len(s.metrics)
}

func (s *ContainerMetricsSorter) Swap(i, j int) {
	s.metrics[i], s.metrics[j] = s.metrics[j], s.metrics[i]
}

func (s *ContainerMetricsSorter) Less(i, j int) bool {
	switch s.sortBy {
	case "cpu":
		return s.metrics[i].Usage.Cpu().MilliValue() > s.metrics[j].Usage.Cpu().MilliValue()
	case "memory":
		return s.metrics[i].Usage.Memory().Value() > s.metrics[j].Usage.Memory().Value()
	default:
		return s.metrics[i].Name < s.metrics[j].Name
	}
}

func NewContainerMetricsSorter(metrics []metricsapi.ContainerMetrics, sortBy string) *ContainerMetricsSorter {
	return &ContainerMetricsSorter{
		metrics: metrics,
		sortBy:  sortBy,
	}
}

func (printer *TopCmdPrinter) PrintNodeMetrics(metrics []metricsapi.NodeMetrics, availableResources map[string]v1.ResourceList, noHeaders bool, sortBy string) error {
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
		})
		delete(availableResources, m.Name)
	}

	// print lines for nodes of which the metrics is unreachable.
	for nodeName := range availableResources {
		printMissingMetricsNodeLine(w, nodeName)
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metricsapi.PodMetrics, printContainers bool, withNamespace bool, noHeaders bool, sortBy string) error {
	if len(metrics) == 0 {
		return nil
	}
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()
	if !noHeaders {
		if withNamespace {
			printValue(w, NamespaceColumn)
		}
		if printContainers {
			printValue(w, PodColumn)
		}
		printColumnNames(w, PodColumns)
	}

	sort.Sort(NewPodMetricsSorter(metrics, withNamespace, sortBy))

	for _, m := range metrics {
		if printContainers {
			sort.Sort(NewContainerMetricsSorter(m.Containers, sortBy))
			printSinglePodContainerMetrics(w, &m, withNamespace)
		} else {
			printSinglePodMetrics(w, &m, withNamespace)
		}
	}
	return nil
}

func printColumnNames(out io.Writer, names []string) {
	for _, name := range names {
		printValue(out, name)
	}
	fmt.Fprint(out, "\n")
}

func printSinglePodMetrics(out io.Writer, m *metricsapi.PodMetrics, withNamespace bool) {
	podMetrics := getPodMetrics(m)
	if withNamespace {
		printValue(out, m.Namespace)
	}
	printMetricsLine(out, &ResourceMetricsInfo{
		Name:      m.Name,
		Metrics:   podMetrics,
		Available: v1.ResourceList{},
	})
}

func printSinglePodContainerMetrics(out io.Writer, m *metricsapi.PodMetrics, withNamespace bool) {
	for _, c := range m.Containers {
		if withNamespace {
			printValue(out, m.Namespace)
		}
		printValue(out, m.Name)
		printMetricsLine(out, &ResourceMetricsInfo{
			Name:      c.Name,
			Metrics:   c.Usage,
			Available: v1.ResourceList{},
		})
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

func printMetricsLine(out io.Writer, metrics *ResourceMetricsInfo) {
	printValue(out, metrics.Name)
	printAllResourceUsages(out, metrics)
	fmt.Fprint(out, "\n")
}

func printMissingMetricsNodeLine(out io.Writer, nodeName string) {
	printValue(out, nodeName)
	unknownMetricsStatus := "<unknown>"
	for i := 0; i < len(MeasuredResources); i++ {
		printValue(out, unknownMetricsStatus)
		printValue(out, "\t")
		printValue(out, unknownMetricsStatus)
		printValue(out, "\t")
	}
	fmt.Fprint(out, "\n")
}

func printValue(out io.Writer, value interface{}) {
	fmt.Fprintf(out, "%v\t", value)
}

func printAllResourceUsages(out io.Writer, metrics *ResourceMetricsInfo) {
	for _, res := range MeasuredResources {
		quantity := metrics.Metrics[res]
		printSingleResourceUsage(out, res, quantity)
		fmt.Fprint(out, "\t")
		if available, found := metrics.Available[res]; found {
			fraction := float64(quantity.MilliValue()) / float64(available.MilliValue()) * 100
			fmt.Fprintf(out, "%d%%\t", int64(fraction))
		}
	}
}

func printSingleResourceUsage(out io.Writer, resourceType v1.ResourceName, quantity resource.Quantity) {
	switch resourceType {
	case v1.ResourceCPU:
		fmt.Fprintf(out, "%vm", quantity.MilliValue())
	case v1.ResourceMemory:
		fmt.Fprintf(out, "%vMi", quantity.Value()/(1024*1024))
	default:
		fmt.Fprintf(out, "%v", quantity.Value())
	}
}
