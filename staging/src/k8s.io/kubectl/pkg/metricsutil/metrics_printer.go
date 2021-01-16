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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sort"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/cli-runtime/pkg/printers"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

var (
	NodeColumns     = []string{"NAME", "CPU(cores)", "CPU%", "MEMORY(bytes)", "MEMORY%"}
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
		// TODO: reimplement node columns
		// printColumnNames(w, NodeColumns)
	}
	var usage v1.ResourceList
	for _, m := range metrics {
		m.Usage.DeepCopyInto(&usage)
		// TODO:
		//printMetricsLine(w, NodeColumns, &ResourceMetricsInfo{
		//	Name:      m.Name,
		//	Metrics:   usage,
		//	Available: availableResources[m.Name],
		//})
		delete(availableResources, m.Name)
	}

	// print lines for nodes of which the metrics is unreachable.
	for nodeName := range availableResources {
		printMissingMetricsNodeLine(w, NodeColumns, nodeName)
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metricsapi.PodMetrics, printContainers bool, withNamespace bool, noHeaders bool, sortBy string, customMetrics string) error {
	if len(metrics) == 0 {
		return nil
	}
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()

	columnList := []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory}
	for _, column := range strings.Split(customMetrics, ",") {
		if column != v1.ResourceCPU.String() && column != v1.ResourceMemory.String() {
			columnList = append(columnList, v1.ResourceName(column))
		}
	}

	if !noHeaders {
		if withNamespace {
			printValue(w, NamespaceColumn)
		}
		if printContainers {
			printValue(w, PodColumn)
		}
		printColumnNames(w, columnList)
	}

	for _, m := range metrics {
		if printContainers {
			sort.Sort(NewContainerMetricsSorter(m.Containers, sortBy))
			printSinglePodContainerMetrics(w, columnList, &m, withNamespace)
		} else {
			printSinglePodMetrics(w, columnList, &m, withNamespace)
		}
	}
	return nil
}

func (printer *TopCmdPrinter) PrintMetrics(metrics []metav1.APIResource, noHeaders bool, kind string) error {
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()

	mgks := []MetricGroupKind{}
	if kind == "" || kind == "pods" {
		mgks = append(mgks,
			MetricGroupKind{GroupKind: metav1.GroupKind{Kind: "pods"}, Metric: "cpu"},
			MetricGroupKind{GroupKind: metav1.GroupKind{Kind: "pods"}, Metric: "memory"},
			)
	}
	if kind == "" || kind == "nodes" {
		mgks = append(mgks,
			MetricGroupKind{GroupKind: metav1.GroupKind{Kind: "nodes"}, Metric: "cpu"},
			MetricGroupKind{GroupKind: metav1.GroupKind{Kind: "nodes"}, Metric: "memory"},
		)
	}

	for _, m := range metrics {
		mgk := metricGroupKind(m)
		if kind == "" || mgk.Kind == kind {
			mgks = append(mgks, mgk)
		}
	}
	sort.Slice(mgks, func(i, j int) bool {
		return mgks[i].Group < mgks[j].Group || (
			mgks[i].Group == mgks[j].Group && (mgks[i].Kind < mgks[j].Kind || (
				mgks[i].Kind == mgks[j].Kind && mgks[i].Metric < mgks[j].Metric)))
	})

	if !noHeaders {
		printValue(w, "GROUP")
		printValue(w, "KIND")
		printValue(w, "METRIC")
		fmt.Fprint(w, "\n")
	}

	for _, mk := range mgks {
		printValue(w, mk.Group)
		printValue(w, mk.Kind)
		printValue(w, mk.Metric)
		fmt.Fprint(w, "\n")
	}
	return nil
}

func printMetricHeaders() {

}


func printColumnNames(out io.Writer, resourceType []v1.ResourceName) {
	printValue(out, "NAME")
	for _, r := range resourceType {
		printResource(out, r)
	}
	fmt.Fprint(out, "\n")
}

func printSinglePodMetrics(out io.Writer, columnList []v1.ResourceName, m *metricsapi.PodMetrics, withNamespace bool) {
	podMetrics := getPodMetrics(m, columnList)
	if withNamespace {
		printValue(out, m.Namespace)
	}
	printMetricsLine(out, columnList, &ResourceMetricsInfo{
		Name:      m.Name,
		Metrics:   podMetrics,
		Available: v1.ResourceList{},
	})
}

func printSinglePodContainerMetrics(out io.Writer, columnList []v1.ResourceName, m *metricsapi.PodMetrics, withNamespace bool) {
	for _, c := range m.Containers {
		if withNamespace {
			printValue(out, m.Namespace)
		}
		printValue(out, m.Name)
		printMetricsLine(out, columnList, &ResourceMetricsInfo{
			Name:      c.Name,
			Metrics:   c.Usage,
			Available: v1.ResourceList{},
		})
	}
}

func getPodMetrics(m *metricsapi.PodMetrics, columnList []v1.ResourceName) v1.ResourceList {
	podMetrics := make(v1.ResourceList)
	for _, res := range columnList {
		podMetrics[res], _ = resource.ParseQuantity("0")
	}

	for _, c := range m.Containers {
		for _, res := range columnList {
			quantity := podMetrics[res]
			quantity.Add(c.Usage[res])
			podMetrics[res] = quantity
		}
	}
	return podMetrics
}

func printMetricsLine(out io.Writer, columnList []v1.ResourceName, metrics *ResourceMetricsInfo) {
	printValue(out, metrics.Name)
	printAllResourceUsages(out, columnList, metrics)
	fmt.Fprint(out, "\n")
}

func printMissingMetricsNodeLine(out io.Writer, columnName []string, nodeName string) {
	printValue(out, nodeName)
	unknownMetricsStatus := "<unknown>"
	for i := 0; i < len(columnName); i++ {
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

func printAllResourceUsages(out io.Writer, columnList []v1.ResourceName, metrics *ResourceMetricsInfo) {
	for _, res := range columnList {
		quantity := metrics.Metrics[res]
		printSingleResourceUsage(out, res, quantity)
		fmt.Fprint(out, "\t")
		if available, found := metrics.Available[res]; found {
			fraction := float64(quantity.MilliValue()) / float64(available.MilliValue()) * 100
			fmt.Fprintf(out, "%d%%\t", int64(fraction))
		}
	}
}

func printResource(out io.Writer, resourceType v1.ResourceName) {
	switch resourceType {
	case v1.ResourceCPU:
		fmt.Fprint(out, "CPU(cores)\t")
	case v1.ResourceMemory:
		fmt.Fprint(out, "MEMORY(bytes)\t")
	default:
		fmt.Fprintf(out, "%v\t", resourceType)
	}
}

func printSingleResourceUsage(out io.Writer, resourceType v1.ResourceName, quantity resource.Quantity) {
	switch {
	case resourceType == v1.ResourceCPU:
		fmt.Fprintf(out, "%vm", quantity.MilliValue())
	case resourceType == v1.ResourceMemory:
		fmt.Fprintf(out, "%vMi", quantity.Value()/(1024*1024))
	case strings.HasSuffix(resourceType.String(), "bytes"):
		value := quantity.Value()
		if value > 1024 * 1024 {
			fmt.Fprintf(out, "%.2fMi", float64(value / 1024)/1024 )
			break
		}
		if value > 1024 {
			fmt.Fprintf(out, "%.2fKi", float64(value)/1024 )
			break
		}
		fmt.Fprintf(out, "%v", value )
	default:
		fmt.Fprintf(out, "%s", quantity.String())
	}
}

type MetricGroupKind struct {
	metav1.GroupKind
	Metric string
}

func metricGroupKind(m metav1.APIResource) MetricGroupKind {
	GroupKindAndMetric := strings.SplitN(m.Name, "/", 2)
	metric := ""
	if len(GroupKindAndMetric) == 2 {
		metric = GroupKindAndMetric[1]
	}
	KindAndGroup := strings.SplitN(GroupKindAndMetric[0], ".", 2)
	group := ""
	if len(KindAndGroup) == 2 {
		group = KindAndGroup[1]
	}
	return MetricGroupKind{
		Metric: metric,
		GroupKind: metav1.GroupKind{
			Group: group,
			Kind:  KindAndGroup[0],
		},
	}
}
