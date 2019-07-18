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
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubectl/pkg/util/printers"
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
	usages  []v1.ResourceList
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
		qi := n.usages[i][v1.ResourceCPU]
		qj := n.usages[j][v1.ResourceCPU]
		return qi.Value() > qj.Value()
	case "memory":
		qi := n.usages[i][v1.ResourceMemory]
		qj := n.usages[j][v1.ResourceMemory]
		return qi.Value() > qj.Value()
	default:
		return n.metrics[i].Name < n.metrics[j].Name
	}
}

func NewNodeMetricsSorter(metrics []metricsapi.NodeMetrics, sortBy string) (*NodeMetricsSorter, error) {
	var usages = make([]v1.ResourceList, len(metrics))
	if len(sortBy) > 0 {
		for i, v := range metrics {
			if err := scheme.Scheme.Convert(&v.Usage, &usages[i], nil); err != nil {
				return nil, err
			}
		}
	}

	return &NodeMetricsSorter{
		metrics: metrics,
		sortBy:  sortBy,
		usages:  usages,
	}, nil
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
}

func (p *PodMetricsSorter) Less(i, j int) bool {
	switch p.sortBy {
	case "cpu":
		qi := p.podMetrics[i][v1.ResourceCPU]
		qj := p.podMetrics[j][v1.ResourceCPU]
		return qi.Value() > qj.Value()
	case "memory":
		qi := p.podMetrics[i][v1.ResourceMemory]
		qj := p.podMetrics[j][v1.ResourceMemory]
		return qi.Value() > qj.Value()
	default:
		if p.withNamespace && p.metrics[i].Namespace != p.metrics[j].Namespace {
			return p.metrics[i].Namespace < p.metrics[j].Namespace
		}
		return p.metrics[i].Name < p.metrics[j].Name
	}
}

func NewPodMetricsSorter(metrics []metricsapi.PodMetrics, printContainers bool, withNamespace bool, sortBy string) (*PodMetricsSorter, error) {
	var podMetrics = make([]v1.ResourceList, len(metrics))
	if len(sortBy) > 0 {
		for i, v := range metrics {
			podMetrics[i], _, _ = getPodMetrics(&v, printContainers)
		}
	}

	return &PodMetricsSorter{
		metrics:       metrics,
		sortBy:        sortBy,
		withNamespace: withNamespace,
		podMetrics:    podMetrics,
	}, nil
}

func (printer *TopCmdPrinter) PrintNodeMetrics(metrics []metricsapi.NodeMetrics, availableResources map[string]v1.ResourceList, noHeaders bool, sortBy string) error {
	if len(metrics) == 0 {
		return nil
	}
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()

	n, err := NewNodeMetricsSorter(metrics, sortBy)
	if err != nil {
		return err
	}
	sort.Sort(n)

	if !noHeaders {
		printColumnNames(w, NodeColumns)
	}
	var usage v1.ResourceList
	for _, m := range metrics {
		err := scheme.Scheme.Convert(&m.Usage, &usage, nil)
		if err != nil {
			return err
		}
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

	p, err := NewPodMetricsSorter(metrics, printContainers, withNamespace, sortBy)
	if err != nil {
		return err
	}
	sort.Sort(p)

	for _, m := range metrics {
		err := printSinglePodMetrics(w, &m, printContainers, withNamespace)
		if err != nil {
			return err
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

func printSinglePodMetrics(out io.Writer, m *metricsapi.PodMetrics, printContainersOnly bool, withNamespace bool) error {
	podMetrics, containers, err := getPodMetrics(m, printContainersOnly)
	if err != nil {
		return err
	}
	if printContainersOnly {
		for contName := range containers {
			if withNamespace {
				printValue(out, m.Namespace)
			}
			printValue(out, m.Name)
			printMetricsLine(out, &ResourceMetricsInfo{
				Name:      contName,
				Metrics:   containers[contName],
				Available: v1.ResourceList{},
			})
		}
	} else {
		if withNamespace {
			printValue(out, m.Namespace)
		}
		printMetricsLine(out, &ResourceMetricsInfo{
			Name:      m.Name,
			Metrics:   podMetrics,
			Available: v1.ResourceList{},
		})
	}
	return nil
}

func getPodMetrics(m *metricsapi.PodMetrics, printContainersOnly bool) (v1.ResourceList, map[string]v1.ResourceList, error) {
	containers := make(map[string]v1.ResourceList)
	podMetrics := make(v1.ResourceList)
	for _, res := range MeasuredResources {
		podMetrics[res], _ = resource.ParseQuantity("0")
	}

	for _, c := range m.Containers {
		var usage v1.ResourceList
		if err := scheme.Scheme.Convert(&c.Usage, &usage, nil); err != nil {
			return nil, nil, err
		}
		containers[c.Name] = usage
		if !printContainersOnly {
			for _, res := range MeasuredResources {
				quantity := podMetrics[res]
				quantity.Add(usage[res])
				podMetrics[res] = quantity
			}
		}
	}
	return podMetrics, containers, nil
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
