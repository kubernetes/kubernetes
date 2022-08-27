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
	"strings"

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
	DefinedResources = []v1.ResourceName{
		v1.ResourceRequestsCPU,
		v1.ResourceRequestsMemory,
		v1.ResourceLimitsCPU,
		v1.ResourceLimitsMemory,
	}
	NodeColumns     = []string{"NAME", "CPU(cores)", "CPU%", "MEMORY(bytes)", "MEMORY%"}
	PodColumns      = []string{"NAME", "CPU(cores)", "MEMORY(bytes)"}
	EnumColumns     = []string{"CPU REQ(cores)", "MEMORY REQ(bytes)", "CPU LIMIT(cores)", "MEMORY LIMIT(bytes)"}
	NamespaceColumn = "NAMESPACE"
	PodColumn       = "POD"
)

type ResourceMetricsInfo struct {
	Name      string
	Metrics   v1.ResourceList
	Available v1.ResourceList
}

type ResourcePodInfo struct {
	Name       string
	Metrics    *metricsapi.PodMetrics
	Pod        *v1.Pod
	Containers map[string]*ResourceContainerInfo
}

type ResourceContainerInfo struct {
	Name      string
	Metrics   *metricsapi.ContainerMetrics
	Container *v1.Container
}

type TopCmdPrinter struct {
	out io.Writer
}

func NewTopCmdPrinter(out io.Writer) *TopCmdPrinter {
	return &TopCmdPrinter{out: out}
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
		}, false)
		delete(availableResources, m.Name)
	}

	// print lines for nodes of which the metrics is unreachable.
	for nodeName := range availableResources {
		printMissingMetricsNodeLine(w, nodeName)
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metricsapi.PodMetrics, pods []v1.Pod, printContainers bool, withNamespace bool, noHeaders bool, sortBy string, sum, enumerate bool) error {
	if len(metrics) == 0 {
		return nil
	}
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()

	podCache := make(map[string]*ResourcePodInfo)
	containerCache := make(map[string]*ResourceContainerInfo)

	columnWidth := len(PodColumns)
	sumCols := len(MeasuredResources)
	if !noHeaders {
		if withNamespace {
			printValue(w, NamespaceColumn)
			columnWidth++
		}
		if printContainers {
			printValue(w, PodColumn)
			columnWidth++
		}
		if enumerate {
			PodColumns = append(PodColumns, EnumColumns...)
			columnWidth += len(EnumColumns)
			sumCols += len(EnumColumns)
		}
		printColumnNames(w, PodColumns)
	}

	sort.Sort(NewPodMetricsSorter(metrics, podCache, withNamespace, enumerate, sortBy))

	if enumerate {
		for _, p := range pods {
			podCache[p.Name] = &ResourcePodInfo{
				Name:       p.Name,
				Pod:        &p,
				Containers: make(map[string]*ResourceContainerInfo),
			}
			for _, c := range p.Spec.Containers {
				containerCache[c.Name] = &ResourceContainerInfo{
					Name:      c.Name,
					Container: &c,
				}
				podCache[p.Name].Containers[c.Name] = containerCache[c.Name]
			}
		}
	}

	for _, m := range metrics {
		if printContainers {
			sort.Sort(NewContainerMetricsSorter(m.Containers, sortBy))
			printSinglePodContainerMetrics(w, &m, containerCache, withNamespace, enumerate)
		} else {
			printSinglePodMetrics(w, &m, podCache, withNamespace, enumerate)
		}
	}

	if sum {
		scopedResources := getScopedResources(enumerate)
		adder := NewResourceAdder(scopedResources)
		for _, m := range metrics {
			if enumerate {
				adder.AddPodMetricsWithResources(&m, podCache[m.Name].Containers)
			} else {
				adder.AddPodMetrics(&m)
			}
		}
		printPodResourcesSum(w, adder.total, columnWidth, sumCols, enumerate)
	}

	return nil
}

func printColumnNames(out io.Writer, names []string) {
	for _, name := range names {
		printValue(out, name)
	}
	fmt.Fprint(out, "\n")
}

func printSinglePodMetrics(out io.Writer, m *metricsapi.PodMetrics, cache map[string]*ResourcePodInfo, withNamespace, enumerate bool) {
	podMetrics := getPodMetrics(m, cache, enumerate)
	if withNamespace {
		printValue(out, m.Namespace)
	}
	printMetricsLine(out, &ResourceMetricsInfo{
		Name:      m.Name,
		Metrics:   podMetrics,
		Available: v1.ResourceList{},
	}, enumerate)
}

func printSinglePodContainerMetrics(out io.Writer, m *metricsapi.PodMetrics, cache map[string]*ResourceContainerInfo, withNamespace, enumerate bool) {
	for _, c := range m.Containers {
		container := &v1.Container{}
		if withNamespace {
			printValue(out, m.Namespace)
		}
		if enumerate {
			container = cache[c.Name].Container
		}
		printValue(out, m.Name)
		printMetricsLine(out, &ResourceMetricsInfo{
			Name:      c.Name,
			Metrics:   getContainerMetrics(&c, container, enumerate),
			Available: v1.ResourceList{},
		}, enumerate)
	}
}

func getPodMetrics(m *metricsapi.PodMetrics, p map[string]*ResourcePodInfo, enumerate bool) v1.ResourceList {
	podMetrics := make(v1.ResourceList)
	scopedResources := getScopedResources(enumerate)
	for _, res := range scopedResources {
		podMetrics[res], _ = resource.ParseQuantity("0")
	}

	for _, c := range m.Containers {
		container := &v1.Container{}
		for _, res := range scopedResources {
			quantity := podMetrics[res]
			if enumerate {
				container = p[m.Name].Containers[c.Name].Container
			}
			quantity.Add(extractResource(&c, container, res))
			podMetrics[res] = quantity
		}
	}
	return podMetrics
}

func getContainerMetrics(metrics *metricsapi.ContainerMetrics, c *v1.Container, enumerate bool) v1.ResourceList {
	containerMetrics := make(v1.ResourceList)
	scopedResources := getScopedResources(enumerate)

	for _, res := range scopedResources {
		containerMetrics[res] = extractResource(metrics, c, res)
	}

	return containerMetrics
}

func getScopedResources(enumerate bool) []v1.ResourceName {
	scopedResources := MeasuredResources
	if enumerate {
		scopedResources = append(scopedResources, DefinedResources...)
	}

	return scopedResources
}

func extractResource(metrics *metricsapi.ContainerMetrics, c *v1.Container, resource v1.ResourceName) resource.Quantity {
	before, after, _ := strings.Cut(resource.String(), ".")
	switch before {
	case "requests":
		return c.Resources.Requests[v1.ResourceName(after)]
	case "limits":
		return c.Resources.Limits[v1.ResourceName(after)]
	default:
		return metrics.Usage[v1.ResourceName(before)]
	}
}

func printMetricsLine(out io.Writer, metrics *ResourceMetricsInfo, enumerate bool) {
	printValue(out, metrics.Name)
	printAllResourceUsages(out, metrics, enumerate)
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

func printAllResourceUsages(out io.Writer, metrics *ResourceMetricsInfo, enumerate bool) {
	scopedResources := getScopedResources(enumerate)
	for _, res := range scopedResources {
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
	case v1.ResourceCPU, v1.ResourceRequestsCPU, v1.ResourceLimitsCPU:
		fmt.Fprintf(out, "%vm", quantity.MilliValue())
	case v1.ResourceMemory, v1.ResourceRequestsMemory, v1.ResourceLimitsMemory:
		fmt.Fprintf(out, "%vMi", quantity.Value()/(1024*1024))
	default:
		fmt.Fprintf(out, "%v", quantity.Value())
	}
}

func printPodResourcesSum(out io.Writer, total v1.ResourceList, columnWidth, sumCols int, enumerate bool) {
	for i := 0; i < columnWidth-sumCols; i++ {
		printValue(out, "")
	}
	for i := 0; i < sumCols; i++ {
		printValue(out, "________")
	}
	fmt.Fprintf(out, "\n")
	for i := 0; i < columnWidth-(sumCols+1); i++ {
		printValue(out, "")
	}
	printMetricsLine(out, &ResourceMetricsInfo{
		Name:      "",
		Metrics:   total,
		Available: v1.ResourceList{},
	}, enumerate)
}
