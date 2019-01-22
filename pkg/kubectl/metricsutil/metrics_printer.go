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
	"k8s.io/kubernetes/pkg/kubectl/util/printers"
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

type PodMetricsInfo struct {
	Namespace     string
	PodName       string
	ContainerName string
	Metrics       v1.ResourceList
}

type NodeMetricsInfo struct {
	NodeName  string
	Metrics   v1.ResourceList
	Available v1.ResourceList
}

type TopCmdPrinter struct {
	out    io.Writer
	sortBy string
}

func NewTopCmdPrinter(out io.Writer, sortBy string) *TopCmdPrinter {
	return &TopCmdPrinter{out, sortBy}
}

func (printer *TopCmdPrinter) PrintNodeMetrics(metrics []metricsapi.NodeMetrics, availableResources map[string]v1.ResourceList, noHeaders bool) error {
	if len(metrics) == 0 {
		return nil
	}
	w := printers.GetNewTabWriter(printer.out)
	defer w.Flush()

	if printer.sortBy == "cpu" {
		sort.Slice(metrics, func(i, j int) bool {
			qi := metrics[i].Usage[v1.ResourceCPU]
			qj := metrics[j].Usage[v1.ResourceCPU]
			return qi.Cmp(qj) < 0
		})
	} else if printer.sortBy == "memory" {
		sort.Slice(metrics, func(i, j int) bool {
			qi := metrics[i].Usage[v1.ResourceMemory]
			qj := metrics[j].Usage[v1.ResourceMemory]
			return qi.Cmp(qj) < 0
		})
	} else {
		sort.Slice(metrics, func(i, j int) bool {
			return metrics[i].Name < metrics[j].Name
		})
	}
	if !noHeaders {
		printColumnNames(w, NodeColumns)
	}
	var usage v1.ResourceList
	for _, m := range metrics {
		err := scheme.Scheme.Convert(&m.Usage, &usage, nil)
		if err != nil {
			return err
		}
		printValue(w, m.Name)
		printAllResourceUsages(w, usage, availableResources[m.Name])
		fmt.Fprint(w, "\n")
		delete(availableResources, m.Name)
	}

	// print lines for nodes of which the metrics is unreachable.
	for nodeName := range availableResources {
		printMissingMetricsNodeLine(w, nodeName)
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metricsapi.PodMetrics, printContainers bool, withNamespace bool, noHeaders bool) error {
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
	infos, err := convert(metrics, printContainers)
	if err != nil {
		return err
	}
	if printer.sortBy == "cpu" {
		sort.Slice(infos, func(i, j int) bool {
			qi := infos[i].Metrics[v1.ResourceCPU]
			qj := infos[j].Metrics[v1.ResourceCPU]
			return qi.Cmp(qj) < 0
		})
	} else if printer.sortBy == "memory" {
		sort.Slice(infos, func(i, j int) bool {
			qi := infos[i].Metrics[v1.ResourceMemory]
			qj := infos[j].Metrics[v1.ResourceMemory]
			return qi.Cmp(qj) < 0
		})
	} else {
		sort.Slice(metrics, func(i, j int) bool {
			return metrics[i].Name < metrics[j].Name
		})
	}

	printPodMetrics(w, infos, printContainers, withNamespace)
	return nil
}

func printColumnNames(out io.Writer, names []string) {
	for _, name := range names {
		printValue(out, name)
	}
	fmt.Fprint(out, "\n")
}
func printPodMetrics(out io.Writer, infos []PodMetricsInfo, printContainersOnly bool, withNamespace bool) {
	for _, info := range infos {
		if withNamespace {
			printValue(out, info.Namespace)
		}
		printValue(out, info.PodName)
		if printContainersOnly {
			printValue(out, info.ContainerName)
		}
		printAllResourceUsages(out, info.Metrics, v1.ResourceList{})
		fmt.Fprint(out, "\n")
	}
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

func printAllResourceUsages(out io.Writer, metrics v1.ResourceList, available v1.ResourceList) {
	for _, res := range MeasuredResources {
		quantity := metrics[res]
		printSingleResourceUsage(out, res, quantity)
		fmt.Fprint(out, "\t")
		if available, found := available[res]; found {
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

func convert(metrics []metricsapi.PodMetrics, printContainersOnly bool) ([]PodMetricsInfo, error) {
	var ret []PodMetricsInfo

	for _, m := range metrics {
		pm := v1.ResourceList{}
		for _, res := range MeasuredResources {
			pm[res], _ = resource.ParseQuantity("0")
		}
		for _, c := range m.Containers {
			var cm v1.ResourceList
			err := scheme.Scheme.Convert(&c.Usage, &cm, nil)
			if err != nil {
				return nil, err
			}
			if !printContainersOnly {
				for _, res := range MeasuredResources {
					quantity := pm[res]
					quantity.Add(cm[res])
					pm[res] = quantity
				}

			} else {
				ret = append(ret, PodMetricsInfo{
					Namespace:     m.Namespace,
					PodName:       m.Name,
					ContainerName: c.Name,
					Metrics:       cm,
				})
			}
		}
		if !printContainersOnly {
			ret = append(ret, PodMetricsInfo{
				Namespace: m.Namespace,
				PodName:   m.Name,
				Metrics:   pm,
			})
		}
	}

	return ret, nil
}
