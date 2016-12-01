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

	metricsapi "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/kubectl"
)

var (
	MeasuredResources = []api.ResourceName{
		api.ResourceCPU,
		api.ResourceMemory,
	}
	NodeColumns     = []string{"NAME", "CPU(cores)", "CPU%", "MEMORY(bytes)", "MEMORY%"}
	PodColumns      = []string{"NAME", "CPU(cores)", "MEMORY(bytes)"}
	NamespaceColumn = "NAMESPACE"
	PodColumn       = "POD"
)

type ResourceMetricsInfo struct {
	Name      string
	Metrics   api.ResourceList
	Available api.ResourceList
}

type TopCmdPrinter struct {
	out io.Writer
}

func NewTopCmdPrinter(out io.Writer) *TopCmdPrinter {
	return &TopCmdPrinter{out: out}
}

func (printer *TopCmdPrinter) PrintNodeMetrics(metrics []metricsapi.NodeMetrics, availableResources map[string]api.ResourceList) error {
	if len(metrics) == 0 {
		return nil
	}
	w := kubectl.GetNewTabWriter(printer.out)
	defer w.Flush()

	printColumnNames(w, NodeColumns)
	var usage api.ResourceList
	for _, m := range metrics {
		err := api.Scheme.Convert(&m.Usage, &usage, nil)
		if err != nil {
			return err
		}
		printMetricsLine(w, &ResourceMetricsInfo{
			Name:      m.Name,
			Metrics:   usage,
			Available: availableResources[m.Name],
		})
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metricsapi.PodMetrics, printContainers bool, withNamespace bool) error {
	if len(metrics) == 0 {
		return nil
	}
	w := kubectl.GetNewTabWriter(printer.out)
	defer w.Flush()

	if withNamespace {
		printValue(w, NamespaceColumn)
	}
	if printContainers {
		printValue(w, PodColumn)
	}
	printColumnNames(w, PodColumns)
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
	containers := make(map[string]api.ResourceList)
	podMetrics := make(api.ResourceList)
	for _, res := range MeasuredResources {
		podMetrics[res], _ = resource.ParseQuantity("0")
	}

	for _, c := range m.Containers {
		var usage api.ResourceList
		err := api.Scheme.Convert(&c.Usage, &usage, nil)
		if err != nil {
			return err
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
	if printContainersOnly {
		for contName := range containers {
			if withNamespace {
				printValue(out, m.Namespace)
			}
			printValue(out, m.Name)
			printMetricsLine(out, &ResourceMetricsInfo{
				Name:      contName,
				Metrics:   containers[contName],
				Available: api.ResourceList{},
			})
		}
	} else {
		if withNamespace {
			printValue(out, m.Namespace)
		}
		printMetricsLine(out, &ResourceMetricsInfo{
			Name:      m.Name,
			Metrics:   podMetrics,
			Available: api.ResourceList{},
		})
	}
	return nil
}

func printMetricsLine(out io.Writer, metrics *ResourceMetricsInfo) {
	printValue(out, metrics.Name)
	printAllResourceUsages(out, metrics)
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

func printSingleResourceUsage(out io.Writer, resourceType api.ResourceName, quantity resource.Quantity) {
	switch resourceType {
	case api.ResourceCPU:
		fmt.Fprintf(out, "%vm", quantity.MilliValue())
	case api.ResourceMemory:
		fmt.Fprintf(out, "%vMi", quantity.Value()/(1024*1024))
	default:
		fmt.Fprintf(out, "%v", quantity.Value())
	}
}
