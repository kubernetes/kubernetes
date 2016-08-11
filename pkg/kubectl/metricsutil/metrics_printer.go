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
	"time"

	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubectl"
)

var (
	MeasuredResources = []v1.ResourceName{
		v1.ResourceCPU,
		v1.ResourceMemory,
		v1.ResourceStorage,
	}
	NodeColumns     = []string{"NAME", "CPU (cores)", "MEMORY (bytes)", "STORAGE (bytes)", "TIMESTAMP"}
	PodColumns      = []string{"NAME", "CPU (cores)", "MEMORY (bytes)", "STORAGE (bytes)", "TIMESTAMP"}
	NamespaceColumn = "NAMESPACE"
	PodColumn       = "POD"
)

type ResourceMetricsInfo struct {
	Name      string
	Metrics   v1.ResourceList
	Timestamp string
}

type TopCmdPrinter struct {
	out io.Writer
}

func NewTopCmdPrinter(out io.Writer) *TopCmdPrinter {
	return &TopCmdPrinter{out: out}
}

func (printer *TopCmdPrinter) PrintNodeMetrics(metrics []metrics_api.NodeMetrics) error {
	if len(metrics) == 0 {
		return nil
	}
	w := kubectl.GetNewTabWriter(printer.out)
	defer w.Flush()

	printColumnNames(w, NodeColumns)
	for _, m := range metrics {
		printMetricsLine(w, &ResourceMetricsInfo{
			Name:      m.Name,
			Metrics:   m.Usage,
			Timestamp: m.Timestamp.Time.Format(time.RFC1123Z),
		})
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metrics_api.PodMetrics, printContainers bool, withNamespace bool) error {
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
		printSinglePodMetrics(w, &m, printContainers, withNamespace)
	}
	return nil
}

func printColumnNames(out io.Writer, names []string) {
	for _, name := range names {
		printValue(out, name)
	}
	fmt.Fprint(out, "\n")
}

func printSinglePodMetrics(out io.Writer, m *metrics_api.PodMetrics, printContainersOnly bool, withNamespace bool) {
	containers := make(map[string]v1.ResourceList)
	podMetrics := make(v1.ResourceList)
	for _, res := range MeasuredResources {
		podMetrics[res], _ = resource.ParseQuantity("0")
	}
	for _, c := range m.Containers {
		containers[c.Name] = c.Usage
		if !printContainersOnly {
			for _, res := range MeasuredResources {
				quantity := podMetrics[res]
				quantity.Add(c.Usage[res])
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
				Timestamp: m.Timestamp.Time.Format(time.RFC1123Z),
			})
		}
	} else {
		if withNamespace {
			printValue(out, m.Namespace)
		}
		printMetricsLine(out, &ResourceMetricsInfo{
			Name:      m.Name,
			Metrics:   podMetrics,
			Timestamp: m.Timestamp.Time.Format(time.RFC1123Z),
		})
	}
}

func printMetricsLine(out io.Writer, metrics *ResourceMetricsInfo) {
	printValue(out, metrics.Name)
	printAllResourceUsages(out, metrics.Metrics)
	printValue(out, metrics.Timestamp)
	fmt.Fprint(out, "\n")
}

func printValue(out io.Writer, value interface{}) {
	fmt.Fprintf(out, "%v\t", value)
}

func printAllResourceUsages(out io.Writer, usage v1.ResourceList) {
	for _, res := range MeasuredResources {
		quantity := usage[res]
		printSingleResourceUsage(out, res, quantity)
		fmt.Fprint(out, "\t")
	}
}

func printSingleResourceUsage(out io.Writer, resourceType v1.ResourceName, quantity resource.Quantity) {
	switch resourceType {
	case v1.ResourceCPU:
		fmt.Fprintf(out, "%vm", quantity.MilliValue())
	case v1.ResourceMemory:
		fmt.Fprintf(out, "%vMi", quantity.Value()/(1024*1024))
	case v1.ResourceStorage:
		// TODO: Change it after storage metrics collection is finished.
		fmt.Fprint(out, "-")
	default:
		fmt.Fprintf(out, "%v", quantity.Value())
	}
}
