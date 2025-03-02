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
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/cli-runtime/pkg/printers"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

var (
	MeasuredResources = []v1.ResourceName{
		v1.ResourceCPU,
		v1.ResourceMemory,
	}
	NodeColumns     = []string{"NAME", "CPU(cores)", "CPU(%)", "MEMORY(bytes)", "MEMORY(%)"}
	PodColumns      = []string{"NAME", "CPU(cores)", "MEMORY(bytes)"}
	NamespaceColumn = "NAMESPACE"
	PodColumn       = "POD"

	PodColumnsDefinitions = []metav1.TableColumnDefinition{
		{Name: "Name", Type: "string", Format: "name", Description: metav1.ObjectMeta{}.SwaggerDoc()["name"]},
		{Name: "CPU", Type: "string", Description: "CPU usage in cores."},
		{Name: "Memory", Type: "string", Description: "Memory usage in bytes."},
	}
	NamespaceColumnDefinition   = metav1.TableColumnDefinition{Name: "Namespace", Type: "string", Description: "Namespace where the pod is scheduled."}
	PodColumnDefinition         = metav1.TableColumnDefinition{Name: "Pod", Type: "string", Description: "This column is pod name, and Name column is container name."}
	PodCustomColumnsDefinitions = []metav1.TableColumnDefinition{
		{Name: "Node", Type: "string", Description: "Node where pod is scheduled."},
	}
)

type ResourceMetricsInfo struct {
	Name      string
	Metrics   v1.ResourceList
	Available v1.ResourceList
}

type TopCmdPrinter struct {
	out io.Writer

	// Value is nil when executing the `top node`
	PodPrintOptions *PodPrintOptions
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
		})
		delete(availableResources, m.Name)
	}

	// print lines for nodes of which the metrics is unreachable.
	for nodeName := range availableResources {
		printMissingMetricsNodeLine(w, nodeName)
	}
	return nil
}

func (printer *TopCmdPrinter) PrintPodMetrics(metrics []metricsapi.PodMetrics, printContainers bool, withNamespace bool, noHeaders bool, sortBy string, sum bool) error {
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
			printSinglePodContainerMetrics(w, &m, withNamespace)
		} else {
			printSinglePodMetrics(w, &m, withNamespace)
		}

	}

	if sum {
		adder := NewResourceAdder(MeasuredResources)
		for _, m := range metrics {
			adder.AddPodMetrics(&m)
		}
		printPodResourcesSum(w, adder.total, columnWidth)
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
		printValue(out, unknownMetricsStatus)
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

func printPodResourcesSum(out io.Writer, total v1.ResourceList, columnWidth int) {
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
	})

}

type PodPrintOptions struct {
	SortBy          string
	PrintContainers bool
	PrintNamespaces bool
	NoHeaders       bool
	Sum             bool

	Wide     bool
	PodsInfo *[]v1.Pod
}

// PrintObj implements printers.ResourcePrinter.
func (printer TopCmdPrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	w := printers.GetNewTabWriter(output)
	defer w.Flush()

	if podMetricsList, ok := obj.(*metricsapi.PodMetricsList); ok {
		options := printer.PodPrintOptions
		if options == nil {
			options = &PodPrintOptions{}
		}

		table, err := printTopPodMetricsList(podMetricsList, *options)
		if err != nil {
			return nil
		}

		if !options.NoHeaders {
			// avoid printing if no rows
			if len(table.Rows) == 0 {
				return nil
			}

			for _, column := range table.ColumnDefinitions {
				if !options.Wide && column.Priority != 0 {
					continue
				}
				// Keep the same output as before: "NAME  CPU(cores)  MEMORY(bytes)"
				columnName := strings.ToUpper(column.Name)
				if column.Name == "CPU" {
					columnName = "CPU(cores)"
				}
				if column.Name == "Memory" {
					columnName = "MEMORY(bytes)"
				}
				fmt.Fprintf(w, "%v\t", columnName)
			}
			fmt.Fprint(w, "\n")
		}
		for _, row := range table.Rows {
			for i, cell := range row.Cells {
				// don't panic if more cells
				if i >= len(table.ColumnDefinitions) {
					break
				}
				column := table.ColumnDefinitions[i]
				if !options.Wide && column.Priority != 0 {
					continue
				}
				// Keep the same output as before
				fmt.Fprintf(w, "%v\t", cell)
			}
			fmt.Fprint(w, "\n")
		}
		return nil
	}

	return errors.New("printing node metrics is not yet supported")
}

func printTopPodMetricsList(podMetricsList *metricsapi.PodMetricsList, options PodPrintOptions) (metav1.Table, error) {
	metrics := podMetricsList.Items

	tableColumns := PodColumnsDefinitions
	columnWidth := len(tableColumns)

	// ->[POD]  NAME  CPU(cores)  MEMORY(bytes)
	if options.PrintContainers {
		columnWidth++
		columns := make([]metav1.TableColumnDefinition, 0, columnWidth)
		columns = append(columns, PodColumnDefinition)
		tableColumns = append(columns, tableColumns...)
	}

	// ->[NAMESPACE]  [POD]  NAME  CPU(cores)  MEMORY(bytes)
	if options.PrintNamespaces {
		columnWidth++
		columns := make([]metav1.TableColumnDefinition, 0, columnWidth)
		columns = append(columns, NamespaceColumnDefinition)
		tableColumns = append(columns, tableColumns...)
	}

	sort.Sort(NewPodMetricsSorter(metrics, options.PrintNamespaces, options.SortBy))

	rows := make([]metav1.TableRow, 0, len(metrics))
	for _, m := range metrics {
		if options.PrintContainers {
			// POD  NAME  CPU(cores)  MEMORY(bytes)
			sort.Sort(NewContainerMetricsSorter(m.Containers, options.SortBy))
			for _, c := range m.Containers {
				cpuQuantity, _ := resource.ParseQuantity("0")
				cpuQuantity.Add(c.Usage[v1.ResourceCPU])
				memQuantity, _ := resource.ParseQuantity("0")
				memQuantity.Add(c.Usage[v1.ResourceMemory])

				r := metav1.TableRow{Object: runtime.RawExtension{Object: &m}}
				r.Cells = append(r.Cells, m.Name, c.Name, printResourceUsage(v1.ResourceCPU, cpuQuantity), printResourceUsage(v1.ResourceMemory, memQuantity))
				rows = append(rows, r)
			}
		} else {
			// NAME  CPU(cores)  MEMORY(bytes)
			cpuQuantity, _ := resource.ParseQuantity("0")
			memQuantity, _ := resource.ParseQuantity("0")
			for _, c := range m.Containers {
				cpuQuantity.Add(c.Usage[v1.ResourceCPU])
				memQuantity.Add(c.Usage[v1.ResourceMemory])
			}

			r := metav1.TableRow{Object: runtime.RawExtension{Object: &m}}
			r.Cells = append(r.Cells, m.Name, printResourceUsage(v1.ResourceCPU, cpuQuantity), printResourceUsage(v1.ResourceMemory, memQuantity))
			rows = append(rows, r)
		}
	}

	// ->[NAMESPACE]  [POD]  NAME  CPU(cores)  MEMORY(bytes)
	if options.PrintNamespaces {
		for i := range rows {
			row := rows[i]

			var m metav1.Object
			if obj := row.Object.Object; obj != nil {
				if acc, err := meta.Accessor(obj); err == nil {
					m = acc
				}
			}

			r := make([]interface{}, 1, columnWidth)
			r[0] = "<unknown>"
			if m != nil {
				r[0] = m.GetNamespace()
			}
			row.Cells = append(r, row.Cells...)

			rows[i] = row
		}
	}

	// [NAMESPACE]  [POD]  NAME  ->[NODE...]  CPU(cores)  MEMORY(bytes)
	if options.Wide {
		// Insert column after NAME and before CPU to align the output of options.Sum
		nameColumn := -1
		for i := range tableColumns {
			if tableColumns[i].Format == "name" && tableColumns[i].Type == "string" {
				nameColumn = i
				break
			}
		}

		if nameColumn != -1 {
			insertIndex := nameColumn + 1

			columnWidth = columnWidth + len(PodCustomColumnsDefinitions)
			columns := make([]metav1.TableColumnDefinition, 0, columnWidth)
			columns = append(columns, tableColumns[:insertIndex]...)
			columns = append(columns, PodCustomColumnsDefinitions...)
			tableColumns = append(columns, tableColumns[insertIndex:]...)

			// obj[namespace][podName][nodeName]
			podsNodeInfo := make(map[string]map[string]string)
			if options.PodsInfo != nil {
				for _, pod := range *options.PodsInfo {
					if podsNodeInfo[pod.Namespace] == nil {
						podsNodeInfo[pod.Namespace] = make(map[string]string)
					}
					podsNodeInfo[pod.Namespace][pod.Name] = pod.Spec.NodeName
				}
			}

			for i := range rows {
				row := rows[i]

				var m metav1.Object
				if obj := row.Object.Object; obj != nil {
					if acc, err := meta.Accessor(obj); err == nil {
						m = acc
					}
				}

				nodeName := "<none>"
				if m != nil {
					if nsPodsNodeInfo, found := podsNodeInfo[m.GetNamespace()]; found {
						if node, found := nsPodsNodeInfo[m.GetName()]; found {
							nodeName = node
						}
					}
				}
				r := make([]interface{}, 0, columnWidth)
				r = append(r, row.Cells[:insertIndex]...)
				// Adding custom data
				r = append(r, nodeName)
				row.Cells = append(r, row.Cells[insertIndex:]...)

				rows[i] = row
			}
		}
	}

	// [NAMESPACE]  [POD]  NAME  [NODE...]  CPU(cores)  MEMORY(bytes)
	//                                   ->[________    ________]
	//                                   ->[sum(CPU)    sum(MEM)]
	if options.Sum {
		cpuQuantity, _ := resource.ParseQuantity("0")
		memQuantity, _ := resource.ParseQuantity("0")
		for _, m := range metrics {
			for _, c := range m.Containers {
				cpuQuantity.Add(c.Usage[v1.ResourceCPU])
				memQuantity.Add(c.Usage[v1.ResourceMemory])
			}
		}

		r1 := make([]interface{}, columnWidth, columnWidth)
		r2 := make([]interface{}, columnWidth, columnWidth)
		for i := 0; i < columnWidth-2; i++ {
			r1[i] = ""
			r2[i] = ""
		}

		r1[columnWidth-2] = "________"
		r1[columnWidth-1] = "________"
		rows = append(rows, metav1.TableRow{Cells: r1})

		r2[columnWidth-2] = printResourceUsage(v1.ResourceCPU, cpuQuantity)
		r2[columnWidth-1] = printResourceUsage(v1.ResourceMemory, memQuantity)
		rows = append(rows, metav1.TableRow{Cells: r2})
	}

	return metav1.Table{ColumnDefinitions: tableColumns, Rows: rows}, nil
}

func printResourceUsage(resourceType v1.ResourceName, quantity resource.Quantity) string {
	switch resourceType {
	case v1.ResourceCPU:
		return fmt.Sprintf("%vm", quantity.MilliValue())
	case v1.ResourceMemory:
		return fmt.Sprintf("%vMi", quantity.Value()/(1024*1024))
	default:
		return fmt.Sprintf("%v", quantity.Value())
	}
}
