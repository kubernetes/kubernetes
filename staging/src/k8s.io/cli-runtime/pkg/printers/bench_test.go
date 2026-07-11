/*
Copyright 2022 The Kubernetes Authors.

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

package printers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
)

func newBenchmarkList() *unstructured.UnstructuredList {
	list := &unstructured.UnstructuredList{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "PodList",
		},
		Items: []unstructured.Unstructured{},
	}
	for i := 0; i < 1000; i++ {
		list.Items = append(list.Items, unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":      fmt.Sprintf("pod%d", i),
					"namespace": "ns",
					"labels": map[string]interface{}{
						"first-label":  "12",
						"second-label": "label-value",
					},
				},
			},
		})
	}
	return list
}

func newBenchmarkTable() *metav1.Table {
	table := &metav1.Table{
		ColumnDefinitions: []metav1.TableColumnDefinition{
			{Name: "Name", Type: "string"},
			{Name: "Ready", Type: "string"},
			{Name: "Status", Type: "string"},
			{Name: "Retries", Type: "integer"},
			{Name: "Age", Type: "string"},
		},
		Rows: []metav1.TableRow{},
	}
	for _, pod := range newBenchmarkList().Items {
		raw, _ := json.Marshal(pod)
		table.Rows = append(table.Rows, metav1.TableRow{
			Cells: []interface{}{pod.GetName(), "1/1", "Ready", int64(0), "20h"},
			Object: runtime.RawExtension{
				Raw:    raw,
				Object: &pod,
			},
		})
	}
	return table
}

func BenchmarkDiscardingPrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter { return NewDiscardingPrinter() }, data)
}

func BenchmarkJSONPrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter { return &JSONPrinter{} }, data)
}

func BenchmarkJSONPathPrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter {
		printer, _ := NewJSONPathPrinter("{range .items[*]}{.metadata.name}\n{end}")
		return printer
	}, data)
}

func BenchmarkYAMLPrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter { return &YAMLPrinter{} }, data)
}

func BenchmarkTablePrinter(b *testing.B) {
	data := newBenchmarkTable()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter { return NewTablePrinter(PrintOptions{}) }, data)
}

func BenchmarkGoTemplatePrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter {
		printer, _ := NewGoTemplatePrinter([]byte("{{range .items}}{{.metadata.name}}{{\"\\n\"}}{{end}}"))
		return printer
	}, data)
}

func BenchmarkNamePrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter { return &NamePrinter{} }, data)
}

func BenchmarkOmitManagedFieldsPrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter { return &OmitManagedFieldsPrinter{Delegate: NewDiscardingPrinter()} }, data)
}

func BenchmarkTypeSetterPrinter(b *testing.B) {
	data := newBenchmarkList()
	b.ResetTimer()
	benchmarkPrinter(b, func() ResourcePrinter { return NewTypeSetter(scheme.Scheme).ToPrinter(NewDiscardingPrinter()) }, data)
}

func benchmarkPrinter(b *testing.B, printerFunc func() ResourcePrinter, data runtime.Object) {
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := &bytes.Buffer{}
			if err := printerFunc().PrintObj(data, buf); err != nil {
				b.Errorf("PrintObj failed: %v", err)
			}
		}
	})
}

// newLargeTable creates a metav1.Table with the given number of rows.
// Each row has 5 string columns with realistic cell widths, representative
// of a typical "kubectl get" response.
func newLargeTable(rowCount int) *metav1.Table {
	table := &metav1.Table{
		ColumnDefinitions: []metav1.TableColumnDefinition{
			{Name: "Name", Type: "string", Format: "name"},
			{Name: "Status", Type: "string"},
			{Name: "Roles", Type: "string"},
			{Name: "Age", Type: "string"},
			{Name: "Version", Type: "string"},
		},
		Rows: make([]metav1.TableRow, rowCount),
	}

	roles := []string{"control-plane", "worker", "worker", "worker", "worker", "infra", "worker", "worker"}
	statuses := []string{"Ready", "Ready", "Ready", "Ready,SchedulingDisabled", "Ready", "NotReady", "Ready", "Ready"}
	ages := []string{"120d", "89d", "45d", "30d", "15d", "7d", "3d", "1d"}
	versions := []string{"v1.32.0", "v1.32.0", "v1.31.4", "v1.32.0", "v1.31.4", "v1.32.0", "v1.32.0", "v1.31.4"}

	for i := range rowCount {
		idx := i % len(roles)
		table.Rows[i] = metav1.TableRow{
			Cells: []interface{}{
				fmt.Sprintf("resource-%05d.us-east-1.compute.internal", i),
				statuses[idx],
				roles[idx],
				ages[idx],
				versions[idx],
			},
		}
	}

	return table
}

func benchmarkLargeTablePrinter(b *testing.B, rowCount int) {
	table := newLargeTable(rowCount)
	printer := NewTablePrinter(PrintOptions{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := printer.PrintObj(table, io.Discard); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTablePrinter_10kRows(b *testing.B)  { benchmarkLargeTablePrinter(b, 10000) }
func BenchmarkTablePrinter_100kRows(b *testing.B) { benchmarkLargeTablePrinter(b, 100000) }
func BenchmarkTablePrinter_500kRows(b *testing.B) { benchmarkLargeTablePrinter(b, 500000) }

// BenchmarkTablePrinter_100kRows_ToBuffer benchmarks writing to a real buffer
// (not discard) to measure actual output generation overhead.
func BenchmarkTablePrinter_100kRows_ToBuffer(b *testing.B) {
	table := newLargeTable(100000)
	printer := NewTablePrinter(PrintOptions{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf := &bytes.Buffer{}
		buf.Grow(100000 * 100) // pre-allocate ~10MB
		if err := printer.PrintObj(table, buf); err != nil {
			b.Fatal(err)
		}
	}
}
