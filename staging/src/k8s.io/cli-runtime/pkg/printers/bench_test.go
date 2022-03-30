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
