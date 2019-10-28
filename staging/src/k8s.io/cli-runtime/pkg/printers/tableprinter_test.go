/*
Copyright 2019 The Kubernetes Authors.

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
	"regexp"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var testPodName = "test-pod-name"
var testPodNamespace = "test-namespace"

var testPod = &corev1.Pod{
	TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
	ObjectMeta: metav1.ObjectMeta{
		Name:      testPodName,
		Namespace: testPodNamespace,
		Labels: map[string]string{
			"first-label":  "12",
			"second-label": "label-value",
		},
	},
}

var testStatus = &metav1.Status{
	TypeMeta: metav1.TypeMeta{APIVersion: "metav1", Kind: "Status"},
	Status:   "Failure",
	Message:  "test-status-message",
	Reason:   "test-status-reason",
}

func TestPrintTable_MissingColumnsRows(t *testing.T) {
	tests := []struct {
		columns  []metav1.TableColumnDefinition
		rows     []metav1.TableRow
		options  PrintOptions
		expected string
	}{
		// No columns and no rows means table string is empty.
		{
			columns:  []metav1.TableColumnDefinition{},
			rows:     []metav1.TableRow{},
			options:  PrintOptions{},
			expected: "",
		},
		// No rows, means table string is empty (columns aren't printed).
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows:     []metav1.TableRow{},
			options:  PrintOptions{},
			expected: "",
		},
	}

	for _, test := range tests {
		// Create the table from the columns and rows.
		table := &metav1.Table{
			ColumnDefinitions: test.columns,
			Rows:              test.rows,
		}
		// Print the table
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(table, out)

		// Validate the printed table is empty.
		if len(out.String()) > 0 {
			t.Errorf("Error Printing Table. Should be empty; got (%s)", out.String())
		}
	}
}

func TestPrintTable_ColumnPriority(t *testing.T) {
	tests := []struct {
		columns  []metav1.TableColumnDefinition
		rows     []metav1.TableRow
		options  PrintOptions
		expected string
	}{
		// Test a basic single row table. Columns with priority > 0 are not printed.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer", Priority: 1}, // Priority > 0
				{Name: "Age", Type: "string", Priority: 1},      // Priority > 0
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
			},
			options:  PrintOptions{},
			expected: "NAME    READY   STATUS\ntest1   1/1     podPhase\n",
		},
		// Test a basic multi-row table row table. Columns with priority > 0 are not printed.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer", Priority: 1}, // Priority > 0
				{Name: "Age", Type: "string", Priority: 1},      // Priority > 0
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
				{Cells: []interface{}{"test2", "1/2", "podPhase", int64(30), "21h"}},
				{Cells: []interface{}{"test3", "4/4", "podPhase", int64(1), "22h"}},
			},
			options: PrintOptions{},
			expected: `NAME    READY   STATUS
test1   1/1     podPhase
test2   1/2     podPhase
test3   4/4     podPhase
`,
		},
		// Test a single row table with "wide" printing option. Columns with priority > 0 are printed.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer", Priority: 1},
				{Name: "Age", Type: "string", Priority: 1},
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
			},
			// Print with no headers.
			options:  PrintOptions{Wide: true},
			expected: "NAME    READY   STATUS     RETRIES   AGE\ntest1   1/1     podPhase   5         20h\n",
		},
		// Test a multi-row table row table with "wide" printing option. Columns with priority > 0 are printed.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer", Priority: 1},
				{Name: "Age", Type: "string", Priority: 1},
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
				{Cells: []interface{}{"test2", "1/2", "podPhase", int64(30), "21h"}},
				{Cells: []interface{}{"test3", "4/4", "podPhase", int64(1), "22h"}},
			},
			options: PrintOptions{Wide: true},
			expected: `NAME    READY   STATUS     RETRIES   AGE
test1   1/1     podPhase   5         20h
test2   1/2     podPhase   30        21h
test3   4/4     podPhase   1         22h
`,
		},
	}
	for _, test := range tests {
		// Create the table from the columns and rows.
		table := &metav1.Table{
			ColumnDefinitions: test.columns,
			Rows:              test.rows,
		}
		// Print the table
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(table, out)
		// Validate the expected output matches the printed table.
		if test.expected != out.String() {
			t.Errorf("Table printing error: expected (%s), got (%s)", test.expected, out.String())
		}
	}
}

func TestPrintTable_ColumnHeaders(t *testing.T) {
	tests := []struct {
		columns  []metav1.TableColumnDefinition
		rows     []metav1.TableRow
		options  PrintOptions
		expected string
	}{
		// Test a basic single row table, shows columns.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
			},
			options:  PrintOptions{},
			expected: "NAME    READY   STATUS     RETRIES   AGE\ntest1   1/1     podPhase   5         20h\n",
		},
		// Test a basic multi-row table row table, shows columns.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
				{Cells: []interface{}{"test2", "1/2", "podPhase", int64(30), "21h"}},
				{Cells: []interface{}{"test3", "4/4", "podPhase", int64(1), "22h"}},
			},
			options: PrintOptions{},
			expected: `NAME    READY   STATUS     RETRIES   AGE
test1   1/1     podPhase   5         20h
test2   1/2     podPhase   30        21h
test3   4/4     podPhase   1         22h
`,
		},
		// Test a single row table with "NoHeaders" option, doesn't print columns.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
			},
			// Print with no headers.
			options:  PrintOptions{NoHeaders: true},
			expected: "test1   1/1   podPhase   5     20h\n",
		},
		// Test a multi-row table row table with "NoHeaders" option, doesn't print columns.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"}},
				{Cells: []interface{}{"test2", "1/2", "podPhase", int64(30), "21h"}},
				{Cells: []interface{}{"test3", "4/4", "podPhase", int64(1), "22h"}},
			},
			options: PrintOptions{NoHeaders: true},
			expected: `test1   1/1   podPhase   5     20h
test2   1/2   podPhase   30    21h
test3   4/4   podPhase   1     22h
`,
		},
	}
	for _, test := range tests {
		// Create the table from the columns and rows.
		table := &metav1.Table{
			ColumnDefinitions: test.columns,
			Rows:              test.rows,
		}
		// Print the table
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(table, out)
		// Validate the expected output matches the printed table.
		if test.expected != out.String() {
			t.Errorf("Table printing error: expected (%s), got (%s)", test.expected, out.String())
		}
	}
}

func TestPrintTable_WithNamespace(t *testing.T) {
	tests := []struct {
		columns  []metav1.TableColumnDefinition
		rows     []metav1.TableRow
		options  PrintOptions
		expected string
	}{
		// Test a single row table "WithNamespace" option, prepends NAMESPACE column.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{
					Cells:  []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"},
					Object: runtime.RawExtension{Object: testPod},
				},
			},
			// Print with namespace column prepended.
			options: PrintOptions{WithNamespace: true},
			expected: `NAMESPACE        NAME    READY   STATUS     RETRIES   AGE
test-namespace   test1   1/1     podPhase   5         20h
`,
		},
	}
	for _, test := range tests {
		// Create the table from the columns and rows.
		table := &metav1.Table{
			ColumnDefinitions: test.columns,
			Rows:              test.rows,
		}
		// Print the table
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(table, out)
		// Validate the expected output matches the printed table.
		if test.expected != out.String() {
			t.Errorf("Table printing error: expected (%s), got (%s)", test.expected, out.String())
		}
	}
}

func TestPrintTable_WithKind(t *testing.T) {
	tests := []struct {
		columns  []metav1.TableColumnDefinition
		rows     []metav1.TableRow
		options  PrintOptions
		expected string
	}{
		// Test a single row table "WithKind" option, prepends "pod" to name.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string", Format: "name"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{
					Cells: []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"},
				},
			},
			// Print with Kind "pod" prepended to name.
			options: PrintOptions{WithKind: true, Kind: schema.GroupKind{Kind: "Pod"}},
			expected: `NAME        READY   STATUS     RETRIES   AGE
pod/test1   1/1     podPhase   5         20h
`,
		},
	}
	for _, test := range tests {
		// Create the table from the columns and rows.
		table := &metav1.Table{
			ColumnDefinitions: test.columns,
			Rows:              test.rows,
		}
		// Print the table
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(table, out)
		// Validate the expected output matches the printed table.
		if test.expected != out.String() {
			t.Errorf("Table printing error: expected (%s), got (%s)", test.expected, out.String())
		}
	}
}

func TestPrintTable_WithLabels(t *testing.T) {
	tests := []struct {
		columns  []metav1.TableColumnDefinition
		rows     []metav1.TableRow
		options  PrintOptions
		expected string
	}{
		// Test a table "ShowLabels" option, appends labels as columns.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{
					Cells:  []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"},
					Object: runtime.RawExtension{Object: testPod},
				},
			},
			options: PrintOptions{ShowLabels: true},
			expected: `NAME    READY   STATUS     RETRIES   AGE   LABELS
test1   1/1     podPhase   5         20h   first-label=12,second-label=label-value
`,
		},
		// Test a table "ColumnLabels" option, appends labels as columns.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{
					Cells:  []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"},
					Object: runtime.RawExtension{Object: testPod},
				},
			},
			// Print "second-label" as column name, with label value.
			options: PrintOptions{ColumnLabels: []string{"second-label"}},
			expected: `NAME    READY   STATUS     RETRIES   AGE   SECOND-LABEL
test1   1/1     podPhase   5         20h   label-value
`,
		},
		// Test a table "ColumnLabels" option, appends labels as columns.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{
					Cells:  []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"},
					Object: runtime.RawExtension{Object: testPod},
				},
			},
			// Print multiple-labels as columns, with their values.
			options: PrintOptions{ColumnLabels: []string{"first-label", "second-label"}},
			expected: `NAME    READY   STATUS     RETRIES   AGE   FIRST-LABEL   SECOND-LABEL
test1   1/1     podPhase   5         20h   12            label-value
`,
		},
	}
	for _, test := range tests {
		// Create the table from the columns and rows.
		table := &metav1.Table{
			ColumnDefinitions: test.columns,
			Rows:              test.rows,
		}
		// Print the table
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(table, out)
		// Validate the expected output matches the printed table.
		if test.expected != out.String() {
			t.Errorf("Table printing error: expected (%s), got (%s)", test.expected, out.String())
		}
	}
}

func TestPrintTable_NonTable(t *testing.T) {
	tests := []struct {
		object   runtime.Object
		options  PrintOptions
		expected string
	}{
		// Test non-table default printing for a pod.
		{
			object:   testPod,
			options:  PrintOptions{},
			expected: "NAME            AGE\ntest-pod-name   <unknown>\n",
		},
		// Test non-table default printing for a pod with "NoHeaders" option.
		{
			object:   testPod,
			options:  PrintOptions{NoHeaders: true},
			expected: "test-pod-name   <unknown>\n",
		},
		// Test non-table default printing for a pod with "WithNamespace" option.
		{
			object:   testPod,
			options:  PrintOptions{WithNamespace: true},
			expected: "NAMESPACE        NAME            AGE\ntest-namespace   test-pod-name   <unknown>\n",
		},
		// Test non-table default printing for a pod with "ShowLabels" option.
		{
			object:  testPod,
			options: PrintOptions{ShowLabels: true},
			expected: `NAME            AGE         LABELS
test-pod-name   <unknown>   first-label=12,second-label=label-value
`,
		},
		// Test non-table default printing for a Status resource.
		{
			object:  testStatus,
			options: PrintOptions{},
			expected: `STATUS    REASON               MESSAGE
Failure   test-status-reason   test-status-message
`,
		},
		// Test non-table default printing for a Status resource with NoHeaders options.
		{
			object:   testStatus,
			options:  PrintOptions{NoHeaders: true},
			expected: "Failure   test-status-reason   test-status-message\n",
		},
	}
	for _, test := range tests {
		// Print the table
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(test.object, out)
		// Validate the expected output matches the printed table.
		if test.expected != out.String() {
			t.Errorf("Table printing error: expected (%s), got (%s)", test.expected, out.String())
		}
	}
}

func TestPrintTable_WatchEvents(t *testing.T) {
	tests := []struct {
		columns  []metav1.TableColumnDefinition
		rows     []metav1.TableRow
		options  PrintOptions
		expected string
	}{
		// Print WatchEvent with Table resource.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{
					Cells:  []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"},
					Object: runtime.RawExtension{Object: testPod},
				},
			},
			options: PrintOptions{},
			expected: `EVENT   NAME    READY   STATUS     RETRIES   AGE
Added   test1   1/1     podPhase   5         20h
`,
		},
		// Print WatchEvent with Table, and "NoHeaders", "WithNamespace", and "ShowLabels" options.
		{
			columns: []metav1.TableColumnDefinition{
				{Name: "Name", Type: "string"},
				{Name: "Ready", Type: "string"},
				{Name: "Status", Type: "string"},
				{Name: "Retries", Type: "integer"},
				{Name: "Age", Type: "string"},
			},
			rows: []metav1.TableRow{
				{
					Cells:  []interface{}{"test1", "1/1", "podPhase", int64(5), "20h"},
					Object: runtime.RawExtension{Object: testPod},
				},
			},
			options:  PrintOptions{NoHeaders: true, WithNamespace: true, ShowLabels: true},
			expected: "Added   test-namespace   test1   1/1   podPhase   5     20h   first-label=12,second-label=label-value\n",
		},
	}
	for _, test := range tests {
		// Create the table from the columns and rows.
		table := &metav1.Table{
			ColumnDefinitions: test.columns,
			Rows:              test.rows,
		}
		// Add the table to the WatchEvent.
		event := &metav1.WatchEvent{
			Type:   "Added",
			Object: runtime.RawExtension{Object: table},
		}
		// Print the event
		out := bytes.NewBuffer([]byte{})
		printer := NewTablePrinter(test.options)
		printer.PrintObj(event, out)
		// Validate the expected output matches the printed table.
		if test.expected != out.String() {
			t.Errorf("Event/Table printing error: expected (%s), got (%s)", test.expected, out.String())
		}
	}
}

func TestPrintUnstructuredObject(t *testing.T) {
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Test",
			"dummy1":     "present",
			"dummy2":     "present",
			"metadata": map[string]interface{}{
				"name":              "MyName",
				"namespace":         "MyNamespace",
				"creationTimestamp": "2017-04-01T00:00:00Z",
				"resourceVersion":   123,
				"uid":               "00000000-0000-0000-0000-000000000001",
				"dummy3":            "present",
				"labels":            map[string]interface{}{"test": "other"},
			},
			/*"items": []interface{}{
				map[string]interface{}{
					"itemBool": true,
					"itemInt":  42,
				},
			},*/
			"url":    "http://localhost",
			"status": "ok",
		},
	}

	tests := []struct {
		expected string
		options  PrintOptions
		object   runtime.Object
	}{
		{
			expected: "NAME\\s+AGE\nMyName\\s+\\d+",
			object:   obj,
		},
		{
			options: PrintOptions{
				WithNamespace: true,
			},
			expected: "NAMESPACE\\s+NAME\\s+AGE\nMyNamespace\\s+MyName\\s+\\d+",
			object:   obj,
		},
		{
			options: PrintOptions{
				ShowLabels:    true,
				WithNamespace: true,
			},
			expected: "NAMESPACE\\s+NAME\\s+AGE\\s+LABELS\nMyNamespace\\s+MyName\\s+\\d+\\w+\\s+test\\=other",
			object:   obj,
		},
		{
			expected: "NAME\\s+AGE\nMyName\\s+\\d+\\w+\nMyName2\\s+\\d+",
			object: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Test",
					"dummy1":     "present",
					"dummy2":     "present",
					"items": []interface{}{
						map[string]interface{}{
							"metadata": map[string]interface{}{
								"name":              "MyName",
								"namespace":         "MyNamespace",
								"creationTimestamp": "2017-04-01T00:00:00Z",
								"resourceVersion":   123,
								"uid":               "00000000-0000-0000-0000-000000000001",
								"dummy3":            "present",
								"labels":            map[string]interface{}{"test": "other"},
							},
						},
						map[string]interface{}{
							"metadata": map[string]interface{}{
								"name":              "MyName2",
								"namespace":         "MyNamespace",
								"creationTimestamp": "2017-04-01T00:00:00Z",
								"resourceVersion":   123,
								"uid":               "00000000-0000-0000-0000-000000000001",
								"dummy3":            "present",
								"labels":            "badlabel",
							},
						},
					},
					"url":    "http://localhost",
					"status": "ok",
				},
			},
		},
	}
	out := bytes.NewBuffer([]byte{})

	for _, test := range tests {
		out.Reset()
		printer := NewTablePrinter(test.options)
		printer.PrintObj(test.object, out)

		matches, err := regexp.MatchString(test.expected, out.String())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !matches {
			t.Errorf("wanted:\n%s\ngot:\n%s", test.expected, out)
		}
	}
}

type TestUnknownType struct{}

func (obj *TestUnknownType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *TestUnknownType) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	clone := *obj
	return &clone
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := NewTablePrinter(PrintOptions{})
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&TestUnknownType{}, buffer)
	if err == nil {
		t.Errorf("An error was expected from printing unknown type")
	}
}
