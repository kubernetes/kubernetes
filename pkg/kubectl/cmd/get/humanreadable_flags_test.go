/*
Copyright 2018 The Kubernetes Authors.

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

package get

import (
	"bytes"
	"regexp"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestHumanReadablePrinterSupportsExpectedOptions(t *testing.T) {
	testObject := &api.Pod{ObjectMeta: metav1.ObjectMeta{
		Name: "foo",
		Labels: map[string]string{
			"l1": "value",
		},
	}}

	testCases := []struct {
		name       string
		showKind   bool
		showLabels bool

		// TODO(juanvallejo): test sorting once it's moved to the HumanReadablePrinter
		sortBy       string
		columnLabels []string
		extraColumns []string

		noHeaders     bool
		withNamespace bool

		outputFormat string

		expectedError  string
		expectedOutput string
		expectNoMatch  bool
	}{
		{
			name:           "empty output format matches a humanreadable printer",
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\nfoo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:           "\"wide\" output format prints",
			outputFormat:   "wide",
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +IP\\ +NODE\\ +NOMINATED NODE\\ +READINESS GATES\nfoo\\ +0/0\\ +0\\ +<unknown>\\ +<none>\\ +<none>\\ +<none>\\ +<none>\n",
		},
		{
			name:           "no-headers prints output with no headers",
			noHeaders:      true,
			expectedOutput: "foo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:           "no-headers and a \"wide\" output format prints output with no headers and additional columns",
			outputFormat:   "wide",
			noHeaders:      true,
			expectedOutput: "foo\\ +0/0\\ +0\\ +<unknown>\\ +<none>\\ +<none>\\ +<none>\\ +<none>\n",
		},
		{
			name:           "show-kind displays the resource's kind, even when printing a single type of resource",
			showKind:       true,
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\npod/foo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:           "label-columns prints specified label values in new column",
			columnLabels:   []string{"l1"},
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +L1\nfoo\\ +0/0\\ +0\\ +<unknown>\\ +value\n",
		},
		{
			name:           "withNamespace displays an additional NAMESPACE column",
			withNamespace:  true,
			expectedOutput: "NAMESPACE\\ +NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\n\\ +foo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:          "no printer is matched on an invalid outputFormat",
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
		{
			name:          "printer should not match on any other format supported by another printer",
			outputFormat:  "go-template",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := HumanPrintFlags{
				ShowKind:     &tc.showKind,
				ShowLabels:   &tc.showLabels,
				SortBy:       &tc.sortBy,
				ColumnLabels: &tc.columnLabels,
				ExtraColumns: &tc.extraColumns,

				NoHeaders:     tc.noHeaders,
				WithNamespace: tc.withNamespace,
			}

			if tc.showKind {
				printFlags.Kind = schema.GroupKind{Kind: "pod"}
			}

			p, err := printFlags.ToPrinter(tc.outputFormat)
			if tc.expectNoMatch {
				if !genericclioptions.IsNoCompatiblePrinterError(err) {
					t.Fatalf("expected no printer matches for output format %q", tc.outputFormat)
				}
				return
			}
			if genericclioptions.IsNoCompatiblePrinterError(err) {
				t.Fatalf("expected to match template printer for output format %q", tc.outputFormat)
			}

			if len(tc.expectedError) > 0 {
				if err == nil || !strings.Contains(err.Error(), tc.expectedError) {
					t.Errorf("expecting error %q, got %v", tc.expectedError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			out := bytes.NewBuffer([]byte{})
			err = p.PrintObj(testObject, out)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			match, err := regexp.Match(tc.expectedOutput, out.Bytes())
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !match {
				t.Errorf("unexpected output: expecting %q, got %q", tc.expectedOutput, out.String())
			}
		})
	}
}

func TestHumanReadablePrinterSupportsExtraColumnsPrinting(t *testing.T) {
	columns := []metav1beta1.TableColumnDefinition{
		{Name: "Name", Type: "string", Format: "name", Description: metav1.ObjectMeta{}.SwaggerDoc()["name"]},
		{Name: "Ready", Type: "string", Description: "The aggregate readiness state of this pod for accepting traffic."},
		{Name: "Status", Type: "string", Description: "The aggregate status of the containers in this pod."},
		{Name: "Restarts", Type: "integer", Description: "The number of times the containers in this pod have been restarted."},
		{Name: "Age", Type: "string", Description: metav1.ObjectMeta{}.SwaggerDoc()["creationTimestamp"]},
		{Name: "IP", Type: "string", Priority: 1, Description: v1.PodStatus{}.SwaggerDoc()["podIP"]},
		{Name: "Node", Type: "string", Priority: 1, Description: v1.PodSpec{}.SwaggerDoc()["nodeName"]},
		{Name: "Nominated Node", Type: "string", Priority: 1, Description: v1.PodStatus{}.SwaggerDoc()["nominatedNodeName"]},
		{Name: "Readiness Gates", Type: "string", Priority: 1, Description: v1.PodSpec{}.SwaggerDoc()["readinessGates"]},
	}

	pod1 := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "Pod",
			"apiVersion": "v1",
			"metadata": map[string]interface{}{
				"name": "testpod123",
				"labels": map[string]interface{}{
					"app": "testapp123",
				},
			},
			"spec": map[string]interface{}{
				"nodeName": "testnode123",
			},
		},
	}

	pod2 := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "Pod",
			"apiVersion": "v1",
			"metadata": map[string]interface{}{
				"name": "testpod123",
				"labels": map[string]interface{}{
					"app": "testapp123",
				},
			},
			"spec": map[string]interface{}{
				"nodeName": "atestnode456",
			},
		},
	}

	testCases := []struct {
		name       string
		input      *metav1beta1.Table
		showKind   bool
		showLabels bool

		sortBy       string
		columnLabels []string
		extraColumns []string

		noHeaders     bool
		withNamespace bool

		outputFormat string

		expectedError  string
		expectedOutput string
		expectNoMatch  bool
	}{
		{
			name:         "--extra-columns Name:.metadata.name prints name and default columns",
			extraColumns: []string{"Name:.metadata.name"},
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod", "1/1", "Running", int64(1), "24d", "<none>", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
		},
		{
			name:         "--extra-columns Name:.metadata.name, Node:.spec.nodeName prints name, nodeName and default columns",
			extraColumns: []string{"Name:.metadata.name", "Node:.spec.nodeName"},
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod", "1/1", "Running", int64(1), "24d", "<none>", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +Name\\ +Node\ntestpod\\ +1/1\\ +Running\\ +1\\ +24d\\ +testpod123\\ +testnode123\n",
		},
		{
			name:         "--extra-columns Node:.spec.nodeName prints node name and default columns",
			extraColumns: []string{"Node:.spec.nodeName"},
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod", "1/1", "Running", int64(1), "24d", "<none>", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +Node\ntestpod\\ +1/1\\ +Running\\ +1\\ +24d\\ +testnode123\n",
		},
		{
			name:         "nonexistent spec path prints default columns and <none>",
			extraColumns: []string{"NoneSpec:.spec.nonexistent"},
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod", "1/1", "Running", int64(1), "24d", "<none>", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +NoneSpec\ntestpod\\ +1/1\\ +Running\\ +1\\ +24d\\ +<none>\n",
		},
		{
			name:         "-o wide + --extra-columns Node:.spec.nodeName prints default wide columns and node name column",
			extraColumns: []string{"Node:.spec.nodeName"},
			outputFormat: "wide",
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod", "1/1", "Running", int64(1), "24d", "10.209.18.164", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +IP\\ +NODE\\ +NOMINATED NODE\\ +READINESS GATES\\ +Node\ntestpod\\ +1/1\\ +Running\\ +1\\ +24d\\ +10.209.18.164\\ +<none>\\ +<none>\\ +<none>\\ +testnode123\n",
		},
		{
			name:         "-L <label> + --extra-columns Node:.spec.nodeName prints default columns + label column + node name column",
			extraColumns: []string{"Node:.spec.nodeName"},
			columnLabels: []string{"app"},
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod", "1/1", "Running", int64(1), "24d", "10.209.18.164", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +APP\\ +Node\ntestpod\\ +1/1\\ +Running\\ +1\\ +24d\\ +testapp123\\ +testnode123\n",
		},
		{
			name:         "--no-headers + --extra-columns Node:.spec.nodeName prints default columns + node name column without headers",
			extraColumns: []string{"Node:.spec.nodeName"},
			noHeaders:    true,
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod", "1/1", "Running", int64(1), "24d", "10.209.18.164", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
			expectedOutput: "testpod\\ +1/1\\ +Running\\ +1\\ +24d\\ +testnode123\n",
		},
		{
			name:         "--sorty-by .spec.nodeName + --extra-columns Node:.spec.nodeName prints default columns + node name column sorted by nodeName",
			extraColumns: []string{"Node:.spec.nodeName"},
			noHeaders:    true,
			sortBy:       ".spec.nodeName",
			input: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{
						Cells:  []interface{}{"testpod2", "1/1", "Running", int64(1), "12d", "10.209.18.165", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod2},
					},
					{
						Cells:  []interface{}{"testpod1", "1/1", "Running", int64(1), "24d", "10.209.18.164", "<none>", "<none>", "<none>"},
						Object: runtime.RawExtension{Object: pod1},
					},
				},
			},
			expectedOutput: "testpod2\\ +1/1\\ +Running\\ +1\\ +12d\\ +atestnode456\ntestpod1\\ +1/1\\ +Running\\ +1\\ +24d\\ +testnode123\n",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := HumanPrintFlags{
				ShowKind:     &tc.showKind,
				ShowLabels:   &tc.showLabels,
				SortBy:       &tc.sortBy,
				ColumnLabels: &tc.columnLabels,
				ExtraColumns: &tc.extraColumns,

				NoHeaders:     tc.noHeaders,
				WithNamespace: tc.withNamespace,
			}

			if tc.showKind {
				printFlags.Kind = schema.GroupKind{Kind: "pod"}
			}

			p, err := printFlags.ToPrinter(tc.outputFormat)
			if tc.expectNoMatch {
				if !genericclioptions.IsNoCompatiblePrinterError(err) {
					t.Fatalf("expected no printer matches for output format %q", tc.outputFormat)
				}
				return
			}
			if genericclioptions.IsNoCompatiblePrinterError(err) {
				t.Fatalf("expected to match template printer for output format %q", tc.outputFormat)
			}

			if len(tc.expectedError) > 0 {
				if err == nil || !strings.Contains(err.Error(), tc.expectedError) {
					t.Errorf("expecting error %q, got %v", tc.expectedError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			out := bytes.NewBuffer([]byte{})
			err = p.PrintObj(tc.input, out)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			match, err := regexp.Match(tc.expectedOutput, out.Bytes())
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !match {
				t.Errorf("unexpected output: expecting %q, got %q", tc.expectedOutput, out.String())
			}
		})
	}
}
