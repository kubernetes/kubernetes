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
	"fmt"
	"regexp"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
)

func TestHumanReadablePrinterSupportsExpectedOptions(t *testing.T) {
	testTable := &metav1.Table{
		ColumnDefinitions: []metav1.TableColumnDefinition{
			{Name: "Name", Type: "string", Format: "name"},
			{Name: "Ready", Type: "string", Format: ""},
			{Name: "Status", Type: "string", Format: ""},
			{Name: "Restarts", Type: "integer", Format: ""},
			{Name: "Age", Type: "string", Format: ""},
			{Name: "IP", Type: "string", Format: "", Priority: 1},
			{Name: "Node", Type: "string", Format: "", Priority: 1},
			{Name: "Nominated Node", Type: "string", Format: "", Priority: 1},
			{Name: "Readiness Gates", Type: "string", Format: "", Priority: 1},
		},
		Rows: []metav1.TableRow{{
			Object: runtime.RawExtension{
				Object: &corev1.Pod{
					TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
					ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foons", Labels: map[string]string{"l1": "value"}},
				},
			},
			Cells: []interface{}{"foo", "0/0", "", int64(0), "<unknown>", "<none>", "<none>", "<none>", "<none>"},
		}},
	}
	testPod := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "foons",
			Labels: map[string]string{
				"l1": "value",
			},
		},
	}

	testCases := []struct {
		name       string
		testObject runtime.Object
		showKind   bool
		showLabels bool

		// TODO(juanvallejo): test sorting once it's moved to the HumanReadablePrinter
		sortBy       string
		columnLabels []string

		noHeaders     bool
		withNamespace bool

		outputFormat string

		expectedError  string
		expectedOutput string
		expectNoMatch  bool
	}{
		{
			name:           "empty output format matches a humanreadable printer",
			testObject:     testPod.DeepCopy(),
			expectedOutput: "NAME\\ +AGE\nfoo\\ +<unknown>\n",
		},
		{
			name:           "empty output format matches a humanreadable printer",
			testObject:     testTable.DeepCopy(),
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\nfoo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:           "\"wide\" output format prints",
			testObject:     testPod.DeepCopy(),
			outputFormat:   "wide",
			expectedOutput: "NAME\\ +AGE\nfoo\\ +<unknown>\n",
		},
		{
			name:           "\"wide\" output format prints",
			testObject:     testTable.DeepCopy(),
			outputFormat:   "wide",
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +IP\\ +NODE\\ +NOMINATED NODE\\ +READINESS GATES\nfoo\\ +0/0\\ +0\\ +<unknown>\\ +<none>\\ +<none>\\ +<none>\\ +<none>\n",
		},
		{
			name:           "no-headers prints output with no headers",
			testObject:     testPod.DeepCopy(),
			noHeaders:      true,
			expectedOutput: "foo\\ +<unknown>\n",
		},
		{
			name:           "no-headers prints output with no headers",
			testObject:     testTable.DeepCopy(),
			noHeaders:      true,
			expectedOutput: "foo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:           "no-headers and a \"wide\" output format prints output with no headers and additional columns",
			testObject:     testPod.DeepCopy(),
			outputFormat:   "wide",
			noHeaders:      true,
			expectedOutput: "foo\\ +<unknown>\n",
		},
		{
			name:           "no-headers and a \"wide\" output format prints output with no headers and additional columns",
			testObject:     testTable.DeepCopy(),
			outputFormat:   "wide",
			noHeaders:      true,
			expectedOutput: "foo\\ +0/0\\ +0\\ +<unknown>\\ +<none>\\ +<none>\\ +<none>\\ +<none>\n",
		},
		{
			name:           "show-kind displays the resource's kind, even when printing a single type of resource",
			testObject:     testPod.DeepCopy(),
			showKind:       true,
			expectedOutput: "NAME\\ +AGE\npod/foo\\ +<unknown>\n",
		},
		{
			name:           "show-kind displays the resource's kind, even when printing a single type of resource",
			testObject:     testTable.DeepCopy(),
			showKind:       true,
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\npod/foo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:           "label-columns prints specified label values in new column",
			testObject:     testPod.DeepCopy(),
			columnLabels:   []string{"l1"},
			expectedOutput: "NAME\\ +AGE\\ +L1\nfoo\\ +<unknown>\\ +value\n",
		},
		{
			name:           "label-columns prints specified label values in new column",
			testObject:     testTable.DeepCopy(),
			columnLabels:   []string{"l1"},
			expectedOutput: "NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\\ +L1\nfoo\\ +0/0\\ +0\\ +<unknown>\\ +value\n",
		},
		{
			name:           "withNamespace displays an additional NAMESPACE column",
			testObject:     testPod.DeepCopy(),
			withNamespace:  true,
			expectedOutput: "NAMESPACE\\ +NAME\\ +AGE\nfoons\\ +foo\\ +<unknown>\n",
		},
		{
			name:           "withNamespace displays an additional NAMESPACE column",
			testObject:     testTable.DeepCopy(),
			withNamespace:  true,
			expectedOutput: "NAMESPACE\\ +NAME\\ +READY\\ +STATUS\\ +RESTARTS\\ +AGE\nfoons\\ +foo\\ +0/0\\ +0\\ +<unknown>\n",
		},
		{
			name:          "no printer is matched on an invalid outputFormat",
			testObject:    testPod.DeepCopy(),
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
		{
			name:          "printer should not match on any other format supported by another printer",
			testObject:    testPod.DeepCopy(),
			outputFormat:  "go-template",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s %T", tc.name, tc.testObject), func(t *testing.T) {
			printFlags := HumanPrintFlags{
				ShowKind:     &tc.showKind,
				ShowLabels:   &tc.showLabels,
				SortBy:       &tc.sortBy,
				ColumnLabels: &tc.columnLabels,

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
			err = p.PrintObj(tc.testObject, out)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			match, err := regexp.Match(tc.expectedOutput, out.Bytes())
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !match {
				t.Errorf("unexpected output: expecting\n%s\ngot\n%s", tc.expectedOutput, out.String())
			}
		})
	}
}
