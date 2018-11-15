/*
Copyright 2014 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/printers"
)

// UniversalDecoder call must specify parameter versions; otherwise it will decode to internal versions.
var decoder = scheme.Codecs.UniversalDecoder(scheme.Scheme.PrioritizedVersionsAllGroups()...)

func TestMassageJSONPath(t *testing.T) {
	tests := []struct {
		input          string
		expectedOutput string
		expectErr      bool
	}{
		{input: "foo.bar", expectedOutput: "{.foo.bar}"},
		{input: "{foo.bar}", expectedOutput: "{.foo.bar}"},
		{input: ".foo.bar", expectedOutput: "{.foo.bar}"},
		{input: "{.foo.bar}", expectedOutput: "{.foo.bar}"},
		{input: "", expectedOutput: ""},
		{input: "{foo.bar", expectErr: true},
		{input: "foo.bar}", expectErr: true},
		{input: "{foo.bar}}", expectErr: true},
		{input: "{{foo.bar}", expectErr: true},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			output, err := RelaxedJSONPathExpression(test.input)
			if err != nil && !test.expectErr {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if test.expectErr {
				if err == nil {
					t.Error("unexpected non-error")
				}
				return
			}
			if output != test.expectedOutput {
				t.Errorf("input: %s, expected: %s, saw: %s", test.input, test.expectedOutput, output)
			}
		})
	}
}

func TestNewColumnPrinterFromSpec(t *testing.T) {
	tests := []struct {
		spec            string
		expectedColumns []Column
		expectErr       bool
		name            string
		noHeaders       bool
	}{
		{
			spec:      "",
			expectErr: true,
			name:      "empty",
		},
		{
			spec:      "invalid",
			expectErr: true,
			name:      "invalid1",
		},
		{
			spec:      "invalid=foobar",
			expectErr: true,
			name:      "invalid2",
		},
		{
			spec:      "invalid,foobar:blah",
			expectErr: true,
			name:      "invalid3",
		},
		{
			spec: "NAME:metadata.name,API_VERSION:apiVersion",
			name: "ok",
			expectedColumns: []Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
				{
					Header:    "API_VERSION",
					FieldSpec: "{.apiVersion}",
				},
			},
		},
		{
			spec:      "API_VERSION:apiVersion",
			name:      "no-headers",
			noHeaders: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			printer, err := NewCustomColumnsPrinterFromSpec(test.spec, decoder, test.noHeaders)
			if test.expectErr {
				if err == nil {
					t.Errorf("[%s] unexpected non-error", test.name)
				}
				return
			}
			if !test.expectErr && err != nil {
				t.Errorf("[%s] unexpected error: %v", test.name, err)
				return
			}
			if test.noHeaders {
				buffer := &bytes.Buffer{}

				printer.PrintObj(&corev1.Pod{}, buffer)
				if err != nil {
					t.Fatalf("An error occurred printing Pod: %#v", err)
				}

				if contains(strings.Fields(buffer.String()), "API_VERSION") {
					t.Errorf("unexpected header API_VERSION")
				}

			} else if !reflect.DeepEqual(test.expectedColumns, printer.Columns) {
				t.Errorf("[%s]\nexpected:\n%v\nsaw:\n%v\n", test.name, test.expectedColumns, printer.Columns)
			}
		})
	}
}

func contains(arr []string, s string) bool {
	for i := range arr {
		if arr[i] == s {
			return true
		}
	}
	return false
}

const exampleTemplateOne = `NAME               API_VERSION
{metadata.name}    {apiVersion}`

const exampleTemplateTwo = `NAME               		API_VERSION
							{metadata.name}    {apiVersion}`

func TestNewColumnPrinterFromTemplate(t *testing.T) {
	tests := []struct {
		spec            string
		expectedColumns []Column
		expectErr       bool
		name            string
	}{
		{
			spec:      "",
			expectErr: true,
			name:      "empty",
		},
		{
			spec:      "invalid",
			expectErr: true,
			name:      "invalid1",
		},
		{
			spec:      "invalid=foobar",
			expectErr: true,
			name:      "invalid2",
		},
		{
			spec:      "invalid,foobar:blah",
			expectErr: true,
			name:      "invalid3",
		},
		{
			spec: exampleTemplateOne,
			name: "ok",
			expectedColumns: []Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
				{
					Header:    "API_VERSION",
					FieldSpec: "{.apiVersion}",
				},
			},
		},
		{
			spec: exampleTemplateTwo,
			name: "ok-2",
			expectedColumns: []Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
				{
					Header:    "API_VERSION",
					FieldSpec: "{.apiVersion}",
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reader := bytes.NewBufferString(test.spec)
			printer, err := NewCustomColumnsPrinterFromTemplate(reader, decoder)
			if test.expectErr {
				if err == nil {
					t.Errorf("[%s] unexpected non-error", test.name)
				}
				return
			}
			if !test.expectErr && err != nil {
				t.Errorf("[%s] unexpected error: %v", test.name, err)
				return
			}

			if !reflect.DeepEqual(test.expectedColumns, printer.Columns) {
				t.Errorf("[%s]\nexpected:\n%v\nsaw:\n%v\n", test.name, test.expectedColumns, printer.Columns)
			}
		})
	}
}

func TestColumnPrint(t *testing.T) {
	tests := []struct {
		columns        []Column
		obj            runtime.Object
		expectedOutput string
	}{
		{
			columns: []Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
			},
			obj: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			expectedOutput: `NAME
foo
`,
		},
		{
			columns: []Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
			},
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
					{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
				},
			},
			expectedOutput: `NAME
foo
bar
`,
		},
		{
			columns: []Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
				{
					Header:    "API_VERSION",
					FieldSpec: "{.apiVersion}",
				},
			},
			obj: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, TypeMeta: metav1.TypeMeta{APIVersion: "baz"}},
			expectedOutput: `NAME   API_VERSION
foo    baz
`,
		},
		{
			columns: []Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
				{
					Header:    "API_VERSION",
					FieldSpec: "{.apiVersion}",
				},
				{
					Header:    "NOT_FOUND",
					FieldSpec: "{.notFound}",
				},
			},
			obj: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, TypeMeta: metav1.TypeMeta{APIVersion: "baz"}},
			expectedOutput: `NAME   API_VERSION   NOT_FOUND
foo    baz           <none>
`,
		},
	}

	for _, test := range tests {
		t.Run(test.expectedOutput, func(t *testing.T) {
			printer := &CustomColumnsPrinter{
				Columns: test.columns,
				Decoder: decoder,
			}
			buffer := &bytes.Buffer{}
			if err := printer.PrintObj(test.obj, buffer); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if buffer.String() != test.expectedOutput {
				t.Errorf("\nexpected:\n'%s'\nsaw\n'%s'\n", test.expectedOutput, buffer.String())
			}
		})
	}
}

// this mimics how resource/get.go calls the customcolumn printer
func TestIndividualPrintObjOnExistingTabWriter(t *testing.T) {
	columns := []Column{
		{
			Header:    "NAME",
			FieldSpec: "{.metadata.name}",
		},
		{
			Header:    "LONG COLUMN NAME", // name is longer than all values of label1
			FieldSpec: "{.metadata.labels.label1}",
		},
		{
			Header:    "LABEL 2",
			FieldSpec: "{.metadata.labels.label2}",
		},
	}
	objects := []*corev1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "foo", Labels: map[string]string{"label1": "foo", "label2": "foo"}}},
		{ObjectMeta: metav1.ObjectMeta{Name: "bar", Labels: map[string]string{"label1": "bar", "label2": "bar"}}},
	}
	expectedOutput := `NAME   LONG COLUMN NAME   LABEL 2
foo    foo                foo
bar    bar                bar
`

	buffer := &bytes.Buffer{}
	tabWriter := printers.GetNewTabWriter(buffer)
	printer := &CustomColumnsPrinter{
		Columns: columns,
		Decoder: decoder,
	}
	for _, obj := range objects {
		if err := printer.PrintObj(obj, tabWriter); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	tabWriter.Flush()
	if buffer.String() != expectedOutput {
		t.Errorf("\nexpected:\n'%s'\nsaw\n'%s'\n", expectedOutput, buffer.String())
	}
}
