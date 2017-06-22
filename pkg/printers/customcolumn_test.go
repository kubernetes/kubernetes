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

package printers_test

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/printers"
)

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
		output, err := printers.RelaxedJSONPathExpression(test.input)
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if test.expectErr {
			if err == nil {
				t.Error("unexpected non-error")
			}
			continue
		}
		if output != test.expectedOutput {
			t.Errorf("input: %s, expected: %s, saw: %s", test.input, test.expectedOutput, output)
		}
	}
}

func TestNewColumnPrinterFromSpec(t *testing.T) {
	tests := []struct {
		spec            string
		expectedColumns []printers.Column
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
			expectedColumns: []printers.Column{
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
		printer, err := printers.NewCustomColumnsPrinterFromSpec(test.spec, api.Codecs.UniversalDecoder(), test.noHeaders)
		if test.expectErr {
			if err == nil {
				t.Errorf("[%s] unexpected non-error", test.name)
			}
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
			continue
		}
		if test.noHeaders {
			buffer := &bytes.Buffer{}

			printer.PrintObj(&api.Pod{}, buffer)
			if err != nil {
				t.Fatalf("An error occurred printing Pod: %#v", err)
			}

			if contains(strings.Fields(buffer.String()), "API_VERSION") {
				t.Errorf("unexpected header API_VERSION")
			}

		} else if !reflect.DeepEqual(test.expectedColumns, printer.Columns) {
			t.Errorf("[%s]\nexpected:\n%v\nsaw:\n%v\n", test.name, test.expectedColumns, printer.Columns)
		}

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
		expectedColumns []printers.Column
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
			expectedColumns: []printers.Column{
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
			expectedColumns: []printers.Column{
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
		reader := bytes.NewBufferString(test.spec)
		printer, err := printers.NewCustomColumnsPrinterFromTemplate(reader, api.Codecs.UniversalDecoder())
		if test.expectErr {
			if err == nil {
				t.Errorf("[%s] unexpected non-error", test.name)
			}
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
			continue
		}

		if !reflect.DeepEqual(test.expectedColumns, printer.Columns) {
			t.Errorf("[%s]\nexpected:\n%v\nsaw:\n%v\n", test.name, test.expectedColumns, printer.Columns)
		}

	}
}

func TestColumnPrint(t *testing.T) {
	tests := []struct {
		columns        []printers.Column
		obj            runtime.Object
		expectedOutput string
	}{
		{
			columns: []printers.Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
			},
			obj: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			expectedOutput: `NAME
foo
`,
		},
		{
			columns: []printers.Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
			},
			obj: &v1.PodList{
				Items: []v1.Pod{
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
			columns: []printers.Column{
				{
					Header:    "NAME",
					FieldSpec: "{.metadata.name}",
				},
				{
					Header:    "API_VERSION",
					FieldSpec: "{.apiVersion}",
				},
			},
			obj: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, TypeMeta: metav1.TypeMeta{APIVersion: "baz"}},
			expectedOutput: `NAME      API_VERSION
foo       baz
`,
		},
	}

	for _, test := range tests {
		printer := &printers.CustomColumnsPrinter{
			Columns: test.columns,
			Decoder: api.Codecs.UniversalDecoder(),
		}
		buffer := &bytes.Buffer{}
		if err := printer.PrintObj(test.obj, buffer); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if buffer.String() != test.expectedOutput {
			t.Errorf("\nexpected:\n'%s'\nsaw\n'%s'\n", test.expectedOutput, buffer.String())
		}
	}
}
