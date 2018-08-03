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

package cmd

import (
	"bytes"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	genericprinters "k8s.io/kubernetes/pkg/kubectl/genericclioptions/printers"
	"k8s.io/kubernetes/pkg/printers"
)

func TestIllegalPackageSourceCheckerThroughPrintFlags(t *testing.T) {
	testCases := []struct {
		name                 string
		expectInternalObjErr bool
		output               string
		obj                  runtime.Object
		expectedOutput       string
	}{
		{
			name:                 "success printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			obj:                  internalPod(),
		},
		{
			name:                 "success printer: object containing package path with no forbidden prefix returns no error",
			expectInternalObjErr: false,
			obj:                  externalPod(),
			output:               "",
			expectedOutput:       "pod/foo succeeded\n",
		},
		{
			name:                 "name printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			output:               "name",
			obj:                  internalPod(),
		},
		{
			name:                 "json printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			output:               "json",
			obj:                  internalPod(),
		},
		{
			name:                 "json printer: object containing package path with no forbidden prefix returns no error",
			expectInternalObjErr: false,
			obj:                  externalPod(),
			output:               "json",
		},
		{
			name:                 "yaml printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			output:               "yaml",
			obj:                  internalPod(),
		},
		{
			name:                 "yaml printer: object containing package path with no forbidden prefix returns no error",
			expectInternalObjErr: false,
			obj:                  externalPod(),
			output:               "yaml",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := genericclioptions.NewPrintFlags("succeeded").WithTypeSetter(legacyscheme.Scheme)
			printFlags.OutputFormat = &tc.output

			printer, err := printFlags.ToPrinter()
			if err != nil {
				t.Fatalf("unexpected error %v", err)
			}

			output := bytes.NewBuffer([]byte{})

			err = printer.PrintObj(tc.obj, output)
			if err != nil {
				if !tc.expectInternalObjErr {
					t.Fatalf("unexpected error %v", err)
				}

				if !genericprinters.IsInternalObjectError(err) {
					t.Fatalf("unexpected error - expecting internal object printer error, got %q", err)
				}
				return
			}

			if tc.expectInternalObjErr {
				t.Fatalf("expected internal object printer error, but got no error")
			}

			if len(tc.expectedOutput) == 0 {
				return
			}

			if tc.expectedOutput != output.String() {
				t.Fatalf("unexpected output: expecting %q, got %q", tc.expectedOutput, output.String())
			}
		})
	}
}

func TestIllegalPackageSourceCheckerDirectlyThroughPrinters(t *testing.T) {
	jsonPathPrinter, err := genericprinters.NewJSONPathPrinter("{ .metadata.name }")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	goTemplatePrinter, err := genericprinters.NewGoTemplatePrinter([]byte("{{ .metadata.name }}"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	customColumns, err := printers.NewCustomColumnsPrinterFromSpec("NAME:.metadata.name", legacyscheme.Codecs.UniversalDecoder(), true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	testCases := []struct {
		name                 string
		expectInternalObjErr bool
		printer              genericprinters.ResourcePrinter
		obj                  runtime.Object
		expectedOutput       string
	}{
		{
			name:                 "json printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			printer:              &genericprinters.JSONPrinter{},
			obj:                  internalPod(),
		},
		{
			name:                 "yaml printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			printer:              &genericprinters.YAMLPrinter{},
			obj:                  internalPod(),
		},
		{
			name:                 "jsonpath printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			printer:              jsonPathPrinter,
			obj:                  internalPod(),
		},
		{
			name:                 "go-template printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			printer:              goTemplatePrinter,
			obj:                  internalPod(),
		},
		{
			name:                 "go-template printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			printer:              goTemplatePrinter,
			obj:                  internalPod(),
		},
		{
			name:                 "custom-columns printer: object containing package path beginning with forbidden prefix is rejected",
			expectInternalObjErr: true,
			printer:              customColumns,
			obj:                  internalPod(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			output := bytes.NewBuffer([]byte{})

			err := tc.printer.PrintObj(tc.obj, output)
			if err != nil {
				if !tc.expectInternalObjErr {
					t.Fatalf("unexpected error %v", err)
				}

				if !genericprinters.IsInternalObjectError(err) {
					t.Fatalf("unexpected error - expecting internal object printer error, got %q", err)
				}
				return
			}

			if tc.expectInternalObjErr {
				t.Fatalf("expected internal object printer error, but got no error")
			}

			if len(tc.expectedOutput) == 0 {
				return
			}

			if tc.expectedOutput != output.String() {
				t.Fatalf("unexpected output: expecting %q, got %q", tc.expectedOutput, output.String())
			}
		})
	}
}

func internalPod() *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "bar",
				},
			},
		},
		Status: api.PodStatus{
			Phase: api.PodRunning,
		},
	}
}

func externalPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}
}
