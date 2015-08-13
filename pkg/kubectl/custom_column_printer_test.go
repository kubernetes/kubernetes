/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"bytes"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
)

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
			obj: &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo"}},
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
			obj: &v1.PodList{
				Items: []v1.Pod{
					{ObjectMeta: v1.ObjectMeta{Name: "foo"}},
					{ObjectMeta: v1.ObjectMeta{Name: "bar"}},
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
			obj: &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo"}, TypeMeta: v1.TypeMeta{APIVersion: "baz"}},
			expectedOutput: `NAME      API_VERSION
foo       baz
`,
		},
	}

	for _, test := range tests {
		printer := &CustomColumnsPrinter{
			Columns: test.columns,
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
