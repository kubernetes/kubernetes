/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var testNamespaceColumnDefinitions = []metav1beta1.TableColumnDefinition{
	{Name: "Name", Type: "string", Format: "name", Description: metav1.ObjectMeta{}.SwaggerDoc()["name"]},
	{Name: "Status", Type: "string", Description: "The status of the namespace"},
	{Name: "Age", Type: "string", Description: metav1.ObjectMeta{}.SwaggerDoc()["creationTimestamp"]},
}

func testPrintNamespace(obj *api.Namespace, options PrintOptions) ([]metav1beta1.TableRow, error) {
	if options.WithNamespace {
		return nil, fmt.Errorf("namespace is not namespaced")
	}
	row := metav1beta1.TableRow{
		Object: runtime.RawExtension{Object: obj},
	}
	row.Cells = append(row.Cells, obj.Name, obj.Status.Phase, "<unknow>")
	return []metav1beta1.TableRow{row}, nil
}

func TestPrintRowsForHandlerEntry(t *testing.T) {
	printFunc := reflect.ValueOf(testPrintNamespace)

	testCase := []struct {
		name          string
		h             *handlerEntry
		opt           PrintOptions
		obj           runtime.Object
		includeHeader bool
		expectOut     string
		expectErr     string
	}{
		{
			name: "no tablecolumndefinition and includeheader flase",
			h: &handlerEntry{
				columnDefinitions: []metav1beta1.TableColumnDefinition{},
				printRows:         true,
				printFunc:         printFunc,
			},
			opt: PrintOptions{},
			obj: &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			includeHeader: false,
			expectOut:     "test\t\t<unknow>\n",
		},
		{
			name: "no tablecolumndefinition and includeheader true",
			h: &handlerEntry{
				columnDefinitions: []metav1beta1.TableColumnDefinition{},
				printRows:         true,
				printFunc:         printFunc,
			},
			opt: PrintOptions{},
			obj: &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			includeHeader: true,
			expectOut:     "\ntest\t\t<unknow>\n",
		},
		{
			name: "have tablecolumndefinition and includeheader true",
			h: &handlerEntry{
				columnDefinitions: testNamespaceColumnDefinitions,
				printRows:         true,
				printFunc:         printFunc,
			},
			opt: PrintOptions{},
			obj: &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			includeHeader: true,
			expectOut:     "NAME\tSTATUS\tAGE\ntest\t\t<unknow>\n",
		},
		{
			name: "print namespace and withnamespace true, should not print header",
			h: &handlerEntry{
				columnDefinitions: testNamespaceColumnDefinitions,
				printRows:         true,
				printFunc:         printFunc,
			},
			opt: PrintOptions{
				WithNamespace: true,
			},
			obj: &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			includeHeader: true,
			expectOut:     "",
			expectErr:     "namespace is not namespaced",
		},
	}
	for _, test := range testCase {
		t.Run(test.name, func(t *testing.T) {
			buffer := &bytes.Buffer{}
			err := printRowsForHandlerEntry(buffer, test.h, test.obj, test.opt, test.includeHeader)
			if err != nil {
				if err.Error() != test.expectErr {
					t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", test.name, test.expectErr, err)
				}
			}
			if test.expectOut != buffer.String() {
				t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", test.name, test.expectOut, buffer.String())
			}
		})
	}
}
