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
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
)

var testNamespaceColumnDefinitions = []metav1alpha1.TableColumnDefinition{
	{Name: "Name", Type: "string", Format: "name", Description: metav1.ObjectMeta{}.SwaggerDoc()["name"]},
	{Name: "Status", Type: "string", Description: "The status of the namespace"},
	{Name: "Age", Type: "string", Description: metav1.ObjectMeta{}.SwaggerDoc()["creationTimestamp"]},
}

func testPrintNamespace(obj *api.Namespace, options PrintOptions) ([]metav1alpha1.TableRow, error) {
	if options.WithNamespace {
		return nil, fmt.Errorf("namespace is not namespaced")
	}
	row := metav1alpha1.TableRow{
		Object: runtime.RawExtension{Object: obj},
	}
	row.Cells = append(row.Cells, obj.Name, obj.Status.Phase, "<unknow>")
	return []metav1alpha1.TableRow{row}, nil
}

func testPrintNamespaceList(list *api.NamespaceList, options PrintOptions) ([]metav1alpha1.TableRow, error) {
	rows := make([]metav1alpha1.TableRow, 0, len(list.Items))
	for i := range list.Items {
		r, err := testPrintNamespace(&list.Items[i], options)
		if err != nil {
			return nil, err
		}
		rows = append(rows, r...)
	}
	return rows, nil
}
func TestPrintRowsForHandlerEntry(t *testing.T) {
	printFunc := reflect.ValueOf(testPrintNamespace)

	testCase := map[string]struct {
		h             *handlerEntry
		opt           PrintOptions
		obj           runtime.Object
		includeHeader bool
		expectOut     string
		expectErr     string
	}{
		"no tablecolumndefinition and includeheader flase": {
			h: &handlerEntry{
				columnDefinitions: []metav1alpha1.TableColumnDefinition{},
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
		"no tablecolumndefinition and includeheader true": {
			h: &handlerEntry{
				columnDefinitions: []metav1alpha1.TableColumnDefinition{},
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
		"have tablecolumndefinition and includeheader true": {
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
		"print namespace and withnamespace true, should not print header": {
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
	for name, test := range testCase {
		buffer := &bytes.Buffer{}
		err := printRowsForHandlerEntry(buffer, test.h, test.obj, test.opt, test.includeHeader)
		if err != nil {
			if err.Error() != test.expectErr {
				t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", name, test.expectErr, err)
			}
		}
		if test.expectOut != buffer.String() {
			t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", name, test.expectOut, buffer.String())
		}
	}
}
