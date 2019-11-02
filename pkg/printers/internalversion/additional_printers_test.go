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

package internalversion

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	genericprinters "k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/printers"
)

///////////////////////////////////////////////////////////////
//
// These tests do not belong in this package, and they
// should be moved (mostly to cli-runtime). seans3.
//
/////////////////////////////////////////////////////////////

type TestPrintType struct {
	Data string
}

func (obj *TestPrintType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *TestPrintType) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	clone := *obj
	return &clone
}

func PrintCustomType(obj *TestPrintType, options printers.GenerateOptions) ([]metav1beta1.TableRow, error) {
	return []metav1beta1.TableRow{{Cells: []interface{}{obj.Data}}}, nil
}

func ErrorPrintHandler(obj *TestPrintType, options printers.GenerateOptions) ([]metav1beta1.TableRow, error) {
	return nil, fmt.Errorf("ErrorPrintHandler error")
}

func TestCustomTypePrinting(t *testing.T) {
	columns := []metav1beta1.TableColumnDefinition{{Name: "Data"}}
	generator := printers.NewTableGenerator()
	generator.TableHandler(columns, PrintCustomType)

	obj := TestPrintType{"test object"}
	table, err := generator.GenerateTable(&obj, printers.GenerateOptions{})
	if err != nil {
		t.Fatalf("An error occurred generating the table for custom type: %#v", err)
	}

	expectedTable := &metav1.Table{
		ColumnDefinitions: []metav1.TableColumnDefinition{{Name: "Data"}},
		Rows:              []metav1.TableRow{{Cells: []interface{}{"test object"}}},
	}
	if !reflect.DeepEqual(expectedTable, table) {
		t.Errorf("Error generating table from custom type. Expected (%#v), got (%#v)", expectedTable, table)
	}
}

func TestPrintHandlerError(t *testing.T) {
	columns := []metav1beta1.TableColumnDefinition{{Name: "Data"}}
	generator := printers.NewTableGenerator()
	generator.TableHandler(columns, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	_, err := generator.GenerateTable(&obj, printers.GenerateOptions{})
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

// TODO(seans3): Move this test to cli-runtime/pkg/printers.
func TestPrinters(t *testing.T) {
	om := func(name string) metav1.ObjectMeta { return metav1.ObjectMeta{Name: name} }

	var (
		err             error
		jsonpathPrinter printers.ResourcePrinter
	)

	jsonpathPrinter, err = genericprinters.NewJSONPathPrinter("{.metadata.name}")
	if err != nil {
		t.Fatal(err)
	}

	genericPrinters := map[string]printers.ResourcePrinter{
		// TODO(juanvallejo): move "generic printer" tests to pkg/kubectl/genericclioptions/printers
		"json":     genericprinters.NewTypeSetter(legacyscheme.Scheme).ToPrinter(&genericprinters.JSONPrinter{}),
		"yaml":     genericprinters.NewTypeSetter(legacyscheme.Scheme).ToPrinter(&genericprinters.YAMLPrinter{}),
		"jsonpath": jsonpathPrinter,
	}
	objects := map[string]runtime.Object{
		"pod":             &v1.Pod{ObjectMeta: om("pod")},
		"emptyPodList":    &v1.PodList{},
		"nonEmptyPodList": &v1.PodList{Items: []v1.Pod{{}}},
		"endpoints": &v1.Endpoints{
			Subsets: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
				Ports:     []v1.EndpointPort{{Port: 8080}},
			}}},
	}
	// map of printer name to set of objects it should fail on.
	expectedErrors := map[string]sets.String{
		"jsonpath": sets.NewString("emptyPodList", "nonEmptyPodList", "endpoints"),
	}

	for pName, p := range genericPrinters {
		for oName, obj := range objects {
			b := &bytes.Buffer{}
			if err := p.PrintObj(obj, b); err != nil {
				if set, found := expectedErrors[pName]; found && set.Has(oName) {
					// expected error
					continue
				}
				t.Errorf("printer '%v', object '%v'; error: '%v'", pName, oName, err)
			}
		}
	}
}
