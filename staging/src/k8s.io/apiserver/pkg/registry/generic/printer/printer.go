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

// Package printer helps adapt existing printer code to TableConvertor
package printer

import (
	"bytes"
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/printers"
)

type handlerEntry struct {
	columns         []string
	columnsWithWide []string
	printFunc       reflect.Value
	args            []reflect.Value
}

type PrinterAdapter struct {
	expect   []reflect.Type
	handlers map[reflect.Type]handlerEntry
}

func NewPrinterAdapter(types ...interface{}) *PrinterAdapter {
	a := &PrinterAdapter{
		handlers: make(map[reflect.Type]handlerEntry),
	}
	for _, t := range types {
		a.expect = append(a.expect, reflect.TypeOf(t))
	}
	return a
}

func (a *PrinterAdapter) With(fns ...func(printers.PrintHandler)) *PrinterAdapter {
	for _, fn := range fns {
		fn(a)
	}
	return a
}

func (a *PrinterAdapter) Handler(columns, columnsWithWide []string, printFunc interface{}) error {
	printFuncValue := reflect.ValueOf(printFunc)
	if err := printers.ValidatePrintHandlerFunc(printFuncValue); err != nil {
		return err
	}
	objType := printFuncValue.Type().In(0)
	for _, t := range a.expect {
		if t != objType {
			continue
		}
		a.handlers[objType] = handlerEntry{
			columns:         columns,
			columnsWithWide: columnsWithWide,
			printFunc:       printFuncValue,
		}
		break
	}
	return nil
}

func (a *PrinterAdapter) ConvertToTableList(ctx genericapirequest.Context, obj runtime.Object, tableOptions runtime.Object) (*metav1alpha1.TableList, error) {
	t := reflect.TypeOf(obj)
	handler, ok := a.handlers[t]
	if !ok {
		return nil, fmt.Errorf("no printer registered for object of type %v", t)
	}
	table := &metav1alpha1.TableList{}
	table.Headers = make([]metav1alpha1.TableListHeader, 0, len(handler.columns)+len(handler.columnsWithWide))
	for _, s := range handler.columns {
		table.Headers = append(table.Headers, metav1alpha1.TableListHeader{
			Name: s,
			Type: "string",
		})
	}
	for _, s := range handler.columnsWithWide {
		table.Headers = append(table.Headers, metav1alpha1.TableListHeader{
			Name: s,
			Type: "string",
		})
	}
	options := printers.PrintOptions{
		NoHeaders: true,
		Wide:      true,
	}
	buf := &bytes.Buffer{}
	args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(buf), reflect.ValueOf(options)}

	if meta.IsListType(obj) {
		// TODO: this uses more memory than it has to, as we refactor printers we should remove the need
		// for this.
		args[0] = reflect.ValueOf(obj)
		resultValue := handler.printFunc.Call(args)[0]
		if !resultValue.IsNil() {
			return nil, resultValue.Interface().(error)
		}
		data := buf.Bytes()
		for len(data) > 0 {
			cells, remainder := tabbedLineToCells(data, len(table.Headers))
			table.Items = append(table.Items, metav1alpha1.TableListItem{
				Cells: cells,
			})
			data = remainder
		}
	} else {
		args[0] = reflect.ValueOf(obj)
		resultValue := handler.printFunc.Call(args)[0]
		if !resultValue.IsNil() {
			return nil, resultValue.Interface().(error)
		}
		data := buf.Bytes()
		cells, _ := tabbedLineToCells(data, len(table.Headers))
		table.Items = append(table.Items, metav1alpha1.TableListItem{
			Cells: cells,
		})
	}
	return table, nil
}

func tabbedLineToCells(data []byte, expected int) ([]interface{}, []byte) {
	var remainder []byte
	max := bytes.Index(data, []byte("\n"))
	if max != -1 {
		remainder = data[max+1:]
		data = data[:max]
	}
	cells := make([]interface{}, expected)
	for i := 0; i < expected; i++ {
		next := bytes.Index(data, []byte("\t"))
		if next == -1 {
			cells[i] = string(data)
			// fill the remainder with empty strings, this indicates a printer bug
			for j := i + 1; j < expected; j++ {
				cells[j] = ""
			}
			break
		}
		cells[i] = string(data[:next])
		data = data[next+1:]
	}
	return cells, remainder
}
