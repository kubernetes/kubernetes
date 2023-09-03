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

package printers

import (
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// GenerateOptions encapsulates attributes for table generation.
type GenerateOptions struct {
	NoHeaders bool
	Wide      bool
}

// TableGenerator - an interface for generating metav1.Table provided a runtime.Object
type TableGenerator interface {
	GenerateTable(obj runtime.Object, options GenerateOptions) (*metav1.Table, error)
}

// PrintHandler - interface to handle printing provided an array of metav1.TableColumnDefinition
type PrintHandler interface {
	TableHandler(columns []metav1.TableColumnDefinition, printFunc interface{}) error
}

type handlerEntry struct {
	columnDefinitions []metav1.TableColumnDefinition
	printFunc         reflect.Value
}

// HumanReadableGenerator is an implementation of TableGenerator used to generate
// a table for a specific resource. The table is printed with a TablePrinter using
// PrintObj().
type HumanReadableGenerator struct {
	handlerMap map[reflect.Type]*handlerEntry
}

var _ TableGenerator = &HumanReadableGenerator{}
var _ PrintHandler = &HumanReadableGenerator{}

// NewTableGenerator creates a HumanReadableGenerator suitable for calling GenerateTable().
func NewTableGenerator() *HumanReadableGenerator {
	return &HumanReadableGenerator{
		handlerMap: make(map[reflect.Type]*handlerEntry),
	}
}

// With method - accepts a list of builder functions that modify HumanReadableGenerator
func (h *HumanReadableGenerator) With(fns ...func(PrintHandler)) *HumanReadableGenerator {
	for _, fn := range fns {
		fn(h)
	}
	return h
}

// GenerateTable returns a table for the provided object, using the printer registered for that type. It returns
// a table that includes all of the information requested by options, but will not remove rows or columns. The
// caller is responsible for applying rules related to filtering rows or columns.
func (h *HumanReadableGenerator) GenerateTable(obj runtime.Object, options GenerateOptions) (*metav1.Table, error) {
	t := reflect.TypeOf(obj)
	handler, ok := h.handlerMap[t]
	if !ok {
		return nil, fmt.Errorf("no table handler registered for this type %v", t)
	}

	args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(options)}
	results := handler.printFunc.Call(args)
	if !results[1].IsNil() {
		return nil, results[1].Interface().(error)
	}

	var columns []metav1.TableColumnDefinition
	if !options.NoHeaders {
		columns = handler.columnDefinitions
		if !options.Wide {
			columns = make([]metav1.TableColumnDefinition, 0, len(handler.columnDefinitions))
			for i := range handler.columnDefinitions {
				if handler.columnDefinitions[i].Priority != 0 {
					continue
				}
				columns = append(columns, handler.columnDefinitions[i])
			}
		}
	}
	table := &metav1.Table{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "",
		},
		ColumnDefinitions: columns,
		Rows:              results[0].Interface().([]metav1.TableRow),
	}
	if m, err := meta.ListAccessor(obj); err == nil {
		table.ResourceVersion = m.GetResourceVersion()
		table.Continue = m.GetContinue()
		table.RemainingItemCount = m.GetRemainingItemCount()
	} else {
		if m, err := meta.CommonAccessor(obj); err == nil {
			table.ResourceVersion = m.GetResourceVersion()
		}
	}
	return table, nil
}

// TableHandler adds a print handler with a given set of columns to HumanReadableGenerator instance.
// See ValidateRowPrintHandlerFunc for required method signature.
func (h *HumanReadableGenerator) TableHandler(columnDefinitions []metav1.TableColumnDefinition, printFunc interface{}) error {
	printFuncValue := reflect.ValueOf(printFunc)
	if err := ValidateRowPrintHandlerFunc(printFuncValue); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to register print function: %v", err))
		return err
	}
	entry := &handlerEntry{
		columnDefinitions: columnDefinitions,
		printFunc:         printFuncValue,
	}

	objType := printFuncValue.Type().In(0)
	if _, ok := h.handlerMap[objType]; ok {
		err := fmt.Errorf("registered duplicate printer for %v", objType)
		utilruntime.HandleError(err)
		return err
	}
	h.handlerMap[objType] = entry
	return nil
}

// ValidateRowPrintHandlerFunc validates print handler signature.
// printFunc is the function that will be called to print an object.
// It must be of the following type:
//
//	func printFunc(object ObjectType, options GenerateOptions) ([]metav1.TableRow, error)
//
// where ObjectType is the type of the object that will be printed, and the first
// return value is an array of rows, with each row containing a number of cells that
// match the number of columns defined for that printer function.
func ValidateRowPrintHandlerFunc(printFunc reflect.Value) error {
	if printFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid print handler. %#v is not a function", printFunc)
	}
	funcType := printFunc.Type()
	if funcType.NumIn() != 2 || funcType.NumOut() != 2 {
		return fmt.Errorf("invalid print handler." +
			"Must accept 2 parameters and return 2 value")
	}
	if funcType.In(1) != reflect.TypeOf((*GenerateOptions)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*[]metav1.TableRow)(nil)).Elem() ||
		funcType.Out(1) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("invalid print handler. The expected signature is: "+
			"func handler(obj %v, options GenerateOptions) ([]metav1.TableRow, error)", funcType.In(0))
	}
	return nil
}
