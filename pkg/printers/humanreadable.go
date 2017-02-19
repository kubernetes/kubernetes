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
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/golang/glog"
)

type handlerEntry struct {
	columns         []string
	columnsWithWide []string
	printFunc       reflect.Value
	args            []reflect.Value
}

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide
// more elegant output. It is not threadsafe, but you may call PrintObj repeatedly; headers
// will only be printed if the object type changes. This makes it useful for printing items
// received from watches.
type HumanReadablePrinter struct {
	handlerMap   map[reflect.Type]*handlerEntry
	options      PrintOptions
	lastType     reflect.Type
	hiddenObjNum int
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
func NewHumanReadablePrinter(options PrintOptions) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		handlerMap: make(map[reflect.Type]*handlerEntry),
		options:    options,
	}
	printer.addDefaultHandlers()
	return printer
}

// formatResourceName receives a resource kind, name, and boolean specifying
// whether or not to update the current name to "kind/name"
func formatResourceName(kind, name string, withKind bool) string {
	if !withKind || kind == "" {
		return name
	}

	return kind + "/" + name
}

// GetResourceKind returns the type currently set for a resource
func (h *HumanReadablePrinter) GetResourceKind() string {
	return h.options.Kind
}

// EnsurePrintWithKind sets HumanReadablePrinter options "WithKind" to true
// and "Kind" to the string arg it receives, pre-pending this string
// to the "NAME" column in an output of resources.
func (h *HumanReadablePrinter) EnsurePrintWithKind(kind string) {
	h.options.WithKind = true
	h.options.Kind = kind
}

// EnsurePrintHeaders sets the HumanReadablePrinter option "NoHeaders" to false
// and removes the .lastType that was printed, which forces headers to be
// printed in cases where multiple lists of the same resource are printed
// consecutively, but are separated by non-printer related information.
func (h *HumanReadablePrinter) EnsurePrintHeaders() {
	h.options.NoHeaders = false
	h.lastType = nil
}

// Handler adds a print handler with a given set of columns to HumanReadablePrinter instance.
// See validatePrintHandlerFunc for required method signature.
func (h *HumanReadablePrinter) Handler(columns, columnsWithWide []string, printFunc interface{}) error {
	printFuncValue := reflect.ValueOf(printFunc)
	if err := h.validatePrintHandlerFunc(printFuncValue); err != nil {
		glog.Errorf("Unable to add print handler: %v", err)
		return err
	}

	objType := printFuncValue.Type().In(0)
	h.handlerMap[objType] = &handlerEntry{
		columns:         columns,
		columnsWithWide: columnsWithWide,
		printFunc:       printFuncValue,
	}
	return nil
}

// validatePrintHandlerFunc validates print handler signature.
// printFunc is the function that will be called to print an object.
// It must be of the following type:
//  func printFunc(object ObjectType, w io.Writer, options PrintOptions) error
// where ObjectType is the type of the object that will be printed.
func (h *HumanReadablePrinter) validatePrintHandlerFunc(printFunc reflect.Value) error {
	if printFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid print handler. %#v is not a function", printFunc)
	}
	funcType := printFunc.Type()
	if funcType.NumIn() != 3 || funcType.NumOut() != 1 {
		return fmt.Errorf("invalid print handler." +
			"Must accept 3 parameters and return 1 value.")
	}
	if funcType.In(1) != reflect.TypeOf((*io.Writer)(nil)).Elem() ||
		funcType.In(2) != reflect.TypeOf((*PrintOptions)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("invalid print handler. The expected signature is: "+
			"func handler(obj %v, w io.Writer, options PrintOptions) error", funcType.In(0))
	}
	return nil
}

func (h *HumanReadablePrinter) HandledResources() []string {
	keys := make([]string, 0)

	for k := range h.handlerMap {
		// k.String looks like "*api.PodList" and we want just "pod"
		api := strings.Split(k.String(), ".")
		resource := api[len(api)-1]
		if strings.HasSuffix(resource, "List") {
			continue
		}
		resource = strings.ToLower(resource)
		keys = append(keys, resource)
	}
	return keys
}

func (h *HumanReadablePrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (h *HumanReadablePrinter) unknown(data []byte, w io.Writer) error {
	_, err := fmt.Fprintf(w, "Unknown object: %s", string(data))
	return err
}

func (h *HumanReadablePrinter) printHeader(columnNames []string, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%s\n", strings.Join(columnNames, "\t")); err != nil {
		return err
	}
	return nil
}
