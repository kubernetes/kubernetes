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
	"io"
	"reflect"
	"strings"
	"text/tabwriter"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var withNamespacePrefixColumns = []string{"NAMESPACE"} // TODO(erictune): print cluster name too.

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
	encoder      runtime.Encoder
	decoder      runtime.Decoder
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
// If encoder and decoder are provided, an attempt to convert unstructured types to internal types is made.
func NewHumanReadablePrinter(encoder runtime.Encoder, decoder runtime.Decoder, options PrintOptions) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		handlerMap: make(map[reflect.Type]*handlerEntry),
		options:    options,
		encoder:    encoder,
		decoder:    decoder,
	}
	return printer
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

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	// if output is a tabwriter (when it's called by kubectl get), we use it; create a new tabwriter otherwise
	w, found := output.(*tabwriter.Writer)
	if !found {
		w = GetNewTabWriter(output)
		defer w.Flush()
	}

	// check if the object is unstructured.  If so, let's attempt to convert it to a type we can understand before
	// trying to print, since the printers are keyed by type.  This is extremely expensive.
	if h.encoder != nil && h.decoder != nil {
		obj, _ = decodeUnknownObject(obj, h.encoder, h.decoder)
	}

	t := reflect.TypeOf(obj)
	if handler := h.handlerMap[t]; handler != nil {
		if !h.options.NoHeaders && t != h.lastType {
			headers := handler.columns
			if h.options.Wide {
				headers = append(headers, handler.columnsWithWide...)
			}
			headers = append(headers, formatLabelHeaders(h.options.ColumnLabels)...)
			// LABELS is always the last column.
			headers = append(headers, formatShowLabelsHeader(h.options.ShowLabels, t)...)
			if h.options.WithNamespace {
				headers = append(withNamespacePrefixColumns, headers...)
			}
			h.printHeader(headers, w)
			h.lastType = t
		}
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(w), reflect.ValueOf(h.options)}
		resultValue := handler.printFunc.Call(args)[0]
		if resultValue.IsNil() {
			return nil
		}
		return resultValue.Interface().(error)
	}

	if _, err := meta.Accessor(obj); err == nil {
		if !h.options.NoHeaders && t != h.lastType {
			headers := []string{"NAME", "KIND"}
			headers = append(headers, formatLabelHeaders(h.options.ColumnLabels)...)
			// LABELS is always the last column.
			headers = append(headers, formatShowLabelsHeader(h.options.ShowLabels, t)...)
			if h.options.WithNamespace {
				headers = append(withNamespacePrefixColumns, headers...)
			}
			h.printHeader(headers, w)
			h.lastType = t
		}

		// we don't recognize this type, but we can still attempt to print some reasonable information about.
		unstructured, ok := obj.(runtime.Unstructured)
		if !ok {
			return fmt.Errorf("error: unknown type %#v", obj)
		}
		// if the error isn't nil, report the "I don't recognize this" error
		if err := printUnstructured(unstructured, w, h.options); err != nil {
			return err
		}
		return nil
	}

	// we failed all reasonable printing efforts, report failure
	return fmt.Errorf("error: unknown type %#v", obj)
}

// TODO: this method assumes the meta/v1 server API, so should be refactored out of this package
func printUnstructured(unstructured runtime.Unstructured, w io.Writer, options PrintOptions) error {
	metadata, err := meta.Accessor(unstructured)
	if err != nil {
		return err
	}

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", metadata.GetNamespace()); err != nil {
			return err
		}
	}

	content := unstructured.UnstructuredContent()
	kind := "<missing>"
	if objKind, ok := content["kind"]; ok {
		if str, ok := objKind.(string); ok {
			kind = str
		}
	}
	if objAPIVersion, ok := content["apiVersion"]; ok {
		if str, ok := objAPIVersion.(string); ok {
			version, err := schema.ParseGroupVersion(str)
			if err != nil {
				return err
			}
			kind = kind + "." + version.Version + "." + version.Group
		}
	}
	name := formatResourceName(options.Kind, metadata.GetName(), options.WithKind)

	if _, err := fmt.Fprintf(w, "%s\t%s", name, kind); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, appendLabels(metadata.GetLabels(), options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, appendAllLabels(options.ShowLabels, metadata.GetLabels())); err != nil {
		return err
	}

	return nil
}

func formatLabelHeaders(columnLabels []string) []string {
	formHead := make([]string, len(columnLabels))
	for i, l := range columnLabels {
		p := strings.Split(l, "/")
		formHead[i] = strings.ToUpper((p[len(p)-1]))
	}
	return formHead
}

// headers for --show-labels=true
func formatShowLabelsHeader(showLabels bool, t reflect.Type) []string {
	if showLabels {
		// TODO: this is all sorts of hack, fix
		if t.String() != "*api.ThirdPartyResource" && t.String() != "*api.ThirdPartyResourceList" {
			return []string{"LABELS"}
		}
	}
	return nil
}

// formatResourceName receives a resource kind, name, and boolean specifying
// whether or not to update the current name to "kind/name"
// TODO: dedup this with printers/internalversions
func formatResourceName(kind, name string, withKind bool) string {
	if !withKind || kind == "" {
		return name
	}

	return kind + "/" + name
}

// TODO: dedup this with printers/internalversions
func appendLabels(itemLabels map[string]string, columnLabels []string) string {
	var buffer bytes.Buffer

	for _, cl := range columnLabels {
		buffer.WriteString(fmt.Sprint("\t"))
		if il, ok := itemLabels[cl]; ok {
			buffer.WriteString(fmt.Sprint(il))
		} else {
			buffer.WriteString("<none>")
		}
	}

	return buffer.String()
}

// Append all labels to a single column. We need this even when show-labels flag* is
// false, since this adds newline delimiter to the end of each row.
// TODO: dedup this with printers/internalversions
func appendAllLabels(showLabels bool, itemLabels map[string]string) string {
	var buffer bytes.Buffer

	if showLabels {
		buffer.WriteString(fmt.Sprint("\t"))
		buffer.WriteString(labels.FormatLabels(itemLabels))
	}
	buffer.WriteString("\n")

	return buffer.String()
}

// check if the object is unstructured. If so, attempt to convert it to a type we can understand.
func decodeUnknownObject(obj runtime.Object, encoder runtime.Encoder, decoder runtime.Decoder) (runtime.Object, error) {
	var err error
	switch obj.(type) {
	case runtime.Unstructured, *runtime.Unknown:
		if objBytes, err := runtime.Encode(encoder, obj); err == nil {
			if decodedObj, err := runtime.Decode(decoder, objBytes); err == nil {
				obj = decodedObj
			}
		}
	}

	return obj, err
}
