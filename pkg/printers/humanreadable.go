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
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/fatih/camelcase"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/slice"
)

type TablePrinter interface {
	PrintTable(obj runtime.Object, options PrintOptions) (*metav1alpha1.Table, error)
}

type PrintHandler interface {
	Handler(columns, columnsWithWide []string, printFunc interface{}) error
	TableHandler(columns []metav1alpha1.TableColumnDefinition, printFunc interface{}) error
}

var withNamespacePrefixColumns = []string{"NAMESPACE"} // TODO(erictune): print cluster name too.

type handlerEntry struct {
	columnDefinitions []metav1alpha1.TableColumnDefinition
	printRows         bool
	printFunc         reflect.Value
	args              []reflect.Value
}

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide
// more elegant output. It is not threadsafe, but you may call PrintObj repeatedly; headers
// will only be printed if the object type changes. This makes it useful for printing items
// received from watches.
type HumanReadablePrinter struct {
	handlerMap    map[reflect.Type]*handlerEntry
	options       PrintOptions
	lastType      reflect.Type
	skipTabWriter bool
	encoder       runtime.Encoder
	decoder       runtime.Decoder
}

var _ PrintHandler = &HumanReadablePrinter{}

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

// NewTablePrinter creates a HumanReadablePrinter suitable for calling PrintTable().
func NewTablePrinter() *HumanReadablePrinter {
	return &HumanReadablePrinter{
		handlerMap: make(map[reflect.Type]*handlerEntry),
	}
}

// AddTabWriter sets whether the PrintObj function will format with tabwriter (true
// by default).
func (a *HumanReadablePrinter) AddTabWriter(t bool) *HumanReadablePrinter {
	a.skipTabWriter = !t
	return a
}

func (a *HumanReadablePrinter) With(fns ...func(PrintHandler)) *HumanReadablePrinter {
	for _, fn := range fns {
		fn(a)
	}
	return a
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
// See ValidatePrintHandlerFunc for required method signature.
func (h *HumanReadablePrinter) Handler(columns, columnsWithWide []string, printFunc interface{}) error {
	var columnDefinitions []metav1alpha1.TableColumnDefinition
	for _, column := range columns {
		columnDefinitions = append(columnDefinitions, metav1alpha1.TableColumnDefinition{
			Name: column,
			Type: "string",
		})
	}
	for _, column := range columnsWithWide {
		columnDefinitions = append(columnDefinitions, metav1alpha1.TableColumnDefinition{
			Name:     column,
			Type:     "string",
			Priority: 1,
		})
	}

	printFuncValue := reflect.ValueOf(printFunc)
	if err := ValidatePrintHandlerFunc(printFuncValue); err != nil {
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

// TableHandler adds a print handler with a given set of columns to HumanReadablePrinter instance.
// See ValidateRowPrintHandlerFunc for required method signature.
func (h *HumanReadablePrinter) TableHandler(columnDefinitions []metav1alpha1.TableColumnDefinition, printFunc interface{}) error {
	printFuncValue := reflect.ValueOf(printFunc)
	if err := ValidateRowPrintHandlerFunc(printFuncValue); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to register print function: %v", err))
		return err
	}
	entry := &handlerEntry{
		columnDefinitions: columnDefinitions,
		printRows:         true,
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
//  func printFunc(object ObjectType, options PrintOptions) ([]metav1alpha1.TableRow, error)
// where ObjectType is the type of the object that will be printed, and the first
// return value is an array of rows, with each row containing a number of cells that
// match the number of coulmns defined for that printer function.
func ValidateRowPrintHandlerFunc(printFunc reflect.Value) error {
	if printFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid print handler. %#v is not a function", printFunc)
	}
	funcType := printFunc.Type()
	if funcType.NumIn() != 2 || funcType.NumOut() != 2 {
		return fmt.Errorf("invalid print handler." +
			"Must accept 2 parameters and return 2 value.")
	}
	if funcType.In(1) != reflect.TypeOf((*PrintOptions)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*[]metav1alpha1.TableRow)(nil)).Elem() ||
		funcType.Out(1) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("invalid print handler. The expected signature is: "+
			"func handler(obj %v, options PrintOptions) ([]metav1alpha1.TableRow, error)", funcType.In(0))
	}
	return nil
}

// ValidatePrintHandlerFunc validates print handler signature.
// printFunc is the function that will be called to print an object.
// It must be of the following type:
//  func printFunc(object ObjectType, w io.Writer, options PrintOptions) error
// where ObjectType is the type of the object that will be printed.
// DEPRECATED: will be replaced with ValidateRowPrintHandlerFunc
func ValidatePrintHandlerFunc(printFunc reflect.Value) error {
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

func (h *HumanReadablePrinter) IsGeneric() bool {
	return false
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
// TODO: unify the behavior of PrintObj, which often expects single items and tracks
// headers and filtering, with other printers, that expect list objects. The tracking
// behavior should probably be a higher level wrapper (MultiObjectTablePrinter) that
// calls into the PrintTable method and then displays consistent output.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	if w, found := output.(*tabwriter.Writer); !found && !h.skipTabWriter {
		w = GetNewTabWriter(output)
		output = w
		defer w.Flush()
	}

	// display tables following the rules of options
	if table, ok := obj.(*metav1alpha1.Table); ok {
		if err := DecorateTable(table, h.options); err != nil {
			return err
		}
		return PrintTable(table, output, h.options)
	}

	// check if the object is unstructured.  If so, let's attempt to convert it to a type we can understand before
	// trying to print, since the printers are keyed by type.  This is extremely expensive.
	if h.encoder != nil && h.decoder != nil {
		obj, _ = decodeUnknownObject(obj, h.encoder, h.decoder)
	}

	t := reflect.TypeOf(obj)
	if handler := h.handlerMap[t]; handler != nil {
		if !h.options.NoHeaders && t != h.lastType {
			var headers []string
			for _, column := range handler.columnDefinitions {
				if column.Priority != 0 && !h.options.Wide {
					continue
				}
				headers = append(headers, strings.ToUpper(column.Name))
			}
			headers = append(headers, formatLabelHeaders(h.options.ColumnLabels)...)
			// LABELS is always the last column.
			headers = append(headers, formatShowLabelsHeader(h.options.ShowLabels, t)...)
			if h.options.WithNamespace {
				headers = append(withNamespacePrefixColumns, headers...)
			}
			h.printHeader(headers, output)
			h.lastType = t
		}

		if handler.printRows {
			args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(h.options)}
			results := handler.printFunc.Call(args)
			if results[1].IsNil() {
				rows := results[0].Interface().([]metav1alpha1.TableRow)
				for _, row := range rows {

					if h.options.WithNamespace {
						if obj := row.Object.Object; obj != nil {
							if m, err := meta.Accessor(obj); err == nil {
								fmt.Fprint(output, m.GetNamespace())
							}
						}
						fmt.Fprint(output, "\t")
					}

					for i, cell := range row.Cells {
						if i != 0 {
							fmt.Fprint(output, "\t")
						} else {
							// TODO: remove this once we drop the legacy printers
							if h.options.WithKind && len(h.options.Kind) > 0 {
								fmt.Fprintf(output, "%s/%s", h.options.Kind, cell)
								continue
							}
						}
						fmt.Fprint(output, cell)
					}

					hasLabels := len(h.options.ColumnLabels) > 0
					if obj := row.Object.Object; obj != nil && (hasLabels || h.options.ShowLabels) {
						if m, err := meta.Accessor(obj); err == nil {
							for _, value := range labelValues(m.GetLabels(), h.options) {
								output.Write([]byte("\t"))
								output.Write([]byte(value))
							}
						}
					}

					output.Write([]byte("\n"))
				}
				return nil
			}
			return results[1].Interface().(error)
		}

		// TODO: this code path is deprecated and will be removed when all handlers are row printers
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(output), reflect.ValueOf(h.options)}
		resultValue := handler.printFunc.Call(args)[0]
		if resultValue.IsNil() {
			return nil
		}
		return resultValue.Interface().(error)
	}

	if _, err := meta.Accessor(obj); err == nil {
		// we don't recognize this type, but we can still attempt to print some reasonable information about.
		unstructured, ok := obj.(runtime.Unstructured)
		if !ok {
			return fmt.Errorf("error: unknown type %T, expected unstructured in %#v", obj, h.handlerMap)
		}

		content := unstructured.UnstructuredContent()

		// we'll elect a few more fields to print depending on how much columns are already taken
		maxDiscoveredFieldsToPrint := 3
		maxDiscoveredFieldsToPrint = maxDiscoveredFieldsToPrint - len(h.options.ColumnLabels)
		if h.options.WithNamespace { // where's my ternary
			maxDiscoveredFieldsToPrint--
		}
		if h.options.ShowLabels {
			maxDiscoveredFieldsToPrint--
		}
		if maxDiscoveredFieldsToPrint < 0 {
			maxDiscoveredFieldsToPrint = 0
		}

		var discoveredFieldNames []string                    // we want it predictable so this will be used to sort
		ignoreIfDiscovered := []string{"kind", "apiVersion"} // these are already covered
		for field, value := range content {
			if slice.ContainsString(ignoreIfDiscovered, field, nil) {
				continue
			}
			switch value.(type) {
			case map[string]interface{}:
				// just simpler types
				continue
			}
			discoveredFieldNames = append(discoveredFieldNames, field)
		}
		sort.Strings(discoveredFieldNames)
		if len(discoveredFieldNames) > maxDiscoveredFieldsToPrint {
			discoveredFieldNames = discoveredFieldNames[:maxDiscoveredFieldsToPrint]
		}

		if !h.options.NoHeaders && t != h.lastType {
			headers := []string{"NAME", "KIND"}
			for _, discoveredField := range discoveredFieldNames {
				fieldAsHeader := strings.ToUpper(strings.Join(camelcase.Split(discoveredField), " "))
				headers = append(headers, fieldAsHeader)
			}
			headers = append(headers, formatLabelHeaders(h.options.ColumnLabels)...)
			// LABELS is always the last column.
			headers = append(headers, formatShowLabelsHeader(h.options.ShowLabels, t)...)
			if h.options.WithNamespace {
				headers = append(withNamespacePrefixColumns, headers...)
			}
			h.printHeader(headers, output)
			h.lastType = t
		}

		// if the error isn't nil, report the "I don't recognize this" error
		if err := printUnstructured(unstructured, output, discoveredFieldNames, h.options); err != nil {
			return err
		}
		return nil
	}

	// we failed all reasonable printing efforts, report failure
	return fmt.Errorf("error: unknown type %#v", obj)
}

func hasCondition(conditions []metav1alpha1.TableRowCondition, t metav1alpha1.RowConditionType) bool {
	for _, condition := range conditions {
		if condition.Type == t {
			return condition.Status == metav1alpha1.ConditionTrue
		}
	}
	return false
}

// PrintTable prints a table to the provided output respecting the filtering rules for options
// for wide columns and filetred rows. It filters out rows that are Completed. You should call
// DecorateTable if you receive a table from a remote server before calling PrintTable.
func PrintTable(table *metav1alpha1.Table, output io.Writer, options PrintOptions) error {
	if !options.NoHeaders {
		first := true
		for _, column := range table.ColumnDefinitions {
			if !options.Wide && column.Priority != 0 {
				continue
			}
			if first {
				first = false
			} else {
				fmt.Fprint(output, "\t")
			}
			fmt.Fprint(output, strings.ToUpper(column.Name))
		}
		fmt.Fprintln(output)
	}
	for _, row := range table.Rows {
		if !options.ShowAll && hasCondition(row.Conditions, metav1alpha1.RowCompleted) {
			continue
		}
		first := true
		for i, cell := range row.Cells {
			column := table.ColumnDefinitions[i]
			if !options.Wide && column.Priority != 0 {
				continue
			}
			if first {
				first = false
			} else {
				fmt.Fprint(output, "\t")
			}
			if cell != nil {
				fmt.Fprint(output, cell)
			}
		}
		fmt.Fprintln(output)
	}
	return nil
}

// DecorateTable takes a table and attempts to add label columns and the
// namespace column. It will fill empty columns with nil (if the object
// does not expose metadata). It returns an error if the table cannot
// be decorated.
func DecorateTable(table *metav1alpha1.Table, options PrintOptions) error {
	width := len(table.ColumnDefinitions) + len(options.ColumnLabels)
	if options.WithNamespace {
		width++
	}
	if options.ShowLabels {
		width++
	}

	columns := table.ColumnDefinitions

	nameColumn := -1
	if options.WithKind && len(options.Kind) > 0 {
		for i := range columns {
			if columns[i].Format == "name" && columns[i].Type == "string" {
				nameColumn = i
				fmt.Printf("found name column: %d\n", i)
				break
			}
		}
	}

	if width != len(table.ColumnDefinitions) {
		columns = make([]metav1alpha1.TableColumnDefinition, 0, width)
		if options.WithNamespace {
			columns = append(columns, metav1alpha1.TableColumnDefinition{
				Name: "Namespace",
				Type: "string",
			})
		}
		columns = append(columns, table.ColumnDefinitions...)
		for _, label := range formatLabelHeaders(options.ColumnLabels) {
			columns = append(columns, metav1alpha1.TableColumnDefinition{
				Name: label,
				Type: "string",
			})
		}
		if options.ShowLabels {
			columns = append(columns, metav1alpha1.TableColumnDefinition{
				Name: "Labels",
				Type: "string",
			})
		}
	}

	rows := table.Rows

	includeLabels := len(options.ColumnLabels) > 0 || options.ShowLabels
	if includeLabels || options.WithNamespace || nameColumn != -1 {
		for i := range rows {
			row := rows[i]

			if nameColumn != -1 {
				row.Cells[nameColumn] = fmt.Sprintf("%s/%s", options.Kind, row.Cells[nameColumn])
			}

			var m metav1.Object
			if obj := row.Object.Object; obj != nil {
				if acc, err := meta.Accessor(obj); err == nil {
					m = acc
				}
			}
			// if we can't get an accessor, fill out the appropriate columns with empty spaces
			if m == nil {
				if options.WithNamespace {
					r := make([]interface{}, 1, width)
					row.Cells = append(r, row.Cells...)
				}
				for j := 0; j < width-len(row.Cells); j++ {
					row.Cells = append(row.Cells, nil)
				}
				rows[i] = row
				continue
			}

			if options.WithNamespace {
				r := make([]interface{}, 1, width)
				r[0] = m.GetNamespace()
				row.Cells = append(r, row.Cells...)
			}
			if includeLabels {
				row.Cells = appendLabelCells(row.Cells, m.GetLabels(), options)
			}
			rows[i] = row
		}
	}

	table.ColumnDefinitions = columns
	table.Rows = rows
	return nil
}

// PrintTable returns a table for the provided object, using the printer registered for that type. It returns
// a table that includes all of the information requested by options, but will not remove rows or columns. The
// caller is responsible for applying rules related to filtering rows or columns.
func (h *HumanReadablePrinter) PrintTable(obj runtime.Object, options PrintOptions) (*metav1alpha1.Table, error) {
	t := reflect.TypeOf(obj)
	handler, ok := h.handlerMap[t]
	if !ok {
		return nil, fmt.Errorf("no table handler registered for this type %v", t)
	}
	if !handler.printRows {
		return h.legacyPrinterToTable(obj, handler)
	}

	args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(options)}
	results := handler.printFunc.Call(args)
	if !results[1].IsNil() {
		return nil, results[1].Interface().(error)
	}

	columns := handler.columnDefinitions
	if !options.Wide {
		columns = make([]metav1alpha1.TableColumnDefinition, 0, len(handler.columnDefinitions))
		for i := range handler.columnDefinitions {
			if handler.columnDefinitions[i].Priority != 0 {
				continue
			}
			columns = append(columns, handler.columnDefinitions[i])
		}
	}
	table := &metav1alpha1.Table{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "",
		},
		ColumnDefinitions: columns,
		Rows:              results[0].Interface().([]metav1alpha1.TableRow),
	}
	if err := DecorateTable(table, options); err != nil {
		return nil, err
	}
	return table, nil
}

// legacyPrinterToTable uses the old printFunc with tabbed writer to generate a table.
// TODO: remove when all legacy printers are removed.
func (h *HumanReadablePrinter) legacyPrinterToTable(obj runtime.Object, handler *handlerEntry) (*metav1alpha1.Table, error) {
	printFunc := handler.printFunc
	table := &metav1alpha1.Table{
		ColumnDefinitions: handler.columnDefinitions,
	}

	options := PrintOptions{
		NoHeaders: true,
		Wide:      true,
	}
	buf := &bytes.Buffer{}
	args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(buf), reflect.ValueOf(options)}

	if meta.IsListType(obj) {
		// TODO: this uses more memory than it has to, as we refactor printers we should remove the need
		// for this.
		args[0] = reflect.ValueOf(obj)
		resultValue := printFunc.Call(args)[0]
		if !resultValue.IsNil() {
			return nil, resultValue.Interface().(error)
		}
		data := buf.Bytes()
		i := 0
		items, err := meta.ExtractList(obj)
		if err != nil {
			return nil, err
		}
		for len(data) > 0 {
			cells, remainder := tabbedLineToCells(data, len(table.ColumnDefinitions))
			table.Rows = append(table.Rows, metav1alpha1.TableRow{
				Cells:  cells,
				Object: runtime.RawExtension{Object: items[i]},
			})
			data = remainder
			i++
		}
	} else {
		args[0] = reflect.ValueOf(obj)
		resultValue := printFunc.Call(args)[0]
		if !resultValue.IsNil() {
			return nil, resultValue.Interface().(error)
		}
		data := buf.Bytes()
		cells, _ := tabbedLineToCells(data, len(table.ColumnDefinitions))
		table.Rows = append(table.Rows, metav1alpha1.TableRow{
			Cells:  cells,
			Object: runtime.RawExtension{Object: obj},
		})
	}
	return table, nil
}

// TODO: this method assumes the meta/v1 server API, so should be refactored out of this package
func printUnstructured(unstructured runtime.Unstructured, w io.Writer, additionalFields []string, options PrintOptions) error {
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

	name := FormatResourceName(options.Kind, metadata.GetName(), options.WithKind)

	if _, err := fmt.Fprintf(w, "%s\t%s", name, kind); err != nil {
		return err
	}
	for _, field := range additionalFields {
		if value, ok := content[field]; ok {
			var formattedValue string
			switch typedValue := value.(type) {
			case []interface{}:
				formattedValue = fmt.Sprintf("%d item(s)", len(typedValue))
			default:
				formattedValue = fmt.Sprintf("%v", value)
			}
			if _, err := fmt.Fprintf(w, "\t%s", formattedValue); err != nil {
				return err
			}
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(metadata.GetLabels(), options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, metadata.GetLabels())); err != nil {
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

// labelValues returns a slice of value columns matching the requested print options.
func labelValues(itemLabels map[string]string, opts PrintOptions) []string {
	var values []string
	for _, key := range opts.ColumnLabels {
		values = append(values, itemLabels[key])
	}
	if opts.ShowLabels {
		values = append(values, labels.FormatLabels(itemLabels))
	}
	return values
}

// appendLabelCells returns a slice of value columns matching the requested print options.
// Intended for use with tables.
func appendLabelCells(values []interface{}, itemLabels map[string]string, opts PrintOptions) []interface{} {
	for _, key := range opts.ColumnLabels {
		values = append(values, itemLabels[key])
	}
	if opts.ShowLabels {
		values = append(values, labels.FormatLabels(itemLabels))
	}
	return values
}

// FormatResourceName receives a resource kind, name, and boolean specifying
// whether or not to update the current name to "kind/name"
func FormatResourceName(kind, name string, withKind bool) string {
	if !withKind || kind == "" {
		return name
	}

	return kind + "/" + name
}

func AppendLabels(itemLabels map[string]string, columnLabels []string) string {
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
func AppendAllLabels(showLabels bool, itemLabels map[string]string) string {
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
