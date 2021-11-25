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
	"io"
	"reflect"
	"strings"
	"time"

	"github.com/liggitt/tabwriter"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/duration"
	"k8s.io/apimachinery/pkg/watch"
)

var _ ResourcePrinter = &HumanReadablePrinter{}

type printHandler struct {
	columnDefinitions []metav1.TableColumnDefinition
	printFunc         reflect.Value
}

var (
	statusHandlerEntry = &printHandler{
		columnDefinitions: statusColumnDefinitions,
		printFunc:         reflect.ValueOf(printStatus),
	}

	statusColumnDefinitions = []metav1.TableColumnDefinition{
		{Name: "Status", Type: "string"},
		{Name: "Reason", Type: "string"},
		{Name: "Message", Type: "string"},
	}

	defaultHandlerEntry = &printHandler{
		columnDefinitions: objectMetaColumnDefinitions,
		printFunc:         reflect.ValueOf(printObjectMeta),
	}

	objectMetaColumnDefinitions = []metav1.TableColumnDefinition{
		{Name: "Name", Type: "string", Format: "name", Description: metav1.ObjectMeta{}.SwaggerDoc()["name"]},
		{Name: "Age", Type: "string", Description: metav1.ObjectMeta{}.SwaggerDoc()["creationTimestamp"]},
	}

	withEventTypePrefixColumns = []string{"EVENT"}
	withNamespacePrefixColumns = []string{"NAMESPACE"} // TODO(erictune): print cluster name too.
)

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide
// more elegant output. It is not threadsafe, but you may call PrintObj repeatedly; headers
// will only be printed if the object type changes. This makes it useful for printing items
// received from watches.
type HumanReadablePrinter struct {
	options        PrintOptions
	lastType       interface{}
	lastColumns    []metav1.TableColumnDefinition
	printedHeaders bool
}

// NewTablePrinter creates a printer suitable for calling PrintObj().
func NewTablePrinter(options PrintOptions) ResourcePrinter {
	printer := &HumanReadablePrinter{
		options: options,
	}
	return printer
}

func printHeader(columnNames []string, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%s\n", strings.Join(columnNames, "\t")); err != nil {
		return err
	}
	return nil
}

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {

	if _, found := output.(*tabwriter.Writer); !found {
		w := GetNewTabWriter(output)
		output = w
		defer w.Flush()
	}

	var eventType string
	if event, isEvent := obj.(*metav1.WatchEvent); isEvent {
		eventType = event.Type
		obj = event.Object.Object
	}

	// Parameter "obj" is a table from server; print it.
	// display tables following the rules of options
	if table, ok := obj.(*metav1.Table); ok {
		// Do not print headers if this table has no column definitions, or they are the same as the last ones we printed
		localOptions := h.options
		if h.printedHeaders && (len(table.ColumnDefinitions) == 0 || reflect.DeepEqual(table.ColumnDefinitions, h.lastColumns)) {
			localOptions.NoHeaders = true
		}

		if len(table.ColumnDefinitions) == 0 {
			// If this table has no column definitions, use the columns from the last table we printed for decoration and layout.
			// This is done when receiving tables in watch events to save bandwidth.
			table.ColumnDefinitions = h.lastColumns
		} else if !reflect.DeepEqual(table.ColumnDefinitions, h.lastColumns) {
			// If this table has column definitions, remember them for future use.
			h.lastColumns = table.ColumnDefinitions
			h.printedHeaders = false
		}

		if len(table.Rows) > 0 {
			h.printedHeaders = true
		}

		if err := decorateTable(table, localOptions); err != nil {
			return err
		}
		if len(eventType) > 0 {
			if err := addColumns(beginning, table,
				[]metav1.TableColumnDefinition{{Name: "Event", Type: "string"}},
				[]cellValueFunc{func(metav1.TableRow) (interface{}, error) { return formatEventType(eventType), nil }},
			); err != nil {
				return err
			}
		}
		return printTable(table, output, localOptions)
	}

	// Could not find print handler for "obj"; use the default or status print handler.
	// Print with the default or status handler, and use the columns from the last time
	var handler *printHandler
	if _, isStatus := obj.(*metav1.Status); isStatus {
		handler = statusHandlerEntry
	} else {
		handler = defaultHandlerEntry
	}

	includeHeaders := h.lastType != handler && !h.options.NoHeaders

	if h.lastType != nil && h.lastType != handler && !h.options.NoHeaders {
		fmt.Fprintln(output)
	}

	if err := printRowsForHandlerEntry(output, handler, eventType, obj, h.options, includeHeaders); err != nil {
		return err
	}
	h.lastType = handler

	return nil
}

// printTable prints a table to the provided output respecting the filtering rules for options
// for wide columns and filtered rows. It filters out rows that are Completed. You should call
// decorateTable if you receive a table from a remote server before calling printTable.
func printTable(table *metav1.Table, output io.Writer, options PrintOptions) error {
	if !options.NoHeaders {
		// avoid printing headers if we have no rows to display
		if len(table.Rows) == 0 {
			return nil
		}

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
		first := true
		for i, cell := range row.Cells {
			if i >= len(table.ColumnDefinitions) {
				// https://issue.k8s.io/66379
				// don't panic in case of bad output from the server, with more cells than column definitions
				break
			}
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
				switch val := cell.(type) {
				case string:
					print := val
					truncated := false
					// truncate at newlines
					newline := strings.Index(print, "\n")
					if newline >= 0 {
						truncated = true
						print = print[:newline]
					}
					fmt.Fprint(output, print)
					if truncated {
						fmt.Fprint(output, "...")
					}
				default:
					fmt.Fprint(output, val)
				}
			}
		}
		fmt.Fprintln(output)
	}
	return nil
}

type cellValueFunc func(metav1.TableRow) (interface{}, error)

type columnAddPosition int

const (
	beginning columnAddPosition = 1
	end       columnAddPosition = 2
)

func addColumns(pos columnAddPosition, table *metav1.Table, columns []metav1.TableColumnDefinition, valueFuncs []cellValueFunc) error {
	if len(columns) != len(valueFuncs) {
		return fmt.Errorf("cannot prepend columns, unmatched value functions")
	}
	if len(columns) == 0 {
		return nil
	}

	// Compute the new rows
	newRows := make([][]interface{}, len(table.Rows))
	for i := range table.Rows {
		newCells := make([]interface{}, 0, len(columns)+len(table.Rows[i].Cells))

		if pos == end {
			// If we're appending, start with the existing cells,
			// then add nil cells to match the number of columns
			newCells = append(newCells, table.Rows[i].Cells...)
			for len(newCells) < len(table.ColumnDefinitions) {
				newCells = append(newCells, nil)
			}
		}

		// Compute cells for new columns
		for _, f := range valueFuncs {
			newCell, err := f(table.Rows[i])
			if err != nil {
				return err
			}
			newCells = append(newCells, newCell)
		}

		if pos == beginning {
			// If we're prepending, add existing cells
			newCells = append(newCells, table.Rows[i].Cells...)
		}

		// Remember the new cells for this row
		newRows[i] = newCells
	}

	// All cells successfully computed, now replace columns and rows
	newColumns := make([]metav1.TableColumnDefinition, 0, len(columns)+len(table.ColumnDefinitions))
	switch pos {
	case beginning:
		newColumns = append(newColumns, columns...)
		newColumns = append(newColumns, table.ColumnDefinitions...)
	case end:
		newColumns = append(newColumns, table.ColumnDefinitions...)
		newColumns = append(newColumns, columns...)
	default:
		return fmt.Errorf("invalid column add position: %v", pos)
	}
	table.ColumnDefinitions = newColumns
	for i := range table.Rows {
		table.Rows[i].Cells = newRows[i]
	}

	return nil
}

// decorateTable takes a table and attempts to add label columns and the
// namespace column. It will fill empty columns with nil (if the object
// does not expose metadata). It returns an error if the table cannot
// be decorated.
func decorateTable(table *metav1.Table, options PrintOptions) error {
	width := len(table.ColumnDefinitions) + len(options.ColumnLabels)
	if options.WithNamespace {
		width++
	}
	if options.ShowLabels {
		width++
	}

	columns := table.ColumnDefinitions

	nameColumn := -1
	if options.WithKind && !options.Kind.Empty() {
		for i := range columns {
			if columns[i].Format == "name" && columns[i].Type == "string" {
				nameColumn = i
				break
			}
		}
	}

	if width != len(table.ColumnDefinitions) {
		columns = make([]metav1.TableColumnDefinition, 0, width)
		if options.WithNamespace {
			columns = append(columns, metav1.TableColumnDefinition{
				Name: "Namespace",
				Type: "string",
			})
		}
		columns = append(columns, table.ColumnDefinitions...)
		for _, label := range formatLabelHeaders(options.ColumnLabels) {
			columns = append(columns, metav1.TableColumnDefinition{
				Name: label,
				Type: "string",
			})
		}
		if options.ShowLabels {
			columns = append(columns, metav1.TableColumnDefinition{
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
				row.Cells[nameColumn] = fmt.Sprintf("%s/%s", strings.ToLower(options.Kind.String()), row.Cells[nameColumn])
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

// printRowsForHandlerEntry prints the incremental table output (headers if the current type is
// different from lastType) including all the rows in the object. It returns the current type
// or an error, if any.
func printRowsForHandlerEntry(output io.Writer, handler *printHandler, eventType string, obj runtime.Object, options PrintOptions, includeHeaders bool) error {
	var results []reflect.Value

	args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(options)}
	results = handler.printFunc.Call(args)
	if !results[1].IsNil() {
		return results[1].Interface().(error)
	}

	if includeHeaders {
		var headers []string
		for _, column := range handler.columnDefinitions {
			if column.Priority != 0 && !options.Wide {
				continue
			}
			headers = append(headers, strings.ToUpper(column.Name))
		}
		headers = append(headers, formatLabelHeaders(options.ColumnLabels)...)
		// LABELS is always the last column.
		headers = append(headers, formatShowLabelsHeader(options.ShowLabels)...)
		// prepend namespace header
		if options.WithNamespace {
			headers = append(withNamespacePrefixColumns, headers...)
		}
		// prepend event type header
		if len(eventType) > 0 {
			headers = append(withEventTypePrefixColumns, headers...)
		}
		printHeader(headers, output)
	}

	if results[1].IsNil() {
		rows := results[0].Interface().([]metav1.TableRow)
		printRows(output, eventType, rows, options)
		return nil
	}
	return results[1].Interface().(error)
}

var formattedEventType = map[string]string{
	string(watch.Added):    "ADDED   ",
	string(watch.Modified): "MODIFIED",
	string(watch.Deleted):  "DELETED ",
	string(watch.Error):    "ERROR   ",
}

func formatEventType(eventType string) string {
	if formatted, ok := formattedEventType[eventType]; ok {
		return formatted
	}
	return string(eventType)
}

// printRows writes the provided rows to output.
func printRows(output io.Writer, eventType string, rows []metav1.TableRow, options PrintOptions) {
	for _, row := range rows {
		if len(eventType) > 0 {
			fmt.Fprint(output, formatEventType(eventType))
			fmt.Fprint(output, "\t")
		}
		if options.WithNamespace {
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
				if options.WithKind && !options.Kind.Empty() {
					fmt.Fprintf(output, "%s/%s", strings.ToLower(options.Kind.String()), cell)
					continue
				}
			}
			fmt.Fprint(output, cell)
		}

		hasLabels := len(options.ColumnLabels) > 0
		if obj := row.Object.Object; obj != nil && (hasLabels || options.ShowLabels) {
			if m, err := meta.Accessor(obj); err == nil {
				for _, value := range labelValues(m.GetLabels(), options) {
					output.Write([]byte("\t"))
					output.Write([]byte(value))
				}
			}
		}

		output.Write([]byte("\n"))
	}
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
func formatShowLabelsHeader(showLabels bool) []string {
	if showLabels {
		return []string{"LABELS"}
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

func printStatus(obj runtime.Object, options PrintOptions) ([]metav1.TableRow, error) {
	status, ok := obj.(*metav1.Status)
	if !ok {
		return nil, fmt.Errorf("expected *v1.Status, got %T", obj)
	}
	return []metav1.TableRow{{
		Object: runtime.RawExtension{Object: obj},
		Cells:  []interface{}{status.Status, status.Reason, status.Message},
	}}, nil
}

func printObjectMeta(obj runtime.Object, options PrintOptions) ([]metav1.TableRow, error) {
	if meta.IsListType(obj) {
		rows := make([]metav1.TableRow, 0, 16)
		err := meta.EachListItem(obj, func(obj runtime.Object) error {
			nestedRows, err := printObjectMeta(obj, options)
			if err != nil {
				return err
			}
			rows = append(rows, nestedRows...)
			return nil
		})
		if err != nil {
			return nil, err
		}
		return rows, nil
	}

	rows := make([]metav1.TableRow, 0, 1)
	m, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}
	row := metav1.TableRow{
		Object: runtime.RawExtension{Object: obj},
	}
	row.Cells = append(row.Cells, m.GetName(), translateTimestampSince(m.GetCreationTimestamp()))
	rows = append(rows, row)
	return rows, nil
}

// translateTimestampSince returns the elapsed time since timestamp in
// human-readable approximation.
func translateTimestampSince(timestamp metav1.Time) string {
	if timestamp.IsZero() {
		return "<unknown>"
	}

	return duration.HumanDuration(time.Since(timestamp.Time))
}
