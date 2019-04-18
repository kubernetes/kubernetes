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

	"github.com/liggitt/tabwriter"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
)

var _ ResourcePrinter = &HumanReadablePrinter{}

var withNamespacePrefixColumns = []string{"NAMESPACE"} // TODO(erictune): print cluster name too.

// NewHumanReadablePrinter creates a printer suitable for calling PrintObj().
// TODO(seans3): Change return type to ResourcePrinter interface once we no longer need
// to constuct the "handlerMap".
func NewHumanReadablePrinter(options PrintOptions) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		handlerMap: make(map[reflect.Type]*handlerEntry),
		options:    options,
	}
	return printer
}

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	w, found := output.(*tabwriter.Writer)
	if !found {
		w = GetNewTabWriter(output)
		output = w
		defer w.Flush()
	}

	// Case 1: Parameter "obj" is a table from server; print it.
	// display tables following the rules of options
	if table, ok := obj.(*metav1beta1.Table); ok {
		// Do not print headers if this table has no column definitions, or they are the same as the last ones we printed
		localOptions := h.options
		if len(table.ColumnDefinitions) == 0 || reflect.DeepEqual(table.ColumnDefinitions, h.lastColumns) {
			localOptions.NoHeaders = true
		}

		if len(table.ColumnDefinitions) == 0 {
			// If this table has no column definitions, use the columns from the last table we printed for decoration and layout.
			// This is done when receiving tables in watch events to save bandwidth.
			localOptions.NoHeaders = true
			table.ColumnDefinitions = h.lastColumns
		} else {
			// If this table has column definitions, remember them for future use.
			h.lastColumns = table.ColumnDefinitions
		}

		if err := decorateTable(table, localOptions); err != nil {
			return err
		}
		return PrintTable(table, output, localOptions)
	}

	// Case 2: Parameter "obj" is not a table; search for a handler to print it.
	// TODO(seans3): Remove this case in 1.16, since table should be returned from server-side printing.
	// print with a registered handler
	t := reflect.TypeOf(obj)
	if handler := h.handlerMap[t]; handler != nil {
		includeHeaders := h.lastType != t && !h.options.NoHeaders

		if h.lastType != nil && h.lastType != t && !h.options.NoHeaders {
			fmt.Fprintln(output)
		}

		if err := printRowsForHandlerEntry(output, handler, obj, h.options, includeHeaders); err != nil {
			return err
		}
		h.lastType = t
		return nil
	}

	// Case 3: Could not find print handler for "obj"; use the default print handler.
	// print with the default handler if set, and use the columns from the last time
	if h.defaultHandler != nil {
		includeHeaders := h.lastType != h.defaultHandler && !h.options.NoHeaders

		if h.lastType != nil && h.lastType != h.defaultHandler && !h.options.NoHeaders {
			fmt.Fprintln(output)
		}

		if err := printRowsForHandlerEntry(output, h.defaultHandler, obj, h.options, includeHeaders); err != nil {
			return err
		}
		h.lastType = h.defaultHandler
		return nil
	}

	// we failed all reasonable printing efforts, report failure
	return fmt.Errorf("error: unknown type %#v", obj)
}

// PrintTable prints a table to the provided output respecting the filtering rules for options
// for wide columns and filtered rows. It filters out rows that are Completed. You should call
// decorateTable if you receive a table from a remote server before calling PrintTable.
func PrintTable(table *metav1beta1.Table, output io.Writer, options PrintOptions) error {
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
				fmt.Fprint(output, cell)
			}
		}
		fmt.Fprintln(output)
	}
	return nil
}

// decorateTable takes a table and attempts to add label columns and the
// namespace column. It will fill empty columns with nil (if the object
// does not expose metadata). It returns an error if the table cannot
// be decorated.
func decorateTable(table *metav1beta1.Table, options PrintOptions) error {
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
		columns = make([]metav1beta1.TableColumnDefinition, 0, width)
		if options.WithNamespace {
			columns = append(columns, metav1beta1.TableColumnDefinition{
				Name: "Namespace",
				Type: "string",
			})
		}
		columns = append(columns, table.ColumnDefinitions...)
		for _, label := range formatLabelHeaders(options.ColumnLabels) {
			columns = append(columns, metav1beta1.TableColumnDefinition{
				Name: label,
				Type: "string",
			})
		}
		if options.ShowLabels {
			columns = append(columns, metav1beta1.TableColumnDefinition{
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
func printRowsForHandlerEntry(output io.Writer, handler *handlerEntry, obj runtime.Object, options PrintOptions, includeHeaders bool) error {
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
		if options.WithNamespace {
			headers = append(withNamespacePrefixColumns, headers...)
		}
		printHeader(headers, output)
	}

	if results[1].IsNil() {
		rows := results[0].Interface().([]metav1beta1.TableRow)
		printRows(output, rows, options)
		return nil
	}
	return results[1].Interface().(error)
}

func printHeader(columnNames []string, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%s\n", strings.Join(columnNames, "\t")); err != nil {
		return err
	}
	return nil
}

// printRows writes the provided rows to output.
func printRows(output io.Writer, rows []metav1beta1.TableRow, options PrintOptions) {
	for _, row := range rows {
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
