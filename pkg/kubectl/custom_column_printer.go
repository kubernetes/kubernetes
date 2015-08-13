/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"fmt"
	"io"
	"reflect"
	"strings"
	"text/tabwriter"

	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/jsonpath"
)

const (
	columnwidth       = 10
	tabwidth          = 4
	padding           = 3
	padding_character = ' '
	flags             = 0
)

// Column represents a user specified column
type Column struct {
	// The header to print above the column, general style is ALL_CAPS
	Header string
	// The pointer to the field in the object to print in JSONPath form
	// e.g. {.ObjectMeta.Name}, see pkg/util/jsonpath for more details.
	FieldSpec string
}

// CustomColumnPrinter is a printer that knows how to print arbitrary columns
// of data from templates specified in the `Columns` array
type CustomColumnsPrinter struct {
	Columns []Column
}

func (s *CustomColumnsPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	w := tabwriter.NewWriter(out, columnwidth, tabwidth, padding, padding_character, flags)
	headers := make([]string, len(s.Columns))
	for ix := range s.Columns {
		headers[ix] = s.Columns[ix].Header
	}
	fmt.Fprintln(w, strings.Join(headers, "\t"))
	parsers := make([]*jsonpath.JSONPath, len(s.Columns))
	for ix := range s.Columns {
		parsers[ix] = jsonpath.New(fmt.Sprintf("column%d", ix))
		if err := parsers[ix].Parse(s.Columns[ix].FieldSpec); err != nil {
			return err
		}
	}

	if runtime.IsListType(obj) {
		objs, err := runtime.ExtractList(obj)
		if err != nil {
			return err
		}
		for ix := range objs {
			if err := s.printOneObject(objs[ix], parsers, w); err != nil {
				return err
			}
		}
	} else {
		if err := s.printOneObject(obj, parsers, w); err != nil {
			return err
		}
	}
	return w.Flush()
}

func (s *CustomColumnsPrinter) printOneObject(obj runtime.Object, parsers []*jsonpath.JSONPath, out io.Writer) error {
	columns := make([]string, len(parsers))
	for ix := range parsers {
		parser := parsers[ix]
		values, err := parser.FindResults(reflect.ValueOf(obj).Elem().Interface())
		if err != nil {
			return err
		}
		if len(values) == 0 || len(values[0]) == 0 {
			fmt.Fprintf(out, "<none>\t")
		}
		valueStrings := []string{}
		for arrIx := range values {
			for valIx := range values[arrIx] {
				valueStrings = append(valueStrings, fmt.Sprintf("%v", values[arrIx][valIx].Interface()))
			}
		}
		columns[ix] = strings.Join(valueStrings, ",")
	}
	fmt.Fprintln(out, strings.Join(columns, "\t"))
	return nil
}
