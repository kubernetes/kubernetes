/*
Copyright 2014 The Kubernetes Authors.

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

package get

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"strings"

	"github.com/liggitt/tabwriter"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/util/jsonpath"
)

var jsonRegexp = regexp.MustCompile(`^\{\.?([^{}]+)\}$|^\.?([^{}]+)$`)

// RelaxedJSONPathExpression attempts to be flexible with JSONPath expressions, it accepts:
//   - metadata.name (no leading '.' or curly braces '{...}'
//   - {metadata.name} (no leading '.')
//   - .metadata.name (no curly braces '{...}')
//   - {.metadata.name} (complete expression)
//
// And transforms them all into a valid jsonpath expression:
//
//	{.metadata.name}
func RelaxedJSONPathExpression(pathExpression string) (string, error) {
	if len(pathExpression) == 0 {
		return pathExpression, nil
	}
	submatches := jsonRegexp.FindStringSubmatch(pathExpression)
	if submatches == nil {
		return "", fmt.Errorf("unexpected path string, expected a 'name1.name2' or '.name1.name2' or '{name1.name2}' or '{.name1.name2}'")
	}
	if len(submatches) != 3 {
		return "", fmt.Errorf("unexpected submatch list: %v", submatches)
	}
	var fieldSpec string
	if len(submatches[1]) != 0 {
		fieldSpec = submatches[1]
	} else {
		fieldSpec = submatches[2]
	}
	return fmt.Sprintf("{.%s}", fieldSpec), nil
}

// NewCustomColumnsPrinterFromSpec creates a custom columns printer from a comma separated list of <header>:<jsonpath-field-spec> pairs.
// e.g. NAME:metadata.name,API_VERSION:apiVersion creates a printer that prints:
//
//	NAME               API_VERSION
//	foo                bar
func NewCustomColumnsPrinterFromSpec(spec string, decoder runtime.Decoder, noHeaders bool) (*CustomColumnsPrinter, error) {
	if len(spec) == 0 {
		return nil, fmt.Errorf("custom-columns format specified but no custom columns given")
	}
	parts := strings.Split(spec, ",")
	columns := make([]Column, len(parts))
	for ix := range parts {
		colSpec := strings.SplitN(parts[ix], ":", 2)
		if len(colSpec) != 2 {
			return nil, fmt.Errorf("unexpected custom-columns spec: %s, expected <header>:<json-path-expr>", parts[ix])
		}
		spec, err := RelaxedJSONPathExpression(colSpec[1])
		if err != nil {
			return nil, err
		}
		columns[ix] = Column{Header: colSpec[0], FieldSpec: spec}
	}
	return &CustomColumnsPrinter{Columns: columns, Decoder: decoder, NoHeaders: noHeaders}, nil
}

func splitOnWhitespace(line string) []string {
	lineScanner := bufio.NewScanner(bytes.NewBufferString(line))
	lineScanner.Split(bufio.ScanWords)
	result := []string{}
	for lineScanner.Scan() {
		result = append(result, lineScanner.Text())
	}
	return result
}

// NewCustomColumnsPrinterFromTemplate creates a custom columns printer from a template stream.  The template is expected
// to consist of two lines, whitespace separated.  The first line is the header line, the second line is the jsonpath field spec
// For example, the template below:
// NAME               API_VERSION
// {metadata.name}    {apiVersion}
func NewCustomColumnsPrinterFromTemplate(templateReader io.Reader, decoder runtime.Decoder) (*CustomColumnsPrinter, error) {
	scanner := bufio.NewScanner(templateReader)
	if !scanner.Scan() {
		return nil, fmt.Errorf("invalid template, missing header line. Expected format is one line of space separated headers, one line of space separated column specs.")
	}
	headers := splitOnWhitespace(scanner.Text())

	if !scanner.Scan() {
		return nil, fmt.Errorf("invalid template, missing spec line. Expected format is one line of space separated headers, one line of space separated column specs.")
	}
	specs := splitOnWhitespace(scanner.Text())

	if len(headers) != len(specs) {
		return nil, fmt.Errorf("number of headers (%d) and field specifications (%d) don't match", len(headers), len(specs))
	}

	columns := make([]Column, len(headers))
	for ix := range headers {
		spec, err := RelaxedJSONPathExpression(specs[ix])
		if err != nil {
			return nil, err
		}
		columns[ix] = Column{
			Header:    headers[ix],
			FieldSpec: spec,
		}
	}
	return &CustomColumnsPrinter{Columns: columns, Decoder: decoder, NoHeaders: false}, nil
}

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
	Columns   []Column
	Decoder   runtime.Decoder
	NoHeaders bool
	// lastType records type of resource printed last so that we don't repeat
	// header while printing same type of resources.
	lastType reflect.Type
}

func (s *CustomColumnsPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	// we use reflect.Indirect here in order to obtain the actual value from a pointer.
	// we need an actual value in order to retrieve the package path for an object.
	// using reflect.Indirect indiscriminately is valid here, as all runtime.Objects are supposed to be pointers.
	if printers.InternalObjectPreventer.IsForbidden(reflect.Indirect(reflect.ValueOf(obj)).Type().PkgPath()) {
		return errors.New(printers.InternalObjectPrinterErr)
	}

	if _, found := out.(*tabwriter.Writer); !found {
		w := printers.GetNewTabWriter(out)
		out = w
		defer w.Flush()
	}

	t := reflect.TypeOf(obj)
	if !s.NoHeaders && t != s.lastType {
		headers := make([]string, len(s.Columns))
		for ix := range s.Columns {
			headers[ix] = s.Columns[ix].Header
		}
		fmt.Fprintln(out, strings.Join(headers, "\t"))
		s.lastType = t
	}
	parsers := make([]*jsonpath.JSONPath, len(s.Columns))
	for ix := range s.Columns {
		parsers[ix] = jsonpath.New(fmt.Sprintf("column%d", ix)).AllowMissingKeys(true)
		if err := parsers[ix].Parse(s.Columns[ix].FieldSpec); err != nil {
			return err
		}
	}

	if meta.IsListType(obj) {
		objs, err := meta.ExtractList(obj)
		if err != nil {
			return err
		}
		for ix := range objs {
			if err := s.printOneObject(objs[ix], parsers, out); err != nil {
				return err
			}
		}
	} else {
		if err := s.printOneObject(obj, parsers, out); err != nil {
			return err
		}
	}
	return nil
}

func (s *CustomColumnsPrinter) printOneObject(obj runtime.Object, parsers []*jsonpath.JSONPath, out io.Writer) error {
	columns := make([]string, len(parsers))
	switch u := obj.(type) {
	case *metav1.WatchEvent:
		if printers.InternalObjectPreventer.IsForbidden(reflect.Indirect(reflect.ValueOf(u.Object.Object)).Type().PkgPath()) {
			return errors.New(printers.InternalObjectPrinterErr)
		}
		unstructuredObject, err := runtime.DefaultUnstructuredConverter.ToUnstructured(u.Object.Object)
		if err != nil {
			return err
		}
		obj = &unstructured.Unstructured{
			Object: map[string]interface{}{
				"type":   u.Type,
				"object": unstructuredObject,
			},
		}

	case *runtime.Unknown:
		if len(u.Raw) > 0 {
			var err error
			if obj, err = runtime.Decode(s.Decoder, u.Raw); err != nil {
				return fmt.Errorf("can't decode object for printing: %v (%s)", err, u.Raw)
			}
		}
	}

	for ix := range parsers {
		parser := parsers[ix]

		var values [][]reflect.Value
		var err error
		if unstructured, ok := obj.(runtime.Unstructured); ok {
			values, err = parser.FindResults(unstructured.UnstructuredContent())
		} else {
			values, err = parser.FindResults(reflect.ValueOf(obj).Elem().Interface())
		}

		if err != nil {
			return err
		}
		valueStrings := []string{}
		if len(values) == 0 || len(values[0]) == 0 {
			valueStrings = append(valueStrings, "<none>")
		}
		for arrIx := range values {
			for valIx := range values[arrIx] {
				valueStrings = append(valueStrings, printers.EscapeTerminal(fmt.Sprint(values[arrIx][valIx].Interface())))
			}
		}
		columns[ix] = strings.Join(valueStrings, ",")
	}
	fmt.Fprintln(out, strings.Join(columns, "\t"))
	return nil
}
