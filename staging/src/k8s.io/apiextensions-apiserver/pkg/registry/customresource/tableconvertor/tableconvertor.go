/*
Copyright 2018 The Kubernetes Authors.

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

package tableconvertor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metatable "k8s.io/apimachinery/pkg/api/meta/table"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/util/jsonpath"
)

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// New creates a new table convertor for the provided CRD column definition. If the printer definition cannot be parsed,
// error will be returned along with a default table convertor.
func New(crdColumns []apiextensionsv1.CustomResourceColumnDefinition) (rest.TableConvertor, error) {
	headers := []metav1.TableColumnDefinition{
		{Name: "Name", Type: "string", Format: "name", Description: swaggerMetadataDescriptions["name"]},
	}
	c := &convertor{
		headers: headers,
	}

	for _, col := range crdColumns {
		path := jsonpath.New(col.Name)
		if err := path.Parse(fmt.Sprintf("{%s}", col.JSONPath)); err != nil {
			return c, fmt.Errorf("unrecognized column definition %q", col.JSONPath)
		}
		path.AllowMissingKeys(true)

		desc := fmt.Sprintf("Custom resource definition column (in JSONPath format): %s", col.JSONPath)
		if len(col.Description) > 0 {
			desc = col.Description
		}

		c.additionalColumns = append(c.additionalColumns, path)
		c.headers = append(c.headers, metav1.TableColumnDefinition{
			Name:        col.Name,
			Type:        col.Type,
			Format:      col.Format,
			Description: desc,
			Priority:    col.Priority,
		})
	}

	return c, nil
}

type columnPrinter interface {
	FindResults(data interface{}) ([][]reflect.Value, error)
	PrintResults(w io.Writer, results []reflect.Value) error
}

type convertor struct {
	headers           []metav1.TableColumnDefinition
	additionalColumns []columnPrinter
}

func (c *convertor) ConvertToTable(ctx context.Context, obj runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	table := &metav1.Table{}
	opt, ok := tableOptions.(*metav1.TableOptions)
	noHeaders := ok && opt != nil && opt.NoHeaders
	if !noHeaders {
		table.ColumnDefinitions = c.headers
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

	var err error
	buf := &bytes.Buffer{}
	table.Rows, err = metatable.MetaToTableRow(obj, func(obj runtime.Object, m metav1.Object, name, age string) ([]interface{}, error) {
		cells := make([]interface{}, 1, 1+len(c.additionalColumns))
		cells[0] = name
		customHeaders := c.headers[1:]
		us, ok := obj.(runtime.Unstructured)
		if !ok {
			m, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
			if err != nil {
				return nil, err
			}
			us = &unstructured.Unstructured{Object: m}
		}
		for i, column := range c.additionalColumns {
			results, err := column.FindResults(us.UnstructuredContent())
			if err != nil || len(results) == 0 || len(results[0]) == 0 {
				cells = append(cells, nil)
				continue
			}

			// as we only support simple JSON path, we can assume to have only one result (or none, filtered out above)
			value := results[0][0].Interface()
			if customHeaders[i].Type == "string" {
				if err := column.PrintResults(buf, []reflect.Value{reflect.ValueOf(value)}); err == nil {
					cells = append(cells, buf.String())
					buf.Reset()
				} else {
					cells = append(cells, nil)
				}
			} else {
				cells = append(cells, cellForJSONValue(customHeaders[i].Type, value))
			}
		}
		return cells, nil
	})
	return table, err
}

func cellForJSONValue(headerType string, value interface{}) interface{} {
	if value == nil {
		return nil
	}

	switch headerType {
	case "integer":
		switch typed := value.(type) {
		case int64:
			return typed
		case float64:
			return int64(typed)
		case json.Number:
			if i64, err := typed.Int64(); err == nil {
				return i64
			}
		}
	case "number":
		switch typed := value.(type) {
		case int64:
			return float64(typed)
		case float64:
			return typed
		case json.Number:
			if f, err := typed.Float64(); err == nil {
				return f
			}
		}
	case "boolean":
		if b, ok := value.(bool); ok {
			return b
		}
	case "string":
		if s, ok := value.(string); ok {
			return s
		}
	case "date":
		if typed, ok := value.(string); ok {
			var timestamp metav1.Time
			err := timestamp.UnmarshalQueryParameter(typed)
			if err != nil {
				return "<invalid>"
			}
			return metatable.ConvertToHumanReadableDateType(timestamp)
		}
	}

	return nil
}
