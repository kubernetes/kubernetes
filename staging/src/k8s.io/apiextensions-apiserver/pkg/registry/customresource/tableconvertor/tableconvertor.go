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
	"fmt"
	"strings"

	"github.com/go-openapi/spec"

	"k8s.io/apimachinery/pkg/api/meta"
	metatable "k8s.io/apimachinery/pkg/api/meta/table"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/util/jsonpath"
)

const printColumnsKey = "x-kubernetes-print-columns"

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// New creates a new table convertor for the provided OpenAPI schema. If the printer definition cannot be parsed,
// error will be returned along with a default table convertor.
func New(extensions spec.Extensions) (rest.TableConvertor, error) {
	headers := []metav1beta1.TableColumnDefinition{
		{Name: "Name", Type: "string", Format: "name", Description: swaggerMetadataDescriptions["name"]},
		{Name: "Created At", Type: "date", Description: swaggerMetadataDescriptions["creationTimestamp"]},
	}
	c := &convertor{
		headers: headers,
	}
	format, ok := extensions.GetString(printColumnsKey)
	if !ok {
		return c, nil
	}
	// "x-kubernetes-print-columns": "custom-columns=NAME:.metadata.name,RSRC:.metadata.resourceVersion"
	parts := strings.SplitN(format, "=", 2)
	if len(parts) != 2 || parts[0] != "custom-columns" {
		return c, fmt.Errorf("unrecognized column definition in 'x-kubernetes-print-columns', only support 'custom-columns=NAME=JSONPATH[,NAME=JSONPATH]'")
	}
	columnSpecs := strings.Split(parts[1], ",")
	var columns []*jsonpath.JSONPath
	for _, spec := range columnSpecs {
		parts := strings.SplitN(spec, ":", 2)
		if len(parts) != 2 || len(parts[0]) == 0 || len(parts[1]) == 0 {
			return c, fmt.Errorf("unrecognized column definition in 'x-kubernetes-print-columns', must specify NAME=JSONPATH: %s", spec)
		}
		path := jsonpath.New(parts[0])
		if err := path.Parse(parts[1]); err != nil {
			return c, fmt.Errorf("unrecognized column definition in 'x-kubernetes-print-columns': %v", spec)
		}
		path.AllowMissingKeys(true)
		columns = append(columns, path)
		headers = append(headers, metav1beta1.TableColumnDefinition{
			Name:        parts[0],
			Type:        "string",
			Description: fmt.Sprintf("Custom resource definition column from OpenAPI (in JSONPath format): %s", parts[1]),
		})
	}
	c.columns = columns
	c.headers = headers
	return c, nil
}

type convertor struct {
	headers []metav1beta1.TableColumnDefinition
	columns []*jsonpath.JSONPath
}

func (c *convertor) ConvertToTable(ctx genericapirequest.Context, obj runtime.Object, tableOptions runtime.Object) (*metav1beta1.Table, error) {
	table := &metav1beta1.Table{
		ColumnDefinitions: c.headers,
	}
	if m, err := meta.ListAccessor(obj); err == nil {
		table.ResourceVersion = m.GetResourceVersion()
		table.SelfLink = m.GetSelfLink()
		table.Continue = m.GetContinue()
	} else {
		if m, err := meta.CommonAccessor(obj); err == nil {
			table.ResourceVersion = m.GetResourceVersion()
			table.SelfLink = m.GetSelfLink()
		}
	}

	var err error
	buf := &bytes.Buffer{}
	table.Rows, err = metatable.MetaToTableRow(obj, func(obj runtime.Object, m metav1.Object, name, age string) ([]interface{}, error) {
		cells := make([]interface{}, 2, 2+len(c.columns))
		cells[0] = name
		cells[1] = age
		for _, column := range c.columns {
			if err := column.Execute(buf, obj); err != nil {
				cells = append(cells, nil)
				continue
			}
			cells = append(cells, buf.String())
			buf.Reset()
		}
		return cells, nil
	})
	return table, err
}
