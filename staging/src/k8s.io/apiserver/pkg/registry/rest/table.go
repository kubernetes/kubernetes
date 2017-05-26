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

package rest

import (
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

var DefaultTableConvertor TableConvertor = defaultTableConvertor{}

type defaultTableConvertor struct{}

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

func (defaultTableConvertor) ConvertToTable(ctx genericapirequest.Context, object runtime.Object, tableOptions runtime.Object) (*metav1alpha1.Table, error) {
	var table metav1alpha1.Table
	fn := func(obj runtime.Object) error {
		m, err := meta.Accessor(obj)
		if err != nil {
			// TODO: skip objects we don't recognize
			return nil
		}
		table.Rows = append(table.Rows, metav1alpha1.TableRow{
			Cells:  []interface{}{m.GetClusterName(), m.GetNamespace(), m.GetName(), m.GetCreationTimestamp().Time.UTC().Format(time.RFC3339)},
			Object: runtime.RawExtension{Object: obj},
		})
		return nil
	}
	switch {
	case meta.IsListType(object):
		if err := meta.EachListItem(object, fn); err != nil {
			return nil, err
		}
	default:
		if err := fn(object); err != nil {
			return nil, err
		}
	}
	table.ColumnDefinitions = []metav1alpha1.TableColumnDefinition{
		{Name: "Cluster Name", Type: "string", Description: swaggerMetadataDescriptions["clusterName"]},
		{Name: "Namespace", Type: "string", Description: swaggerMetadataDescriptions["namespace"]},
		{Name: "Name", Type: "string", Description: swaggerMetadataDescriptions["name"]},
		{Name: "Created At", Type: "date", Description: swaggerMetadataDescriptions["creationTimestamp"]},
	}
	// trim the left two columns if completely empty
	if trimColumn(0, &table) {
		trimColumn(0, &table)
	} else {
		trimColumn(1, &table)
	}
	return &table, nil
}

func trimColumn(column int, table *metav1alpha1.Table) bool {
	for _, item := range table.Rows {
		switch t := item.Cells[column].(type) {
		case string:
			if len(t) > 0 {
				return false
			}
		case interface{}:
			if t == nil {
				return false
			}
		}
	}
	if column == 0 {
		table.ColumnDefinitions = table.ColumnDefinitions[1:]
	} else {
		for j := column; j < len(table.ColumnDefinitions); j++ {
			table.ColumnDefinitions[j] = table.ColumnDefinitions[j+1]
		}
	}
	for i := range table.Rows {
		cells := table.Rows[i].Cells
		if column == 0 {
			table.Rows[i].Cells = cells[1:]
			continue
		}
		for j := column; j < len(cells); j++ {
			cells[j] = cells[j+1]
		}
		table.Rows[i].Cells = cells[:len(cells)-1]
	}
	return true
}
