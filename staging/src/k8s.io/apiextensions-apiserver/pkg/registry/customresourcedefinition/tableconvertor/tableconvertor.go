/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"sort"
	"strings"
	"time"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/rest"
)

var metaDocs = metav1.ObjectMeta{}.SwaggerDoc()

func New() rest.TableConvertor { return &crdConvertor{} }

type crdConvertor struct{}

// ConvertToTable satisfies rest.TableConvertor.
func (c *crdConvertor) ConvertToTable(ctx context.Context, obj runtime.Object, tableOpts runtime.Object) (*metav1.Table, error) {
	table := &metav1.Table{}

	if opt, _ := tableOpts.(*metav1.TableOptions); opt == nil || !opt.NoHeaders {
		table.ColumnDefinitions = []metav1.TableColumnDefinition{
			{Name: "Name", Type: "string", Format: "name", Description: metaDocs["name"]},
			{Name: "Group", Type: "string", Description: "API group"},
			{Name: "Scope", Type: "string", Description: "Cluster/Namespaced"},
			{Name: "Versions", Type: "string", Description: "Served versions"},
			{Name: "Created At", Type: "date", Description: metaDocs["creationTimestamp"]},
		}
	}

	addRow := func(cr *apiextensions.CustomResourceDefinition) {
		versions := make([]string, 0, len(cr.Spec.Versions))
		for _, v := range cr.Spec.Versions {
			versions = append(versions, v.Name)
		}
		sort.Strings(versions)

		table.Rows = append(table.Rows, metav1.TableRow{
			Object: runtime.RawExtension{Object: cr},
			Cells: []any{
				cr.Name,
				cr.Spec.Group,
				string(cr.Spec.Scope),
				strings.Join(versions, ","),
				cr.CreationTimestamp.Time.UTC().Format(time.RFC3339),
			},
		})
	}

	switch {
	case meta.IsListType(obj):
		if err := meta.EachListItem(obj, func(item runtime.Object) error {
			if typed, ok := item.(*apiextensions.CustomResourceDefinition); ok {
				addRow(typed)
			}
			return nil
		}); err != nil {
			return nil, err
		}
	default:
		if typed, ok := obj.(*apiextensions.CustomResourceDefinition); ok {
			addRow(typed)
		}
	}

	if l, err := meta.ListAccessor(obj); err == nil {
		table.ResourceVersion = l.GetResourceVersion()
		table.Continue = l.GetContinue()
		table.RemainingItemCount = l.GetRemainingItemCount()
	} else if m, err := meta.CommonAccessor(obj); err == nil {
		table.ResourceVersion = m.GetResourceVersion()
	}

	return table, nil
}
