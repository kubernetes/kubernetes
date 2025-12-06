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
			{Name: "Scope", Type: "string", Description: "Cluster/Namespaced"},
			{Name: "Versions", Type: "string", Description: "Served versions"},
			{Name: "Created At", Type: "date", Description: metaDocs["creationTimestamp"]},
			{Name: "Group", Type: "string", Priority: 1, Description: "API group"},
			{Name: "Kind", Type: "string", Priority: 1, Description: "CustomResource kind"},
			{Name: "ShortNames", Type: "string", Priority: 1, Description: "Short names"},
			{Name: "Established", Type: "boolean", Priority: 1, Description: "Established status"},
		}
	}

	addRow := func(cr *apiextensions.CustomResourceDefinition) {
		// versions: only served=true
		var versions []string
		for _, v := range cr.Spec.Versions {
			if v.Served {
				label := v.Name
				if v.Storage {
					label += "(storage)"
				}
				versions = append(versions, label)
			}
		}
		sort.Strings(versions)

		shortNames := strings.Join(cr.Spec.Names.ShortNames, ",")

		table.Rows = append(table.Rows, metav1.TableRow{
			Object: runtime.RawExtension{Object: cr},
			Cells: []interface{}{
				cr.Name,
				string(cr.Spec.Scope),
				strings.Join(versions, ","),
				cr.CreationTimestamp.Time.UTC().Format(time.RFC3339),
				cr.Spec.Group,
				cr.Spec.Names.Kind,
				shortNames,
				isEstablished(cr),
			},
		})
	}

	switch {
	case meta.IsListType(obj):
		_ = meta.EachListItem(obj, func(item runtime.Object) error {
			if cr, ok := item.(*apiextensions.CustomResourceDefinition); ok {
				addRow(cr)
			}
			return nil
		})
	default:
		if cr, ok := obj.(*apiextensions.CustomResourceDefinition); ok {
			addRow(cr)
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

// isEstablished returns true if the CRD has the Established condition with Status=True.
func isEstablished(crd *apiextensions.CustomResourceDefinition) bool {
	for _, cond := range crd.Status.Conditions {
		if cond.Type == apiextensions.Established && cond.Status == apiextensions.ConditionTrue {
			return true
		}
	}
	return false
}
