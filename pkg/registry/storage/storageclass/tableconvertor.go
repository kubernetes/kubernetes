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

package storageclass

import (
	"context"
	"fmt"
	"sort"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
)

// TableConvertor implements rest.TableConvertor for StorageClass.
// It properly handles effective default detection for both single and list queries.
// When multiple StorageClasses have the default annotation, only the most recently
// created one (or alphabetically first if timestamps are equal) is shown as "(default)".
type TableConvertor struct {
	tableGenerator printers.TableGenerator
	lister         Lister
}

// Lister is an interface for listing StorageClasses
type Lister interface {
	List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error)
}

// NewTableConvertor creates a new TableConvertor for StorageClass.
// The lister can be set later using SetLister if not available at creation time.
func NewTableConvertor() *TableConvertor {
	return &TableConvertor{
		tableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers),
	}
}

// SetLister sets the Lister used to list all StorageClasses for effective default detection.
// This must be called before the TableConvertor is used for single object queries.
func (c *TableConvertor) SetLister(lister Lister) {
	c.lister = lister
}

// ConvertToTable converts StorageClass objects to a Table with proper effective default handling
func (c *TableConvertor) ConvertToTable(ctx context.Context, obj runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	noHeaders := false
	if tableOptions != nil {
		switch t := tableOptions.(type) {
		case *metav1.TableOptions:
			if t != nil {
				noHeaders = t.NoHeaders
			}
		default:
			return nil, fmt.Errorf("unrecognized type %T for table options, can't display tabular output", tableOptions)
		}
	}

	// Determine the effective default StorageClass
	effectiveDefault, err := c.getEffectiveDefault(ctx, obj)
	if err != nil {
		// If we can't determine effective default, fall back to standard behavior
		return c.tableGenerator.GenerateTable(obj, printers.GenerateOptions{Wide: true, NoHeaders: noHeaders})
	}

	// Generate table with effective default information
	return c.generateTableWithEffectiveDefault(obj, effectiveDefault, noHeaders)
}

// getEffectiveDefault determines which StorageClass is the effective default
func (c *TableConvertor) getEffectiveDefault(ctx context.Context, obj runtime.Object) (string, error) {
	var allStorageClasses []storageapi.StorageClass

	switch t := obj.(type) {
	case *storageapi.StorageClass:
		// For single object query, we need to list all StorageClasses to determine effective default
		if c.lister != nil {
			listObj, err := c.lister.List(ctx, &metainternalversion.ListOptions{})
			if err != nil {
				return "", err
			}
			list, ok := listObj.(*storageapi.StorageClassList)
			if !ok {
				return "", fmt.Errorf("unexpected list type: %T", listObj)
			}
			allStorageClasses = list.Items
		} else {
			// Fallback: if no lister, use annotation on the single object
			if storageutil.IsDefaultAnnotation(t.ObjectMeta) {
				return t.Name, nil
			}
			return "", nil
		}
	case *storageapi.StorageClassList:
		allStorageClasses = t.Items
	default:
		return "", fmt.Errorf("unexpected object type: %T", obj)
	}

	// Find all StorageClasses with default annotation
	var defaultClasses []*storageapi.StorageClass
	for i := range allStorageClasses {
		if storageutil.IsDefaultAnnotation(allStorageClasses[i].ObjectMeta) {
			defaultClasses = append(defaultClasses, &allStorageClasses[i])
		}
	}

	if len(defaultClasses) == 0 {
		return "", nil
	}

	// Sort by creation timestamp (newest first), then by name (ascending) for tie-breaking
	sort.Slice(defaultClasses, func(i, j int) bool {
		if defaultClasses[i].CreationTimestamp.UnixNano() == defaultClasses[j].CreationTimestamp.UnixNano() {
			return defaultClasses[i].Name < defaultClasses[j].Name
		}
		return defaultClasses[i].CreationTimestamp.UnixNano() > defaultClasses[j].CreationTimestamp.UnixNano()
	})

	return defaultClasses[0].Name, nil
}

// generateTableWithEffectiveDefault generates a table with the effective default marked
func (c *TableConvertor) generateTableWithEffectiveDefault(obj runtime.Object, effectiveDefault string, noHeaders bool) (*metav1.Table, error) {
	// Use the tableGenerator to get base table structure
	table, err := c.tableGenerator.GenerateTable(obj, printers.GenerateOptions{Wide: true, NoHeaders: noHeaders})
	if err != nil {
		return nil, err
	}

	// Update the Name column to reflect effective default status
	for i := range table.Rows {
		if len(table.Rows[i].Cells) > 0 {
			name, ok := table.Rows[i].Cells[0].(string)
			if ok {
				// Remove any existing (default) suffix first
				cleanName := removeDefaultSuffix(name)
				// Add (default) only if this is the effective default
				if cleanName == effectiveDefault {
					table.Rows[i].Cells[0] = cleanName + " (default)"
				} else {
					table.Rows[i].Cells[0] = cleanName
				}
			}
		}
	}

	return table, nil
}

// removeDefaultSuffix removes the " (default)" suffix from a name if present
func removeDefaultSuffix(name string) string {
	const suffix = " (default)"
	if len(name) > len(suffix) && name[len(name)-len(suffix):] == suffix {
		return name[:len(name)-len(suffix)]
	}
	return name
}
