/*
Copyright 2022 The Kubernetes Authors.

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

package noderesourceslice

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
)

// nodeResourceSliceStrategy implements behavior for NodeResourceSlice objects
type nodeResourceSliceStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var Strategy = nodeResourceSliceStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (nodeResourceSliceStrategy) NamespaceScoped() bool {
	return false
}

func (nodeResourceSliceStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (nodeResourceSliceStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	slice := obj.(*resource.NodeResourceSlice)
	return validation.ValidateNodeResourceSlice(slice)
}

func (nodeResourceSliceStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (nodeResourceSliceStrategy) Canonicalize(obj runtime.Object) {
}

func (nodeResourceSliceStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (nodeResourceSliceStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (nodeResourceSliceStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateNodeResourceSliceUpdate(obj.(*resource.NodeResourceSlice), old.(*resource.NodeResourceSlice))
}

func (nodeResourceSliceStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (nodeResourceSliceStrategy) AllowUnconditionalUpdate() bool {
	return true
}

var TriggerFunc = map[string]storage.IndexerFunc{
	// Only one index is supported:
	// https://github.com/kubernetes/kubernetes/blob/3aa8c59fec0bf339e67ca80ea7905c817baeca85/staging/src/k8s.io/apiserver/pkg/storage/cacher/cacher.go#L346-L350
	"nodeName": nodeNameTriggerFunc,
}

func nodeNameTriggerFunc(obj runtime.Object) string {
	return obj.(*resource.NodeResourceSlice).NodeName
}

// Indexers returns the indexers for NodeResourceSlice.
func Indexers() *cache.Indexers {
	return &cache.Indexers{
		storage.FieldIndex("nodeName"): nodeNameIndexFunc,
	}
}

func nodeNameIndexFunc(obj interface{}) ([]string, error) {
	slice, ok := obj.(*resource.NodeResourceSlice)
	if !ok {
		return nil, fmt.Errorf("not a NodeResourceSlice")
	}
	return []string{slice.NodeName}, nil
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	slice, ok := obj.(*resource.NodeResourceSlice)
	if !ok {
		return nil, nil, fmt.Errorf("not a NodeResourceSlice")
	}
	return labels.Set(slice.ObjectMeta.Labels), toSelectableFields(slice), nil
}

// Match returns a generic matcher for a given label and field selector.
func Match(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:       label,
		Field:       field,
		GetAttrs:    GetAttrs,
		IndexFields: []string{"nodeName"},
	}
}

// toSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func toSelectableFields(slice *resource.NodeResourceSlice) fields.Set {
	// The purpose of allocation with a given number of elements is to reduce
	// amount of allocations needed to create the fields.Set. If you add any
	// field here or the number of object-meta related fields changes, this should
	// be adjusted.
	fields := make(fields.Set, 3)
	fields["nodeName"] = slice.NodeName
	fields["driverName"] = slice.DriverName

	// Adds one field.
	return generic.AddObjectMetaFieldsSet(fields, &slice.ObjectMeta, false)
}
