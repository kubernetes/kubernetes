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

package fieldmanager

import (
	"fmt"
	"reflect"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	openapiproto "k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/value"
)

// DefaultMaxUpdateManagers defines the default maximum retained number of managedFields entries from updates
// if the number of update managers exceeds this, the oldest entries will be merged until the number is below the maximum.
// TODO(jennybuckley): Determine if this is really the best value. Ideally we wouldn't unnecessarily merge too many entries.
const DefaultMaxUpdateManagers int = 10

// Managed groups a fieldpath.ManagedFields together with the timestamps associated with each operation.
type Managed interface {
	// Fields gets the fieldpath.ManagedFields.
	Fields() fieldpath.ManagedFields

	// Times gets the timestamps associated with each operation.
	Times() map[string]*metav1.Time
}

// Manager updates the managed fields and merges applied configurations.
type Manager interface {
	// Update is used when the object has already been merged (non-apply
	// use-case), and simply updates the managed fields in the output
	// object.
	Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error)

	// Apply is used when server-side apply is called, as it merges the
	// object and updates the managed fields.
	Apply(liveObj runtime.Object, patch []byte, managed Managed, fieldManager string, force bool) (runtime.Object, Managed, error)
}

// FieldManager updates the managed fields and merge applied
// configurations.
type FieldManager struct {
	fieldManager Manager
}

// NewFieldManager creates a new FieldManager that decodes, manages, then re-encodes managedFields
// on update and apply requests.
func NewFieldManager(f Manager) *FieldManager {
	return &FieldManager{f}
}

// NewDefaultFieldManager creates a new FieldManager that merges apply requests
// and update managed fields for other types of requests.
func NewDefaultFieldManager(models openapiproto.Models, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind, hub schema.GroupVersion) (*FieldManager, error) {
	f, err := NewStructuredMergeManager(models, objectConverter, objectDefaulter, kind.GroupVersion(), hub)
	if err != nil {
		return nil, fmt.Errorf("failed to create field manager: %v", err)
	}
	return newDefaultFieldManager(f, objectCreater, kind), nil
}

// NewDefaultCRDFieldManager creates a new FieldManager specifically for
// CRDs. This allows for the possibility of fields which are not defined
// in models, as well as having no models defined at all.
func NewDefaultCRDFieldManager(models openapiproto.Models, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind, hub schema.GroupVersion, preserveUnknownFields bool) (_ *FieldManager, err error) {
	f, err := NewCRDStructuredMergeManager(models, objectConverter, objectDefaulter, kind.GroupVersion(), hub, preserveUnknownFields)
	if err != nil {
		return nil, fmt.Errorf("failed to create field manager: %v", err)
	}
	return newDefaultFieldManager(f, objectCreater, kind), nil
}

// newDefaultFieldManager is a helper function which wraps a Manager with certain default logic.
func newDefaultFieldManager(f Manager, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind) *FieldManager {
	f = NewStripMetaManager(f)
	f = NewBuildManagerInfoManager(f, kind.GroupVersion())
	f = NewCapManagersManager(f, DefaultMaxUpdateManagers)
	// DO NOT MERGE -- this is to test the performance impact of maintaining field managers on every modifying request.
	// f = NewSkipNonAppliedManager(f, objectCreater, kind)
	return NewFieldManager(f)
}

// Update is used when the object has already been merged (non-apply
// use-case), and simply updates the managed fields in the output
// object.
func (f *FieldManager) Update(liveObj, newObj runtime.Object, manager string) (object runtime.Object, err error) {
	// If the object doesn't have metadata, we should just return without trying to
	// set the managedFields at all, so creates/updates/patches will work normally.
	if _, err = meta.Accessor(newObj); err != nil {
		return newObj, nil
	}

	// First try to decode the managed fields provided in the update,
	// This is necessary to allow directly updating managed fields.
	var managed Managed
	if managed, err = internal.DecodeObjectManagedFields(newObj); err != nil || len(managed.Fields()) == 0 {
		// If the managed field is empty or we failed to decode it,
		// let's try the live object. This is to prevent clients who
		// don't understand managedFields from deleting it accidentally.
		managed, err = internal.DecodeObjectManagedFields(liveObj)
		if err != nil {
			return nil, fmt.Errorf("failed to decode managed fields: %v", err)
		}
	}

	internal.RemoveObjectManagedFields(liveObj)
	internal.RemoveObjectManagedFields(newObj)

	if object, managed, err = f.fieldManager.Update(liveObj, newObj, managed, manager); err != nil {
		return nil, err
	}

	if err = internal.EncodeObjectManagedFields(object, managed); err != nil {
		return nil, fmt.Errorf("failed to encode managed fields: %v", err)
	}

	return object, nil
}

// Apply is used when server-side apply is called, as it merges the
// object and updates the managed fields.
func (f *FieldManager) Apply(liveObj runtime.Object, patch []byte, manager string, force bool) (object runtime.Object, err error) {
	// If the object doesn't have metadata, apply isn't allowed.
	if _, err = meta.Accessor(liveObj); err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}

	// Decode the managed fields in the live object, since it isn't allowed in the patch.
	var managed Managed
	if managed, err = internal.DecodeObjectManagedFields(liveObj); err != nil {
		return nil, fmt.Errorf("failed to decode managed fields: %v", err)
	}

	internal.RemoveObjectManagedFields(liveObj)

	if object, managed, err = f.fieldManager.Apply(liveObj, patch, managed, manager, force); err != nil {
		return nil, err
	}

	if err = internal.EncodeObjectManagedFields(object, managed); err != nil {
		return nil, fmt.Errorf("failed to encode managed fields: %v", err)
	}

	return object, nil
}

// TODO(jpbetz): For experimenting only. If we provide conversions, we'll need a cleaner way to do it.
func init() {
	value.CustomTypeConverters[reflect.TypeOf(metav1.Time{})] = timeConverter{}
	value.CustomTypeConverters[reflect.TypeOf(&metav1.Time{})] = timeConverter{}
	value.CustomTypeConverters[reflect.TypeOf(resource.Quantity{})] = quantityConverter{}
	value.CustomTypeConverters[reflect.TypeOf(&resource.Quantity{})] = quantityConverter{}
	// TODO: IntOrString, &JSONSchemaPropsOrArray what else?
}

type timeConverter struct{}

func (timeConverter) IsNull(v reflect.Value) bool {
	t, ok := v.Interface().(metav1.Time)
	if !ok {
		panic("value is not a metav1.Time")
	}
	return t.IsZero()
}

func (timeConverter) ToString(v reflect.Value) string {
	t, ok := v.Interface().(metav1.Time)
	if !ok {
		panic("value is not a metav1.Time")
	}
	if t.IsZero() {
		panic("value is null, not string")
	}
	buf := make([]byte, 0, len(time.RFC3339))
	// time cannot contain non escapable JSON characters
	buf = t.UTC().AppendFormat(buf, time.RFC3339)
	return string(buf)
}

// func (_ timeConverter) Equal(lhs reflect.Value, rhs reflect.Value) bool {
// 	lhsTime, ok := lhs.Interface().(metav1.Time)
// 	if !ok {
// 		panic("Expected lhs value to be of type metav1.Time")
// 	}
// 	rhsInterface := rhs.Interface()
// 	if rhsInterface == nil {
// 		return false
// 	}
// 	rhsTime, ok := rhsInterface.(metav1.Time)
// 	if !ok {
// 		return false
// 	}
// 	return lhsTime.Equal(&rhsTime)
// }

type quantityConverter struct{}

func (quantityConverter) IsNull(v reflect.Value) bool {
	return false
}

func (quantityConverter) ToString(v reflect.Value) string {
	quantity, ok := v.Interface().(resource.Quantity)
	if !ok {
		panic("value is not a resource.Quantity")
	}
	return quantity.String()
}

// func (_ quantityConverter) Equal(lhs reflect.Value, rhs reflect.Value) bool {
// 	lhsTime, ok := lhs.Interface().(resource.Quantity)
// 	if !ok {
// 		panic("Expected lhs value to be of type time.Time")
// 	}
// 	rhsInterface := rhs.Interface()
// 	if rhsInterface == nil {
// 		return false
// 	}
// 	rhsTime, ok := rhsInterface.(resource.Quantity)
// 	if !ok {
// 		return false
// 	}
// 	return lhsTime.Equal(rhsTime)
// }
