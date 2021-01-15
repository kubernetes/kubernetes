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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"k8s.io/klog/v2"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
)

// DefaultMaxUpdateManagers defines the default maximum retained number of managedFields entries from updates
// if the number of update managers exceeds this, the oldest entries will be merged until the number is below the maximum.
// TODO(jennybuckley): Determine if this is really the best value. Ideally we wouldn't unnecessarily merge too many entries.
const DefaultMaxUpdateManagers int = 10

// DefaultTrackOnCreateProbability defines the default probability that the field management of an object
// starts being tracked from the object's creation, instead of from the first time the object is applied to.
const DefaultTrackOnCreateProbability float32 = 1

var atMostEverySecond = internal.NewAtMostEvery(time.Second)

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
	Apply(liveObj, appliedObj runtime.Object, managed Managed, fieldManager string, force bool) (runtime.Object, Managed, error)
}

// FieldManager updates the managed fields and merge applied
// configurations.
type FieldManager struct {
	fieldManager                         Manager
	ignoreManagedFieldsFromRequestObject bool
}

// NewFieldManager creates a new FieldManager that decodes, manages, then re-encodes managedFields
// on update and apply requests.
func NewFieldManager(f Manager, ignoreManagedFieldsFromRequestObject bool) *FieldManager {
	return &FieldManager{fieldManager: f, ignoreManagedFieldsFromRequestObject: ignoreManagedFieldsFromRequestObject}
}

// NewDefaultFieldManager creates a new FieldManager that merges apply requests
// and update managed fields for other types of requests.
func NewDefaultFieldManager(typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind, hub schema.GroupVersion, ignoreManagedFieldsFromRequestObject bool) (*FieldManager, error) {
	f, err := NewStructuredMergeManager(typeConverter, objectConverter, objectDefaulter, kind.GroupVersion(), hub)
	if err != nil {
		return nil, fmt.Errorf("failed to create field manager: %v", err)
	}
	return newDefaultFieldManager(f, typeConverter, objectConverter, objectCreater, kind, ignoreManagedFieldsFromRequestObject), nil
}

// NewDefaultCRDFieldManager creates a new FieldManager specifically for
// CRDs. This allows for the possibility of fields which are not defined
// in models, as well as having no models defined at all.
func NewDefaultCRDFieldManager(typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind, hub schema.GroupVersion, ignoreManagedFieldsFromRequestObject bool) (_ *FieldManager, err error) {
	f, err := NewCRDStructuredMergeManager(typeConverter, objectConverter, objectDefaulter, kind.GroupVersion(), hub)
	if err != nil {
		return nil, fmt.Errorf("failed to create field manager: %v", err)
	}
	return newDefaultFieldManager(f, typeConverter, objectConverter, objectCreater, kind, ignoreManagedFieldsFromRequestObject), nil
}

// newDefaultFieldManager is a helper function which wraps a Manager with certain default logic.
func newDefaultFieldManager(f Manager, typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind, ignoreManagedFieldsFromRequestObject bool) *FieldManager {
	f = NewStripMetaManager(f)
	f = NewManagedFieldsUpdater(f)
	f = NewBuildManagerInfoManager(f, kind.GroupVersion())
	f = NewCapManagersManager(f, DefaultMaxUpdateManagers)
	f = NewProbabilisticSkipNonAppliedManager(f, objectCreater, kind, DefaultTrackOnCreateProbability)
	f = NewLastAppliedManager(f, typeConverter, objectConverter, kind.GroupVersion())
	f = NewLastAppliedUpdater(f)

	return NewFieldManager(f, ignoreManagedFieldsFromRequestObject)
}

func decodeLiveManagedFields(liveObj runtime.Object) (Managed, error) {
	liveAccessor, err := meta.Accessor(liveObj)
	if err != nil {
		return nil, err
	}
	managed, err := internal.DecodeObjectManagedFields(liveAccessor.GetManagedFields())
	if err != nil {
		return internal.NewEmptyManaged(), nil
	}
	return managed, nil
}

func decodeManagedFields(liveObj, newObj runtime.Object, ignoreManagedFieldsFromRequestObject bool) (Managed, error) {
	// We take the managedFields of the live object in case the request tries to
	// manually set managedFields via a subresource.
	if ignoreManagedFieldsFromRequestObject {
		return decodeLiveManagedFields(liveObj)
	}

	// If the object doesn't have metadata, we should just return without trying to
	// set the managedFields at all, so creates/updates/patches will work normally.
	newAccessor, err := meta.Accessor(newObj)
	if err != nil {
		return nil, err
	}

	if isResetManagedFields(newAccessor.GetManagedFields()) {
		return internal.NewEmptyManaged(), nil
	}

	managed, err := internal.DecodeObjectManagedFields(newAccessor.GetManagedFields())
	// If the managed field is empty or we failed to decode it,
	// let's try the live object. This is to prevent clients who
	// don't understand managedFields from deleting it accidentally.
	if err != nil || len(managed.Fields()) == 0 {
		return decodeLiveManagedFields(liveObj)
	}

	return managed, nil
}

// Update is used when the object has already been merged (non-apply
// use-case), and simply updates the managed fields in the output
// object.
func (f *FieldManager) Update(liveObj, newObj runtime.Object, manager string) (object runtime.Object, err error) {
	// First try to decode the managed fields provided in the update,
	// This is necessary to allow directly updating managed fields.
	managed, err := decodeManagedFields(liveObj, newObj, f.ignoreManagedFieldsFromRequestObject)
	if err != nil {
		return newObj, nil
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

// UpdateNoErrors is the same as Update, but it will not return
// errors. If an error happens, the object is returned with
// managedFields cleared.
func (f *FieldManager) UpdateNoErrors(liveObj, newObj runtime.Object, manager string) runtime.Object {
	obj, err := f.Update(liveObj, newObj, manager)
	if err != nil {
		atMostEverySecond.Do(func() {
			klog.Errorf("[SHOULD NOT HAPPEN] failed to update managedFields for %v: %v",
				newObj.GetObjectKind().GroupVersionKind(),
				err)
		})
		// Explicitly remove managedFields on failure, so that
		// we can't have garbage in it.
		internal.RemoveObjectManagedFields(newObj)
		return newObj
	}
	return obj
}

// Returns true if the managedFields indicate that the user is trying to
// reset the managedFields, i.e. if the list is non-nil but empty, or if
// the list has one empty item.
func isResetManagedFields(managedFields []metav1.ManagedFieldsEntry) bool {
	if len(managedFields) == 0 {
		return managedFields != nil
	}

	if len(managedFields) == 1 {
		return reflect.DeepEqual(managedFields[0], metav1.ManagedFieldsEntry{})
	}

	return false
}

// Apply is used when server-side apply is called, as it merges the
// object and updates the managed fields.
func (f *FieldManager) Apply(liveObj, appliedObj runtime.Object, manager string, force bool) (object runtime.Object, err error) {
	// If the object doesn't have metadata, apply isn't allowed.
	accessor, err := meta.Accessor(liveObj)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}

	// Decode the managed fields in the live object, since it isn't allowed in the patch.
	managed, err := internal.DecodeObjectManagedFields(accessor.GetManagedFields())
	if err != nil {
		return nil, fmt.Errorf("failed to decode managed fields: %v", err)
	}

	internal.RemoveObjectManagedFields(liveObj)

	object, managed, err = f.fieldManager.Apply(liveObj, appliedObj, managed, manager, force)
	if err != nil {
		if conflicts, ok := err.(merge.Conflicts); ok {
			return nil, internal.NewConflictError(conflicts)
		}
		return nil, err
	}

	if err = internal.EncodeObjectManagedFields(object, managed); err != nil {
		return nil, fmt.Errorf("failed to encode managed fields: %v", err)
	}

	return object, nil
}
