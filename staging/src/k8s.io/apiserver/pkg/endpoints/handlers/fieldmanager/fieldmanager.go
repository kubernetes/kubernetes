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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"k8s.io/klog"
	openapiproto "k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/merge"
	"sigs.k8s.io/yaml"
)

var atMostEverySecond = internal.NewAtMostEvery(time.Second)

// FieldManager updates the managed fields and merge applied
// configurations.
type FieldManager struct {
	typeConverter   internal.TypeConverter
	objectConverter runtime.ObjectConvertor
	objectDefaulter runtime.ObjectDefaulter
	groupVersion    schema.GroupVersion
	hubVersion      schema.GroupVersion
	updater         merge.Updater
}

// NewFieldManager creates a new FieldManager that merges apply requests
// and update managed fields for other types of requests.
func NewFieldManager(models openapiproto.Models, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, gv schema.GroupVersion, hub schema.GroupVersion) (*FieldManager, error) {
	typeConverter, err := internal.NewTypeConverter(models, false)
	if err != nil {
		return nil, err
	}

	return &FieldManager{
		typeConverter:   typeConverter,
		objectConverter: objectConverter,
		objectDefaulter: objectDefaulter,
		groupVersion:    gv,
		hubVersion:      hub,
		updater: merge.Updater{
			Converter: internal.NewVersionConverter(typeConverter, objectConverter, hub),
		},
	}, nil
}

// NewCRDFieldManager creates a new FieldManager specifically for
// CRDs. This allows for the possibility of fields which are not defined
// in models, as well as having no models defined at all.
func NewCRDFieldManager(models openapiproto.Models, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, gv schema.GroupVersion, hub schema.GroupVersion, preserveUnknownFields bool) (_ *FieldManager, err error) {
	var typeConverter internal.TypeConverter = internal.DeducedTypeConverter{}
	if models != nil {
		typeConverter, err = internal.NewTypeConverter(models, preserveUnknownFields)
		if err != nil {
			return nil, err
		}
	}
	return &FieldManager{
		typeConverter:   typeConverter,
		objectConverter: objectConverter,
		objectDefaulter: objectDefaulter,
		groupVersion:    gv,
		hubVersion:      hub,
		updater: merge.Updater{
			Converter: internal.NewCRDVersionConverter(typeConverter, objectConverter, hub),
		},
	}, nil
}

// Update is used when the object has already been merged (non-apply
// use-case), and simply updates the managed fields in the output
// object.
func (f *FieldManager) Update(liveObj, newObj runtime.Object, manager string) (runtime.Object, error) {
	// If the object doesn't have metadata, we should just return without trying to
	// set the managedFields at all, so creates/updates/patches will work normally.
	newAccessor, err := meta.Accessor(newObj)
	if err != nil {
		return newObj, nil
	}

	// First try to decode the managed fields provided in the update,
	// This is necessary to allow directly updating managed fields.
	var managed internal.Managed
	if isResetManagedFields(newAccessor.GetManagedFields()) {
		managed = internal.NewEmptyManaged()
	} else if managed, err = internal.DecodeObjectManagedFields(newAccessor.GetManagedFields()); err != nil || len(managed.Fields) == 0 {
		liveAccessor, err := meta.Accessor(liveObj)
		if err != nil {
			return newObj, nil
		}
		// If the managed field is empty or we failed to decode it,
		// let's try the live object. This is to prevent clients who
		// don't understand managedFields from deleting it accidentally.
		if managed, err = internal.DecodeObjectManagedFields(liveAccessor.GetManagedFields()); err != nil {
			managed = internal.NewEmptyManaged()
		}
	}
	// if managed field is still empty, skip updating managed fields altogether
	if len(managed.Fields) == 0 {
		newAccessor.SetManagedFields(nil)
		return newObj, nil
	}
	newObjVersioned, err := f.toVersioned(newObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert new object to proper version: %v", err)
	}
	liveObjVersioned, err := f.toVersioned(liveObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert live object to proper version: %v", err)
	}
	internal.RemoveObjectManagedFields(liveObjVersioned)
	internal.RemoveObjectManagedFields(newObjVersioned)
	newObjTyped, err := f.typeConverter.ObjectToTyped(newObjVersioned)
	if err != nil {
		// Return newObj and just by-pass fields update. This really shouldn't happen.
		klog.Errorf("[SHOULD NOT HAPPEN] failed to create typed new object: %v", err)
		return newObj, nil
	}
	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		// Return newObj and just by-pass fields update. This really shouldn't happen.
		klog.Errorf("[SHOULD NOT HAPPEN] failed to create typed live object: %v", err)
		return newObj, nil
	}
	apiVersion := fieldpath.APIVersion(f.groupVersion.String())

	// TODO(apelisse) use the first return value when unions are implemented
	_, managed.Fields, err = f.updater.Update(liveObjTyped, newObjTyped, apiVersion, managed.Fields, manager)
	if err != nil {
		return nil, fmt.Errorf("failed to update ManagedFields: %v", err)
	}
	managed.Fields = f.stripFields(managed.Fields, manager)

	// If the current operation took any fields from anything, it means the object changed,
	// so update the timestamp of the managedFieldsEntry and merge with any previous updates from the same manager
	if vs, ok := managed.Fields[manager]; ok {
		delete(managed.Fields, manager)

		// Build a manager identifier which will only match previous updates from the same manager
		manager, err = f.buildManagerInfo(manager, metav1.ManagedFieldsOperationUpdate)
		if err != nil {
			return nil, fmt.Errorf("failed to build manager identifier: %v", err)
		}

		managed.Times[manager] = &metav1.Time{Time: time.Now().UTC()}
		if previous, ok := managed.Fields[manager]; ok {
			managed.Fields[manager] = fieldpath.NewVersionedSet(vs.Set().Union(previous.Set()), vs.APIVersion(), vs.Applied())
		} else {
			managed.Fields[manager] = vs
		}
	}

	if err := internal.EncodeObjectManagedFields(newObj, managed); err != nil {
		return nil, fmt.Errorf("failed to encode managed fields: %v", err)
	}

	return newObj, nil
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
// object and update the managed fields.
func (f *FieldManager) Apply(liveObj runtime.Object, patch []byte, fieldManager string, force bool) (runtime.Object, error) {
	// If the object doesn't have metadata, apply isn't allowed.
	accessor, err := meta.Accessor(liveObj)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}
	missingManagedFields := (len(accessor.GetManagedFields()) == 0)

	// Decode the managed fields in the live object, since it isn't allowed in the patch.
	managed, err := internal.DecodeObjectManagedFields(accessor.GetManagedFields())
	if err != nil {
		return nil, fmt.Errorf("failed to decode managed fields: %v", err)
	}
	// Check that the patch object has the same version as the live object
	patchObj := &unstructured.Unstructured{Object: map[string]interface{}{}}

	if err := yaml.Unmarshal(patch, &patchObj.Object); err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("error decoding YAML: %v", err))
	}

	if patchObj.GetManagedFields() != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("metadata.managedFields must be nil"))
	}

	if patchObj.GetAPIVersion() != f.groupVersion.String() {
		return nil,
			errors.NewBadRequest(
				fmt.Sprintf("Incorrect version specified in apply patch. "+
					"Specified patch version: %s, expected: %s",
					patchObj.GetAPIVersion(), f.groupVersion.String()))
	}

	liveObjVersioned, err := f.toVersioned(liveObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert live object to proper version: %v", err)
	}
	internal.RemoveObjectManagedFields(liveObjVersioned)

	patchObjTyped, err := f.typeConverter.YAMLToTyped(patch)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed patch object: %v", err)
	}
	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed live object: %v", err)
	}
	manager, err := f.buildManagerInfo(fieldManager, metav1.ManagedFieldsOperationApply)
	if err != nil {
		return nil, fmt.Errorf("failed to build manager identifier: %v", err)
	}

	apiVersion := fieldpath.APIVersion(f.groupVersion.String())
	// if managed field is missing, create a single entry for all the fields
	if missingManagedFields {
		unknownManager, err := internal.BuildManagerIdentifier(&metav1.ManagedFieldsEntry{
			Manager:    "before-first-apply",
			Operation:  metav1.ManagedFieldsOperationUpdate,
			APIVersion: f.groupVersion.String(),
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create manager for existing fields: %v", err)
		}
		unknownFieldSet, err := liveObjTyped.ToFieldSet()
		if err != nil {
			return nil, fmt.Errorf("failed to create fieldset for existing fields: %v", err)
		}
		managed.Fields[unknownManager] = fieldpath.NewVersionedSet(unknownFieldSet, apiVersion, false)
		f.stripFields(managed.Fields, unknownManager)
	}
	newObjTyped, managedFields, err := f.updater.Apply(liveObjTyped, patchObjTyped, apiVersion, managed.Fields, manager, force)
	if err != nil {
		if conflicts, ok := err.(merge.Conflicts); ok {
			return nil, internal.NewConflictError(conflicts)
		}
		return nil, err
	}
	managed.Fields = f.stripFields(managedFields, manager)

	// Update the time in the managedFieldsEntry for this operation
	managed.Times[manager] = &metav1.Time{Time: time.Now().UTC()}

	newObj, err := f.typeConverter.TypedToObject(newObjTyped)
	if err != nil {
		return nil, fmt.Errorf("failed to convert new typed object to object: %v", err)
	}

	if err := internal.EncodeObjectManagedFields(newObj, managed); err != nil {
		return nil, fmt.Errorf("failed to encode managed fields: %v", err)
	}

	newObjVersioned, err := f.toVersioned(newObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert new object to proper version: %v", err)
	}
	f.objectDefaulter.Default(newObjVersioned)

	newObjUnversioned, err := f.toUnversioned(newObjVersioned)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to unversioned: %v", err)
	}
	return newObjUnversioned, nil
}

func (f *FieldManager) toVersioned(obj runtime.Object) (runtime.Object, error) {
	return f.objectConverter.ConvertToVersion(obj, f.groupVersion)
}

func (f *FieldManager) toUnversioned(obj runtime.Object) (runtime.Object, error) {
	return f.objectConverter.ConvertToVersion(obj, f.hubVersion)
}

func (f *FieldManager) buildManagerInfo(prefix string, operation metav1.ManagedFieldsOperationType) (string, error) {
	managerInfo := metav1.ManagedFieldsEntry{
		Manager:    prefix,
		Operation:  operation,
		APIVersion: f.groupVersion.String(),
	}
	if managerInfo.Manager == "" {
		managerInfo.Manager = "unknown"
	}
	return internal.BuildManagerIdentifier(&managerInfo)
}

// stripSet is the list of fields that should never be part of a mangedFields.
var stripSet = fieldpath.NewSet(
	fieldpath.MakePathOrDie("apiVersion"),
	fieldpath.MakePathOrDie("kind"),
	fieldpath.MakePathOrDie("metadata"),
	fieldpath.MakePathOrDie("metadata", "name"),
	fieldpath.MakePathOrDie("metadata", "namespace"),
	fieldpath.MakePathOrDie("metadata", "creationTimestamp"),
	fieldpath.MakePathOrDie("metadata", "selfLink"),
	fieldpath.MakePathOrDie("metadata", "uid"),
	fieldpath.MakePathOrDie("metadata", "clusterName"),
	fieldpath.MakePathOrDie("metadata", "generation"),
	fieldpath.MakePathOrDie("metadata", "managedFields"),
	fieldpath.MakePathOrDie("metadata", "resourceVersion"),
)

// stripFields removes a predefined set of paths found in typed from managed and returns the updated ManagedFields
func (f *FieldManager) stripFields(managed fieldpath.ManagedFields, manager string) fieldpath.ManagedFields {
	vs, ok := managed[manager]
	if ok {
		if vs == nil {
			panic(fmt.Sprintf("Found unexpected nil manager which should never happen: %s", manager))
		}
		newSet := vs.Set().Difference(stripSet)
		if newSet.Empty() {
			delete(managed, manager)
		} else {
			managed[manager] = fieldpath.NewVersionedSet(newSet, vs.APIVersion(), vs.Applied())
		}
	}

	return managed
}
