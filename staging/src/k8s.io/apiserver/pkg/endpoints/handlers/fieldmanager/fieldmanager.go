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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	openapiproto "k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/merge"
)

const applyManager = "apply"

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
	typeConverter, err := internal.NewTypeConverter(models)
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

// Update is used when the object has already been merged (non-apply
// use-case), and simply updates the managed fields in the output
// object.
func (f *FieldManager) Update(liveObj, newObj runtime.Object, manager string) (runtime.Object, error) {
	managed, err := internal.DecodeObjectManagedFields(newObj)
	// If the managed field is empty or we failed to decode it,
	// let's try the live object
	if err != nil || len(managed) == 0 {
		managed, err = internal.DecodeObjectManagedFields(liveObj)
		if err != nil {
			return nil, fmt.Errorf("failed to decode managed fields: %v", err)
		}
	}
	newObjVersioned, err := f.toVersioned(newObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert new object to proper version: %v", err)
	}
	liveObjVersioned, err := f.toVersioned(liveObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert live object to proper version: %v", err)
	}
	if err := internal.RemoveObjectManagedFields(liveObjVersioned); err != nil {
		return nil, fmt.Errorf("failed to remove managed fields from live obj: %v", err)
	}
	if err := internal.RemoveObjectManagedFields(newObjVersioned); err != nil {
		return nil, fmt.Errorf("failed to remove managed fields from new obj: %v", err)
	}

	newObjTyped, err := f.typeConverter.ObjectToTyped(newObjVersioned)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed new object: %v", err)
	}
	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed live object: %v", err)
	}
	apiVersion := fieldpath.APIVersion(f.groupVersion.String())
	managed, err = f.updater.Update(liveObjTyped, newObjTyped, apiVersion, managed, manager)
	if err != nil {
		return nil, fmt.Errorf("failed to update ManagedFields: %v", err)
	}

	if err := internal.EncodeObjectManagedFields(newObj, managed); err != nil {
		return nil, fmt.Errorf("failed to encode managed fields: %v", err)
	}

	return newObj, nil
}

// Apply is used when server-side apply is called, as it merges the
// object and update the managed fields.
func (f *FieldManager) Apply(liveObj runtime.Object, patch []byte, force bool) (runtime.Object, error) {
	managed, err := internal.DecodeObjectManagedFields(liveObj)
	if err != nil {
		return nil, fmt.Errorf("failed to decode managed fields: %v", err)
	}
	// We can assume that patchObj is already on the proper version:
	// it shouldn't have to be converted so that it's not defaulted.
	liveObjVersioned, err := f.toVersioned(liveObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert live object to proper version: %v", err)
	}
	if err := internal.RemoveObjectManagedFields(liveObjVersioned); err != nil {
		return nil, fmt.Errorf("failed to remove managed fields from live obj: %v", err)
	}

	patchObjTyped, err := f.typeConverter.YAMLToTyped(patch)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed patch object: %v", err)
	}
	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed live object: %v", err)
	}
	apiVersion := fieldpath.APIVersion(f.groupVersion.String())
	newObjTyped, managed, err := f.updater.Apply(liveObjTyped, patchObjTyped, apiVersion, managed, applyManager, force)
	if err != nil {
		if conflicts, ok := err.(merge.Conflicts); ok {
			return nil, errors.NewApplyConflict(conflicts)
		}
		return nil, err
	}

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
