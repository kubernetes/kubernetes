/*
Copyright 2019 The Kubernetes Authors.

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
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
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

type structuredMergeManager struct {
	typeConverter   internal.TypeConverter
	objectConverter runtime.ObjectConvertor
	objectDefaulter runtime.ObjectDefaulter
	groupVersion    schema.GroupVersion
	hubVersion      schema.GroupVersion
	updater         merge.Updater
}

var _ Manager = &structuredMergeManager{}

// NewStructuredMergeManager creates a new Manager that merges apply requests
// and update managed fields for other types of requests.
func NewStructuredMergeManager(models openapiproto.Models, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, gv schema.GroupVersion, hub schema.GroupVersion) (Manager, error) {
	typeConverter, err := internal.NewTypeConverter(models, false)
	if err != nil {
		return nil, err
	}

	return &structuredMergeManager{
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

// NewCRDStructuredMergeManager creates a new Manager specifically for
// CRDs. This allows for the possibility of fields which are not defined
// in models, as well as having no models defined at all.
func NewCRDStructuredMergeManager(models openapiproto.Models, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, gv schema.GroupVersion, hub schema.GroupVersion, preserveUnknownFields bool) (_ Manager, err error) {
	var typeConverter internal.TypeConverter = internal.DeducedTypeConverter{}
	if models != nil {
		typeConverter, err = internal.NewTypeConverter(models, preserveUnknownFields)
		if err != nil {
			return nil, err
		}
	}
	return &structuredMergeManager{
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

// Update implements Manager.
func (f *structuredMergeManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	newObjVersioned, err := f.toVersioned(newObj)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert new object to proper version: %v", err)
	}
	liveObjVersioned, err := f.toVersioned(liveObj)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert live object to proper version: %v", err)
	}
	newObjTyped, err := f.typeConverter.ObjectToTyped(newObjVersioned)
	if err != nil {
		// Return newObj and just by-pass fields update. This really shouldn't happen.
		klog.Errorf("[SHOULD NOT HAPPEN] failed to create typed new object: %v", err)
		return newObj, managed, nil
	}
	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		// Return newObj and just by-pass fields update. This really shouldn't happen.
		klog.Errorf("[SHOULD NOT HAPPEN] failed to create typed live object: %v", err)
		return newObj, managed, nil
	}
	apiVersion := fieldpath.APIVersion(f.groupVersion.String())

	// TODO(apelisse) use the first return value when unions are implemented
	self := "current-operation"
	_, managedFields, err := f.updater.Update(liveObjTyped, newObjTyped, apiVersion, managed.Fields(), self)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to update ManagedFields: %v", err)
	}
	managed = internal.NewManaged(managedFields, managed.Times())

	// If the current operation took any fields from anything, it means the object changed,
	// so update the timestamp of the managedFieldsEntry and merge with any previous updates from the same manager
	if vs, ok := managed.Fields()[self]; ok {
		delete(managed.Fields(), self)

		managed.Times()[manager] = &metav1.Time{Time: time.Now().UTC()}
		if previous, ok := managed.Fields()[manager]; ok {
			managed.Fields()[manager] = fieldpath.NewVersionedSet(vs.Set().Union(previous.Set()), vs.APIVersion(), vs.Applied())
		} else {
			managed.Fields()[manager] = vs
		}
	}

	return newObj, managed, nil
}

// Apply implements Manager.
func (f *structuredMergeManager) Apply(liveObj runtime.Object, patch []byte, managed Managed, manager string, force bool) (runtime.Object, Managed, error) {
	patchObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal(patch, &patchObj.Object); err != nil {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("error decoding YAML: %v", err))
	}

	// Check that the patch object has the same version as the live object
	if patchObj.GetAPIVersion() != f.groupVersion.String() {
		return nil, nil,
			errors.NewBadRequest(
				fmt.Sprintf("Incorrect version specified in apply patch. "+
					"Specified patch version: %s, expected: %s",
					patchObj.GetAPIVersion(), f.groupVersion.String()))
	}

	if patchObj.GetManagedFields() != nil {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("metadata.managedFields must be nil"))
	}

	liveObjVersioned, err := f.toVersioned(liveObj)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert live object to proper version: %v", err)
	}

	patchObjTyped, err := f.typeConverter.ObjectToTyped(patchObj)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create typed patch object: %v", err)
	}
	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create typed live object: %v", err)
	}

	apiVersion := fieldpath.APIVersion(f.groupVersion.String())
	newObjTyped, managedFields, err := f.updater.Apply(liveObjTyped, patchObjTyped, apiVersion, managed.Fields(), manager, force)
	if err != nil {
		if conflicts, ok := err.(merge.Conflicts); ok {
			return nil, nil, internal.NewConflictError(conflicts)
		}
		return nil, nil, err
	}
	managed = internal.NewManaged(managedFields, managed.Times())

	// Update the time in the managedFieldsEntry for this operation
	managed.Times()[manager] = &metav1.Time{Time: time.Now().UTC()}

	newObj, err := f.typeConverter.TypedToObject(newObjTyped)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert new typed object to object: %v", err)
	}

	newObjVersioned, err := f.toVersioned(newObj)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert new object to proper version: %v", err)
	}
	f.objectDefaulter.Default(newObjVersioned)

	newObjUnversioned, err := f.toUnversioned(newObjVersioned)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert to unversioned: %v", err)
	}
	return newObjUnversioned, managed, nil
}

func (f *structuredMergeManager) toVersioned(obj runtime.Object) (runtime.Object, error) {
	return f.objectConverter.ConvertToVersion(obj, f.groupVersion)
}

func (f *structuredMergeManager) toUnversioned(obj runtime.Object) (runtime.Object, error) {
	return f.objectConverter.ConvertToVersion(obj, f.hubVersion)
}
