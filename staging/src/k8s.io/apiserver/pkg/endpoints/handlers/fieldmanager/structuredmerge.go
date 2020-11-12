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
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

type structuredMergeManager struct {
	typeConverter   TypeConverter
	objectConverter runtime.ObjectConvertor
	objectDefaulter runtime.ObjectDefaulter
	groupVersion    schema.GroupVersion
	hubVersion      schema.GroupVersion
	updater         merge.Updater
	preparator      Preparator
}

var _ Manager = &structuredMergeManager{}

// NewStructuredMergeManager creates a new Manager that merges apply requests
// and update managed fields for other types of requests.
func NewStructuredMergeManager(typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, gv schema.GroupVersion, hub schema.GroupVersion, preparator Preparator) (Manager, error) {
	return &structuredMergeManager{
		typeConverter:   typeConverter,
		objectConverter: objectConverter,
		objectDefaulter: objectDefaulter,
		groupVersion:    gv,
		hubVersion:      hub,
		updater: merge.Updater{
			Converter: newVersionConverter(typeConverter, objectConverter, hub), // This is the converter provided to SMD from k8s
		},
		preparator: preparator,
	}, nil
}

// NewCRDStructuredMergeManager creates a new Manager specifically for
// CRDs. This allows for the possibility of fields which are not defined
// in models, as well as having no models defined at all.
func NewCRDStructuredMergeManager(typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, gv schema.GroupVersion, hub schema.GroupVersion, preparator Preparator) (_ Manager, err error) {
	return &structuredMergeManager{
		typeConverter:   typeConverter,
		objectConverter: objectConverter,
		objectDefaulter: objectDefaulter,
		groupVersion:    gv,
		hubVersion:      hub,
		updater: merge.Updater{
			Converter: newCRDVersionConverter(typeConverter, objectConverter, hub),
		},
		preparator: preparator,
	}, nil
}

// Update implements Manager.
func (f *structuredMergeManager) Update(ctx context.Context, liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
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
		return nil, nil, fmt.Errorf("failed to convert new object (%v) to smd typed: %v", newObjVersioned.GetObjectKind().GroupVersionKind(), err)
	}
	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert live object (%v) to smd typed: %v", liveObjVersioned.GetObjectKind().GroupVersionKind(), err)
	}
	apiVersion := fieldpath.APIVersion(f.groupVersion.String())

	liveManagedFields := managed.Fields()
	// TODO(apelisse) use the first return value when unions are implemented
	_, managedFields, err := f.updater.Update(liveObjTyped, newObjTyped, apiVersion, managed.Fields(), manager)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to update ManagedFields: %v", err)
	}

	if f.preparator != nil {
		preparedPatchTyped, err := f.preparedPatchTyped(ctx, liveObj, newObj)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to get preparedPatchTyped: %v", err)
		}

		managedFields, err = merge.WipeManagedFields(liveManagedFields, managedFields, manager, liveObjTyped, preparedPatchTyped)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to wipe managedFields: %w", err)
		}
	}

	managed = internal.NewManaged(managedFields, managed.Times())

	return newObj, managed, nil
}

// Apply implements Manager.
func (f *structuredMergeManager) Apply(ctx context.Context, liveObj, patchObj runtime.Object, managed Managed, manager string, force bool) (runtime.Object, Managed, error) {
	// Check that the patch object has the same version as the live object
	if patchVersion := patchObj.GetObjectKind().GroupVersionKind().GroupVersion(); patchVersion != f.groupVersion {
		return nil, nil,
			errors.NewBadRequest(
				fmt.Sprintf("Incorrect version specified in apply patch. "+
					"Specified patch version: %s, expected: %s",
					patchVersion, f.groupVersion))
	}

	patchObjMeta, err := meta.Accessor(patchObj)
	if err != nil {
		return nil, nil, fmt.Errorf("couldn't get accessor: %v", err)
	}
	if patchObjMeta.GetManagedFields() != nil {
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

	liveManagedFields := managed.Fields()
	apiVersion := fieldpath.APIVersion(f.groupVersion.String())
	newObjTyped, managedFields, err := f.updater.Apply(liveObjTyped, patchObjTyped, apiVersion, managed.Fields(), manager, force)
	if err != nil {
		return nil, nil, err
	}

	if f.preparator != nil {
		preparedPatchTyped, err := f.preparedPatchTyped(ctx, liveObj, patchObj)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to get preparedPatchTyped: %v", err)
		}

		managedFields, err = merge.WipeManagedFields(liveManagedFields, managedFields, manager, liveObjTyped, preparedPatchTyped)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to wipe managedFields: %w", err)
		}
	}

	managed = internal.NewManaged(managedFields, managed.Times())

	if newObjTyped == nil {
		return nil, managed, nil
	}

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

func (f *structuredMergeManager) preparedPatchTyped(ctx context.Context, liveObj, patchObj runtime.Object) (*typed.TypedValue, error) {
	liveObjUnversioned, err := f.toUnversioned(liveObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert live object to unversioned: %w", err)
	}
	patchObjUnversioned, err := f.toUnversioned(patchObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert patch object to unversioned: %w", err)
	}

	f.preparator.Prepare(ctx, patchObjUnversioned, liveObjUnversioned)

	preparedPatchObjVersioned, err := f.toVersioned(patchObjUnversioned)
	if err != nil {
		return nil, fmt.Errorf("failed to convert prepared patch object to proper version: %w", err)
	}
	preparedPatchObjTyped, err := f.typeConverter.ObjectToTyped(preparedPatchObjVersioned)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed patch object: %v", err)
	}

	return preparedPatchObjTyped, nil
}
