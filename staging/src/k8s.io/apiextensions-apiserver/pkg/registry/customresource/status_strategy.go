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

package customresource

import (
	"context"
	"fmt"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	structurallisttype "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/listtype"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
)

type statusStrategy struct {
	customResourceStrategy
}

func NewStatusStrategy(strategy customResourceStrategy) statusStrategy {
	return statusStrategy{strategy}
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (a statusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		fieldpath.APIVersion(a.customResourceStrategy.kind.GroupVersion().String()): fieldpath.NewSet(
			// Note that if there are other top level fields unique to CRDs,
			// those will also get removed by the apiserver prior to persisting,
			// but won't be added to the resetFields set.

			// This isn't an issue now, but if it becomes an issue in the future
			// we might need a mechanism that is the inverse of resetFields where
			// you specify only the fields to be kept rather than the fields to be wiped
			// that way you could wipe everything but the status in this case.
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (a statusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	// update is only allowed to set status
	newCustomResourceObject := obj.(*unstructured.Unstructured)
	newCustomResource := newCustomResourceObject.UnstructuredContent()
	status, ok := newCustomResource["status"]

	// managedFields must be preserved since it's been modified to
	// track changed fields in the status update.
	managedFields := newCustomResourceObject.GetManagedFields()

	// KCP PATCH START
	// Get the syncer view diff internal annotations
	// before overriding the new object with the old,
	// because we want to update them.
	newObjectSyncerViewDiffAnnotations := getSyncerViewDiffAnnotations(newCustomResourceObject)
	// KCP PATCH END

	// copy old object into new object
	oldCustomResourceObject := old.(*unstructured.Unstructured)
	// overridding the resourceVersion in metadata is safe here, we have already checked that
	// new object and old object have the same resourceVersion.
	*newCustomResourceObject = *oldCustomResourceObject.DeepCopy()

	// set status
	newCustomResourceObject.SetManagedFields(managedFields)

	// KCP PATCH START
	// Update the the syncer view diff annotations even on status update
	updateSyncerViewDiffAnnotations(newCustomResourceObject, newObjectSyncerViewDiffAnnotations)
	// KCP PATCH END

	newCustomResource = newCustomResourceObject.UnstructuredContent()
	if ok {
		newCustomResource["status"] = status
	} else {
		delete(newCustomResource, "status")
	}
}

// ValidateUpdate is the default update validation for an end user updating status.
func (a statusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	uNew, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return field.ErrorList{field.Invalid(field.NewPath(""), obj, fmt.Sprintf("has type %T. Must be a pointer to an Unstructured type", obj))}
	}
	uOld, ok := old.(*unstructured.Unstructured)
	if !ok {
		return field.ErrorList{field.Invalid(field.NewPath(""), old, fmt.Sprintf("has type %T. Must be a pointer to an Unstructured type", old))}
	}

	var errs field.ErrorList
	errs = append(errs, a.customResourceStrategy.validator.ValidateStatusUpdate(ctx, uNew, uOld, a.scale)...)

	// ratcheting validation of x-kubernetes-list-type value map and set
	if newErrs := structurallisttype.ValidateListSetsAndMaps(nil, a.structuralSchema, uNew.Object); len(newErrs) > 0 {
		if oldErrs := structurallisttype.ValidateListSetsAndMaps(nil, a.structuralSchema, uOld.Object); len(oldErrs) == 0 {
			errs = append(errs, newErrs...)
		}
	}

	// validate x-kubernetes-validations rules
	if celValidator := a.customResourceStrategy.celValidator; celValidator != nil {
		if has, err := hasBlockingErr(errs); has {
			errs = append(errs, err)
		} else {
			err, _ := celValidator.Validate(ctx, nil, a.customResourceStrategy.structuralSchema, uNew.Object, uOld.Object, celconfig.RuntimeCELCostBudget)
			errs = append(errs, err...)
		}
	}
	return errs
}

// WarningsOnUpdate returns warnings for the given update.
func (statusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
