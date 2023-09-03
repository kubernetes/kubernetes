/*
Copyright 2020 The Kubernetes Authors.

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

package storageversion

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apiserverinternal"
	"k8s.io/kubernetes/pkg/apis/apiserverinternal/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// storageVersionStrategy implements verification logic for StorageVersion.
type storageVersionStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating StorageVersion objects.
var Strategy = storageVersionStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all StorageVersion's need to be cluster scoped
func (storageVersionStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (storageVersionStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"internal.apiserver.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of an StorageVersion before creation.
func (storageVersionStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	sv := obj.(*apiserverinternal.StorageVersion)
	sv.Status = apiserverinternal.StorageVersionStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (storageVersionStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	sv := obj.(*apiserverinternal.StorageVersion)
	sv.Status = old.(*apiserverinternal.StorageVersion).Status
}

// Validate validates a new storageVersion.
func (storageVersionStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	sv := obj.(*apiserverinternal.StorageVersion)
	return validation.ValidateStorageVersion(sv)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (storageVersionStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (storageVersionStrategy) Canonicalize(obj runtime.Object) {
}

// Does not allow creating a StorageVersion object with a PUT request.
func (storageVersionStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (storageVersionStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newStorageVersion := obj.(*apiserverinternal.StorageVersion)
	oldStorageVersion := old.(*apiserverinternal.StorageVersion)
	validationErrorList := validation.ValidateStorageVersionUpdate(newStorageVersion, oldStorageVersion)
	return validationErrorList
}

// WarningsOnUpdate returns warnings for the given update.
func (storageVersionStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for storageVersion objects. Status update should
// only be allowed if version match.
func (storageVersionStrategy) AllowUnconditionalUpdate() bool {
	return false
}

type storageVersionStatusStrategy struct {
	storageVersionStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = storageVersionStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (storageVersionStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"internal.apiserver.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (storageVersionStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newSV := obj.(*apiserverinternal.StorageVersion)
	oldSV := old.(*apiserverinternal.StorageVersion)

	newSV.Spec = oldSV.Spec
	metav1.ResetObjectMetaForStatus(&newSV.ObjectMeta, &oldSV.ObjectMeta)
}

func (storageVersionStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateStorageVersionStatusUpdate(obj.(*apiserverinternal.StorageVersion), old.(*apiserverinternal.StorageVersion))
}

// WarningsOnUpdate returns warnings for the given update.
func (storageVersionStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
