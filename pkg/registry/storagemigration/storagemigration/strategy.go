/*
Copyright 2024 The Kubernetes Authors.

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

// Package storagemigration provides Registry interface and its RESTStorage
// implementation for storing StorageVersionMigration objects.
package storagemigration

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storagemigration"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	svmvalidation "k8s.io/kubernetes/pkg/apis/storagemigration/validation"
)

// strategy implements behavior for ClusterTrustBundles.
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the create, update, and delete strategy for ClusterTrustBundles.
var Strategy = strategy{legacyscheme.Scheme, names.SimpleNameGenerator}

var _ rest.RESTCreateStrategy = Strategy
var _ rest.RESTUpdateStrategy = Strategy
var _ rest.RESTDeleteStrategy = Strategy

func (strategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (strategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"storagemigration.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of an StorageVersion before creation.
func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	svm := obj.(*storagemigration.StorageVersionMigration)
	svm.Status = storagemigration.StorageVersionMigrationStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (s strategy) PrepareForUpdate(ctx context.Context, new, old runtime.Object) {
	svm := new.(*storagemigration.StorageVersionMigration)
	svm.Status = old.(*storagemigration.StorageVersionMigration).Status
}

func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	bundle := obj.(*storagemigration.StorageVersionMigration)
	return svmvalidation.ValidateStorageVersionMigration(bundle)
}

func (strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (strategy) Canonicalize(obj runtime.Object) {}

func (strategy) AllowCreateOnUpdate() bool {
	return false
}

func (s strategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	newBundle := new.(*storagemigration.StorageVersionMigration)
	oldBundle := old.(*storagemigration.StorageVersionMigration)
	return svmvalidation.ValidateStorageVersionMigrationUpdate(newBundle, oldBundle)
}

func (strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (strategy) AllowUnconditionalUpdate() bool {
	return false
}

type statusStrategy struct {
	strategy
}

var StatusStrategy = statusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (statusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"storagemigration.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (statusStrategy) PrepareForUpdate(ctx context.Context, new, old runtime.Object) {
	newBundle := new.(*storagemigration.StorageVersionMigration)
	oldBundle := old.(*storagemigration.StorageVersionMigration)

	newBundle.Spec = oldBundle.Spec
	metav1.ResetObjectMetaForStatus(&newBundle.ObjectMeta, &oldBundle.ObjectMeta)
}

func (s statusStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	newSVM := new.(*storagemigration.StorageVersionMigration)
	oldSVM := old.(*storagemigration.StorageVersionMigration)

	return svmvalidation.ValidateStorageVersionMigrationStatusUpdate(newSVM, oldSVM)
}

// WarningsOnUpdate returns warnings for the given update.
func (statusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
