/*
Copyright 2014 The Kubernetes Authors.

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

package resourcequota

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/core"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// resourcequotaStrategy implements behavior for ResourceQuota objects
type resourcequotaStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating ResourceQuota
// objects via the REST API.
var Strategy = resourcequotaStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is true for resourcequotas.
func (resourcequotaStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (resourcequotaStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (resourcequotaStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	resourcequota := obj.(*api.ResourceQuota)
	resourcequota.Status = api.ResourceQuotaStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (resourcequotaStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newResourcequota := obj.(*api.ResourceQuota)
	oldResourcequota := old.(*api.ResourceQuota)
	newResourcequota.Status = oldResourcequota.Status
}

// Validate validates a new resourcequota.
func (resourcequotaStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	resourcequota := obj.(*api.ResourceQuota)
	opts := getValidationOptionsFromResourceQuota(resourcequota, nil)
	return validation.ValidateResourceQuota(resourcequota, opts)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (resourcequotaStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (resourcequotaStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for resourcequotas.
func (resourcequotaStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (resourcequotaStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newObj, oldObj := obj.(*api.ResourceQuota), old.(*api.ResourceQuota)
	opts := getValidationOptionsFromResourceQuota(newObj, oldObj)
	return validation.ValidateResourceQuotaUpdate(newObj, oldObj, opts)
}

// WarningsOnUpdate returns warnings for the given update.
func (resourcequotaStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (resourcequotaStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type resourcequotaStatusStrategy struct {
	resourcequotaStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = resourcequotaStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (resourcequotaStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (resourcequotaStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newResourcequota := obj.(*api.ResourceQuota)
	oldResourcequota := old.(*api.ResourceQuota)
	newResourcequota.Spec = oldResourcequota.Spec
}

func (resourcequotaStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateResourceQuotaStatusUpdate(obj.(*api.ResourceQuota), old.(*api.ResourceQuota))
}

// WarningsOnUpdate returns warnings for the given update.
func (resourcequotaStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func getValidationOptionsFromResourceQuota(newObj *api.ResourceQuota, oldObj *api.ResourceQuota) validation.ResourceQuotaValidationOptions {
	opts := validation.ResourceQuotaValidationOptions{
		AllowPodAffinityNamespaceSelector:  utilfeature.DefaultFeatureGate.Enabled(features.PodAffinityNamespaceSelector),
		AllowEphemeralStorageInScopedQuota: false,
	}

	if oldObj == nil {
		return opts
	}

	opts.AllowPodAffinityNamespaceSelector = opts.AllowPodAffinityNamespaceSelector || hasCrossNamespacePodAffinityScope(&oldObj.Spec)
	opts.AllowEphemeralStorageInScopedQuota = hasEphemeralStorage(&oldObj.Spec)
	return opts
}

func hasEphemeralStorage(spec *api.ResourceQuotaSpec) bool {
	if spec == nil {
		return false
	}

	for resourceName := range spec.Hard {
		switch resourceName {
		case core.ResourceEphemeralStorage, core.ResourceLimitsEphemeralStorage, core.ResourceRequestsEphemeralStorage:
			return true
		}
	}
	return false
}

func hasCrossNamespacePodAffinityScope(spec *api.ResourceQuotaSpec) bool {
	if spec == nil {
		return false
	}
	for _, scope := range spec.Scopes {
		if scope == api.ResourceQuotaScopeCrossNamespacePodAffinity {
			return true
		}
	}

	if spec.ScopeSelector == nil {
		return false
	}
	for _, req := range spec.ScopeSelector.MatchExpressions {
		if req.ScopeName == api.ResourceQuotaScopeCrossNamespacePodAffinity {
			return true
		}
	}
	return false
}
