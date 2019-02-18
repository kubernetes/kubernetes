/*
Copyright 2017 The Kubernetes Authors.

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

package customresourcedefinition

import (
	"context"
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// strategy implements behavior for CustomResources.
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func NewStrategy(typer runtime.ObjectTyper) strategy {
	return strategy{typer, names.SimpleNameGenerator}
}

func (strategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of a CustomResourceDefinition before creation.
func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	crd := obj.(*apiextensions.CustomResourceDefinition)
	crd.Status = apiextensions.CustomResourceDefinitionStatus{}
	crd.Generation = 1

	dropDisabledFields(&crd.Spec, nil)

	for _, v := range crd.Spec.Versions {
		if v.Storage {
			if !apiextensions.IsStoredVersion(crd, v.Name) {
				crd.Status.StoredVersions = append(crd.Status.StoredVersions, v.Name)
			}
			break
		}
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (strategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCRD := obj.(*apiextensions.CustomResourceDefinition)
	oldCRD := old.(*apiextensions.CustomResourceDefinition)
	newCRD.Status = oldCRD.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the controller has *seen*.
	if !apiequality.Semantic.DeepEqual(oldCRD.Spec, newCRD.Spec) {
		newCRD.Generation = oldCRD.Generation + 1
	}

	dropDisabledFields(&newCRD.Spec, &oldCRD.Spec)

	for _, v := range newCRD.Spec.Versions {
		if v.Storage {
			if !apiextensions.IsStoredVersion(newCRD, v.Name) {
				newCRD.Status.StoredVersions = append(newCRD.Status.StoredVersions, v.Name)
			}
			break
		}
	}
}

// Validate validates a new CustomResourceDefinition.
func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateCustomResourceDefinition(obj.(*apiextensions.CustomResourceDefinition))
}

// AllowCreateOnUpdate is false for CustomResourceDefinition; this means a POST is
// needed to create one.
func (strategy) AllowCreateOnUpdate() bool {
	return false
}

// AllowUnconditionalUpdate is the default update policy for CustomResourceDefinition objects.
func (strategy) AllowUnconditionalUpdate() bool {
	return false
}

// Canonicalize normalizes the object after validation.
func (strategy) Canonicalize(obj runtime.Object) {
}

// ValidateUpdate is the default update validation for an end user updating status.
func (strategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateCustomResourceDefinitionUpdate(obj.(*apiextensions.CustomResourceDefinition), old.(*apiextensions.CustomResourceDefinition))
}

type statusStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func NewStatusStrategy(typer runtime.ObjectTyper) statusStrategy {
	return statusStrategy{typer, names.SimpleNameGenerator}
}

func (statusStrategy) NamespaceScoped() bool {
	return false
}

func (statusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newObj := obj.(*apiextensions.CustomResourceDefinition)
	oldObj := old.(*apiextensions.CustomResourceDefinition)
	newObj.Spec = oldObj.Spec

	// Status updates are for only for updating status, not objectmeta.
	// TODO: Update after ResetObjectMetaForStatus is added to meta/v1.
	newObj.Labels = oldObj.Labels
	newObj.Annotations = oldObj.Annotations
	newObj.OwnerReferences = oldObj.OwnerReferences
	newObj.Generation = oldObj.Generation
	newObj.SelfLink = oldObj.SelfLink
}

func (statusStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (statusStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (statusStrategy) Canonicalize(obj runtime.Object) {
}

func (statusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateUpdateCustomResourceDefinitionStatus(obj.(*apiextensions.CustomResourceDefinition), old.(*apiextensions.CustomResourceDefinition))
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	apiserver, ok := obj.(*apiextensions.CustomResourceDefinition)
	if !ok {
		return nil, nil, false, fmt.Errorf("given object is not a CustomResourceDefinition")
	}
	return labels.Set(apiserver.ObjectMeta.Labels), CustomResourceDefinitionToSelectableFields(apiserver), apiserver.Initializers != nil, nil
}

// MatchCustomResourceDefinition is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchCustomResourceDefinition(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// CustomResourceDefinitionToSelectableFields returns a field set that represents the object.
func CustomResourceDefinitionToSelectableFields(obj *apiextensions.CustomResourceDefinition) fields.Set {
	return generic.ObjectMetaFieldsSet(&obj.ObjectMeta, true)
}

func dropDisabledFields(crdSpec, oldCrdSpec *apiextensions.CustomResourceDefinitionSpec) {
	// if the feature gate is disabled, drop the feature.
	if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceValidation) &&
		!validationInUse(oldCrdSpec) {
		crdSpec.Validation = nil
		for i := range crdSpec.Versions {
			crdSpec.Versions[i].Schema = nil
		}
	}
	if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceSubresources) &&
		!subresourceInUse(oldCrdSpec) {
		crdSpec.Subresources = nil
		for i := range crdSpec.Versions {
			crdSpec.Versions[i].Subresources = nil
		}
	}

	// 1. On CREATE (in which case the old CRD spec is nil), if the CustomResourceWebhookConversion feature gate is off, we auto-clear
	// the per-version fields. This is to be consistent with the other built-in types, as the
	// apiserver drops unknown fields.
	// 2. On UPDATE, if the CustomResourceWebhookConversion feature gate is off, we auto-clear
	// the per-version fields if the old CRD doesn't use per-version fields already.
	// This is to be consistent with the other built-in types, as the apiserver drops unknown
	// fields. If the old CRD already uses per-version fields, the CRD is allowed to continue
	// use per-version fields.
	if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceWebhookConversion) &&
		!hasPerVersionField(oldCrdSpec) {
		for i := range crdSpec.Versions {
			crdSpec.Versions[i].Schema = nil
			crdSpec.Versions[i].Subresources = nil
			crdSpec.Versions[i].AdditionalPrinterColumns = nil
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceWebhookConversion) &&
		!conversionWebhookInUse(oldCrdSpec) {
		if crdSpec.Conversion != nil {
			crdSpec.Conversion.WebhookClientConfig = nil
		}
	}

}

func validationInUse(crdSpec *apiextensions.CustomResourceDefinitionSpec) bool {
	if crdSpec == nil {
		return false
	}
	if crdSpec.Validation != nil {
		return true
	}

	for i := range crdSpec.Versions {
		if crdSpec.Versions[i].Schema != nil {
			return true
		}
	}
	return false
}

func subresourceInUse(crdSpec *apiextensions.CustomResourceDefinitionSpec) bool {
	if crdSpec == nil {
		return false
	}
	if crdSpec.Subresources != nil {
		return true
	}

	for i := range crdSpec.Versions {
		if crdSpec.Versions[i].Subresources != nil {
			return true
		}
	}
	return false
}

// hasPerVersionField returns true if a CRD uses per-version schema/subresources/columns fields.
//func hasPerVersionField(versions []apiextensions.CustomResourceDefinitionVersion) bool {
func hasPerVersionField(crdSpec *apiextensions.CustomResourceDefinitionSpec) bool {
	if crdSpec == nil {
		return false
	}
	for _, v := range crdSpec.Versions {
		if v.Schema != nil || v.Subresources != nil || len(v.AdditionalPrinterColumns) > 0 {
			return true
		}
	}
	return false
}

func conversionWebhookInUse(crdSpec *apiextensions.CustomResourceDefinitionSpec) bool {
	if crdSpec == nil {
		return false
	}
	if crdSpec.Conversion == nil {
		return false
	}
	return crdSpec.Conversion.WebhookClientConfig != nil
}
