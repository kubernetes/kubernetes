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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
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
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}

	return validation.ValidateCustomResourceDefinition(obj.(*apiextensions.CustomResourceDefinition), groupVersion)
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
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}

	return validation.ValidateCustomResourceDefinitionUpdate(obj.(*apiextensions.CustomResourceDefinition), old.(*apiextensions.CustomResourceDefinition), groupVersion)
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
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	apiserver, ok := obj.(*apiextensions.CustomResourceDefinition)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a CustomResourceDefinition")
	}
	return labels.Set(apiserver.ObjectMeta.Labels), CustomResourceDefinitionToSelectableFields(apiserver), nil
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
