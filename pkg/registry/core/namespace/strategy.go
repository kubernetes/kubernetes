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

package namespace

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// namespaceStrategy implements behavior for Namespaces
type namespaceStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Namespace
// objects via the REST API.
var Strategy = namespaceStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is false for namespaces.
func (namespaceStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (namespaceStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (namespaceStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	// on create, status is active
	namespace := obj.(*api.Namespace)
	namespace.Status = api.NamespaceStatus{
		Phase: api.NamespaceActive,
	}
	// on create, we require the kubernetes value
	// we cannot use this in defaults conversion because we let it get removed over life of object
	hasKubeFinalizer := false
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] == api.FinalizerKubernetes {
			hasKubeFinalizer = true
			break
		}
	}
	if !hasKubeFinalizer {
		if len(namespace.Spec.Finalizers) == 0 {
			namespace.Spec.Finalizers = []api.FinalizerName{api.FinalizerKubernetes}
		} else {
			namespace.Spec.Finalizers = append(namespace.Spec.Finalizers, api.FinalizerKubernetes)
		}
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (namespaceStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newNamespace := obj.(*api.Namespace)
	oldNamespace := old.(*api.Namespace)
	newNamespace.Spec.Finalizers = oldNamespace.Spec.Finalizers
	newNamespace.Status = oldNamespace.Status
}

// Validate validates a new namespace.
func (namespaceStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	namespace := obj.(*api.Namespace)
	return validation.ValidateNamespace(namespace)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (namespaceStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (namespaceStrategy) Canonicalize(obj runtime.Object) {
	// Ensure the label matches the name for namespaces just created using GenerateName,
	// where the final name wasn't available for defaulting to make this change.
	// This code needs to be kept in sync with the implementation that exists
	// in Namespace defaulting (pkg/apis/core/v1)
	// Why this hook *and* defaults.go?
	//
	// CREATE:
	// Defaulting and PrepareForCreate happen before generateName is completed
	// (i.e. the name is not yet known). Validation happens after generateName
	// but should not modify objects. Canonicalize happens later, which makes
	// it the best hook for setting the label.
	//
	// UPDATE:
	// Defaulting and Canonicalize will both trigger with the full name.
	//
	// GET:
	// Only defaulting will be applied.
	ns := obj.(*api.Namespace)
	if ns.Labels == nil {
		ns.Labels = map[string]string{}
	}
	ns.Labels[v1.LabelMetadataName] = ns.Name
}

// AllowCreateOnUpdate is false for namespaces.
func (namespaceStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (namespaceStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidateNamespace(obj.(*api.Namespace))
	return append(errorList, validation.ValidateNamespaceUpdate(obj.(*api.Namespace), old.(*api.Namespace))...)
}

// WarningsOnUpdate returns warnings for the given update.
func (namespaceStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (namespaceStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type namespaceStatusStrategy struct {
	namespaceStrategy
}

var StatusStrategy = namespaceStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (namespaceStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
}

func (namespaceStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newNamespace := obj.(*api.Namespace)
	oldNamespace := old.(*api.Namespace)
	newNamespace.Spec = oldNamespace.Spec
}

func (namespaceStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateNamespaceStatusUpdate(obj.(*api.Namespace), old.(*api.Namespace))
}

// WarningsOnUpdate returns warnings for the given update.
func (namespaceStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

type namespaceFinalizeStrategy struct {
	namespaceStrategy
}

var FinalizeStrategy = namespaceFinalizeStrategy{Strategy}

func (namespaceFinalizeStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateNamespaceFinalizeUpdate(obj.(*api.Namespace), old.(*api.Namespace))
}

// WarningsOnUpdate returns warnings for the given update.
func (namespaceFinalizeStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (namespaceFinalizeStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (namespaceFinalizeStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newNamespace := obj.(*api.Namespace)
	oldNamespace := old.(*api.Namespace)
	newNamespace.Status = oldNamespace.Status
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	namespaceObj, ok := obj.(*api.Namespace)
	if !ok {
		return nil, nil, fmt.Errorf("not a namespace")
	}
	return labels.Set(namespaceObj.Labels), NamespaceToSelectableFields(namespaceObj), nil
}

// MatchNamespace returns a generic matcher for a given label and field selector.
func MatchNamespace(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// NamespaceToSelectableFields returns a field set that represents the object
func NamespaceToSelectableFields(namespace *api.Namespace) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&namespace.ObjectMeta, false)
	specificFieldsSet := fields.Set{
		"status.phase": string(namespace.Status.Phase),
		// This is a bug, but we need to support it for backward compatibility.
		"name": namespace.Name,
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}
