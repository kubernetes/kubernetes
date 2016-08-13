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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// namespaceStrategy implements behavior for Namespaces
type namespaceStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Namespace
// objects via the REST API.
var Strategy = namespaceStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for namespaces.
func (namespaceStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (namespaceStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
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
func (namespaceStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newNamespace := obj.(*api.Namespace)
	oldNamespace := old.(*api.Namespace)
	newNamespace.Spec.Finalizers = oldNamespace.Spec.Finalizers
	newNamespace.Status = oldNamespace.Status
}

// Validate validates a new namespace.
func (namespaceStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	namespace := obj.(*api.Namespace)
	return validation.ValidateNamespace(namespace)
}

// Canonicalize normalizes the object after validation.
func (namespaceStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for namespaces.
func (namespaceStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (namespaceStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidateNamespace(obj.(*api.Namespace))
	return append(errorList, validation.ValidateNamespaceUpdate(obj.(*api.Namespace), old.(*api.Namespace))...)
}

func (namespaceStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type namespaceStatusStrategy struct {
	namespaceStrategy
}

var StatusStrategy = namespaceStatusStrategy{Strategy}

func (namespaceStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newNamespace := obj.(*api.Namespace)
	oldNamespace := old.(*api.Namespace)
	newNamespace.Spec = oldNamespace.Spec
}

func (namespaceStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateNamespaceStatusUpdate(obj.(*api.Namespace), old.(*api.Namespace))
}

type namespaceFinalizeStrategy struct {
	namespaceStrategy
}

var FinalizeStrategy = namespaceFinalizeStrategy{Strategy}

func (namespaceFinalizeStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateNamespaceFinalizeUpdate(obj.(*api.Namespace), old.(*api.Namespace))
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (namespaceFinalizeStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newNamespace := obj.(*api.Namespace)
	oldNamespace := old.(*api.Namespace)
	newNamespace.Status = oldNamespace.Status
}

// MatchNamespace returns a generic matcher for a given label and field selector.
func MatchNamespace(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		namespaceObj, ok := obj.(*api.Namespace)
		if !ok {
			return false, fmt.Errorf("not a namespace")
		}
		fields := NamespaceToSelectableFields(namespaceObj)
		return label.Matches(labels.Set(namespaceObj.Labels)) && field.Matches(fields), nil
	})
}

// NamespaceToSelectableFields returns a label set that represents the object
func NamespaceToSelectableFields(namespace *api.Namespace) labels.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(namespace.ObjectMeta, false)
	specificFieldsSet := fields.Set{
		"status.phase": string(namespace.Status.Phase),
		// This is a bug, but we need to support it for backward compatibility.
		"name": namespace.Name,
	}
	return labels.Set(generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet))
}
