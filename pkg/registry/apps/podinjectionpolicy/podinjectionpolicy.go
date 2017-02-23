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

package podinjectionpolicy

import (
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/apps/validation"
)

// podInjectionPolicyStrategy implements verification logic for Pod Injection Policies.
type podInjectionPolicyStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Pod Injection Policy objects.
var Strategy = podInjectionPolicyStrategy{api.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all Pod Injection Policies need to be within a namespace.
func (podInjectionPolicyStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a Pod Injection Policy before creation.
func (podInjectionPolicyStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	pip := obj.(*apps.PodInjectionPolicy)
	pip.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podInjectionPolicyStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newPodInjectionPolicy := obj.(*apps.PodInjectionPolicy)
	oldPodInjectionPolicy := old.(*apps.PodInjectionPolicy)

	// Update is not allowed
	newPodInjectionPolicy = oldPodInjectionPolicy

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldPodInjectionPolicy.Spec, newPodInjectionPolicy.Spec) {
		newPodInjectionPolicy.Generation = oldPodInjectionPolicy.Generation + 1
	}

}

// Validate validates a new PodInjectionPolicy.
func (podInjectionPolicyStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	pip := obj.(*apps.PodInjectionPolicy)
	return validation.ValidatePodInjectionPolicy(pip)
}

// Canonicalize normalizes the object after validation.
func (podInjectionPolicyStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for PodInjectionPolicy; this means POST is needed to create one.
func (podInjectionPolicyStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (podInjectionPolicyStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidatePodInjectionPolicy(obj.(*apps.PodInjectionPolicy))
	updateErrorList := validation.ValidatePodInjectionPolicyUpdate(obj.(*apps.PodInjectionPolicy), old.(*apps.PodInjectionPolicy))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for Pod Injection Policy objects.
func (podInjectionPolicyStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// PodInjectionPolicyToSelectableFields returns a field set that represents the object.
func PodInjectionPolicyToSelectableFields(pip *apps.PodInjectionPolicy) fields.Set {
	return generic.ObjectMetaFieldsSet(&pip.ObjectMeta, true)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pip, ok := obj.(*apps.PodInjectionPolicy)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a PodInjectionPolicy.")
	}
	return labels.Set(pip.ObjectMeta.Labels), PodInjectionPolicyToSelectableFields(pip), nil
}

// MatchPodInjectionPolicy is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchPodInjectionPolicy(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}
