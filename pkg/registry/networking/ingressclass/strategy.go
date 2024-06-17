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

package ingressclass

import (
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/apis/networking/validation"
)

// ingressClassStrategy implements verification logic for IngressClass
// resources.
type ingressClassStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// IngressClass objects.
var Strategy = ingressClassStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because IngressClass is a non-namespaced
// resource.
func (ingressClassStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate prepares an IngressClass for creation.
func (ingressClassStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ingressClass := obj.(*networking.IngressClass)
	ingressClass.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on
// update.
func (ingressClassStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIngressClass := obj.(*networking.IngressClass)
	oldIngressClass := old.(*networking.IngressClass)

	// Any changes to the spec increment the generation number.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(oldIngressClass.Spec, newIngressClass.Spec) {
		newIngressClass.Generation = oldIngressClass.Generation + 1
	}
}

// Validate validates a new IngressClass.
func (ingressClassStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ingressClass := obj.(*networking.IngressClass)
	return validation.ValidateIngressClass(ingressClass)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (ingressClassStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (ingressClassStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for IngressClass; this means POST is needed to
// create one.
func (ingressClassStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (ingressClassStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newIngressClass := obj.(*networking.IngressClass)
	oldIngressClass := old.(*networking.IngressClass)

	return validation.ValidateIngressClassUpdate(newIngressClass, oldIngressClass)
}

// WarningsOnUpdate returns warnings for the given update.
func (ingressClassStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for IngressClass
// objects.
func (ingressClassStrategy) AllowUnconditionalUpdate() bool {
	return true
}
