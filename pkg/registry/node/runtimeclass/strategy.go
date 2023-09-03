/*
Copyright 2019 The Kubernetes Authors.

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

package runtimeclass

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	nodeapi "k8s.io/kubernetes/pkg/api/node"
	"k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/apis/node/validation"
)

// strategy implements verification logic for RuntimeClass.
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating RuntimeClass objects.
var Strategy = strategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped is false for RuntimeClasses
func (strategy) NamespaceScoped() bool {
	return false
}

// AllowCreateOnUpdate is true for RuntimeClasses.
func (strategy) AllowCreateOnUpdate() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users
// on creation.
func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (strategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newRuntimeClass := obj.(*node.RuntimeClass)
	oldRuntimeClass := old.(*node.RuntimeClass)

	_, _ = newRuntimeClass, oldRuntimeClass
}

// Validate validates a new RuntimeClass. Validation must check for a correct signature.
func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	runtimeClass := obj.(*node.RuntimeClass)
	return validation.ValidateRuntimeClass(runtimeClass)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nodeapi.GetWarningsForRuntimeClass(obj.(*node.RuntimeClass))
}

// Canonicalize normalizes the object after validation.
func (strategy) Canonicalize(obj runtime.Object) {
	_ = obj.(*node.RuntimeClass)
}

// ValidateUpdate is the default update validation for an end user.
func (strategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newObj := obj.(*node.RuntimeClass)
	errorList := validation.ValidateRuntimeClass(newObj)
	return append(errorList, validation.ValidateRuntimeClassUpdate(newObj, old.(*node.RuntimeClass))...)
}

// WarningsOnUpdate returns warnings for the given update.
func (strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nodeapi.GetWarningsForRuntimeClass(obj.(*node.RuntimeClass))
}

// If AllowUnconditionalUpdate() is true and the object specified by
// the user does not have a resource version, then generic Update()
// populates it with the latest version. Else, it checks that the
// version specified by the user matches the version of latest etcd
// object.
func (strategy) AllowUnconditionalUpdate() bool {
	return false
}
