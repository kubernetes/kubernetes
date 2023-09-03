/*
Copyright 2022 The Kubernetes Authors.

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

package resourceclass

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
)

// resourceClassStrategy implements behavior for ResourceClass objects
type resourceClassStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var Strategy = resourceClassStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (resourceClassStrategy) NamespaceScoped() bool {
	return false
}

func (resourceClassStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (resourceClassStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	resourceClass := obj.(*resource.ResourceClass)
	return validation.ValidateClass(resourceClass)
}

func (resourceClassStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (resourceClassStrategy) Canonicalize(obj runtime.Object) {
}

func (resourceClassStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (resourceClassStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (resourceClassStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidateClass(obj.(*resource.ResourceClass))
	return append(errorList, validation.ValidateClassUpdate(obj.(*resource.ResourceClass), old.(*resource.ResourceClass))...)
}

func (resourceClassStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (resourceClassStrategy) AllowUnconditionalUpdate() bool {
	return true
}
