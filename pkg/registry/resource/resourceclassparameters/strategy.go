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

package resourceclassparameters

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
)

// resourceClassParametersStrategy implements behavior for ResourceClassParameters objects
type resourceClassParametersStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var Strategy = resourceClassParametersStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (resourceClassParametersStrategy) NamespaceScoped() bool {
	return true
}

func (resourceClassParametersStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (resourceClassParametersStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	resourceClassParameters := obj.(*resource.ResourceClassParameters)
	return validation.ValidateResourceClassParameters(resourceClassParameters)
}

func (resourceClassParametersStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (resourceClassParametersStrategy) Canonicalize(obj runtime.Object) {
}

func (resourceClassParametersStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (resourceClassParametersStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (resourceClassParametersStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateResourceClassParametersUpdate(obj.(*resource.ResourceClassParameters), old.(*resource.ResourceClassParameters))
}

func (resourceClassParametersStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (resourceClassParametersStrategy) AllowUnconditionalUpdate() bool {
	return true
}
