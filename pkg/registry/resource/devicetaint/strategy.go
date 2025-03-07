/*
Copyright 2025 The Kubernetes Authors.

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

package devicetaint

import (
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
)

// deviceTaintStrategy implements behavior for DeviceTaint objects
type deviceTaintStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var Strategy = deviceTaintStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (deviceTaintStrategy) NamespaceScoped() bool {
	return false
}

func (deviceTaintStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	patch := obj.(*resource.DeviceTaint)
	patch.Generation = 1
}

func (deviceTaintStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	patch := obj.(*resource.DeviceTaint)
	return validation.ValidateDeviceTaint(patch)
}

func (deviceTaintStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (deviceTaintStrategy) Canonicalize(obj runtime.Object) {
}

func (deviceTaintStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (deviceTaintStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	patch := obj.(*resource.DeviceTaint)
	oldPatch := old.(*resource.DeviceTaint)

	// Any changes to the spec increment the generation number.
	if !apiequality.Semantic.DeepEqual(oldPatch.Spec, patch.Spec) {
		patch.Generation = oldPatch.Generation + 1
	}
}

func (deviceTaintStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateDeviceTaintUpdate(obj.(*resource.DeviceTaint), old.(*resource.DeviceTaint))
}

func (deviceTaintStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (deviceTaintStrategy) AllowUnconditionalUpdate() bool {
	return true
}
