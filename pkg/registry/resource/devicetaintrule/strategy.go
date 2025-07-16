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

package devicetaintrule

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

// deviceTaintRuleStrategy implements behavior for DeviceTaintRule objects
type deviceTaintRuleStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var Strategy = deviceTaintRuleStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (deviceTaintRuleStrategy) NamespaceScoped() bool {
	return false
}

func (deviceTaintRuleStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	patch := obj.(*resource.DeviceTaintRule)
	patch.Generation = 1
}

func (deviceTaintRuleStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	patch := obj.(*resource.DeviceTaintRule)
	return validation.ValidateDeviceTaintRule(patch)
}

func (deviceTaintRuleStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (deviceTaintRuleStrategy) Canonicalize(obj runtime.Object) {
}

func (deviceTaintRuleStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (deviceTaintRuleStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	patch := obj.(*resource.DeviceTaintRule)
	oldPatch := old.(*resource.DeviceTaintRule)

	// Any changes to the spec increment the generation number.
	if !apiequality.Semantic.DeepEqual(oldPatch.Spec, patch.Spec) {
		patch.Generation = oldPatch.Generation + 1
	}
}

func (deviceTaintRuleStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateDeviceTaintRuleUpdate(obj.(*resource.DeviceTaintRule), old.(*resource.DeviceTaintRule))
}

func (deviceTaintRuleStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (deviceTaintRuleStrategy) AllowUnconditionalUpdate() bool {
	return true
}
