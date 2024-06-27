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

package deviceclass

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

// deviceClassStrategy implements behavior for DeviceClass objects
type deviceClassStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var Strategy = deviceClassStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (deviceClassStrategy) NamespaceScoped() bool {
	return false
}

func (deviceClassStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	class := obj.(*resource.DeviceClass)
	class.Generation = 1
}

func (deviceClassStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	deviceClass := obj.(*resource.DeviceClass)
	return validation.ValidateDeviceClass(deviceClass)
}

func (deviceClassStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (deviceClassStrategy) Canonicalize(obj runtime.Object) {
}

func (deviceClassStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (deviceClassStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	class := obj.(*resource.DeviceClass)
	oldClass := old.(*resource.DeviceClass)

	// Any changes to the spec increment the generation number.
	if !apiequality.Semantic.DeepEqual(oldClass.Spec, class.Spec) {
		class.Generation = oldClass.Generation + 1
	}
}

func (deviceClassStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidateDeviceClass(obj.(*resource.DeviceClass))
	return append(errorList, validation.ValidateDeviceClassUpdate(obj.(*resource.DeviceClass), old.(*resource.DeviceClass))...)
}

func (deviceClassStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (deviceClassStrategy) AllowUnconditionalUpdate() bool {
	return true
}
