/*
Copyright 2024 The Kubernetes Authors.

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
	"bytes"
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	jsonutil "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"k8s.io/kubernetes/pkg/features"
)

// deviceClassStrategy implements behavior for DeviceClass objects
type deviceClassStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

var Strategy = deviceClassStrategy{rest.DeclarativeValidation{Scheme: legacyscheme.Scheme}, names.SimpleNameGenerator}

func (deviceClassStrategy) NamespaceScoped() bool {
	return false
}

func (deviceClassStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	class := obj.(*resource.DeviceClass)
	class.Generation = 1
	dropDisabledFields(class, nil)
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

func (deviceClassStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

func (deviceClassStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	class := obj.(*resource.DeviceClass)
	oldClass := old.(*resource.DeviceClass)

	dropDisabledFields(class, oldClass)
	preserveOpaqueParameters(&class.Spec, &oldClass.Spec)

	// Any changes to the spec increment the generation number.
	if !apiequality.Semantic.DeepEqual(oldClass.Spec, class.Spec) {
		class.Generation = oldClass.Generation + 1
	}
}

func (deviceClassStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newClass := obj.(*resource.DeviceClass)
	oldClass := old.(*resource.DeviceClass)
	return validation.ValidateDeviceClassUpdate(newClass, oldClass)
}

func (deviceClassStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (deviceClassStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return true
}

// dropDisabledFields removes fields which are covered by a feature gate.
func dropDisabledFields(newClass, oldClass *resource.DeviceClass) {
	dropDisabledDRAExtendedResourceFields(newClass, oldClass)
	dropDisabledDRAExtendedResourceCapacityFields(newClass, oldClass)
}

func dropDisabledDRAExtendedResourceFields(newClass, oldClass *resource.DeviceClass) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
		return
	}
	if draExtendedResourceFeatureInUse(oldClass) {
		return
	}
	newClass.Spec.ExtendedResourceName = nil
}

func draExtendedResourceFeatureInUse(class *resource.DeviceClass) bool {
	if class == nil {
		return false
	}

	if class.Spec.ExtendedResourceName != nil {
		return true
	}

	return false
}

func dropDisabledDRAExtendedResourceCapacityFields(newClass, oldClass *resource.DeviceClass) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResourceCapacity) || draExtendedResourceCapacityInUse(oldClass) {
		return
	}
	newClass.Spec.ExtendedResourceRequest = nil
}

func draExtendedResourceCapacityInUse(class *resource.DeviceClass) bool {
	if class == nil {
		return false
	}

	if class.Spec.ExtendedResourceRequest != nil {
		return true
	}

	return false
}

// Preserve the original bytes of semantically equivalent opaque parameters
// to avoid unnecessary generation increments after a JSON round-trip.
func preserveOpaqueParameters(new, old *resource.DeviceClassSpec) {
	if len(new.Config) != len(old.Config) {
		return
	}
	for i := range new.Config {
		newConfig, oldConfig := &new.Config[i], &old.Config[i]
		if newConfig.Opaque == nil || oldConfig.Opaque == nil || bytes.Equal(newConfig.Opaque.Parameters.Raw, oldConfig.Opaque.Parameters.Raw) {
			continue
		}
		newOpaque, oldOpaque := *newConfig.Opaque, *oldConfig.Opaque
		newRaw, oldRaw := newOpaque.Parameters.Raw, oldOpaque.Parameters.Raw

		// Pair atomic, ordered configurations only when their surrounding fields
		// (including any decoded Object) are unchanged, as for ResourceClaims.
		newOpaque.Parameters.Raw, oldOpaque.Parameters.Raw = nil, nil
		newIdentity, oldIdentity := *newConfig, *oldConfig
		newIdentity.Opaque, oldIdentity.Opaque = &newOpaque, &oldOpaque
		if !apiequality.Semantic.DeepEqual(newIdentity, oldIdentity) {
			continue
		}
		if jsonutil.EqualWhenNormalized(oldRaw, newRaw) {
			newConfig.Opaque.Parameters.Raw = bytes.Clone(oldRaw)
		}
	}
}
