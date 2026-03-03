/*
Copyright 2016 The Kubernetes Authors.

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

package storage

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apimachineryversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/util/compatibility"
	basecompatibility "k8s.io/component-base/compatibility"
)

type ResourceEncodingConfig interface {
	// StorageEncodingFor returns the serialization format for the resource.
	// TODO this should actually return a GroupVersionKind since you can logically have multiple "matching" Kinds
	// For now, it returns just the GroupVersion for consistency with old behavior
	StorageEncodingFor(schema.GroupResource) (schema.GroupVersion, error)

	// InMemoryEncodingFor returns the groupVersion for the in memory representation the storage should convert to.
	InMemoryEncodingFor(schema.GroupResource) (schema.GroupVersion, error)
}

type CompatibilityResourceEncodingConfig interface {
	BackwardCompatibileStorageEncodingFor(schema.GroupResource, runtime.Object) (schema.GroupVersion, error)
}

type DefaultResourceEncodingConfig struct {
	// resources records the overriding encoding configs for individual resources.
	resources        map[schema.GroupResource]*OverridingResourceEncoding
	scheme           *runtime.Scheme
	effectiveVersion basecompatibility.EffectiveVersion
}

type OverridingResourceEncoding struct {
	ExternalResourceEncoding schema.GroupVersion
	InternalResourceEncoding schema.GroupVersion
}

var _ ResourceEncodingConfig = &DefaultResourceEncodingConfig{}

func NewDefaultResourceEncodingConfig(scheme *runtime.Scheme) *DefaultResourceEncodingConfig {
	return NewDefaultResourceEncodingConfigForEffectiveVersion(scheme, compatibility.DefaultComponentGlobalsRegistry.EffectiveVersionFor(basecompatibility.DefaultKubeComponent))
}

func NewDefaultResourceEncodingConfigForEffectiveVersion(scheme *runtime.Scheme, effectiveVersion basecompatibility.EffectiveVersion) *DefaultResourceEncodingConfig {
	return &DefaultResourceEncodingConfig{resources: map[schema.GroupResource]*OverridingResourceEncoding{}, scheme: scheme, effectiveVersion: effectiveVersion}
}

func (o *DefaultResourceEncodingConfig) SetResourceEncoding(resourceBeingStored schema.GroupResource, externalEncodingVersion, internalVersion schema.GroupVersion) {
	o.resources[resourceBeingStored] = &OverridingResourceEncoding{
		ExternalResourceEncoding: externalEncodingVersion,
		InternalResourceEncoding: internalVersion,
	}
}

func (o *DefaultResourceEncodingConfig) SetEffectiveVersion(effectiveVersion basecompatibility.EffectiveVersion) {
	o.effectiveVersion = effectiveVersion
}

func (o *DefaultResourceEncodingConfig) StorageEncodingFor(resource schema.GroupResource) (schema.GroupVersion, error) {
	if !o.scheme.IsGroupRegistered(resource.Group) {
		return schema.GroupVersion{}, fmt.Errorf("group %q is not registered in scheme", resource.Group)
	}

	resourceOverride, resourceExists := o.resources[resource]
	if resourceExists {
		return resourceOverride.ExternalResourceEncoding, nil
	}

	// return the most preferred external version for the group
	return o.scheme.PrioritizedVersionsForGroup(resource.Group)[0], nil
}

func (o *DefaultResourceEncodingConfig) BackwardCompatibileStorageEncodingFor(resource schema.GroupResource, example runtime.Object) (schema.GroupVersion, error) {
	if !o.scheme.IsGroupRegistered(resource.Group) {
		return schema.GroupVersion{}, fmt.Errorf("group %q is not registered in scheme", resource.Group)
	}

	// Always respect overrides
	resourceOverride, resourceExists := o.resources[resource]
	if resourceExists {
		return resourceOverride.ExternalResourceEncoding, nil
	}

	return emulatedStorageVersion(
		o.scheme.PrioritizedVersionsForGroup(resource.Group)[0],
		example,
		o.effectiveVersion,
		o.scheme)
}

func (o *DefaultResourceEncodingConfig) InMemoryEncodingFor(resource schema.GroupResource) (schema.GroupVersion, error) {
	if !o.scheme.IsGroupRegistered(resource.Group) {
		return schema.GroupVersion{}, fmt.Errorf("group %q is not registered in scheme", resource.Group)
	}

	resourceOverride, resourceExists := o.resources[resource]
	if resourceExists {
		return resourceOverride.InternalResourceEncoding, nil
	}
	return schema.GroupVersion{Group: resource.Group, Version: runtime.APIVersionInternal}, nil
}

// Object interface generated from "k8s:prerelease-lifecycle-gen:introduced" tags in types.go.
type introducedInterface interface {
	APILifecycleIntroduced() (major, minor int)
}

func emulatedStorageVersion(binaryVersionOfResource schema.GroupVersion, example runtime.Object, effectiveVersion basecompatibility.EffectiveVersion, scheme *runtime.Scheme) (schema.GroupVersion, error) {
	if example == nil || effectiveVersion == nil {
		return binaryVersionOfResource, nil
	}

	// Look up example in scheme to find all objects of the same Group-Kind.
	// We track two candidates across all versions (in priority order):
	//   - compat: highest-priority non-alpha version introduced by MinCompatibilityVersion
	//   - best:   highest-priority version introduced by EmulationVersion
	//
	// This ensures that beta APIs are used for storage as soon as they are
	// introduced, while non-alpha versions (beta/GA) still respect n-1
	// compatibility for safe rollback.
	gvks, _, err := scheme.ObjectKinds(example)
	if err != nil {
		return schema.GroupVersion{}, err
	}

	var gvk schema.GroupVersionKind
	for _, item := range gvks {
		if item.Group != binaryVersionOfResource.Group {
			continue
		}

		gvk = item
		break
	}

	if len(gvk.Kind) == 0 {
		return schema.GroupVersion{}, fmt.Errorf("object %T has no GVKs registered in scheme", example)
	}

	// VersionsForGroupKind returns versions in priority order.
	// Use the kind's priority version as the fallback instead of the group's,
	// because the kind may not exist in the group's top version.
	versions := scheme.VersionsForGroupKind(schema.GroupKind{Group: gvk.Group, Kind: gvk.Kind})
	for _, gv := range versions {
		if gv.Version != runtime.APIVersionInternal {
			binaryVersionOfResource = gv
			break
		}
	}

	compatibilityVersion := effectiveVersion.MinCompatibilityVersion()
	emulationVersion := effectiveVersion.EmulationVersion()

	var best, compat schema.GroupVersion
	for _, gv := range versions {
		if gv.Version == runtime.APIVersionInternal {
			continue
		}

		candidateGVK := schema.GroupVersionKind{
			Group:   gv.Group,
			Version: gv.Version,
			Kind:    gvk.Kind,
		}

		exampleOfGVK, err := scheme.New(candidateGVK)
		if err != nil {
			return schema.GroupVersion{}, err
		}

		// Determine when this version was introduced.
		// Skip the introduced check for test when current compatibility version is 0.0 to test all apis.
		var introducedVer *apimachineryversion.Version
		if introduced, hasIntroduced := exampleOfGVK.(introducedInterface); hasIntroduced && (compatibilityVersion.Major() > 0 || compatibilityVersion.Minor() > 0) {
			majorIntroduced, minorIntroduced := introduced.APILifecycleIntroduced()
			introducedVer = apimachineryversion.MajorMinor(uint(majorIntroduced), uint(minorIntroduced))
		}

		if best.Empty() {
			if introducedVer == nil || !introducedVer.GreaterThan(emulationVersion) {
				best = gv
			}
		}

		if compat.Empty() && !strings.Contains(gv.Version, "alpha") {
			if introducedVer == nil || !introducedVer.GreaterThan(compatibilityVersion) {
				compat = gv
			}
		}

		if !best.Empty() && !compat.Empty() {
			break
		}
	}

	if !compat.Empty() {
		return compat, nil
	}
	if !best.Empty() {
		return best, nil
	}
	return binaryVersionOfResource, nil
}
