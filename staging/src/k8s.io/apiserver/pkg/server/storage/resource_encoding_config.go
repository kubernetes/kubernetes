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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/util/feature"
	utilversion "k8s.io/apiserver/pkg/util/version"
)

type ResourceEncodingConfig interface {
	// StorageEncoding returns the serialization format for the resource.
	// TODO this should actually return a GroupVersionKind since you can logically have multiple "matching" Kinds
	// For now, it returns just the GroupVersion for consistency with old behavior
	StorageEncodingFor(schema.GroupResource) (schema.GroupVersion, error)

	// InMemoryEncodingFor returns the groupVersion for the in memory representation the storage should convert to.
	InMemoryEncodingFor(schema.GroupResource) (schema.GroupVersion, error)
}

type DefaultResourceEncodingConfig struct {
	// resources records the overriding encoding configs for individual resources.
	resources               map[schema.GroupResource]*OverridingResourceEncoding
	scheme                  *runtime.Scheme
	emulationVersion        *version.Version
	minCompatibilityVersion *version.Version
}

type OverridingResourceEncoding struct {
	ExternalResourceEncoding schema.GroupVersion
	InternalResourceEncoding schema.GroupVersion
}

var _ ResourceEncodingConfig = &DefaultResourceEncodingConfig{}

func NewDefaultResourceEncodingConfig(scheme *runtime.Scheme) *DefaultResourceEncodingConfig {
	emuVer := utilversion.Effective.EmulationVersion()
	compatVer := utilversion.Effective.MinCompatibilityVersion()
	return &DefaultResourceEncodingConfig{resources: map[schema.GroupResource]*OverridingResourceEncoding{}, scheme: scheme,
		emulationVersion: version.MajorMinor(emuVer.Major(), emuVer.Minor()), minCompatibilityVersion: version.MajorMinor(compatVer.Major(), compatVer.Minor())}
}

func (o *DefaultResourceEncodingConfig) SetResourceEncoding(resourceBeingStored schema.GroupResource, externalEncodingVersion, internalVersion schema.GroupVersion) {
	o.resources[resourceBeingStored] = &OverridingResourceEncoding{
		ExternalResourceEncoding: externalEncodingVersion,
		InternalResourceEncoding: internalVersion,
	}
}

func (o *DefaultResourceEncodingConfig) StorageEncodingFor(resource schema.GroupResource) (schema.GroupVersion, error) {
	if !o.scheme.IsGroupRegistered(resource.Group) {
		return schema.GroupVersion{}, fmt.Errorf("group %q is not registered in scheme", resource.Group)
	}

	resourceOverride, resourceExists := o.resources[resource]
	if resourceExists {
		return resourceOverride.ExternalResourceEncoding, nil
	}

	if feature.DefaultFeatureGate.Enabled(features.EmulationVersion) && o.emulationVersion != nil && o.minCompatibilityVersion != nil {
		// List all versions for this group.
		knownVersionsToBinary := o.scheme.PrioritizedVersionsForGroup(resource.Group)
		emulationVersions := enabledVersions(resource.Resource, o.scheme, o.emulationVersion, knownVersionsToBinary)
		minCompatibilityVersions := enabledVersions(resource.Resource, o.scheme, o.minCompatibilityVersion, knownVersionsToBinary)

		// Return the first GV that is common to both the emulation and the
		// minimum compatibility versions. The lists are sorted by priority
		// so returning the first match will give the highest priority version
		for _, emulationGV := range emulationVersions {
			for _, minWindowGV := range minCompatibilityVersions {
				if emulationGV == minWindowGV {
					return emulationGV, nil
				}
			}
		}

		return schema.GroupVersion{}, fmt.Errorf("resource not codable by both emulation version and min compatibility version: %v", resource)
	}

	// return the most preferred external version for the group
	return o.scheme.PrioritizedVersionsForGroup(resource.Group)[0], nil
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

func enabledVersions(
	resource string,
	registry GroupVersionRegistry,
	emulationVersion *version.Version,
	prioritizedVersions []schema.GroupVersion,
) []schema.GroupVersion {
	var enabledVersions []schema.GroupVersion
	for _, gv := range prioritizedVersions {
		gvr := schema.GroupVersionResource{
			Group:    gv.Group,
			Version:  gv.Version,
			Resource: resource,
		}

		if enabled, _ := isAPIAvailable(registry.GroupVersionLifecycle(gv), emulationVersion); !enabled {
			continue
		} else if enabled, _ := isAPIAvailable(registry.ResourceLifecycle(gvr), emulationVersion); !enabled {
			continue
		}

		enabledVersions = append(enabledVersions, gv)
	}
	return enabledVersions
}
