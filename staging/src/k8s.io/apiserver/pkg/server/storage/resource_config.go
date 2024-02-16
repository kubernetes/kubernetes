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

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/version"
	utilversion "k8s.io/apiserver/pkg/util/version"
	"k8s.io/klog/v2"
)

// APIResourceConfigSource is the interface to determine which groups and versions are enabled
type APIResourceConfigSource interface {
	ResourceEnabled(resource schema.GroupVersionResource) bool
	AnyResourceForGroupEnabled(group string) bool
}

// GroupVersionRegistry provides access to registered group versions.
type GroupVersionRegistry interface {
	// IsGroupRegistered returns true if given group is registered.
	IsGroupRegistered(group string) bool
	// IsVersionRegistered returns true if given version is registered.
	IsVersionRegistered(v schema.GroupVersion) bool
	// PrioritizedVersionsAllGroups returns all registered group versions.
	PrioritizedVersionsAllGroups() []schema.GroupVersion
	// GroupVersionLifecycle returns the APILifecycle for the GroupVersion.
	GroupVersionLifecycle(gv schema.GroupVersion) schema.APILifecycle
	// ResourceLifecycle returns the APILifecycle for the GroupVersionResource.
	ResourceLifecycle(gvr schema.GroupVersionResource) schema.APILifecycle
}

var _ APIResourceConfigSource = &ResourceConfig{}

type ResourceConfig struct {
	GroupVersionConfigs map[schema.GroupVersion]bool
	ResourceConfigs     map[schema.GroupVersionResource]bool
	emulationVersion    *version.Version
	GroupVersionRegistry
}

func NewResourceConfig(registry GroupVersionRegistry) *ResourceConfig {
	emuVer := utilversion.Effective.EmulationVersion()
	return &ResourceConfig{GroupVersionConfigs: map[schema.GroupVersion]bool{}, ResourceConfigs: map[schema.GroupVersionResource]bool{},
		emulationVersion: emuVer, GroupVersionRegistry: registry}
}

// NewResourceConfigIgnoreLifecycle creates a ResourceConfig that allows enabling/disabling resources regardless of their APILifecycle.
// Mainly used in tests.
func NewResourceConfigIgnoreLifecycle() *ResourceConfig {
	return &ResourceConfig{GroupVersionConfigs: map[schema.GroupVersion]bool{}, ResourceConfigs: map[schema.GroupVersionResource]bool{}}
}

// DisableMatchingVersions disables all group/versions for which the matcher function returns true.
// This will remove any preferences previously set on individual resources.
func (o *ResourceConfig) DisableMatchingVersions(matcher func(gv schema.GroupVersion) bool) {
	for version := range o.GroupVersionConfigs {
		if matcher(version) {
			o.GroupVersionConfigs[version] = false
			o.removeMatchingResourcePreferences(resourceMatcherForVersion(version))
		}
	}
}

// EnableMatchingVersions enables all group/versions for which the matcher function returns true.
// This will remove any preferences previously set on individual resources.
func (o *ResourceConfig) EnableMatchingVersions(matcher func(gv schema.GroupVersion) bool) {
	for version := range o.GroupVersionConfigs {
		if matcher(version) {
			if available, err := o.versionAvailable(version); available {
				o.GroupVersionConfigs[version] = true
			} else {
				o.GroupVersionConfigs[version] = false
				klog.V(1).Infof("version %s cannot be enabled because: %s", version.String(), err.Error())
			}
			o.removeMatchingResourcePreferences(resourceMatcherForVersion(version))
		}
	}
}

// resourceMatcherForVersion matches resources in the specified version
func resourceMatcherForVersion(gv schema.GroupVersion) func(gvr schema.GroupVersionResource) bool {
	return func(gvr schema.GroupVersionResource) bool {
		return gv == gvr.GroupVersion()
	}
}

// removeMatchingResourcePreferences removes individual resource preferences that match.  This is useful when an override of a version or level enablement should
// override the previously individual preferences.
func (o *ResourceConfig) removeMatchingResourcePreferences(matcher func(gvr schema.GroupVersionResource) bool) {
	keysToRemove := []schema.GroupVersionResource{}
	for k := range o.ResourceConfigs {
		if matcher(k) {
			keysToRemove = append(keysToRemove, k)
		}
	}
	for _, k := range keysToRemove {
		delete(o.ResourceConfigs, k)
	}
}

// DisableVersions disables the versions entirely.
// This will remove any preferences previously set on individual resources.
func (o *ResourceConfig) DisableVersions(versions ...schema.GroupVersion) {
	for _, version := range versions {
		o.GroupVersionConfigs[version] = false
		// a preference about a version takes priority over the previously set resources
		o.removeMatchingResourcePreferences(resourceMatcherForVersion(version))
	}
}

// EnableVersions enables all resources in a given groupVersion.
// A groupVersion can only be enabled if its APILifecyle is available for the emulation version.
// This will remove any preferences previously set on individual resources.
func (o *ResourceConfig) EnableVersions(versions ...schema.GroupVersion) {
	for _, version := range versions {
		if available, err := o.versionAvailable(version); available {
			o.GroupVersionConfigs[version] = true
		} else {
			o.GroupVersionConfigs[version] = false
			klog.V(1).Infof("version %s cannot be enabled because: %s", version.String(), err.Error())
		}
		// a preference about a version takes priority over the previously set resources
		o.removeMatchingResourcePreferences(resourceMatcherForVersion(version))
	}
}

// TODO this must be removed and we enable/disable individual resources.
func (o *ResourceConfig) versionEnabled(version schema.GroupVersion) bool {
	enabled, _ := o.GroupVersionConfigs[version]
	return enabled
}

// apiAvailable compares the APILifecycle against the emulationVersion.
// An API is unavailable if it introduced after the emulationVersion,
// or removed before the emulationVersion.
func (o *ResourceConfig) apiAvailable(lifecycle schema.APILifecycle) (bool, error) {
	// emulationVersion is not set.
	if o.emulationVersion == nil {
		return true, nil
	}
	// GroupVersion is introduced after the emulationVersion.
	if lifecycle.IntroducedVersion != nil && o.emulationVersion.LessThan(lifecycle.IntroducedVersion) {
		return false, fmt.Errorf("api introduced at %s, after the emulationVersion %s", lifecycle.IntroducedVersion.String(), o.emulationVersion.String())
	}
	// GroupVersion is removed before the emulationVersion.
	if lifecycle.RemovedVersion != nil && o.emulationVersion.GreaterThan(lifecycle.RemovedVersion) {
		return false, fmt.Errorf("api removed at %s, before the emulationVersion %s", lifecycle.RemovedVersion.String(), o.emulationVersion.String())
	}
	// TODO: handle remove when RemovedVersion equals emulationVersion like resourceExpirationEvaluator.ShouldServeForVersion.
	return true, nil
}

// versionAvailable checks if a GroupVersion is available based on its APILifecycle.
func (o *ResourceConfig) versionAvailable(version schema.GroupVersion) (bool, error) {
	if o.GroupVersionRegistry == nil {
		return true, nil
	}
	return o.apiAvailable(o.GroupVersionLifecycle(version))
}

// versionAvailable checks if a GroupVersionResource is available based on its APILifecycle.
func (o *ResourceConfig) resourceAvailable(resource schema.GroupVersionResource) (bool, error) {
	if o.GroupVersionRegistry == nil {
		return true, nil
	}
	// both the group version and resource have to avaible.
	if available, err := o.apiAvailable(o.ResourceLifecycle(resource)); !available {
		return available, err
	}
	return o.versionAvailable(resource.GroupVersion())
}

func (o *ResourceConfig) DisableResources(resources ...schema.GroupVersionResource) {
	for _, resource := range resources {
		o.ResourceConfigs[resource] = false
	}
}

// EnableResources enables resources explicitly.
// A resource can only be enabled if its APILifecyle is available for the emulation version.
func (o *ResourceConfig) EnableResources(resources ...schema.GroupVersionResource) {
	for _, resource := range resources {
		if available, err := o.resourceAvailable(resource); available {
			o.ResourceConfigs[resource] = true
		} else {
			o.ResourceConfigs[resource] = false
			klog.V(1).Infof("resource %s cannot be enabled because: %s", resource.String(), err.Error())
		}
	}
}

func (o *ResourceConfig) ResourceEnabled(resource schema.GroupVersionResource) bool {
	// if a resource is explicitly set, that takes priority over the preference of the version.
	resourceEnabled, explicitlySet := o.ResourceConfigs[resource]
	if explicitlySet {
		return resourceEnabled
	}
	if available, _ := o.resourceAvailable(resource); !available {
		return false
	}
	if !o.versionEnabled(resource.GroupVersion()) {
		return false
	}
	// they are enabled by default.
	return true
}

func (o *ResourceConfig) AnyResourceForGroupEnabled(group string) bool {
	for version := range o.GroupVersionConfigs {
		if version.Group == group {
			if o.versionEnabled(version) {
				return true
			}
		}
	}
	for resource := range o.ResourceConfigs {
		if resource.Group == group && o.ResourceEnabled(resource) {
			return true
		}
	}

	return false
}
