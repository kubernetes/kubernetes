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

package config

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"fmt"
	"strconv"
	"strings"
)

// APIResourceConfigSource is the interface to determine which versions and resources are enabled
type APIResourceConfigSource interface {
	AnyVersionOfResourceEnabled(resource unversioned.GroupResource) bool
	ResourceEnabled(resource unversioned.GroupVersionResource) bool
	AllResourcesForVersionEnabled(version unversioned.GroupVersion) bool
	AnyResourcesForVersionEnabled(version unversioned.GroupVersion) bool
	AnyResourcesForGroupEnabled(group string) bool
}

// Specifies the overrides for various API group versions.
// This can be used to enable/disable entire group versions or specific resources.
type GroupVersionResourceConfig struct {
	// Whether to enable or disable this entire group version.  This dominates any enablement check.
	// Enable=true means the group version is enabled, and EnabledResources/DisabledResources are considered.
	// Enable=false means the group version is disabled, and EnabledResources/DisabledResources are not considered.
	Enable bool

	// DisabledResources lists the resources that are specifically disabled for a group/version
	// DisabledResources trumps EnabledResources
	DisabledResources sets.String

	// EnabledResources lists the resources that should be enabled by default.  This is a little
	// unusual, but we need it for compatibility with old code for now.  An empty set means
	// enable all, a non-empty set means that all other resources are disabled.
	EnabledResources sets.String
}

var _ APIResourceConfigSource = &ResourceConfig{}

type ResourceConfig struct {
	GroupVersionResourceConfigs map[unversioned.GroupVersion]*GroupVersionResourceConfig
}

func NewResourceConfig() *ResourceConfig {
	return &ResourceConfig{GroupVersionResourceConfigs: map[unversioned.GroupVersion]*GroupVersionResourceConfig{}}
}

func NewGroupVersionResourceConfig() *GroupVersionResourceConfig {
	return &GroupVersionResourceConfig{Enable: true, DisabledResources: sets.String{}, EnabledResources: sets.String{}}
}

// DisableVersions disables the versions entirely.  No resources (even those whitelisted in EnabledResources) will be enabled
func (o *ResourceConfig) DisableVersions(versions ...unversioned.GroupVersion) {
	for _, version := range versions {
		_, versionExists := o.GroupVersionResourceConfigs[version]
		if !versionExists {
			o.GroupVersionResourceConfigs[version] = NewGroupVersionResourceConfig()
		}

		o.GroupVersionResourceConfigs[version].Enable = false
	}
}

func (o *ResourceConfig) EnableVersions(versions ...unversioned.GroupVersion) {
	for _, version := range versions {
		_, versionExists := o.GroupVersionResourceConfigs[version]
		if !versionExists {
			o.GroupVersionResourceConfigs[version] = NewGroupVersionResourceConfig()
		}

		o.GroupVersionResourceConfigs[version].Enable = true
	}
}

func (o *ResourceConfig) DisableResources(resources ...unversioned.GroupVersionResource) {
	for _, resource := range resources {
		version := resource.GroupVersion()
		_, versionExists := o.GroupVersionResourceConfigs[version]
		if !versionExists {
			o.GroupVersionResourceConfigs[version] = NewGroupVersionResourceConfig()
		}

		o.GroupVersionResourceConfigs[version].DisabledResources.Insert(resource.Resource)
	}
}

func (o *ResourceConfig) EnableResources(resources ...unversioned.GroupVersionResource) {
	for _, resource := range resources {
		version := resource.GroupVersion()
		_, versionExists := o.GroupVersionResourceConfigs[version]
		if !versionExists {
			o.GroupVersionResourceConfigs[version] = NewGroupVersionResourceConfig()
		}

		o.GroupVersionResourceConfigs[version].EnabledResources.Insert(resource.Resource)
		o.GroupVersionResourceConfigs[version].DisabledResources.Delete(resource.Resource)
	}
}

// AnyResourcesForVersionEnabled only considers matches based on exactly group/resource lexical matching.  This means that
// resource renames across versions are NOT considered to be the same resource by this method. You'll need to manually check
// using the ResourceEnabled function.
func (o *ResourceConfig) AnyVersionOfResourceEnabled(resource unversioned.GroupResource) bool {
	for version := range o.GroupVersionResourceConfigs {
		if version.Group != resource.Group {
			continue
		}

		if o.ResourceEnabled(version.WithResource(resource.Resource)) {
			return true
		}
	}

	return false
}

func (o *ResourceConfig) ResourceEnabled(resource unversioned.GroupVersionResource) bool {
	versionOverride, versionExists := o.GroupVersionResourceConfigs[resource.GroupVersion()]
	if !versionExists {
		return false
	}
	if !versionOverride.Enable {
		return false
	}

	if versionOverride.DisabledResources.Has(resource.Resource) {
		return false
	}

	if len(versionOverride.EnabledResources) > 0 {
		return versionOverride.EnabledResources.Has(resource.Resource)
	}

	return true
}

func (o *ResourceConfig) AllResourcesForVersionEnabled(version unversioned.GroupVersion) bool {
	versionOverride, versionExists := o.GroupVersionResourceConfigs[version]
	if !versionExists {
		return false
	}
	if !versionOverride.Enable {
		return false
	}

	if len(versionOverride.EnabledResources) == 0 && len(versionOverride.DisabledResources) == 0 {
		return true
	}

	return false
}

func (o *ResourceConfig) AnyResourcesForVersionEnabled(version unversioned.GroupVersion) bool {
	versionOverride, versionExists := o.GroupVersionResourceConfigs[version]
	if !versionExists {
		return false
	}

	return versionOverride.Enable
}

func (o *ResourceConfig) AnyResourcesForGroupEnabled(group string) bool {
	for version := range o.GroupVersionResourceConfigs {
		if version.Group == group {
			if o.AnyResourcesForVersionEnabled(version) {
				return true
			}
		}
	}

	return false
}


// Merges with the given resourceConfigOverrides.
func (o *ResourceConfig) MergeResourceConfigs(resourceConfigOverrides ConfigurationMap) error {
	overrides := resourceConfigOverrides

	// "api/all=false" allows users to selectively enable specific api versions.
	allAPIFlagValue, ok := overrides["api/all"]
	if ok {
		if allAPIFlagValue == "false" {
			// Disable all group versions.
			o.DisableVersions(registered.RegisteredGroupVersions()...)
		} else if allAPIFlagValue == "true" {
			o.EnableVersions(registered.RegisteredGroupVersions()...)
		}
	}

	// "api/legacy=false" allows users to disable legacy api versions.
	disableLegacyAPIs := false
	legacyAPIFlagValue, ok := overrides["api/legacy"]
	if ok && legacyAPIFlagValue == "false" {
		disableLegacyAPIs = true
	}
	_ = disableLegacyAPIs // hush the compiler while we don't have legacy APIs to disable.

	// "<resourceSpecifier>={true|false} allows users to enable/disable API.
	// This takes preference over api/all and api/legacy, if specified.
	// Iterate through all group/version overrides specified in runtimeConfig.
	for key := range overrides {
		if key == "api/all" || key == "api/legacy" {
			// Have already handled them above. Can skip them here.
			continue
		}
		tokens := strings.Split(key, "/")
		if len(tokens) != 2 {
			continue
		}
		groupVersionString := tokens[0] + "/" + tokens[1]
		// HACK: Hack for "v1" legacy group version.
		// Remove when we stop supporting the legacy group version.
		if groupVersionString == "api/v1" {
			groupVersionString = "v1"
		}
		groupVersion, err := unversioned.ParseGroupVersion(groupVersionString)
		if err != nil {
			return fmt.Errorf("invalid key %s", key)
		}
		// Verify that the groupVersion is registered.
		if !registered.IsRegisteredVersion(groupVersion) {
			return fmt.Errorf("group version %s that has not been registered", groupVersion.String())
		}
		enabled, err := getRuntimeConfigValue(overrides, key, false)
		if err != nil {
			return err
		}
		if enabled {
			o.EnableVersions(groupVersion)
		} else {
			o.DisableVersions(groupVersion)
		}
	}

	// Iterate through all group/version/resource overrides specified in runtimeConfig.
	for key := range overrides {
		tokens := strings.Split(key, "/")
		if len(tokens) != 3 {
			continue
		}
		groupVersionString := tokens[0] + "/" + tokens[1]
		// HACK: Hack for "v1" legacy group version.
		// Remove when we stop supporting the legacy group version.
		if groupVersionString == "api/v1" {
			groupVersionString = "v1"
		}
		groupVersion, err := unversioned.ParseGroupVersion(groupVersionString)
		if err != nil {
			return fmt.Errorf("invalid key %s", key)
		}
		resource := tokens[2]
		// Verify that the groupVersion is registered.
		if !registered.IsRegisteredVersion(groupVersion) {
			return fmt.Errorf("group version %s that has not been registered", groupVersion.String())
		}

		if !o.AnyResourcesForVersionEnabled(groupVersion) {
			return fmt.Errorf("%v is disabled, you cannot configure its resources individually", groupVersion)
		}

		enabled, err := getRuntimeConfigValue(overrides, key, false)
		if err != nil {
			return err
		}
		if enabled {
			o.EnableResources(groupVersion.WithResource(resource))
		} else {
			o.DisableResources(groupVersion.WithResource(resource))
		}
	}
	return nil
}

func getRuntimeConfigValue(overrides ConfigurationMap, apiKey string, defaultValue bool) (bool, error) {
	flagValue, ok := overrides[apiKey]
	if ok {
		if flagValue == "" {
			return true, nil
		}
		boolValue, err := strconv.ParseBool(flagValue)
		if err != nil {
			return false, fmt.Errorf("invalid value of %s: %s, err: %v", apiKey, flagValue, err)
		}
		return boolValue, nil
	}
	return defaultValue, nil
}
