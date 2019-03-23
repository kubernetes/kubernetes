/*
Copyright 2017 The Kubernetes Authors.

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

package resourceconfig

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	serverstore "k8s.io/apiserver/pkg/server/storage"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/klog"
)

// GroupVersionRegistry provides access to registered group versions.
type GroupVersionRegistry interface {
	// IsGroupRegistered returns true if given group is registered.
	IsGroupRegistered(group string) bool
	// IsVersionRegistered returns true if given version is registered.
	IsVersionRegistered(v schema.GroupVersion) bool
	// PrioritizedVersionsAllGroups returns all registered group versions.
	PrioritizedVersionsAllGroups() []schema.GroupVersion
}

// MergeResourceEncodingConfigs merges the given defaultResourceConfig with specific GroupVersionResource overrides.
func MergeResourceEncodingConfigs(
	defaultResourceEncoding *serverstore.DefaultResourceEncodingConfig,
	resourceEncodingOverrides []schema.GroupVersionResource,
) *serverstore.DefaultResourceEncodingConfig {
	resourceEncodingConfig := defaultResourceEncoding
	for _, gvr := range resourceEncodingOverrides {
		resourceEncodingConfig.SetResourceEncoding(gvr.GroupResource(), gvr.GroupVersion(),
			schema.GroupVersion{Group: gvr.Group, Version: runtime.APIVersionInternal})
	}
	return resourceEncodingConfig
}

// MergeAPIResourceConfigs merges the given defaultAPIResourceConfig with the given resourceConfigOverrides.
// Exclude the groups not registered in registry, and check if version is
// not registered in group, then it will fail.
func MergeAPIResourceConfigs(
	defaultAPIResourceConfig *serverstore.ResourceConfig,
	resourceConfigOverrides cliflag.ConfigurationMap,
	registry GroupVersionRegistry,
) (*serverstore.ResourceConfig, error) {
	resourceConfig := defaultAPIResourceConfig
	overrides := resourceConfigOverrides

	// "api/all=false" allows users to selectively enable specific api versions.
	allAPIFlagValue, ok := overrides["api/all"]
	if ok {
		if allAPIFlagValue == "false" {
			// Disable all group versions.
			resourceConfig.DisableAll()
		} else if allAPIFlagValue == "true" {
			resourceConfig.EnableAll()
		}
	}

	// "<resourceSpecifier>={true|false} allows users to enable/disable API.
	// This takes preference over api/all, if specified.
	// Iterate through all group/version overrides specified in runtimeConfig.
	for key := range overrides {
		// Have already handled them above. Can skip them here.
		if key == "api/all" {
			continue
		}

		tokens := strings.Split(key, "/")
		if len(tokens) < 2 {
			continue
		}
		groupVersionString := tokens[0] + "/" + tokens[1]
		groupVersion, err := schema.ParseGroupVersion(groupVersionString)
		if err != nil {
			return nil, fmt.Errorf("invalid key %s", key)
		}

		// individual resource enablement/disablement is only supported in the extensions/v1beta1 API group for legacy reasons.
		// all other API groups are expected to contain coherent sets of resources that are enabled/disabled together.
		if len(tokens) > 2 && (groupVersion != schema.GroupVersion{Group: "extensions", Version: "v1beta1"}) {
			klog.Warningf("ignoring invalid key %s, individual resource enablement/disablement is not supported in %s, and will prevent starting in future releases", key, groupVersion.String())
			continue
		}

		// Exclude group not registered into the registry.
		if !registry.IsGroupRegistered(groupVersion.Group) {
			continue
		}

		// Verify that the groupVersion is registered into registry.
		if !registry.IsVersionRegistered(groupVersion) {
			return nil, fmt.Errorf("group version %s that has not been registered", groupVersion.String())
		}
		enabled, err := getRuntimeConfigValue(overrides, key, false)
		if err != nil {
			return nil, err
		}
		if enabled {
			// enable the groupVersion for "group/version=true" and "group/version/resource=true"
			resourceConfig.EnableVersions(groupVersion)
		} else if len(tokens) == 2 {
			// disable the groupVersion only for "group/version=false", not "group/version/resource=false"
			resourceConfig.DisableVersions(groupVersion)
		}

		if len(tokens) < 3 {
			continue
		}
		groupVersionResource := groupVersion.WithResource(tokens[2])
		if enabled {
			resourceConfig.EnableResources(groupVersionResource)
		} else {
			resourceConfig.DisableResources(groupVersionResource)
		}
	}

	return resourceConfig, nil
}

func getRuntimeConfigValue(overrides cliflag.ConfigurationMap, apiKey string, defaultValue bool) (bool, error) {
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

// ParseGroups takes in resourceConfig and returns parsed groups.
func ParseGroups(resourceConfig cliflag.ConfigurationMap) ([]string, error) {
	groups := []string{}
	for key := range resourceConfig {
		if key == "api/all" {
			continue
		}
		tokens := strings.Split(key, "/")
		if len(tokens) != 2 && len(tokens) != 3 {
			return groups, fmt.Errorf("runtime-config invalid key %s", key)
		}
		groupVersionString := tokens[0] + "/" + tokens[1]
		groupVersion, err := schema.ParseGroupVersion(groupVersionString)
		if err != nil {
			return nil, fmt.Errorf("runtime-config invalid key %s", key)
		}
		groups = append(groups, groupVersion.Group)
	}

	return groups, nil
}
