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
	"regexp"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	serverstore "k8s.io/apiserver/pkg/server/storage"
	cliflag "k8s.io/component-base/cli/flag"
)

// GroupVersionRegistry provides access to registered group versions.
type GroupVersionRegistry interface {
	// IsGroupRegistered returns true if given group is registered.
	IsGroupRegistered(group string) bool
	// IsVersionRegistered returns true if given version is registered.
	IsVersionRegistered(v schema.GroupVersion) bool
	// PrioritizedVersionsAllGroups returns all registered group versions.
	PrioritizedVersionsAllGroups() []schema.GroupVersion
	// PrioritizedVersionsForGroup returns versions for a single group in priority order
	PrioritizedVersionsForGroup(group string) []schema.GroupVersion
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

// Recognized values for the --runtime-config parameter to enable/disable groups of APIs
const (
	APIAll   = "api/all"
	APIGA    = "api/ga"
	APIBeta  = "api/beta"
	APIAlpha = "api/alpha"
)

var (
	gaPattern    = regexp.MustCompile(`^v\d+$`)
	betaPattern  = regexp.MustCompile(`^v\d+beta\d+$`)
	alphaPattern = regexp.MustCompile(`^v\d+alpha\d+$`)

	groupVersionMatchers = map[string]func(gv schema.GroupVersion) bool{
		// allows users to address all api versions
		APIAll: func(gv schema.GroupVersion) bool { return true },
		// allows users to address all api versions in the form v[0-9]+
		APIGA: func(gv schema.GroupVersion) bool { return gaPattern.MatchString(gv.Version) },
		// allows users to address all beta api versions
		APIBeta: func(gv schema.GroupVersion) bool { return betaPattern.MatchString(gv.Version) },
		// allows users to address all alpha api versions
		APIAlpha: func(gv schema.GroupVersion) bool { return alphaPattern.MatchString(gv.Version) },
	}

	groupVersionMatchersOrder = []string{APIAll, APIGA, APIBeta, APIAlpha}
)

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

	for _, flag := range groupVersionMatchersOrder {
		if value, ok := overrides[flag]; ok {
			if value == "false" {
				resourceConfig.DisableMatchingVersions(groupVersionMatchers[flag])
			} else if value == "true" {
				resourceConfig.EnableMatchingVersions(groupVersionMatchers[flag])
			} else {
				return nil, fmt.Errorf("invalid value %v=%v", flag, value)
			}
		}
	}
	if err := applyVersionAndResourcePreferences(resourceConfig, overrides, registry); err != nil {
		return nil, err
	}
	return resourceConfig, nil
}

func applyVersionAndResourcePreferences(
	resourceConfig *serverstore.ResourceConfig,
	overrides cliflag.ConfigurationMap,
	registry GroupVersionRegistry,
) error {
	type versionEnablementPreference struct {
		key          string
		enabled      bool
		groupVersion schema.GroupVersion
	}
	type resourceEnablementPreference struct {
		key                  string
		enabled              bool
		groupVersionResource schema.GroupVersionResource
	}
	versionPreferences := []versionEnablementPreference{}
	resourcePreferences := []resourceEnablementPreference{}

	// "<resourceSpecifier>={true|false} allows users to enable/disable API.
	// This takes preference over api/all, if specified.
	// Iterate through all group/version overrides specified in runtimeConfig.
	for key := range overrides {
		// Have already handled them above. Can skip them here.
		if _, ok := groupVersionMatchers[key]; ok {
			continue
		}

		tokens := strings.Split(key, "/")
		if len(tokens) < 2 || len(tokens) > 3 {
			continue
		}
		groupVersionString := tokens[0] + "/" + tokens[1]
		groupVersion, err := schema.ParseGroupVersion(groupVersionString)
		if err != nil {
			return fmt.Errorf("invalid key %s", key)
		}

		// Exclude group not registered into the registry.
		if !registry.IsGroupRegistered(groupVersion.Group) {
			continue
		}

		// Verify that the groupVersion is registered into registry.
		if !registry.IsVersionRegistered(groupVersion) {
			return fmt.Errorf("group version %s that has not been registered", groupVersion.String())
		}
		enabled, err := getRuntimeConfigValue(overrides, key, false)
		if err != nil {
			return err
		}

		switch len(tokens) {
		case 2:
			versionPreferences = append(versionPreferences, versionEnablementPreference{
				key:          key,
				enabled:      enabled,
				groupVersion: groupVersion,
			})
		case 3:
			if strings.ToLower(tokens[2]) != tokens[2] {
				return fmt.Errorf("invalid key %v: group/version/resource and resource is always lowercase plural, not %q", key, tokens[2])
			}
			resourcePreferences = append(resourcePreferences, resourceEnablementPreference{
				key:                  key,
				enabled:              enabled,
				groupVersionResource: groupVersion.WithResource(tokens[2]),
			})
		}
	}

	// apply version preferences first, so that we can remove the hardcoded resource preferences that are being overridden
	for _, versionPreference := range versionPreferences {
		if versionPreference.enabled {
			// enable the groupVersion for "group/version=true"
			resourceConfig.EnableVersions(versionPreference.groupVersion)

		} else {
			// disable the groupVersion only for "group/version=false"
			resourceConfig.DisableVersions(versionPreference.groupVersion)
		}
	}

	// apply resource preferences last, so they have the highest priority
	for _, resourcePreference := range resourcePreferences {
		if resourcePreference.enabled {
			// enable the resource for "group/version/resource=true"
			resourceConfig.EnableResources(resourcePreference.groupVersionResource)
		} else {
			resourceConfig.DisableResources(resourcePreference.groupVersionResource)
		}
	}
	return nil
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
		if _, ok := groupVersionMatchers[key]; ok {
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

// EmulationForwardCompatibleResourceConfig creates a new ResourceConfig that besides all the enabled resources in resourceConfig,
// enables all higher priority versions of enabled resources, excluding alpha versions.
// This is useful for ensuring forward compatibility when a new version of an API is introduced.
func EmulationForwardCompatibleResourceConfig(
	resourceConfig *serverstore.ResourceConfig,
	resourceConfigOverrides cliflag.ConfigurationMap,
	registry GroupVersionRegistry,
) (*serverstore.ResourceConfig, error) {
	ret := serverstore.NewResourceConfig()
	for gv, enabled := range resourceConfig.GroupVersionConfigs {
		ret.GroupVersionConfigs[gv] = enabled
		if !enabled {
			continue
		}
		// EmulationForwardCompatibility is not applicable to alpha apis.
		if alphaPattern.MatchString(gv.Version) {
			continue
		}
		for _, pgv := range registry.PrioritizedVersionsForGroup(gv.Group) {
			if pgv.Version == gv.Version {
				break
			}
			ret.EnableVersions(pgv)
		}
	}
	for gvr, enabled := range resourceConfig.ResourceConfigs {
		ret.ResourceConfigs[gvr] = enabled
		if !enabled {
			continue
		}
		// EmulationForwardCompatibility is not applicable to alpha apis.
		if alphaPattern.MatchString(gvr.Version) {
			continue
		}
		for _, pgv := range registry.PrioritizedVersionsForGroup(gvr.Group) {
			if pgv.Version == gvr.Version {
				break
			}
			ret.EnableResources(pgv.WithResource(gvr.Resource))
		}
	}
	// need to reapply the version preferences if there is an override of a higher priority version.
	if err := applyVersionAndResourcePreferences(ret, resourceConfigOverrides, registry); err != nil {
		return nil, err
	}
	return ret, nil
}
