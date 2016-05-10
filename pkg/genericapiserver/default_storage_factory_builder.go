/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package genericapiserver

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/util/config"
)

// Builds the DefaultStorageFactory.
// Merges defaultResourceConfig with the user specified overrides and merges
// defaultAPIResourceConfig with the corresponding user specified overrides as well.
func BuildDefaultStorageFactory(storageConfig storagebackend.Config, defaultMediaType string, serializer runtime.StorageSerializer,
	defaultResourceEncoding *DefaultResourceEncodingConfig, storageEncodingOverrides map[string]unversioned.GroupVersion, defaultAPIResourceConfig *ResourceConfig, resourceConfigOverrides config.ConfigurationMap) (*DefaultStorageFactory, error) {

	resourceEncodingConfig := mergeResourceEncodingConfigs(defaultResourceEncoding, storageEncodingOverrides)
	apiResourceConfig, err := mergeAPIResourceConfigs(defaultAPIResourceConfig, resourceConfigOverrides)
	if err != nil {
		return nil, err
	}
	return NewDefaultStorageFactory(storageConfig, defaultMediaType, serializer, resourceEncodingConfig, apiResourceConfig), nil
}

// Merges the given defaultAPIResourceConfig with the given storageEncodingOverrides.
func mergeResourceEncodingConfigs(defaultResourceEncoding *DefaultResourceEncodingConfig, storageEncodingOverrides map[string]unversioned.GroupVersion) *DefaultResourceEncodingConfig {
	resourceEncodingConfig := defaultResourceEncoding
	for group, storageEncodingVersion := range storageEncodingOverrides {
		resourceEncodingConfig.SetVersionEncoding(group, storageEncodingVersion, unversioned.GroupVersion{Group: group, Version: runtime.APIVersionInternal})
	}
	return resourceEncodingConfig
}

// Merges the given defaultAPIResourceConfig with the given resourceConfigOverrides.
func mergeAPIResourceConfigs(defaultAPIResourceConfig *ResourceConfig, resourceConfigOverrides config.ConfigurationMap) (*ResourceConfig, error) {
	resourceConfig := defaultAPIResourceConfig
	overrides := resourceConfigOverrides

	// "api/all=false" allows users to selectively enable specific api versions.
	allAPIFlagValue, ok := overrides["api/all"]
	if ok && allAPIFlagValue == "false" {
		// Disable all group versions.
		for _, groupVersion := range registered.RegisteredGroupVersions() {
			if resourceConfig.AnyResourcesForVersionEnabled(groupVersion) {
				resourceConfig.DisableVersions(groupVersion)
			}
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
			return nil, fmt.Errorf("invalid key %s", key)
		}
		// Verify that the groupVersion is registered.
		if !registered.IsRegisteredVersion(groupVersion) {
			return nil, fmt.Errorf("group version %s that has not been registered", groupVersion.String())
		}
		enabled, err := getRuntimeConfigValue(overrides, key, false)
		if err != nil {
			return nil, err
		}
		if enabled {
			resourceConfig.EnableVersions(groupVersion)
		} else {
			resourceConfig.DisableVersions(groupVersion)
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
			return nil, fmt.Errorf("invalid key %s", key)
		}
		resource := tokens[2]
		// Verify that the groupVersion is registered.
		if !registered.IsRegisteredVersion(groupVersion) {
			return nil, fmt.Errorf("group version %s that has not been registered", groupVersion.String())
		}

		if !resourceConfig.AnyResourcesForVersionEnabled(groupVersion) {
			return nil, fmt.Errorf("%v is disabled, you cannot configure its resources individually", groupVersion)
		}

		enabled, err := getRuntimeConfigValue(overrides, key, false)
		if err != nil {
			return nil, err
		}
		if enabled {
			resourceConfig.EnableResources(groupVersion.WithResource(resource))
		} else {
			resourceConfig.DisableResources(groupVersion.WithResource(resource))
		}
	}
	return resourceConfig, nil
}

func getRuntimeConfigValue(overrides config.ConfigurationMap, apiKey string, defaultValue bool) (bool, error) {
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
