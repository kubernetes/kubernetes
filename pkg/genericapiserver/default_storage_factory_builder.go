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

package genericapiserver

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/util/config"
)

// Builds the DefaultStorageFactory.
// Merges defaultResourceConfig with the user specified overrides and merges
// defaultAPIResourceConfig with the corresponding user specified overrides as well.
func BuildDefaultStorageFactory(storageConfig storagebackend.Config, defaultMediaType string, serializer runtime.StorageSerializer,
	defaultResourceEncoding *DefaultResourceEncodingConfig, storageEncodingOverrides map[string]unversioned.GroupVersion, resourceEncodingOverrides []unversioned.GroupVersionResource,
	defaultAPIResourceConfig *config.ResourceConfig, resourceConfigOverrides config.ConfigurationMap) (*DefaultStorageFactory, error) {

	resourceEncodingConfig := mergeGroupEncodingConfigs(defaultResourceEncoding, storageEncodingOverrides)
	resourceEncodingConfig = mergeResourceEncodingConfigs(resourceEncodingConfig, resourceEncodingOverrides)
	apiResourceConfig := defaultAPIResourceConfig
	err := apiResourceConfig.MergeResourceConfigs(resourceConfigOverrides)
	if err != nil {
		return nil, err
	}
	return NewDefaultStorageFactory(storageConfig, defaultMediaType, serializer, resourceEncodingConfig, apiResourceConfig), nil
}

// Merges the given defaultResourceConfig with specifc GroupvVersionResource overrides.
func mergeResourceEncodingConfigs(defaultResourceEncoding *DefaultResourceEncodingConfig, resourceEncodingOverrides []unversioned.GroupVersionResource) *DefaultResourceEncodingConfig {
	resourceEncodingConfig := defaultResourceEncoding
	for _, gvr := range resourceEncodingOverrides {
		resourceEncodingConfig.SetResourceEncoding(gvr.GroupResource(), gvr.GroupVersion(),
			unversioned.GroupVersion{Group: gvr.Group, Version: runtime.APIVersionInternal})
	}
	return resourceEncodingConfig
}

// Merges the given defaultResourceConfig with specifc GroupVersion overrides.
func mergeGroupEncodingConfigs(defaultResourceEncoding *DefaultResourceEncodingConfig, storageEncodingOverrides map[string]unversioned.GroupVersion) *DefaultResourceEncodingConfig {
	resourceEncodingConfig := defaultResourceEncoding
	for group, storageEncodingVersion := range storageEncodingOverrides {
		resourceEncodingConfig.SetVersionEncoding(group, storageEncodingVersion, unversioned.GroupVersion{Group: group, Version: runtime.APIVersionInternal})
	}
	return resourceEncodingConfig
}
