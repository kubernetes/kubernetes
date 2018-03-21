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

package kubeapiserver

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/server/resourceconfig"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

// SpecialDefaultResourcePrefixes are prefixes compiled into Kubernetes.
var SpecialDefaultResourcePrefixes = map[schema.GroupResource]string{
	{Group: "", Resource: "replicationcontrollers"}:        "controllers",
	{Group: "", Resource: "endpoints"}:                     "services/endpoints",
	{Group: "", Resource: "nodes"}:                         "minions",
	{Group: "", Resource: "services"}:                      "services/specs",
	{Group: "extensions", Resource: "ingresses"}:           "ingress",
	{Group: "extensions", Resource: "podsecuritypolicies"}: "podsecuritypolicy",
	{Group: "policy", Resource: "podsecuritypolicies"}:     "podsecuritypolicy",
}

// NewStorageFactory builds the DefaultStorageFactory.
// Merges defaultResourceEncoding with the user specified overrides.
func NewStorageFactory(
	storageConfig storagebackend.Config,
	defaultMediaType string,
	serializer runtime.StorageSerializer,
	defaultResourceEncoding *serverstorage.DefaultResourceEncodingConfig,
	storageEncodingOverrides map[string]schema.GroupVersion,
	resourceEncodingOverrides []schema.GroupVersionResource,
	apiResourceConfig *serverstorage.ResourceConfig,
) (*serverstorage.DefaultStorageFactory, error) {
	resourceEncodingConfig := resourceconfig.MergeGroupEncodingConfigs(defaultResourceEncoding, storageEncodingOverrides)
	resourceEncodingConfig = resourceconfig.MergeResourceEncodingConfigs(resourceEncodingConfig, resourceEncodingOverrides)
	return serverstorage.NewDefaultStorageFactory(storageConfig, defaultMediaType, serializer, resourceEncodingConfig, apiResourceConfig, SpecialDefaultResourcePrefixes), nil
}
