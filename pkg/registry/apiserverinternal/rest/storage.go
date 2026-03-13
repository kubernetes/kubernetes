/*
Copyright 2020 The Kubernetes Authors.

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

package rest

import (
	apiserverv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apiserverinternal"
	storageversionstorage "k8s.io/kubernetes/pkg/registry/apiserverinternal/storageversion/storage"
)

// StorageProvider is a REST storage provider for internal.apiserver.k8s.io
type StorageProvider struct{}

// NewRESTStorage returns a StorageProvider
func (p StorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiserverinternal.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)

	if storageMap, err := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[apiserverv1alpha1.SchemeGroupVersion.Version] = storageMap
	}

	return apiGroupInfo, nil
}

func (p StorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	if resource := "storageversions"; apiResourceConfigSource.ResourceEnabled(apiserverv1alpha1.SchemeGroupVersion.WithResource(resource)) {
		s, status, err := storageversionstorage.NewREST(restOptionsGetter)
		if err != nil {
			return nil, err
		}
		storage[resource] = s
		storage[resource+"/status"] = status
	}

	return storage, nil
}

// GroupName is the group name for the storage provider
func (p StorageProvider) GroupName() string {
	return apiserverinternal.GroupName
}
