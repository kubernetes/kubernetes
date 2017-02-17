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

package rest

import (
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	storageapiv1 "k8s.io/kubernetes/pkg/apis/storage/v1"
	storageapiv1beta1 "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	storageclassstore "k8s.io/kubernetes/pkg/registry/storage/storageclass/storage"
)

type RESTStorageProvider struct {
}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(storageapi.GroupName, api.Registry, api.Scheme, api.ParameterCodec, api.Codecs)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(storageapiv1beta1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[storageapiv1beta1.SchemeGroupVersion.Version] = p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = storageapiv1beta1.SchemeGroupVersion
	}
	if apiResourceConfigSource.AnyResourcesForVersionEnabled(storageapiv1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[storageapiv1.SchemeGroupVersion.Version] = p.v1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = storageapiv1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	version := storageapiv1beta1.SchemeGroupVersion

	storage := map[string]rest.Storage{}

	if apiResourceConfigSource.ResourceEnabled(version.WithResource("storageclasses")) {
		storageClassStorage := storageclassstore.NewREST(restOptionsGetter)
		storage["storageclasses"] = storageClassStorage
	}

	return storage
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	version := storageapiv1.SchemeGroupVersion

	storage := map[string]rest.Storage{}

	if apiResourceConfigSource.ResourceEnabled(version.WithResource("storageclasses")) {
		storageClassStorage := storageclassstore.NewREST(restOptionsGetter)
		storage["storageclasses"] = storageClassStorage
	}

	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return storageapi.GroupName
}
