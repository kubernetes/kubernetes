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

package master

import (
	"k8s.io/kubernetes/pkg/api/rest"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	storageapiv1beta1 "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	"k8s.io/kubernetes/pkg/genericapiserver"
	storageclassetcd "k8s.io/kubernetes/pkg/registry/storageclass/etcd"
)

type StorageRESTStorageProvider struct {
}

var _ RESTStorageProvider = &StorageRESTStorageProvider{}

func (p StorageRESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(storageapi.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(storageapiv1beta1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[storageapiv1beta1.SchemeGroupVersion.Version] = p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = storageapiv1beta1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p StorageRESTStorageProvider) v1beta1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) map[string]rest.Storage {
	version := storageapiv1beta1.SchemeGroupVersion

	storage := map[string]rest.Storage{}

	if apiResourceConfigSource.ResourceEnabled(version.WithResource("storageclasses")) {
		storageClassStorage := storageclassetcd.NewREST(restOptionsGetter(storageapi.Resource("storageclasses")))
		storage["storageclasses"] = storageClassStorage
	}

	return storage
}
