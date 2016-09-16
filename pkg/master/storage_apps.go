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
	"k8s.io/kubernetes/pkg/apis/apps"
	appsapiv1alpha1 "k8s.io/kubernetes/pkg/apis/apps/v1alpha1"
	"k8s.io/kubernetes/pkg/genericapiserver"
	petsetetcd "k8s.io/kubernetes/pkg/registry/petset/etcd"
)

type AppsRESTStorageProvider struct{}

var _ RESTStorageProvider = &AppsRESTStorageProvider{}

func (p AppsRESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apps.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(appsapiv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[appsapiv1alpha1.SchemeGroupVersion.Version] = p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = appsapiv1alpha1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p AppsRESTStorageProvider) v1alpha1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) map[string]rest.Storage {
	version := appsapiv1alpha1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("petsets")) {
		petsetStorage, petsetStatusStorage := petsetetcd.NewREST(restOptionsGetter(apps.Resource("petsets")))
		storage["petsets"] = petsetStorage
		storage["petsets/status"] = petsetStatusStorage
	}
	return storage
}
