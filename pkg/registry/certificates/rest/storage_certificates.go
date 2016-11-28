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
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/certificates"
	certificatesapiv1alpha1 "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry"
	certificateetcd "k8s.io/kubernetes/pkg/registry/certificates/certificates/etcd"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter registry.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(certificates.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(certificatesapiv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[certificatesapiv1alpha1.SchemeGroupVersion.Version] = p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = certificatesapiv1alpha1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter registry.RESTOptionsGetter) map[string]rest.Storage {
	version := certificatesapiv1alpha1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("certificatesigningrequests")) {
		csrStorage, csrStatusStorage, csrApprovalStorage := certificateetcd.NewREST(restOptionsGetter(certificates.Resource("certificatesigningrequests")))
		storage["certificatesigningrequests"] = csrStorage
		storage["certificatesigningrequests/status"] = csrStatusStorage
		storage["certificatesigningrequests/approval"] = csrApprovalStorage
	}
	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return certificates.GroupName
}
