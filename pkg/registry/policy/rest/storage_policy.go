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
	"k8s.io/kubernetes/pkg/apis/policy"
	policyapiv1beta1 "k8s.io/kubernetes/pkg/apis/policy/v1beta1"
	"k8s.io/kubernetes/pkg/genericapiserver"
	poddisruptionbudgetetcd "k8s.io/kubernetes/pkg/registry/policy/poddisruptionbudget/etcd"
)

type RESTStorageProvider struct{}

var _ genericapiserver.RESTStorageProvider = &RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(policy.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(policyapiv1beta1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[policyapiv1beta1.SchemeGroupVersion.Version] = p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = policyapiv1beta1.SchemeGroupVersion
	}
	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) map[string]rest.Storage {
	version := policyapiv1beta1.SchemeGroupVersion
	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("poddisruptionbudgets")) {
		poddisruptionbudgetStorage, poddisruptionbudgetStatusStorage := poddisruptionbudgetetcd.NewREST(restOptionsGetter(policy.Resource("poddisruptionbudgets")))
		storage["poddisruptionbudgets"] = poddisruptionbudgetStorage
		storage["poddisruptionbudgets/status"] = poddisruptionbudgetStatusStorage
	}
	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return policy.GroupName
}
