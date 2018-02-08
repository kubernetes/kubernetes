/*
Copyright 2018 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	apiservicestorage "k8s.io/kube-aggregator/pkg/registry/apiservice/etcd"
)

// NewRESTStorage returns an APIGroupInfo object that will work against apiservice.
func NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) genericapiserver.APIGroupInfo {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiregistration.GroupName, aggregatorscheme.Registry, aggregatorscheme.Scheme, metav1.ParameterCodec, aggregatorscheme.Codecs)

	if apiResourceConfigSource.VersionEnabled(v1beta1.SchemeGroupVersion) {
		apiGroupInfo.GroupMeta.GroupVersion = v1beta1.SchemeGroupVersion
		storage := map[string]rest.Storage{}
		apiServiceREST := apiservicestorage.NewREST(aggregatorscheme.Scheme, restOptionsGetter)
		storage["apiservices"] = apiServiceREST
		storage["apiservices/status"] = apiservicestorage.NewStatusREST(aggregatorscheme.Scheme, apiServiceREST)
		apiGroupInfo.VersionedResourcesStorageMap["v1beta1"] = storage
	}

	if apiResourceConfigSource.VersionEnabled(v1.SchemeGroupVersion) {
		apiGroupInfo.GroupMeta.GroupVersion = v1.SchemeGroupVersion
		storage := map[string]rest.Storage{}
		apiServiceREST := apiservicestorage.NewREST(aggregatorscheme.Scheme, restOptionsGetter)
		storage["apiservices"] = apiServiceREST
		storage["apiservices/status"] = apiservicestorage.NewStatusREST(aggregatorscheme.Scheme, apiServiceREST)
		apiGroupInfo.VersionedResourcesStorageMap["v1"] = storage
	}

	return apiGroupInfo
}
