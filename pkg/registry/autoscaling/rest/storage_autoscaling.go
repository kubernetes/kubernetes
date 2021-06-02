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
	autoscalingapiv1 "k8s.io/api/autoscaling/v1"
	autoscalingapiv2beta1 "k8s.io/api/autoscaling/v2beta1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingapiv2beta2 "k8s.io/kubernetes/pkg/apis/autoscaling/v2beta2"
	horizontalpodautoscalerstore "k8s.io/kubernetes/pkg/registry/autoscaling/horizontalpodautoscaler/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(autoscaling.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.VersionEnabled(autoscalingapiv2beta2.SchemeGroupVersion) {
		if storageMap, err := p.v2beta2Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[autoscalingapiv2beta2.SchemeGroupVersion.Version] = storageMap
		}
	}
	if apiResourceConfigSource.VersionEnabled(autoscalingapiv2beta1.SchemeGroupVersion) {
		if storageMap, err := p.v2beta1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[autoscalingapiv2beta1.SchemeGroupVersion.Version] = storageMap
		}
	}
	if apiResourceConfigSource.VersionEnabled(autoscalingapiv1.SchemeGroupVersion) {
		if storageMap, err := p.v1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[autoscalingapiv1.SchemeGroupVersion.Version] = storageMap
		}
	}

	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// horizontalpodautoscalers
	hpaStorage, hpaStatusStorage, err := horizontalpodautoscalerstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["horizontalpodautoscalers"] = hpaStorage
	storage["horizontalpodautoscalers/status"] = hpaStatusStorage

	return storage, err
}

func (p RESTStorageProvider) v2beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// horizontalpodautoscalers
	hpaStorage, hpaStatusStorage, err := horizontalpodautoscalerstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["horizontalpodautoscalers"] = hpaStorage
	storage["horizontalpodautoscalers/status"] = hpaStatusStorage

	return storage, err
}

func (p RESTStorageProvider) v2beta2Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// horizontalpodautoscalers
	hpaStorage, hpaStatusStorage, err := horizontalpodautoscalerstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["horizontalpodautoscalers"] = hpaStorage
	storage["horizontalpodautoscalers/status"] = hpaStatusStorage

	return storage, err
}

func (p RESTStorageProvider) GroupName() string {
	return autoscaling.GroupName
}
