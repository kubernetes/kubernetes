/*
Copyright The Kubernetes Authors.

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
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/lifecycle"
	"k8s.io/kubernetes/pkg/features"
	evictionstorage "k8s.io/kubernetes/pkg/registry/lifecycle/eviction/storage"
	evictionrequeststorage "k8s.io/kubernetes/pkg/registry/lifecycle/evictionrequest/storage"
	"k8s.io/utils/clock"
)

type RESTStorageProvider struct {
}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(lifecycle.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if storageMap, err := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[lifecyclev1alpha1.SchemeGroupVersion.Version] = storageMap
	}

	return apiGroupInfo, nil
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	if resource := "evictions"; apiResourceConfigSource.ResourceEnabled(lifecyclev1alpha1.SchemeGroupVersion.WithResource(resource)) {
		if utilfeature.DefaultFeatureGate.Enabled(features.EvictionRequestAPI) {
			evictionStorage, evictionStatusStorage, err := evictionstorage.NewREST(restOptionsGetter, clock.RealClock{})
			if err != nil {
				return storage, err
			}
			storage[resource] = evictionStorage
			storage[resource+"/status"] = evictionStatusStorage
		} else {
			klog.Warning("Eviction storage is disabled because the EvictionRequestAPI feature gate is disabled")
		}
	}
	if resource := "evictionrequests"; apiResourceConfigSource.ResourceEnabled(lifecyclev1alpha1.SchemeGroupVersion.WithResource(resource)) {
		if utilfeature.DefaultFeatureGate.Enabled(features.EvictionRequestAPI) {
			evictionRequestStorage, evictionRequestStatusStorage, err := evictionrequeststorage.NewREST(restOptionsGetter)
			if err != nil {
				return storage, err
			}
			storage[resource] = evictionRequestStorage
			storage[resource+"/status"] = evictionRequestStatusStorage
		} else {
			klog.Warning("EvictionRequest storage is disabled because the EvictionRequestAPI feature gate is disabled")
		}
	}

	return storage, nil
}

func (p RESTStorageProvider) GroupName() string {
	return lifecycle.GroupName
}
