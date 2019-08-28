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
	policyapiv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/policy"
	poddisruptionbudgetstore "k8s.io/kubernetes/pkg/registry/policy/poddisruptionbudget/storage"
	pspstore "k8s.io/kubernetes/pkg/registry/policy/podsecuritypolicy/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(policy.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.VersionEnabled(policyapiv1beta1.SchemeGroupVersion) {
		if storageMap, err := p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[policyapiv1beta1.SchemeGroupVersion.Version] = storageMap
		}
	}
	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// poddisruptionbudgets
	poddisruptionbudgetStorage, poddisruptionbudgetStatusStorage, err := poddisruptionbudgetstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["poddisruptionbudgets"] = poddisruptionbudgetStorage
	storage["poddisruptionbudgets/status"] = poddisruptionbudgetStatusStorage

	rest, err := pspstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["podsecuritypolicies"] = rest

	return storage, err
}

func (p RESTStorageProvider) GroupName() string {
	return policy.GroupName
}
