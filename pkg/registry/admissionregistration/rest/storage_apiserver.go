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
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	mutatingwebhookconfigurationstorage "k8s.io/kubernetes/pkg/registry/admissionregistration/mutatingwebhookconfiguration/storage"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
	validatingadmissionpolicystorage "k8s.io/kubernetes/pkg/registry/admissionregistration/validatingadmissionpolicy/storage"
	policybindingstorage "k8s.io/kubernetes/pkg/registry/admissionregistration/validatingadmissionpolicybinding/storage"
	validatingwebhookconfigurationstorage "k8s.io/kubernetes/pkg/registry/admissionregistration/validatingwebhookconfiguration/storage"
)

type RESTStorageProvider struct {
	Authorizer      authorizer.Authorizer
	DiscoveryClient discovery.DiscoveryInterface
}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(admissionregistration.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if storageMap, err := p.v1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[admissionregistrationv1.SchemeGroupVersion.Version] = storageMap
	}

	if storageMap, err := p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[admissionregistrationv1beta1.SchemeGroupVersion.Version] = storageMap
	}

	if storageMap, err := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[admissionregistrationv1alpha1.SchemeGroupVersion.Version] = storageMap
	}
	return apiGroupInfo, nil
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	// validatingwebhookconfigurations
	if resource := "validatingwebhookconfigurations"; apiResourceConfigSource.ResourceEnabled(admissionregistrationv1.SchemeGroupVersion.WithResource(resource)) {
		validatingStorage, err := validatingwebhookconfigurationstorage.NewREST(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage[resource] = validatingStorage
	}

	// mutatingwebhookconfigurations
	if resource := "mutatingwebhookconfigurations"; apiResourceConfigSource.ResourceEnabled(admissionregistrationv1.SchemeGroupVersion.WithResource(resource)) {
		mutatingStorage, err := mutatingwebhookconfigurationstorage.NewREST(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage[resource] = mutatingStorage
	}

	return storage, nil
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	// use a simple wrapper so that initialization order won't cause a nil getter
	var policyGetter rest.Getter

	r, err := resolver.NewDiscoveryResourceResolver(p.DiscoveryClient)
	if err != nil {
		return storage, err
	}

	// validatingadmissionpolicies
	if resource := "validatingadmissionpolicies"; apiResourceConfigSource.ResourceEnabled(admissionregistrationv1alpha1.SchemeGroupVersion.WithResource(resource)) {
		policyStorage, policyStatusStorage, err := validatingadmissionpolicystorage.NewREST(restOptionsGetter, p.Authorizer, r)
		if err != nil {
			return storage, err
		}
		policyGetter = policyStorage
		storage[resource] = policyStorage
		storage[resource+"/status"] = policyStatusStorage
	}

	// validatingadmissionpolicybindings
	if resource := "validatingadmissionpolicybindings"; apiResourceConfigSource.ResourceEnabled(admissionregistrationv1alpha1.SchemeGroupVersion.WithResource(resource)) {
		policyBindingStorage, err := policybindingstorage.NewREST(restOptionsGetter, p.Authorizer, &policybindingstorage.DefaultPolicyGetter{Getter: policyGetter}, r)
		if err != nil {
			return storage, err
		}
		storage[resource] = policyBindingStorage
	}

	return storage, nil
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	// use a simple wrapper so that initialization order won't cause a nil getter
	var policyGetter rest.Getter

	r, err := resolver.NewDiscoveryResourceResolver(p.DiscoveryClient)
	if err != nil {
		return storage, err
	}

	// validatingadmissionpolicies
	if resource := "validatingadmissionpolicies"; apiResourceConfigSource.ResourceEnabled(admissionregistrationv1beta1.SchemeGroupVersion.WithResource(resource)) {
		policyStorage, policyStatusStorage, err := validatingadmissionpolicystorage.NewREST(restOptionsGetter, p.Authorizer, r)
		if err != nil {
			return storage, err
		}
		policyGetter = policyStorage
		storage[resource] = policyStorage
		storage[resource+"/status"] = policyStatusStorage
	}

	// validatingadmissionpolicybindings
	if resource := "validatingadmissionpolicybindings"; apiResourceConfigSource.ResourceEnabled(admissionregistrationv1beta1.SchemeGroupVersion.WithResource(resource)) {
		policyBindingStorage, err := policybindingstorage.NewREST(restOptionsGetter, p.Authorizer, &policybindingstorage.DefaultPolicyGetter{Getter: policyGetter}, r)
		if err != nil {
			return storage, err
		}
		storage[resource] = policyBindingStorage
	}

	return storage, nil
}

func (p RESTStorageProvider) GroupName() string {
	return admissionregistration.GroupName
}
