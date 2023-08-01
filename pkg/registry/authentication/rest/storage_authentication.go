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
	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1alpha1 "k8s.io/api/authentication/v1alpha1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/authentication"
	"k8s.io/kubernetes/pkg/registry/authentication/selfsubjectreview"
	"k8s.io/kubernetes/pkg/registry/authentication/tokenreview"
)

type RESTStorageProvider struct {
	Authenticator authenticator.Request
	APIAudiences  authenticator.Audiences
}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	// TODO figure out how to make the swagger generation stable, while allowing this endpoint to be disabled.
	// if p.Authenticator == nil {
	// 	return genericapiserver.APIGroupInfo{}, false
	// }

	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(authentication.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if storageMap := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter); len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[authenticationv1alpha1.SchemeGroupVersion.Version] = storageMap
	}

	if storageMap := p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter); len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[authenticationv1beta1.SchemeGroupVersion.Version] = storageMap
	}

	if storageMap := p.v1Storage(apiResourceConfigSource, restOptionsGetter); len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[authenticationv1.SchemeGroupVersion.Version] = storageMap
	}

	return apiGroupInfo, nil
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	storage := map[string]rest.Storage{}

	// tokenreviews
	if resource := "tokenreviews"; apiResourceConfigSource.ResourceEnabled(authenticationv1.SchemeGroupVersion.WithResource(resource)) {
		tokenReviewStorage := tokenreview.NewREST(p.Authenticator, p.APIAudiences)
		storage[resource] = tokenReviewStorage
	}
	if resource := "selfsubjectreviews"; apiResourceConfigSource.ResourceEnabled(authenticationv1.SchemeGroupVersion.WithResource(resource)) {
		selfSRStorage := selfsubjectreview.NewREST()
		storage[resource] = selfSRStorage
	}

	return storage
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	storage := map[string]rest.Storage{}

	// selfsubjectreviews
	if resource := "selfsubjectreviews"; apiResourceConfigSource.ResourceEnabled(authenticationv1alpha1.SchemeGroupVersion.WithResource(resource)) {
		selfSRStorage := selfsubjectreview.NewREST()
		storage[resource] = selfSRStorage
	}
	return storage
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	storage := map[string]rest.Storage{}

	// selfsubjectreviews
	if resource := "selfsubjectreviews"; apiResourceConfigSource.ResourceEnabled(authenticationv1beta1.SchemeGroupVersion.WithResource(resource)) {
		selfSRStorage := selfsubjectreview.NewREST()
		storage[resource] = selfSRStorage
	}
	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return authentication.GroupName
}
