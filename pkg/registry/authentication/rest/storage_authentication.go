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
	"k8s.io/kubernetes/pkg/apis/authentication"
	authenticationv1beta1 "k8s.io/kubernetes/pkg/apis/authentication/v1beta1"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/authentication/tokenreview"
)

type RESTStorageProvider struct {
	Authenticator authenticator.Request
}

var _ genericapiserver.RESTStorageProvider = &RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	// TODO figure out how to make the swagger generation stable, while allowing this endpoint to be disabled.
	// if p.Authenticator == nil {
	// 	return genericapiserver.APIGroupInfo{}, false
	// }

	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(authentication.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(authenticationv1beta1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[authenticationv1beta1.SchemeGroupVersion.Version] = p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = authenticationv1beta1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) map[string]rest.Storage {
	version := authenticationv1beta1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.AnyResourcesForVersionEnabled(authenticationv1beta1.SchemeGroupVersion) {
		if apiResourceConfigSource.ResourceEnabled(version.WithResource("tokenreviews")) {
			tokenReviewStorage := tokenreview.NewREST(p.Authenticator)
			storage["tokenreviews"] = tokenReviewStorage
		}
	}

	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return authentication.GroupName
}
