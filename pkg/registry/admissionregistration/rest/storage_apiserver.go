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
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	mutatingwebhookconfigurationstorage "k8s.io/kubernetes/pkg/registry/admissionregistration/mutatingwebhookconfiguration/storage"
	validatingwebhookconfigurationstorage "k8s.io/kubernetes/pkg/registry/admissionregistration/validatingwebhookconfiguration/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(admissionregistration.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.VersionEnabled(admissionregistrationv1beta1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[admissionregistrationv1beta1.SchemeGroupVersion.Version] = p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter)
	}
	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	storage := map[string]rest.Storage{}
	// validatingwebhookconfigurations
	validatingStorage := validatingwebhookconfigurationstorage.NewREST(restOptionsGetter)
	storage["validatingwebhookconfigurations"] = validatingStorage

	// mutatingwebhookconfigurations
	mutatingStorage := mutatingwebhookconfigurationstorage.NewREST(restOptionsGetter)
	storage["mutatingwebhookconfigurations"] = mutatingStorage

	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return admissionregistration.GroupName
}
