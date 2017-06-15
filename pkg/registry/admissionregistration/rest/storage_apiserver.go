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
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	admissionregistrationv1alpha1 "k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1"
	externaladmissionhookconfigurationstorage "k8s.io/kubernetes/pkg/registry/admissionregistration/externaladmissionhookconfiguration/storage"
	initializerconfigurationstorage "k8s.io/kubernetes/pkg/registry/admissionregistration/initializerconfiguration/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(admissionregistration.GroupName, api.Registry, api.Scheme, api.ParameterCodec, api.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(admissionregistrationv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[admissionregistrationv1alpha1.SchemeGroupVersion.Version] = p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = admissionregistrationv1alpha1.SchemeGroupVersion
	}
	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	version := admissionregistrationv1alpha1.SchemeGroupVersion
	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("initializerconfigurations")) {
		s := initializerconfigurationstorage.NewREST(restOptionsGetter)
		storage["initializerconfigurations"] = s
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("externaladmissionhookconfigurations")) {
		s := externaladmissionhookconfigurationstorage.NewREST(restOptionsGetter)
		storage["externaladmissionhookconfigurations"] = s
	}
	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return admissionregistration.GroupName
}
