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
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	configmapstore "k8s.io/kubernetes/pkg/registry/core/configmap/storage"
	eventstore "k8s.io/kubernetes/pkg/registry/core/event/storage"
	limitrangestore "k8s.io/kubernetes/pkg/registry/core/limitrange/storage"
	namespacestore "k8s.io/kubernetes/pkg/registry/core/namespace/storage"
	resourcequotastore "k8s.io/kubernetes/pkg/registry/core/resourcequota/storage"
	secretstore "k8s.io/kubernetes/pkg/registry/core/secret/storage"
	serviceaccountstore "k8s.io/kubernetes/pkg/registry/core/serviceaccount/storage"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// LegacyRESTStorageProvider provides information needed to build RESTStorage for core, but
// does NOT implement the "normal" RESTStorageProvider (yet!)
type LegacyRESTStorageProvider struct {
	StorageFactory serverstorage.StorageFactory
	EventTTL       time.Duration

	ServiceAccountIssuer        serviceaccount.TokenGenerator
	ServiceAccountMaxExpiration time.Duration
	ExtendExpiration            bool

	APIAudiences authenticator.Audiences
}

// LegacyRESTStorage returns stateful information about particular instances of REST storage to
// master.go for wiring controllers.
// TODO remove this by running the controller as a poststarthook
type LegacyRESTStorage struct{}

func (c LegacyRESTStorageProvider) NewLegacyRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (LegacyRESTStorage, genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.APIGroupInfo{
		PrioritizedVersions:          legacyscheme.Scheme.PrioritizedVersionsForGroup(""),
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		Scheme:                       legacyscheme.Scheme,
		ParameterCodec:               legacyscheme.ParameterCodec,
		NegotiatedSerializer:         legacyscheme.Codecs,
	}

	restStorage := LegacyRESTStorage{}

	eventStorage, err := eventstore.NewREST(restOptionsGetter, uint64(c.EventTTL.Seconds()))
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	limitRangeStorage, err := limitrangestore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	resourceQuotaStorage, resourceQuotaStatusStorage, err := resourcequotastore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	secretStorage, err := secretstore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	configMapStorage, err := configmapstore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	namespaceStorage, namespaceStatusStorage, namespaceFinalizeStorage, err := namespacestore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	var serviceAccountStorage *serviceaccountstore.REST
	if c.ServiceAccountIssuer != nil {
		// TODO: replace with something generic for token
		serviceAccountStorage, err = serviceaccountstore.NewREST(restOptionsGetter, c.ServiceAccountIssuer, c.APIAudiences, c.ServiceAccountMaxExpiration, nil, secretStorage.Store, c.ExtendExpiration)
	} else {
		serviceAccountStorage, err = serviceaccountstore.NewREST(restOptionsGetter, nil, nil, 0, nil, nil, false)
	}
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	storage := map[string]rest.Storage{}
	if resource := "events"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = eventStorage
	}

	if resource := "limitranges"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = limitRangeStorage
	}

	if resource := "resourcequotas"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = resourceQuotaStorage
		storage[resource+"/status"] = resourceQuotaStatusStorage
	}

	if resource := "namespaces"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = namespaceStorage
		storage[resource+"/status"] = namespaceStatusStorage
		storage[resource+"/finalize"] = namespaceFinalizeStorage
	}

	if resource := "secrets"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = secretStorage
	}

	if resource := "serviceaccounts"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = serviceAccountStorage
		if serviceAccountStorage.Token != nil {
			storage[resource+"/token"] = serviceAccountStorage.Token
		}
	}

	if resource := "configmaps"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = configMapStorage
	}

	if len(storage) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap["v1"] = storage
	}

	return restStorage, apiGroupInfo, nil
}

func (p LegacyRESTStorageProvider) GroupName() string {
	return api.GroupName
}
