/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"
	"net/http"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/client-go/informers"
	restclient "k8s.io/client-go/rest"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	configmapstore "k8s.io/kubernetes/pkg/registry/core/configmap/storage"
	eventstore "k8s.io/kubernetes/pkg/registry/core/event/storage"
	namespacestore "k8s.io/kubernetes/pkg/registry/core/namespace/storage"
	resourcequotastore "k8s.io/kubernetes/pkg/registry/core/resourcequota/storage"
	secretstore "k8s.io/kubernetes/pkg/registry/core/secret/storage"
	serviceaccountstore "k8s.io/kubernetes/pkg/registry/core/serviceaccount/storage"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// GenericConfig provides information needed to build RESTStorage
// for generic resources in core. It implements the "normal" RESTStorageProvider interface.
type GenericConfig struct {
	StorageFactory serverstorage.StorageFactory
	// Used for custom proxy dialing, and proxy TLS options
	ProxyTransport http.RoundTripper
	EventTTL       time.Duration

	ServiceAccountIssuer        serviceaccount.TokenGenerator
	ServiceAccountMaxExpiration time.Duration
	ExtendExpiration            bool

	APIAudiences authenticator.Audiences

	LoopbackClientConfig *restclient.Config
	Informers            informers.SharedInformerFactory
}

func (c *GenericConfig) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.APIGroupInfo{
		PrioritizedVersions:          legacyscheme.Scheme.PrioritizedVersionsForGroup(""),
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		Scheme:                       legacyscheme.Scheme,
		ParameterCodec:               legacyscheme.ParameterCodec,
		NegotiatedSerializer:         legacyscheme.Codecs,
	}

	storage := map[string]rest.Storage{}
	if resource := "events"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		eventsStorage, err := eventstore.NewREST(restOptionsGetter, uint64(c.EventTTL.Seconds()))
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}

		storage[resource] = eventsStorage
	}

	if resource := "resourcequotas"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		resourceQuotaStorage, resourceQuotaStorageStatus, err := resourcequotastore.NewREST(restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}

		storage[resource] = resourceQuotaStorage
		storage[resource+"/status"] = resourceQuotaStorageStatus
	}

	if resource := "namespaces"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		namespaceStorage, namespaceStorageStatus, namespaceStorageFinalize, err := namespacestore.NewREST(restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}

		storage[resource] = namespaceStorage
		storage[resource+"/status"] = namespaceStorageStatus
		storage[resource+"/finalize"] = namespaceStorageFinalize
	}

	var secretStorage *secretstore.REST
	if resource := "secrets"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		var err error
		secretStorage, err = secretstore.NewREST(restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}

		storage[resource] = secretStorage
	}

	if resource := "serviceaccounts"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		if secretStorage == nil {
			return genericapiserver.APIGroupInfo{}, fmt.Errorf("serviceaccounts depend on secrets, but secrets are disabled")
		}
		var serviceAccountStorage *serviceaccountstore.REST
		var err error
		if c.ServiceAccountIssuer != nil {
			serviceAccountStorage, err = serviceaccountstore.NewREST(restOptionsGetter, c.ServiceAccountIssuer, c.APIAudiences, c.ServiceAccountMaxExpiration, nil, secretStorage.Store, c.ExtendExpiration)
		} else {
			serviceAccountStorage, err = serviceaccountstore.NewREST(restOptionsGetter, nil, nil, 0, nil, nil, false)
		}
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}

		storage[resource] = serviceAccountStorage
		if serviceAccountStorage.Token != nil {
			storage[resource+"/token"] = serviceAccountStorage.Token
		}
	}

	if resource := "configmaps"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		configmapsStorage, err := configmapstore.NewREST(restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}

		storage[resource] = configmapsStorage
	}

	if len(storage) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap["v1"] = storage
	}

	return apiGroupInfo, nil
}

func (c *GenericConfig) GroupName() string {
	return api.GroupName
}
