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

package apiserver

import (
	"net/http"
	"os"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	apiserverfilters "k8s.io/kubernetes/pkg/apiserver/filters"
	authhandlers "k8s.io/kubernetes/pkg/auth/handlers"
	"k8s.io/kubernetes/pkg/genericapiserver"
	genericfilters "k8s.io/kubernetes/pkg/genericapiserver/filters"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"

	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration"
	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1"
	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/clientset_generated/internalclientset"
	clientset "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/informers"
	listers "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/listers/apiregistration/internalversion"
	apiservicestorage "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/registry/apiservice"
)

// legacyAPIServiceName is the fixed name of the only non-groupified API version
const legacyAPIServiceName = "v1."

// TODO move to genericapiserver or something like that
// RESTOptionsGetter is used to construct storage for a particular resource
type RESTOptionsGetter interface {
	NewFor(resource schema.GroupResource) generic.RESTOptions
}

type Config struct {
	GenericConfig *genericapiserver.Config

	// RESTOptionsGetter is used to construct storage for a particular resource
	RESTOptionsGetter RESTOptionsGetter
}

// APIDiscoveryServer contains state for a Kubernetes cluster master/api server.
type APIDiscoveryServer struct {
	GenericAPIServer *genericapiserver.GenericAPIServer

	// handledAPIServices tracks which APIServices have already been handled.  Once endpoints are added,
	// the listers that are used keep bits in sync automatically.
	handledAPIServices sets.String

	// lister is used to add group handling for /apis/<group> discovery lookups based on
	// controller state
	lister listers.APIServiceLister
}

type completedConfig struct {
	*Config
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() completedConfig {
	c.GenericConfig.Complete()

	version := version.Get()
	c.GenericConfig.Version = &version

	return completedConfig{c}
}

// SkipComplete provides a way to construct a server instance without config completion.
func (c *Config) SkipComplete() completedConfig {
	return completedConfig{c}
}

// New returns a new instance of APIDiscoveryServer from the given config.
func (c completedConfig) New() (*APIDiscoveryServer, error) {
	informerFactory := informers.NewSharedInformerFactory(
		internalclientset.NewForConfigOrDie(c.Config.GenericConfig.LoopbackClientConfig),
		clientset.NewForConfigOrDie(c.Config.GenericConfig.LoopbackClientConfig),
		5*time.Minute, // this is effectively used as a refresh interval right now.  Might want to do something nicer later on.
	)

	// most API servers don't need to do this, but we need a custom handler chain to handle the special /apis handling here
	c.Config.GenericConfig.BuildHandlerChainsFunc = (&handlerChainConfig{
		informers: informerFactory,
	}).handlerChain

	genericServer, err := c.Config.GenericConfig.SkipComplete().New() // completion is done in Complete, no need for a second time
	if err != nil {
		return nil, err
	}

	s := &APIDiscoveryServer{
		GenericAPIServer:   genericServer,
		handledAPIServices: sets.String{},
		lister:             informerFactory.Apiregistration().InternalVersion().APIServices().Lister(),
	}

	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiregistration.GroupName)
	apiGroupInfo.GroupMeta.GroupVersion = v1alpha1.SchemeGroupVersion
	v1alpha1storage := map[string]rest.Storage{}
	v1alpha1storage["apiservices"] = apiservicestorage.NewREST(c.RESTOptionsGetter.NewFor(apiregistration.Resource("apiservices")))
	apiGroupInfo.VersionedResourcesStorageMap["v1alpha1"] = v1alpha1storage

	if err := s.GenericAPIServer.InstallAPIGroup(&apiGroupInfo); err != nil {
		return nil, err
	}

	apiserviceRegistrationController := NewAPIServiceRegistrationController(informerFactory.Apiregistration().InternalVersion().APIServices(), s)

	s.GenericAPIServer.AddPostStartHook("start-informers", func(context genericapiserver.PostStartHookContext) error {
		informerFactory.Start(wait.NeverStop)
		return nil
	})
	s.GenericAPIServer.AddPostStartHook("apiservice-registration-controller", func(context genericapiserver.PostStartHookContext) error {
		apiserviceRegistrationController.Run(wait.NeverStop)
		return nil
	})

	return s, nil
}

// handlerChainConfig is the config used to build the custom handler chain for this api server
type handlerChainConfig struct {
	informers informers.SharedInformerFactory
}

// handlerChain is a method to build the handler chain for this API server.  We need a custom handler chain so that we
// can have custom handling for `/apis`, since we're hosting discovery differently from anyone else and we're hosting
// the endpoints differently, since we're proxying all groups except for apiregistration.k8s.io.
func (h *handlerChainConfig) handlerChain(apiHandler http.Handler, c *genericapiserver.Config) (secure, insecure http.Handler) {
	// add this as a filter so that we never collide with "already registered" failures on `/apis`
	handler := WithAPIs(apiHandler, h.informers.Apiregistration().InternalVersion().APIServices())

	handler = apiserverfilters.WithAuthorization(handler, c.RequestContextMapper, c.Authorizer)
	handler = apiserverfilters.WithImpersonation(handler, c.RequestContextMapper, c.Authorizer)
	// audit to stdout to help with debugging as we get this started
	handler = apiserverfilters.WithAudit(handler, c.RequestContextMapper, os.Stdout)
	handler = authhandlers.WithAuthentication(handler, c.RequestContextMapper, c.Authenticator, authhandlers.Unauthorized(c.SupportsBasicAuth))

	handler = genericfilters.WithCORS(handler, c.CorsAllowedOriginList, nil, nil, nil, "true")
	handler = genericfilters.WithPanicRecovery(handler, c.RequestContextMapper)
	handler = genericfilters.WithTimeoutForNonLongRunningRequests(handler, c.RequestContextMapper, c.LongRunningFunc)
	handler = genericfilters.WithMaxInFlightLimit(handler, c.MaxRequestsInFlight, c.MaxMutatingRequestsInFlight, c.RequestContextMapper, c.LongRunningFunc)
	handler = apiserverfilters.WithRequestInfo(handler, genericapiserver.NewRequestInfoResolver(c), c.RequestContextMapper)
	handler = api.WithRequestContext(handler, c.RequestContextMapper)

	return handler, nil
}

// AddAPIService adds an API service.  It is not thread-safe, so only call it on one thread at a time please.
// It's a slow moving API, so its ok to run the controller on a single thread
func (s *APIDiscoveryServer) AddAPIService(apiService *apiregistration.APIService) {
	if s.handledAPIServices.Has(apiService.Name) {
		return
	}
	// if we're dealing with the legacy group, we're done here
	if apiService.Name == legacyAPIServiceName {
		s.handledAPIServices.Insert(apiService.Name)
		return
	}

	// it's time to register the group discovery endpoint
	groupPath := "/apis/" + apiService.Spec.Group
	groupDiscoveryHandler := &apiGroupHandler{
		groupName: apiService.Spec.Group,
		lister:    s.lister,
	}
	// discovery is protected
	s.GenericAPIServer.HandlerContainer.SecretRoutes.Handle(groupPath, groupDiscoveryHandler)
	s.GenericAPIServer.HandlerContainer.SecretRoutes.Handle(groupPath+"/", groupDiscoveryHandler)

}

// RemoveAPIService removes the APIService from being handled.  Later on it will disable the proxy endpoint.
// Right now it does nothing because our handler has to properly 404 itself since muxes don't unregister
func (s *APIDiscoveryServer) RemoveAPIService(apiServiceName string) {
	if !s.handledAPIServices.Has(apiServiceName) {
		return
	}
}
