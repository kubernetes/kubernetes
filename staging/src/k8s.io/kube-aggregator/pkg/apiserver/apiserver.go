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
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/pkg/version"

	"bytes"
	"fmt"
	"io"

	"github.com/go-openapi/spec"
	"github.com/golang/glog"
	"github.com/pkg/errors"
	"k8s.io/apiserver/pkg/server/openapi"
	"k8s.io/client-go/transport"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/install"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
	"k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kube-aggregator/pkg/client/informers/internalversion"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/internalversion"
	statuscontrollers "k8s.io/kube-aggregator/pkg/controllers/status"
	apiservicestorage "k8s.io/kube-aggregator/pkg/registry/apiservice/etcd"
)

var (
	groupFactoryRegistry = make(announced.APIGroupFactoryRegistry)
	registry             = registered.NewOrDie("")
	Scheme               = runtime.NewScheme()
	Codecs               = serializer.NewCodecFactory(Scheme)
)

const (
	LOAD_OPENAPI_SPEC_MAX_RETRIES = 10
)

func init() {
	install.Install(groupFactoryRegistry, registry, Scheme)

	// we need to add the options (like ListOptions) to empty v1
	metav1.AddToGroupVersion(Scheme, schema.GroupVersion{Group: "", Version: "v1"})

	unversioned := schema.GroupVersion{Group: "", Version: "v1"}
	Scheme.AddUnversionedTypes(unversioned,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)
}

// legacyAPIServiceName is the fixed name of the only non-groupified API version
const legacyAPIServiceName = "v1."

type Config struct {
	GenericConfig *genericapiserver.Config

	// CoreKubeInformers is used to watch kube resources
	CoreKubeInformers kubeinformers.SharedInformerFactory

	// ProxyClientCert/Key are the client cert used to identify this proxy. Backing APIServices use
	// this to confirm the proxy's identity
	ProxyClientCert []byte
	ProxyClientKey  []byte

	// If present, the Dial method will be used for dialing out to delegate
	// apiservers.
	ProxyTransport *http.Transport

	// Mechanism by which the Aggregator will resolve services. Required.
	ServiceResolver ServiceResolver
}

// APIAggregator contains state for a Kubernetes cluster master/api server.
type APIAggregator struct {
	GenericAPIServer *genericapiserver.GenericAPIServer

	delegateHandler http.Handler

	contextMapper genericapirequest.RequestContextMapper

	// proxyClientCert/Key are the client cert used to identify this proxy. Backing APIServices use
	// this to confirm the proxy's identity
	proxyClientCert []byte
	proxyClientKey  []byte
	proxyTransport  *http.Transport

	// proxyHandlers are the proxy handlers that are currently registered, keyed by apiservice.name
	proxyHandlers map[string]*proxyHandler
	// handledGroups are the groups that already have routes
	handledGroups sets.String

	// Swagger spec for each api service
	apiServiceSpecs map[string]*spec.Swagger

	// List of the specs that needs to be loaded. When a spec is successfully loaded
	// it will be removed from this list and added to apiServiceSpecs.
	// Map values are retry counts. After a preset retries, it will stop
	// trying.
	toLoadAPISpec map[string]int

	// protecting toLoadAPISpec and apiServiceSpecs
	specMutex sync.Mutex

	// rootSpec is the OpenAPI spec of the Aggregator server.
	rootSpec *spec.Swagger

	// delegationSpec is the delegation API Server's spec (most of API groups are in this spec).
	delegationSpec *spec.Swagger

	// lister is used to add group handling for /apis/<group> aggregator lookups based on
	// controller state
	lister listers.APIServiceLister

	// provided for easier embedding
	APIRegistrationInformers informers.SharedInformerFactory

	// Information needed to determine routing for the aggregator
	serviceResolver ServiceResolver
}

type completedConfig struct {
	*Config
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() completedConfig {
	// the kube aggregator wires its own discovery mechanism
	// TODO eventually collapse this by extracting all of the discovery out
	c.GenericConfig.EnableDiscovery = false
	c.GenericConfig.Complete()

	version := version.Get()
	c.GenericConfig.Version = &version

	return completedConfig{c}
}

// SkipComplete provides a way to construct a server instance without config completion.
func (c *Config) SkipComplete() completedConfig {
	return completedConfig{c}
}

// New returns a new instance of APIAggregator from the given config.
func (c completedConfig) NewWithDelegate(delegationTarget genericapiserver.DelegationTarget) (*APIAggregator, error) {
	genericServer, err := c.Config.GenericConfig.SkipComplete().New("kube-aggregator", delegationTarget) // completion is done in Complete, no need for a second time
	if err != nil {
		return nil, err
	}

	apiregistrationClient, err := internalclientset.NewForConfig(c.Config.GenericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, err
	}
	informerFactory := informers.NewSharedInformerFactory(
		apiregistrationClient,
		5*time.Minute, // this is effectively used as a refresh interval right now.  Might want to do something nicer later on.
	)

	s := &APIAggregator{
		GenericAPIServer: genericServer,
		delegateHandler:  delegationTarget.UnprotectedHandler(),
		delegationSpec:   delegationTarget.OpenAPISpec(),
		contextMapper:    c.GenericConfig.RequestContextMapper,
		proxyClientCert:  c.ProxyClientCert,
		proxyClientKey:   c.ProxyClientKey,
		proxyTransport:   c.ProxyTransport,
		proxyHandlers:    map[string]*proxyHandler{},
		apiServiceSpecs:  map[string]*spec.Swagger{},
		toLoadAPISpec:    map[string]int{},
		handledGroups:    sets.String{},
		lister:           informerFactory.Apiregistration().InternalVersion().APIServices().Lister(),
		APIRegistrationInformers: informerFactory,
		serviceResolver:          c.ServiceResolver,
	}

	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiregistration.GroupName, registry, Scheme, metav1.ParameterCodec, Codecs)
	apiGroupInfo.GroupMeta.GroupVersion = v1beta1.SchemeGroupVersion
	v1beta1storage := map[string]rest.Storage{}
	apiServiceREST := apiservicestorage.NewREST(Scheme, c.GenericConfig.RESTOptionsGetter)
	v1beta1storage["apiservices"] = apiServiceREST
	v1beta1storage["apiservices/status"] = apiservicestorage.NewStatusREST(Scheme, apiServiceREST)
	apiGroupInfo.VersionedResourcesStorageMap["v1beta1"] = v1beta1storage

	if err := s.GenericAPIServer.InstallAPIGroup(&apiGroupInfo); err != nil {
		return nil, err
	}

	apisHandler := &apisHandler{
		codecs: Codecs,
		lister: s.lister,
		mapper: s.contextMapper,
	}
	s.GenericAPIServer.Handler.NonGoRestfulMux.Handle("/apis", apisHandler)
	s.GenericAPIServer.Handler.NonGoRestfulMux.UnlistedHandle("/apis/", apisHandler)

	apiserviceRegistrationController := NewAPIServiceRegistrationController(informerFactory.Apiregistration().InternalVersion().APIServices(), c.CoreKubeInformers.Core().V1().Services(), s)
	availableController := statuscontrollers.NewAvailableConditionController(
		informerFactory.Apiregistration().InternalVersion().APIServices(),
		c.CoreKubeInformers.Core().V1().Services(),
		c.CoreKubeInformers.Core().V1().Endpoints(),
		apiregistrationClient.Apiregistration(),
	)

	s.GenericAPIServer.AddPostStartHook("start-kube-aggregator-informers", func(context genericapiserver.PostStartHookContext) error {
		informerFactory.Start(context.StopCh)
		c.CoreKubeInformers.Start(context.StopCh)
		return nil
	})
	s.GenericAPIServer.AddPostStartHook("apiservice-registration-controller", func(context genericapiserver.PostStartHookContext) error {
		go apiserviceRegistrationController.Run(context.StopCh)
		return nil
	})
	s.GenericAPIServer.AddPostStartHook("apiservice-status-available-controller", func(context genericapiserver.PostStartHookContext) error {
		go availableController.Run(context.StopCh)
		return nil
	})

	s.GenericAPIServer.PrepareOpenAPIService()

	if s.GenericAPIServer.OpenAPIService != nil {
		s.rootSpec = s.GenericAPIServer.OpenAPIService.GetSpec()
		if err := s.updateOpenAPISpec(); err != nil {
			return nil, err
		}
		s.GenericAPIServer.OpenAPIService.AddUpdateHook(func(r *http.Request) {
			if s.tryLoadingOpenAPISpecs(r) {
				s.updateOpenAPISpec()
			}
		})
	}

	return s, nil
}

// AddAPIService adds an API service.  It is not thread-safe, so only call it on one thread at a time please.
// It's a slow moving API, so its ok to run the controller on a single thread
func (s *APIAggregator) AddAPIService(apiService *apiregistration.APIService) {
	// if the proxyHandler already exists, it needs to be updated. The aggregation bits do not
	// since they are wired against listers because they require multiple resources to respond
	if proxyHandler, exists := s.proxyHandlers[apiService.Name]; exists {
		proxyHandler.updateAPIService(apiService)
		return
	}

	proxyPath := "/apis/" + apiService.Spec.Group + "/" + apiService.Spec.Version
	// v1. is a special case for the legacy API.  It proxies to a wider set of endpoints.
	if apiService.Name == legacyAPIServiceName {
		proxyPath = "/api"
	}

	// register the proxy handler
	proxyHandler := &proxyHandler{
		contextMapper:   s.contextMapper,
		localDelegate:   s.delegateHandler,
		proxyClientCert: s.proxyClientCert,
		proxyClientKey:  s.proxyClientKey,
		proxyTransport:  s.proxyTransport,
		serviceResolver: s.serviceResolver,
	}
	proxyHandler.updateAPIService(apiService)
	s.proxyHandlers[apiService.Name] = proxyHandler

	s.deferLoadAPISpec(apiService.Name)

	s.GenericAPIServer.Handler.NonGoRestfulMux.Handle(proxyPath, proxyHandler)
	s.GenericAPIServer.Handler.NonGoRestfulMux.UnlistedHandlePrefix(proxyPath+"/", proxyHandler)

	// if we're dealing with the legacy group, we're done here
	if apiService.Name == legacyAPIServiceName {
		return
	}

	// if we've already registered the path with the handler, we don't want to do it again.
	if s.handledGroups.Has(apiService.Spec.Group) {
		return
	}

	// it's time to register the group aggregation endpoint
	groupPath := "/apis/" + apiService.Spec.Group
	groupDiscoveryHandler := &apiGroupHandler{
		codecs:        Codecs,
		groupName:     apiService.Spec.Group,
		lister:        s.lister,
		delegate:      s.delegateHandler,
		contextMapper: s.contextMapper,
	}
	// aggregation is protected
	s.GenericAPIServer.Handler.NonGoRestfulMux.Handle(groupPath, groupDiscoveryHandler)
	s.GenericAPIServer.Handler.NonGoRestfulMux.UnlistedHandle(groupPath+"/", groupDiscoveryHandler)
	s.handledGroups.Insert(apiService.Spec.Group)
}

func (s *APIAggregator) deferLoadAPISpec(name string) {
	s.specMutex.Lock()
	defer s.specMutex.Unlock()
	s.toLoadAPISpec[name] = 0
}

func (s *APIAggregator) deleteApiSpec(name string) {
	s.specMutex.Lock()
	defer s.specMutex.Unlock()
	delete(s.apiServiceSpecs, name)
	delete(s.toLoadAPISpec, name)
}

// RemoveAPIService removes the APIService from being handled.  It is not thread-safe, so only call it on one thread at a time please.
// It's a slow moving API, so its ok to run the controller on a single thread.
func (s *APIAggregator) RemoveAPIService(apiServiceName string) {
	version := apiregistration.APIServiceNameToGroupVersion(apiServiceName)

	proxyPath := "/apis/" + version.Group + "/" + version.Version
	// v1. is a special case for the legacy API.  It proxies to a wider set of endpoints.
	if apiServiceName == legacyAPIServiceName {
		proxyPath = "/api"
	}
	s.GenericAPIServer.Handler.NonGoRestfulMux.Unregister(proxyPath)
	s.GenericAPIServer.Handler.NonGoRestfulMux.Unregister(proxyPath + "/")
	delete(s.proxyHandlers, apiServiceName)
	s.deleteApiSpec(apiServiceName)
	s.updateOpenAPISpec()

	// TODO unregister group level discovery when there are no more versions for the group
	// We don't need this right away because the handler properly delegates when no versions are present
}

func (_ *APIAggregator) loadOpenAPISpec(p *proxyHandler, r *http.Request) (*spec.Swagger, error) {
	value := p.handlingInfo.Load()
	if value == nil {
		return nil, nil
	}
	handlingInfo := value.(proxyHandlingInfo)
	if handlingInfo.local {
		return nil, nil
	}
	loc, err := p.serviceResolver.ResolveEndpoint(handlingInfo.serviceNamespace, handlingInfo.serviceName)
	if err != nil {
		return nil, fmt.Errorf("missing route")
	}
	host := loc.Host

	var w io.Reader
	req, err := http.NewRequest("GET", "/swagger.json", w)
	if err != nil {
		return nil, err
	}
	req.URL.Scheme = "https"
	req.URL.Host = host

	req = req.WithContext(context.Background())
	// Get user from the original request
	ctx, ok := p.contextMapper.Get(r)
	if !ok {
		return nil, fmt.Errorf("missing context")
	}
	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return nil, fmt.Errorf("missing user")
	}
	proxyRoundTripper := transport.NewAuthProxyRoundTripper(user.GetName(), user.GetGroups(), user.GetExtra(), handlingInfo.proxyRoundTripper)
	res, err := proxyRoundTripper.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, errors.New(res.Status)
	}
	buf := new(bytes.Buffer)
	buf.ReadFrom(res.Body)
	bytes := buf.Bytes()
	var s spec.Swagger
	if err := json.Unmarshal(bytes, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// Returns true if any Spec is loaded
func (s *APIAggregator) tryLoadingOpenAPISpecs(r *http.Request) bool {
	s.specMutex.Lock()
	defer s.specMutex.Unlock()
	if len(s.toLoadAPISpec) == 0 {
		return false
	}
	loaded := false
	newList := map[string]int{}
	for name, retries := range s.toLoadAPISpec {
		if retries >= LOAD_OPENAPI_SPEC_MAX_RETRIES {
			continue
		}
		proxyHandler := s.proxyHandlers[name]
		if spec, err := s.loadOpenAPISpec(proxyHandler, r); err != nil {
			glog.Warningf("Failed to Load OpenAPI spec (try %d of %d) for %s, err=%s", retries+1, LOAD_OPENAPI_SPEC_MAX_RETRIES, name, err)
			newList[name] = retries + 1
		} else if spec != nil {
			s.apiServiceSpecs[name] = spec
			loaded = true
		}
		s.toLoadAPISpec = newList
	}
	return loaded
}

func (s *APIAggregator) updateOpenAPISpec() error {
	s.specMutex.Lock()
	defer s.specMutex.Unlock()
	if s.GenericAPIServer.OpenAPIService == nil {
		return nil
	}
	sp, err := openapi.CloneSpec(s.rootSpec)
	if err != nil {
		return err
	}
	openapi.FilterSpecByPaths(sp, []string{"/apis/apiregistration.k8s.io/"})
	if _, found := sp.Paths.Paths["/version/"]; found {
		return fmt.Errorf("Cleanup didn't work")
	}
	if err := openapi.MergeSpecs(sp, s.delegationSpec); err != nil {
		return err
	}

	for k, v := range s.apiServiceSpecs {
		version := apiregistration.APIServiceNameToGroupVersion(k)

		proxyPath := "/apis/" + version.Group + "/"
		// v1. is a special case for the legacy API.  It proxies to a wider set of endpoints.
		if k == legacyAPIServiceName {
			proxyPath = "/api/"
		}
		spc, err := openapi.CloneSpec(v)
		if err != nil {
			return err
		}
		openapi.FilterSpecByPaths(spc, []string{proxyPath})
		if err := openapi.MergeSpecs(sp, spc); err != nil {
			return err
		}
	}
	return s.GenericAPIServer.OpenAPIService.UpdateSpec(sp)
}
