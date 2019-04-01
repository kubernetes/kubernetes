/*
Copyright 2014 The Kubernetes Authors.

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

package server

import (
	"fmt"
	"net/http"
	gpath "path"
	"strings"
	"sync"
	"time"

	systemd "github.com/coreos/go-systemd/daemon"
	"github.com/go-openapi/spec"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	utilwaitgroup "k8s.io/apimachinery/pkg/util/waitgroup"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapi "k8s.io/apiserver/pkg/endpoints"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/routes"
	utilopenapi "k8s.io/apiserver/pkg/util/openapi"
	restclient "k8s.io/client-go/rest"
	openapibuilder "k8s.io/kube-openapi/pkg/builder"
	openapicommon "k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
	openapiutil "k8s.io/kube-openapi/pkg/util"
	openapiproto "k8s.io/kube-openapi/pkg/util/proto"
)

// Info about an API group.
type APIGroupInfo struct {
	PrioritizedVersions []schema.GroupVersion
	// Info about the resources in this group. It's a map from version to resource to the storage.
	VersionedResourcesStorageMap map[string]map[string]rest.Storage
	// OptionsExternalVersion controls the APIVersion used for common objects in the
	// schema like api.Status, api.DeleteOptions, and metav1.ListOptions. Other implementors may
	// define a version "v1beta1" but want to use the Kubernetes "v1" internal objects.
	// If nil, defaults to groupMeta.GroupVersion.
	// TODO: Remove this when https://github.com/kubernetes/kubernetes/issues/19018 is fixed.
	OptionsExternalVersion *schema.GroupVersion
	// MetaGroupVersion defaults to "meta.k8s.io/v1" and is the scheme group version used to decode
	// common API implementations like ListOptions. Future changes will allow this to vary by group
	// version (for when the inevitable meta/v2 group emerges).
	MetaGroupVersion *schema.GroupVersion

	// Scheme includes all of the types used by this group and how to convert between them (or
	// to convert objects from outside of this group that are accepted in this API).
	// TODO: replace with interfaces
	Scheme *runtime.Scheme
	// NegotiatedSerializer controls how this group encodes and decodes data
	NegotiatedSerializer runtime.NegotiatedSerializer
	// ParameterCodec performs conversions for query parameters passed to API calls
	ParameterCodec runtime.ParameterCodec
}

// GenericAPIServer contains state for a Kubernetes cluster api server.
type GenericAPIServer struct {
	// discoveryAddresses is used to build cluster IPs for discovery.
	discoveryAddresses discovery.Addresses

	// LoopbackClientConfig is a config for a privileged loopback connection to the API server
	LoopbackClientConfig *restclient.Config

	// minRequestTimeout is how short the request timeout can be.  This is used to build the RESTHandler
	minRequestTimeout time.Duration

	// ShutdownTimeout is the timeout used for server shutdown. This specifies the timeout before server
	// gracefully shutdown returns.
	ShutdownTimeout time.Duration

	// legacyAPIGroupPrefixes is used to set up URL parsing for authorization and for validating requests
	// to InstallLegacyAPIGroup
	legacyAPIGroupPrefixes sets.String

	// admissionControl is used to build the RESTStorage that backs an API Group.
	admissionControl admission.Interface

	// SecureServingInfo holds configuration of the TLS server.
	SecureServingInfo *SecureServingInfo

	// ExternalAddress is the address (hostname or IP and port) that should be used in
	// external (public internet) URLs for this GenericAPIServer.
	ExternalAddress string

	// Serializer controls how common API objects not in a group/version prefix are serialized for this server.
	// Individual APIGroups may define their own serializers.
	Serializer runtime.NegotiatedSerializer

	// "Outputs"
	// Handler holds the handlers being used by this API server
	Handler *APIServerHandler

	// listedPathProvider is a lister which provides the set of paths to show at /
	listedPathProvider routes.ListedPathProvider

	// DiscoveryGroupManager serves /apis
	DiscoveryGroupManager discovery.GroupManager

	// Enable swagger and/or OpenAPI if these configs are non-nil.
	openAPIConfig *openapicommon.Config

	// OpenAPIVersionedService controls the /openapi/v2 endpoint, and can be used to update the served spec.
	// It is set during PrepareRun.
	OpenAPIVersionedService *handler.OpenAPIService

	// StaticOpenAPISpec is the spec derived from the restful container endpoints.
	// It is set during PrepareRun.
	StaticOpenAPISpec *spec.Swagger

	// PostStartHooks are each called after the server has started listening, in a separate go func for each
	// with no guarantee of ordering between them.  The map key is a name used for error reporting.
	// It may kill the process with a panic if it wishes to by returning an error.
	postStartHookLock      sync.Mutex
	postStartHooks         map[string]postStartHookEntry
	postStartHooksCalled   bool
	disabledPostStartHooks sets.String

	preShutdownHookLock    sync.Mutex
	preShutdownHooks       map[string]preShutdownHookEntry
	preShutdownHooksCalled bool

	// healthz checks
	healthzLock    sync.Mutex
	healthzChecks  []healthz.HealthzChecker
	healthzCreated bool

	// auditing. The backend is started after the server starts listening.
	AuditBackend audit.Backend

	// Authorizer determines whether a user is allowed to make a certain request. The Handler does a preliminary
	// authorization check using the request URI but it may be necessary to make additional checks, such as in
	// the create-on-update case
	Authorizer authorizer.Authorizer

	// enableAPIResponseCompression indicates whether API Responses should support compression
	// if the client requests it via Accept-Encoding
	enableAPIResponseCompression bool

	// delegationTarget is the next delegate in the chain. This is never nil.
	delegationTarget DelegationTarget

	// HandlerChainWaitGroup allows you to wait for all chain handlers finish after the server shutdown.
	HandlerChainWaitGroup *utilwaitgroup.SafeWaitGroup

	// The limit on the request body size that would be accepted and decoded in a write request.
	// 0 means no limit.
	maxRequestBodyBytes int64
}

// DelegationTarget is an interface which allows for composition of API servers with top level handling that works
// as expected.
type DelegationTarget interface {
	// UnprotectedHandler returns a handler that is NOT protected by a normal chain
	UnprotectedHandler() http.Handler

	// PostStartHooks returns the post-start hooks that need to be combined
	PostStartHooks() map[string]postStartHookEntry

	// PreShutdownHooks returns the pre-stop hooks that need to be combined
	PreShutdownHooks() map[string]preShutdownHookEntry

	// HealthzChecks returns the healthz checks that need to be combined
	HealthzChecks() []healthz.HealthzChecker

	// ListedPaths returns the paths for supporting an index
	ListedPaths() []string

	// NextDelegate returns the next delegationTarget in the chain of delegations
	NextDelegate() DelegationTarget
}

func (s *GenericAPIServer) UnprotectedHandler() http.Handler {
	// when we delegate, we need the server we're delegating to choose whether or not to use gorestful
	return s.Handler.Director
}
func (s *GenericAPIServer) PostStartHooks() map[string]postStartHookEntry {
	return s.postStartHooks
}
func (s *GenericAPIServer) PreShutdownHooks() map[string]preShutdownHookEntry {
	return s.preShutdownHooks
}
func (s *GenericAPIServer) HealthzChecks() []healthz.HealthzChecker {
	return s.healthzChecks
}
func (s *GenericAPIServer) ListedPaths() []string {
	return s.listedPathProvider.ListedPaths()
}

func (s *GenericAPIServer) NextDelegate() DelegationTarget {
	return s.delegationTarget
}

type emptyDelegate struct {
}

func NewEmptyDelegate() DelegationTarget {
	return emptyDelegate{}
}

func (s emptyDelegate) UnprotectedHandler() http.Handler {
	return nil
}
func (s emptyDelegate) PostStartHooks() map[string]postStartHookEntry {
	return map[string]postStartHookEntry{}
}
func (s emptyDelegate) PreShutdownHooks() map[string]preShutdownHookEntry {
	return map[string]preShutdownHookEntry{}
}
func (s emptyDelegate) HealthzChecks() []healthz.HealthzChecker {
	return []healthz.HealthzChecker{}
}
func (s emptyDelegate) ListedPaths() []string {
	return []string{}
}
func (s emptyDelegate) NextDelegate() DelegationTarget {
	return nil
}

// preparedGenericAPIServer is a private wrapper that enforces a call of PrepareRun() before Run can be invoked.
type preparedGenericAPIServer struct {
	*GenericAPIServer
}

// PrepareRun does post API installation setup steps.
func (s *GenericAPIServer) PrepareRun() preparedGenericAPIServer {
	if s.openAPIConfig != nil {
		s.OpenAPIVersionedService, s.StaticOpenAPISpec = routes.OpenAPI{
			Config: s.openAPIConfig,
		}.Install(s.Handler.GoRestfulContainer, s.Handler.NonGoRestfulMux)
	}

	s.installHealthz()

	// Register audit backend preShutdownHook.
	if s.AuditBackend != nil {
		err := s.AddPreShutdownHook("audit-backend", func() error {
			s.AuditBackend.Shutdown()
			return nil
		})
		if err != nil {
			klog.Errorf("Failed to add pre-shutdown hook for audit-backend %s", err)
		}
	}

	return preparedGenericAPIServer{s}
}

// Run spawns the secure http server. It only returns if stopCh is closed
// or the secure port cannot be listened on initially.
func (s preparedGenericAPIServer) Run(stopCh <-chan struct{}) error {
	err := s.NonBlockingRun(stopCh)
	if err != nil {
		return err
	}

	<-stopCh

	err = s.RunPreShutdownHooks()
	if err != nil {
		return err
	}

	// Wait for all requests to finish, which are bounded by the RequestTimeout variable.
	s.HandlerChainWaitGroup.Wait()

	return nil
}

// NonBlockingRun spawns the secure http server. An error is
// returned if the secure port cannot be listened on.
func (s preparedGenericAPIServer) NonBlockingRun(stopCh <-chan struct{}) error {
	// Use an stop channel to allow graceful shutdown without dropping audit events
	// after http server shutdown.
	auditStopCh := make(chan struct{})

	// Start the audit backend before any request comes in. This means we must call Backend.Run
	// before http server start serving. Otherwise the Backend.ProcessEvents call might block.
	if s.AuditBackend != nil {
		if err := s.AuditBackend.Run(auditStopCh); err != nil {
			return fmt.Errorf("failed to run the audit backend: %v", err)
		}
	}

	// Use an internal stop channel to allow cleanup of the listeners on error.
	internalStopCh := make(chan struct{})
	var stoppedCh <-chan struct{}
	if s.SecureServingInfo != nil && s.Handler != nil {
		var err error
		stoppedCh, err = s.SecureServingInfo.Serve(s.Handler, s.ShutdownTimeout, internalStopCh)
		if err != nil {
			close(internalStopCh)
			return err
		}
	}

	// Now that listener have bound successfully, it is the
	// responsibility of the caller to close the provided channel to
	// ensure cleanup.
	go func() {
		<-stopCh
		close(internalStopCh)
		if stoppedCh != nil {
			<-stoppedCh
		}
		s.HandlerChainWaitGroup.Wait()
		close(auditStopCh)
	}()

	s.RunPostStartHooks(stopCh)

	if _, err := systemd.SdNotify(true, "READY=1\n"); err != nil {
		klog.Errorf("Unable to send systemd daemon successful start message: %v\n", err)
	}

	return nil
}

// installAPIResources is a private method for installing the REST storage backing each api groupversionresource
func (s *GenericAPIServer) installAPIResources(apiPrefix string, apiGroupInfo *APIGroupInfo, openAPIModels openapiproto.Models) error {
	for _, groupVersion := range apiGroupInfo.PrioritizedVersions {
		if len(apiGroupInfo.VersionedResourcesStorageMap[groupVersion.Version]) == 0 {
			klog.Warningf("Skipping API %v because it has no resources.", groupVersion)
			continue
		}

		apiGroupVersion := s.getAPIGroupVersion(apiGroupInfo, groupVersion, apiPrefix)
		if apiGroupInfo.OptionsExternalVersion != nil {
			apiGroupVersion.OptionsExternalVersion = apiGroupInfo.OptionsExternalVersion
		}
		apiGroupVersion.OpenAPIModels = openAPIModels
		apiGroupVersion.MaxRequestBodyBytes = s.maxRequestBodyBytes

		if err := apiGroupVersion.InstallREST(s.Handler.GoRestfulContainer); err != nil {
			return fmt.Errorf("unable to setup API %v: %v", apiGroupInfo, err)
		}
	}

	return nil
}

func (s *GenericAPIServer) InstallLegacyAPIGroup(apiPrefix string, apiGroupInfo *APIGroupInfo) error {
	if !s.legacyAPIGroupPrefixes.Has(apiPrefix) {
		return fmt.Errorf("%q is not in the allowed legacy API prefixes: %v", apiPrefix, s.legacyAPIGroupPrefixes.List())
	}

	openAPIModels, err := s.getOpenAPIModels(apiPrefix, apiGroupInfo)
	if err != nil {
		return fmt.Errorf("unable to get openapi models: %v", err)
	}

	if err := s.installAPIResources(apiPrefix, apiGroupInfo, openAPIModels); err != nil {
		return err
	}

	// Install the version handler.
	// Add a handler at /<apiPrefix> to enumerate the supported api versions.
	s.Handler.GoRestfulContainer.Add(discovery.NewLegacyRootAPIHandler(s.discoveryAddresses, s.Serializer, apiPrefix).WebService())

	return nil
}

// Exposes given api groups in the API.
func (s *GenericAPIServer) InstallAPIGroups(apiGroupInfos ...*APIGroupInfo) error {
	for _, apiGroupInfo := range apiGroupInfos {
		// Do not register empty group or empty version.  Doing so claims /apis/ for the wrong entity to be returned.
		// Catching these here places the error  much closer to its origin
		if len(apiGroupInfo.PrioritizedVersions[0].Group) == 0 {
			return fmt.Errorf("cannot register handler with an empty group for %#v", *apiGroupInfo)
		}
		if len(apiGroupInfo.PrioritizedVersions[0].Version) == 0 {
			return fmt.Errorf("cannot register handler with an empty version for %#v", *apiGroupInfo)
		}
	}

	openAPIModels, err := s.getOpenAPIModels(APIGroupPrefix, apiGroupInfos...)
	if err != nil {
		return fmt.Errorf("unable to get openapi models: %v", err)
	}

	for _, apiGroupInfo := range apiGroupInfos {
		if err := s.installAPIResources(APIGroupPrefix, apiGroupInfo, openAPIModels); err != nil {
			return fmt.Errorf("unable to install api resources: %v", err)
		}

		// setup discovery
		// Install the version handler.
		// Add a handler at /apis/<groupName> to enumerate all versions supported by this group.
		apiVersionsForDiscovery := []metav1.GroupVersionForDiscovery{}
		for _, groupVersion := range apiGroupInfo.PrioritizedVersions {
			// Check the config to make sure that we elide versions that don't have any resources
			if len(apiGroupInfo.VersionedResourcesStorageMap[groupVersion.Version]) == 0 {
				continue
			}
			apiVersionsForDiscovery = append(apiVersionsForDiscovery, metav1.GroupVersionForDiscovery{
				GroupVersion: groupVersion.String(),
				Version:      groupVersion.Version,
			})
		}
		preferredVersionForDiscovery := metav1.GroupVersionForDiscovery{
			GroupVersion: apiGroupInfo.PrioritizedVersions[0].String(),
			Version:      apiGroupInfo.PrioritizedVersions[0].Version,
		}
		apiGroup := metav1.APIGroup{
			Name:             apiGroupInfo.PrioritizedVersions[0].Group,
			Versions:         apiVersionsForDiscovery,
			PreferredVersion: preferredVersionForDiscovery,
		}

		s.DiscoveryGroupManager.AddGroup(apiGroup)
		s.Handler.GoRestfulContainer.Add(discovery.NewAPIGroupHandler(s.Serializer, apiGroup).WebService())
	}
	return nil
}

// Exposes the given api group in the API.
func (s *GenericAPIServer) InstallAPIGroup(apiGroupInfo *APIGroupInfo) error {
	return s.InstallAPIGroups(apiGroupInfo)
}

func (s *GenericAPIServer) getAPIGroupVersion(apiGroupInfo *APIGroupInfo, groupVersion schema.GroupVersion, apiPrefix string) *genericapi.APIGroupVersion {
	storage := make(map[string]rest.Storage)
	for k, v := range apiGroupInfo.VersionedResourcesStorageMap[groupVersion.Version] {
		storage[strings.ToLower(k)] = v
	}
	version := s.newAPIGroupVersion(apiGroupInfo, groupVersion)
	version.Root = apiPrefix
	version.Storage = storage
	return version
}

func (s *GenericAPIServer) newAPIGroupVersion(apiGroupInfo *APIGroupInfo, groupVersion schema.GroupVersion) *genericapi.APIGroupVersion {
	return &genericapi.APIGroupVersion{
		GroupVersion:     groupVersion,
		MetaGroupVersion: apiGroupInfo.MetaGroupVersion,

		ParameterCodec:  apiGroupInfo.ParameterCodec,
		Serializer:      apiGroupInfo.NegotiatedSerializer,
		Creater:         apiGroupInfo.Scheme,
		Convertor:       apiGroupInfo.Scheme,
		UnsafeConvertor: runtime.UnsafeObjectConvertor(apiGroupInfo.Scheme),
		Defaulter:       apiGroupInfo.Scheme,
		Typer:           apiGroupInfo.Scheme,
		Linker:          runtime.SelfLinker(meta.NewAccessor()),

		Admit:                        s.admissionControl,
		MinRequestTimeout:            s.minRequestTimeout,
		EnableAPIResponseCompression: s.enableAPIResponseCompression,
		Authorizer:                   s.Authorizer,
	}
}

// NewDefaultAPIGroupInfo returns an APIGroupInfo stubbed with "normal" values
// exposed for easier composition from other packages
func NewDefaultAPIGroupInfo(group string, scheme *runtime.Scheme, parameterCodec runtime.ParameterCodec, codecs serializer.CodecFactory) APIGroupInfo {
	return APIGroupInfo{
		PrioritizedVersions:          scheme.PrioritizedVersionsForGroup(group),
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		// TODO unhardcode this.  It was hardcoded before, but we need to re-evaluate
		OptionsExternalVersion: &schema.GroupVersion{Version: "v1"},
		Scheme:                 scheme,
		ParameterCodec:         parameterCodec,
		NegotiatedSerializer:   codecs,
	}
}

// getOpenAPIModels is a private method for getting the OpenAPI models
func (s *GenericAPIServer) getOpenAPIModels(apiPrefix string, apiGroupInfos ...*APIGroupInfo) (openapiproto.Models, error) {
	if s.openAPIConfig == nil {
		return nil, nil
	}
	pathsToIgnore := openapiutil.NewTrie(s.openAPIConfig.IgnorePrefixes)
	resourceNames := make([]string, 0)
	for _, apiGroupInfo := range apiGroupInfos {
		groupResources, err := getResourceNamesForGroup(apiPrefix, apiGroupInfo, pathsToIgnore)
		if err != nil {
			return nil, err
		}
		resourceNames = append(resourceNames, groupResources...)
	}

	// Build the openapi definitions for those resources and convert it to proto models
	openAPISpec, err := openapibuilder.BuildOpenAPIDefinitionsForResources(s.openAPIConfig, resourceNames...)
	if err != nil {
		return nil, err
	}
	return utilopenapi.ToProtoModels(openAPISpec)
}

// getResourceNamesForGroup is a private method for getting the canonical names for each resource to build in an api group
func getResourceNamesForGroup(apiPrefix string, apiGroupInfo *APIGroupInfo, pathsToIgnore openapiutil.Trie) ([]string, error) {
	// Get the canonical names of every resource we need to build in this api group
	resourceNames := make([]string, 0)
	for _, groupVersion := range apiGroupInfo.PrioritizedVersions {
		for resource, storage := range apiGroupInfo.VersionedResourcesStorageMap[groupVersion.Version] {
			path := gpath.Join(apiPrefix, groupVersion.Group, groupVersion.Version, resource)
			if !pathsToIgnore.HasPrefix(path) {
				kind, err := genericapi.GetResourceKind(groupVersion, storage, apiGroupInfo.Scheme)
				if err != nil {
					return nil, err
				}
				sampleObject, err := apiGroupInfo.Scheme.New(kind)
				if err != nil {
					return nil, err
				}
				name := openapiutil.GetCanonicalTypeName(sampleObject)
				resourceNames = append(resourceNames, name)
			}
		}
	}

	return resourceNames, nil
}
