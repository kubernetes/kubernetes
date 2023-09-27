/*
Copyright 2017 The Kubernetes Authors.

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

// Package aggregator contains a server that aggregates content from a generic control
// plane server, apiextensions server, and CustomResourceDefinitions.
package miniaggregator

import (
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"

	apiextensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apiextensions-apiserver/pkg/controller/openapi/builder"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kube-aggregator/pkg/controllers/openapi/aggregator"
	"k8s.io/kube-openapi/pkg/handler"

	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
)

var (
	// DiscoveryScheme defines methods for serializing and deserializing API objects.
	DiscoveryScheme = runtime.NewScheme()

	// DiscoveryCodecs provides methods for retrieving codecs and serializers for specific
	// versions and content types.
	DiscoveryCodecs = serializer.NewCodecFactory(DiscoveryScheme)
)

func init() {
	// we need to add the options to empty v1
	// TODO fix the server code to avoid this
	metav1.AddToGroupVersion(DiscoveryScheme, schema.GroupVersion{Version: "v1"})

	// TODO: keep the generic API server from wanting this
	unversioned := schema.GroupVersion{Group: "", Version: "v1"}
	DiscoveryScheme.AddUnversionedTypes(unversioned,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)
}

// MiniAggregatorConfig contains configuration settings for the mini aggregator.
type MiniAggregatorConfig struct {
	GenericConfig genericapiserver.Config
}

// completedMiniAggregatorConfig contains completed configuration settings for
// the mini aggregator. Any fields not filled in by the user that are required
// to have valid data are defaulted. This struct is private and ultimately
// embedded in CompletedMiniAggregatorConfig to require the user to invoke
// Complete() prior to being able to instantiate a MiniAggregatorServer.
type completedMiniAggregatorConfig struct {
	GenericConfig genericapiserver.CompletedConfig
}

// CompletedMiniAggregatorConfig contains completed configuration settings for
// the mini aggregator. Any fields not filled in by the user that are required
// to have valid data are defaulted.
type CompletedMiniAggregatorConfig struct {
	*completedMiniAggregatorConfig
}

// MiniAggregatorServer sits in front of the Apis and
// ApiExtensions servers and aggregates them.
type MiniAggregatorServer struct {
	// GenericAPIServer is the aggregator's server.
	GenericAPIServer *genericapiserver.GenericAPIServer
	// Apis is the server for the minimal control plane. It serves
	// APIs such as core v1, certificates.k8s.io, RBAC, etc.
	Apis *controlplaneapiserver.Server
	// ApiExtensions is the server for API extensions.
	ApiExtensions *apiextensionsapiserver.CustomResourceDefinitions
}

// Complete fills in any fields not set that are required to have valid data.
// It's mutating the receiver.
func (cfg *MiniAggregatorConfig) Complete() CompletedMiniAggregatorConfig {
	// CRITICAL: to be able to provide our own /openapi/v2 implementation that aggregates
	// content from multiple servers, we *must* skip OpenAPI installation. Otherwise,
	// when PrepareRun() is invoked, it will register a handler for /openapi/v2,
	// replacing the aggregator's handler.
	cfg.GenericConfig.SkipOpenAPIInstallation = true

	return CompletedMiniAggregatorConfig{
		completedMiniAggregatorConfig: &completedMiniAggregatorConfig{
			GenericConfig: cfg.GenericConfig.Complete(nil),
		},
	}
}

// New creates a new MiniAggregatorServer.
func (c completedMiniAggregatorConfig) New(
	delegationTarget genericapiserver.DelegationTarget,
	apis *controlplaneapiserver.Server,
	crds *apiextensionsapiserver.CustomResourceDefinitions,
) (*MiniAggregatorServer, error) {
	genericServer, err := c.GenericConfig.New("mini-aggregator", delegationTarget)
	if err != nil {
		return nil, err
	}

	s := &MiniAggregatorServer{
		GenericAPIServer: genericServer,
		Apis:             apis,
		ApiExtensions:    crds,
	}

	// Have to do this as a filter because of how the APIServerHandler.Director serves requests.
	s.GenericAPIServer.Handler.GoRestfulContainer.Filter(s.filterAPIsRequest)

	s.GenericAPIServer.Handler.NonGoRestfulMux.HandleFunc("/openapi/v2", s.serveOpenAPI)

	return s, nil
}

// filterAPIsRequest checks if the request is for /apis, and if so, it aggregates group discovery
// for the generic control plane server, apiextensions server (which provides the apiextensions.k8s.io group),
// and the CRDs themselves.
func (s *MiniAggregatorServer) filterAPIsRequest(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	if req.Request.URL.Path != "/apis" && req.Request.URL.Path != "/apis/" {
		chain.ProcessFilter(req, resp)
		return
	}

	// Discovery for things like core, authentication, authorization, certificates, ...
	gcpGroups, err := s.Apis.GenericAPIServer.DiscoveryGroupManager.Groups(req.Request.Context(), req.Request)
	if err != nil {
		http.Error(resp.ResponseWriter, fmt.Sprintf("error retrieving generic control plane discovery groups: %v", err), http.StatusInternalServerError)
	}

	// Discovery for the apiextensions group itself
	apiextensionsGroups, err := s.ApiExtensions.GenericAPIServer.DiscoveryGroupManager.Groups(req.Request.Context(), req.Request)
	if err != nil {
		http.Error(resp.ResponseWriter, fmt.Sprintf("error retrieving apiextensions discovery groups: %v", err), http.StatusInternalServerError)
	}

	// Discovery for all the groups contributed by CRDs
	crdGroups, err := s.ApiExtensions.DiscoveryGroupLister.Groups(req.Request.Context(), req.Request)
	if err != nil {
		http.Error(resp.ResponseWriter, fmt.Sprintf("error retrieving custom resource discovery groups: %v", err), http.StatusInternalServerError)
	}

	// Combine the slices using copy - more efficient than append
	combined := make([]metav1.APIGroup, len(gcpGroups)+len(apiextensionsGroups)+len(crdGroups))
	var i int
	i += copy(combined[i:], gcpGroups)
	i += copy(combined[i:], apiextensionsGroups)
	i += copy(combined[i:], crdGroups)

	responsewriters.WriteObjectNegotiated(DiscoveryCodecs, negotiation.DefaultEndpointRestrictions, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &metav1.APIGroupList{Groups: combined}, false)
}

// serveOpenAPI aggregates OpenAPI specs from the generic control plane and apiextensions servers.
func (s *MiniAggregatorServer) serveOpenAPI(w http.ResponseWriter, req *http.Request) {
	downloader := aggregator.NewDownloader()

	cluster := genericapirequest.ClusterFrom(req.Context())

	withCluster := func(handler http.Handler) http.HandlerFunc {
		return func(res http.ResponseWriter, req *http.Request) {
			if cluster != nil {
				req = req.Clone(genericapirequest.WithCluster(req.Context(), *cluster))
			}
			handler.ServeHTTP(res, req)
		}
	}

	// Can't use withCluster here because the GenericControlPlane doesn't have APIs coming from multiple logical clusters at this time.
	controlPlaneSpec, _, _, err := downloader.Download(s.Apis.GenericAPIServer.Handler.Director, "")

	// Use withCluster here because each logical cluster can have a distinct set of APIs coming from its CRDs.
	crdSpecs, _, _, err := downloader.Download(withCluster(s.ApiExtensions.GenericAPIServer.Handler.Director), "")

	// TODO(ncdc): merging on the fly is expensive. We may need to optimize this (e.g. caching).
	mergedSpecs, err := builder.MergeSpecs(controlPlaneSpec, crdSpecs)
	if err != nil {
		utilruntime.HandleError(err)
	}

	h := &singlePathHandler{}

	// In order to reuse the kube-openapi API as much as possible, we
	// register the OpenAPI service in the singlePathHandler
	handler.NewOpenAPIService(mergedSpecs).RegisterOpenAPIVersionedService("/openapi/v2", h)

	h.ServeHTTP(w, req)
}

// singlePathHandler is a dummy PathHandler that mainly allows grabbing a http.Handler
// from a PathHandler consumer and then being able to use the http.Handler
// to serve a request.
type singlePathHandler struct {
	handler [1]http.Handler
}

func (sph *singlePathHandler) Handle(path string, handler http.Handler) {
	sph.handler[0] = handler
}
func (sph *singlePathHandler) ServeHTTP(res http.ResponseWriter, req *http.Request) {
	if sph.handler[0] == nil {
		res.WriteHeader(404)
	}
	sph.handler[0].ServeHTTP(res, req)
}
