/*
Copyright 2022 The Kubernetes Authors.

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

package aggregated

import (
	"net/http"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime"

	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

type WrappedHandler struct {
	s                     runtime.NegotiatedSerializer
	handler               http.Handler
	aggHandler            http.Handler
	peerAggregatedHandler http.Handler
}

func (wrapped *WrappedHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	// mediaType.Convert looks at the request accept headers and is used to control whether the discovery document will be aggregated.
	mediaType, _ := negotiation.NegotiateMediaTypeOptions(req.Header.Get("Accept"), wrapped.s.SupportedMediaTypes(), DiscoveryEndpointRestrictions)
	if IsAggregatedDiscoveryGVK(mediaType.Convert) {
		if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.UnknownVersionInteroperabilityProxy) && mediaType.Convert.Version == "v2" {
			if mediaType.Profile == "nopeer" {
				NoPeerDiscoveryRequestCounter.Inc()
				wrapped.aggHandler.ServeHTTP(resp, req)
				return
			}
			if wrapped.peerAggregatedHandler != nil {
				// Serve peer-aggregated discovery by default if provided.
				wrapped.peerAggregatedHandler.ServeHTTP(resp, req)
				return
			}
		}
		wrapped.aggHandler.ServeHTTP(resp, req)
		return
	}
	wrapped.handler.ServeHTTP(resp, req)
}

func (wrapped *WrappedHandler) restfulHandle(req *restful.Request, resp *restful.Response) {
	wrapped.ServeHTTP(resp.ResponseWriter, req.Request)
}

func (wrapped *WrappedHandler) GenerateWebService(prefix string, returnType interface{}) *restful.WebService {
	mediaTypes, _ := negotiation.MediaTypesForSerializer(wrapped.s)
	ws := new(restful.WebService)
	ws.Path(prefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(wrapped.restfulHandle).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(returnType))
	return ws
}

// WrapAggregatedDiscoveryToHandler wraps a handler with an option to
// emit the aggregated discovery by passing in the aggregated
// discovery type in content negotiation headers: eg: (Accept:
// application/json;v=v2;g=apidiscovery.k8s.io;as=APIGroupDiscoveryList)
//
// If peerAggregatedHandler is non-nil, it will be used to serve peer-aggregated
// discovery requests which aggregate discovery information from peer API servers.
// To request a no-peer aggregation, the client must include the "profile=nopeer" parameter in the
// Accept header eg: (Accept: application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList;profile=nopeer).
func WrapAggregatedDiscoveryToHandler(handler, aggHandler, peerAggregatedHandler http.Handler) *WrappedHandler {
	scheme := runtime.NewScheme()
	utilruntime.Must(apidiscoveryv2.AddToScheme(scheme))
	utilruntime.Must(apidiscoveryv2beta1.AddToScheme(scheme))
	codecs := serializer.NewCodecFactory(scheme)
	return &WrappedHandler{codecs, handler, aggHandler, peerAggregatedHandler}
}
