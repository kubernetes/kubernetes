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

package discovery

import (
	"net/http"

	"github.com/emicklei/go-restful"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// legacyRootAPIDiscoveryHandler creates a webservice serving api group discovery.
// Note: during the server runtime apiGroups might change.
type legacyRootAPIDiscoveryHandler struct {
	// discoveryAddresses is used to build cluster IPs for discovery.
	discoveryAddresses DiscoveryAddresses
	apiPrefix          string
	serializer         runtime.NegotiatedSerializer
	apiVersions        []string
}

func NewLegacyRootAPIDiscoveryHandler(discoveryAddresses DiscoveryAddresses, serializer runtime.NegotiatedSerializer, apiPrefix string, apiVersions []string) *legacyRootAPIDiscoveryHandler {
	// Because in release 1.1, /apis returns response with empty APIVersion, we
	// use stripVersionNegotiatedSerializer to keep the response backwards
	// compatible.
	serializer = stripVersionNegotiatedSerializer{serializer}

	return &legacyRootAPIDiscoveryHandler{
		discoveryAddresses: discoveryAddresses,
		apiPrefix:          apiPrefix,
		serializer:         serializer,
		apiVersions:        apiVersions,
	}
}

// AddApiWebService adds a service to return the supported api versions at the legacy /api.
func (s *legacyRootAPIDiscoveryHandler) WebService() *restful.WebService {
	// Because in release 1.1, /api returns response with empty APIVersion, we
	// use stripVersionNegotiatedSerializer to keep the response backwards
	// compatible.
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s.serializer)
	ws := new(restful.WebService)
	ws.Path(s.apiPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(s.handle).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIVersions{}))
	return ws
}

func (s *legacyRootAPIDiscoveryHandler) handle(req *restful.Request, resp *restful.Response) {
	clientIP := utilnet.GetClientIP(req.Request)
	apiVersionsForDiscovery := &metav1.APIVersions{
		ServerAddressByClientCIDRs: s.discoveryAddresses.ServerAddressByClientCIDRs(clientIP),
		Versions:                   s.apiVersions,
	}

	responsewriters.WriteObjectNegotiated(s.serializer, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, apiVersionsForDiscovery)
}
