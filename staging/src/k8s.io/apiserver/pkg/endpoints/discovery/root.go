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
	"sort"
	"sync"

	"github.com/emicklei/go-restful"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

type DiscoveryGroupManager interface {
	AddAPIGroup(apiGroup metav1.APIGroup)
	RemoveAPIGroup(groupName string)

	WebService() *restful.WebService
}

// rootAPIsDiscoveryHandler creates a webservice serving api group discovery.
// Note: during the server runtime apiGroups might change.
type rootAPIsDiscoveryHandler struct {
	// DiscoveryAddresses is used to build cluster IPs for discovery.
	discoveryAddresses DiscoveryAddresses

	serializer runtime.NegotiatedSerializer

	// Map storing information about all groups to be exposed in discovery response.
	// The map is from name to the group.
	lock      sync.RWMutex
	apiGroups map[string]metav1.APIGroup
}

func NewRootAPIsDiscoveryHandler(discoveryAddresses DiscoveryAddresses, serializer runtime.NegotiatedSerializer) *rootAPIsDiscoveryHandler {
	// Because in release 1.1, /apis returns response with empty APIVersion, we
	// use stripVersionNegotiatedSerializer to keep the response backwards
	// compatible.
	serializer = stripVersionNegotiatedSerializer{serializer}

	return &rootAPIsDiscoveryHandler{
		discoveryAddresses: discoveryAddresses,
		serializer:         serializer,
		apiGroups:          map[string]metav1.APIGroup{},
	}
}

func (s *rootAPIsDiscoveryHandler) AddAPIGroup(apiGroup metav1.APIGroup) {
	s.lock.Lock()
	defer s.lock.Unlock()

	s.apiGroups[apiGroup.Name] = apiGroup
}

func (s *rootAPIsDiscoveryHandler) RemoveAPIGroup(groupName string) {
	s.lock.Lock()
	defer s.lock.Unlock()

	delete(s.apiGroups, groupName)
}

func (s *rootAPIsDiscoveryHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	s.lock.RLock()
	defer s.lock.RUnlock()

	// sort to have a deterministic order
	sortedGroups := []metav1.APIGroup{}
	groupNames := make([]string, 0, len(s.apiGroups))
	for groupName := range s.apiGroups {
		groupNames = append(groupNames, groupName)
	}
	sort.Strings(groupNames)
	for _, groupName := range groupNames {
		sortedGroups = append(sortedGroups, s.apiGroups[groupName])
	}

	clientIP := utilnet.GetClientIP(req)
	serverCIDR := s.discoveryAddresses.ServerAddressByClientCIDRs(clientIP)
	groups := make([]metav1.APIGroup, len(sortedGroups))
	for i := range sortedGroups {
		groups[i] = sortedGroups[i]
		groups[i].ServerAddressByClientCIDRs = serverCIDR
	}

	responsewriters.WriteObjectNegotiated(s.serializer, schema.GroupVersion{}, resp, req, http.StatusOK, &metav1.APIGroupList{Groups: groups})
}

func (s *rootAPIsDiscoveryHandler) restfulHandle(req *restful.Request, resp *restful.Response) {
	s.ServeHTTP(resp.ResponseWriter, req.Request)
}

// WebService returns a webservice serving api group discovery.
// Note: during the server runtime apiGroups might change.
func (s *rootAPIsDiscoveryHandler) WebService() *restful.WebService {
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s.serializer)
	ws := new(restful.WebService)
	ws.Path(APIGroupPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(s.restfulHandle).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIGroupList{}))
	return ws
}
