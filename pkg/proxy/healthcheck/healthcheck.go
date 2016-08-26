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

package healthcheck

import (
	"fmt"
	"net"
	"net/http"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

// proxyMutationRequest: Message to request addition/deletion of endpoints for a service
type proxyMutationRequest struct {
	serviceName  types.NamespacedName
	endpointUids *sets.String
}

// proxyListenerRequest: Message to request addition/deletion of a service responder on a listening port
type proxyListenerRequest struct {
	serviceName     types.NamespacedName
	listenPort      uint16
	add             bool
	responseChannel chan bool
}

// serviceEndpointsList: A list of endpoints for a service
type serviceEndpointsList struct {
	serviceName types.NamespacedName
	endpoints   *sets.String
}

// serviceResponder: Contains net/http datastructures necessary for responding to each Service's health check on its aux nodePort
type serviceResponder struct {
	serviceName types.NamespacedName
	listenPort  uint16
	listener    *net.Listener
	server      *http.Server
}

// proxyHC: Handler structure for health check, endpoint add/delete and service listener add/delete requests
type proxyHC struct {
	serviceEndpointsMap    cache.ThreadSafeStore
	serviceResponderMap    map[types.NamespacedName]serviceResponder
	mutationRequestChannel chan *proxyMutationRequest
	listenerRequestChannel chan *proxyListenerRequest
}

// handleHealthCheckRequest - received a health check request - lookup and respond to HC.
func (h *proxyHC) handleHealthCheckRequest(rw http.ResponseWriter, serviceName string) {
	s, ok := h.serviceEndpointsMap.Get(serviceName)
	if !ok {
		glog.V(4).Infof("Service %s not found or has no local endpoints", serviceName)
		sendHealthCheckResponse(rw, http.StatusServiceUnavailable, "No Service Endpoints Found")
		return
	}
	numEndpoints := len(*s.(*serviceEndpointsList).endpoints)
	if numEndpoints > 0 {
		sendHealthCheckResponse(rw, http.StatusOK, fmt.Sprintf("%d Service Endpoints found", numEndpoints))
		return
	}
	sendHealthCheckResponse(rw, http.StatusServiceUnavailable, "0 local Endpoints are alive")
}

// handleMutationRequest - receive requests to mutate the table entry for a service
func (h *proxyHC) handleMutationRequest(req *proxyMutationRequest) {
	numEndpoints := len(*req.endpointUids)
	glog.V(4).Infof("LB service health check mutation request Service: %s - %d Endpoints %v",
		req.serviceName, numEndpoints, (*req.endpointUids).List())
	if numEndpoints == 0 {
		if _, ok := h.serviceEndpointsMap.Get(req.serviceName.String()); ok {
			glog.V(4).Infof("Deleting endpoints map for service %s, all local endpoints gone", req.serviceName.String())
			h.serviceEndpointsMap.Delete(req.serviceName.String())
		}
		return
	}
	var entry *serviceEndpointsList
	e, exists := h.serviceEndpointsMap.Get(req.serviceName.String())
	if exists {
		entry = e.(*serviceEndpointsList)
		if entry.endpoints.Equal(*req.endpointUids) {
			return
		}
		// Compute differences just for printing logs about additions and removals
		deletedEndpoints := entry.endpoints.Difference(*req.endpointUids)
		newEndpoints := req.endpointUids.Difference(*entry.endpoints)
		for _, e := range newEndpoints.List() {
			glog.V(4).Infof("Adding local endpoint %s to LB health check for service %s",
				e, req.serviceName.String())
		}
		for _, d := range deletedEndpoints.List() {
			glog.V(4).Infof("Deleted endpoint %s from service %s LB health check (%d endpoints left)",
				d, req.serviceName.String(), len(*entry.endpoints))
		}
	}
	entry = &serviceEndpointsList{serviceName: req.serviceName, endpoints: req.endpointUids}
	h.serviceEndpointsMap.Add(req.serviceName.String(), entry)
}

// proxyHealthCheckRequest - Factory method to instantiate the health check handler
func proxyHealthCheckFactory() *proxyHC {
	glog.V(2).Infof("Initializing kube-proxy health checker")
	phc := &proxyHC{
		serviceEndpointsMap:    cache.NewThreadSafeStore(cache.Indexers{}, cache.Indices{}),
		serviceResponderMap:    make(map[types.NamespacedName]serviceResponder),
		mutationRequestChannel: make(chan *proxyMutationRequest, 1024),
		listenerRequestChannel: make(chan *proxyListenerRequest, 1024),
	}
	return phc
}
