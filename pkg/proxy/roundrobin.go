/*
Copyright 2014 Google Inc. All rights reserved.

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

package proxy

import (
	"errors"
	"net"
	"reflect"
	"strconv"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/slice"
	"github.com/golang/glog"
)

var (
	ErrMissingServiceEntry = errors.New("missing service entry")
	ErrMissingEndpoints    = errors.New("missing endpoints")
)

type sessionAffinityDetail struct {
	clientIPAddress string
	//clientProtocol  api.Protocol //not yet used
	//sessionCookie   string       //not yet used
	endpoint     string
	lastUsedDTTM time.Time
}

type serviceDetail struct {
	name                string
	sessionAffinityType api.AffinityType
	sessionAffinityMap  map[string]*sessionAffinityDetail
	stickyMaxAgeMinutes int
}

// LoadBalancerRR is a round-robin load balancer.
type LoadBalancerRR struct {
	lock          sync.RWMutex
	endpointsMap  map[string][]string
	rrIndex       map[string]int
	serviceDtlMap map[string]serviceDetail
}

func newServiceDetail(service string, sessionAffinityType api.AffinityType, stickyMaxAgeMinutes int) *serviceDetail {
	return &serviceDetail{
		name:                service,
		sessionAffinityType: sessionAffinityType,
		sessionAffinityMap:  make(map[string]*sessionAffinityDetail),
		stickyMaxAgeMinutes: stickyMaxAgeMinutes,
	}
}

// NewLoadBalancerRR returns a new LoadBalancerRR.
func NewLoadBalancerRR() *LoadBalancerRR {
	return &LoadBalancerRR{
		endpointsMap:  make(map[string][]string),
		rrIndex:       make(map[string]int),
		serviceDtlMap: make(map[string]serviceDetail),
	}
}

func (lb *LoadBalancerRR) NewService(service string, sessionAffinityType api.AffinityType, stickyMaxAgeMinutes int) error {
	if stickyMaxAgeMinutes == 0 {
		stickyMaxAgeMinutes = 180 //default to 3 hours if not specified.  Should 0 be unlimeted instead????
	}
	if _, exists := lb.serviceDtlMap[service]; !exists {
		lb.serviceDtlMap[service] = *newServiceDetail(service, sessionAffinityType, stickyMaxAgeMinutes)
		glog.V(4).Infof("NewService.  Service does not exist.  So I created it: %+v", lb.serviceDtlMap[service])
	}
	return nil
}

// return true if this service detail is using some form of session affinity.
func isSessionAffinity(serviceDtl serviceDetail) bool {
	//Should never be empty string, but chekcing for it to be safe.
	if serviceDtl.sessionAffinityType == "" || serviceDtl.sessionAffinityType == api.AffinityTypeNone {
		return false
	}
	return true
}

// NextEndpoint returns a service endpoint.
// The service endpoint is chosen using the round-robin algorithm.
func (lb *LoadBalancerRR) NextEndpoint(service string, srcAddr net.Addr) (string, error) {
	var ipaddr string
	glog.V(4).Infof("NextEndpoint.  service: %s.  srcAddr: %+v. Endpoints: %+v", service, srcAddr, lb.endpointsMap)

	lb.lock.RLock()
	serviceDtls, exists := lb.serviceDtlMap[service]
	endpoints, _ := lb.endpointsMap[service]
	index := lb.rrIndex[service]
	sessionAffinityEnabled := isSessionAffinity(serviceDtls)

	lb.lock.RUnlock()
	if !exists {
		return "", ErrMissingServiceEntry
	}
	if len(endpoints) == 0 {
		return "", ErrMissingEndpoints
	}
	if sessionAffinityEnabled {
		if _, _, err := net.SplitHostPort(srcAddr.String()); err == nil {
			ipaddr, _, _ = net.SplitHostPort(srcAddr.String())
		}
		sessionAffinity, exists := serviceDtls.sessionAffinityMap[ipaddr]
		glog.V(4).Infof("NextEndpoint.  Key: %s. sessionAffinity: %+v", ipaddr, sessionAffinity)
		if exists && int(time.Now().Sub(sessionAffinity.lastUsedDTTM).Minutes()) < serviceDtls.stickyMaxAgeMinutes {
			endpoint := sessionAffinity.endpoint
			sessionAffinity.lastUsedDTTM = time.Now()
			glog.V(4).Infof("NextEndpoint.  Key: %s. sessionAffinity: %+v", ipaddr, sessionAffinity)
			return endpoint, nil
		}
	}
	endpoint := endpoints[index]
	lb.lock.Lock()
	lb.rrIndex[service] = (index + 1) % len(endpoints)

	if sessionAffinityEnabled {
		var affinity *sessionAffinityDetail
		affinity, _ = lb.serviceDtlMap[service].sessionAffinityMap[ipaddr]
		if affinity == nil {
			affinity = new(sessionAffinityDetail) //&sessionAffinityDetail{ipaddr, "TCP", "", endpoint, time.Now()}
			lb.serviceDtlMap[service].sessionAffinityMap[ipaddr] = affinity
		}
		affinity.lastUsedDTTM = time.Now()
		affinity.endpoint = endpoint
		affinity.clientIPAddress = ipaddr

		glog.V(4).Infof("NextEndpoint. New Affinity key %s: %+v", ipaddr, lb.serviceDtlMap[service].sessionAffinityMap[ipaddr])
	}

	lb.lock.Unlock()
	return endpoint, nil
}

func isValidEndpoint(ep *api.Endpoint) bool {
	return ep.IP != "" && ep.Port > 0
}

func filterValidEndpoints(endpoints []api.Endpoint) []string {
	// Convert Endpoint objects into strings for easier use later.  Ignore
	// the protocol field - we'll get that from the Service objects.
	var result []string
	for i := range endpoints {
		ep := &endpoints[i]
		if isValidEndpoint(ep) {
			result = append(result, net.JoinHostPort(ep.IP, strconv.Itoa(ep.Port)))
		}
	}
	return result
}

//remove any session affinity records associated to a particular endpoint (for example when a pod goes down).
func removeSessionAffinityByEndpoint(lb *LoadBalancerRR, service string, endpoint string) {
	for _, affinityDetail := range lb.serviceDtlMap[service].sessionAffinityMap {
		if affinityDetail.endpoint == endpoint {
			glog.V(4).Infof("Removing client: %s from sessionAffinityMap for service: %s", affinityDetail.endpoint, service)
			delete(lb.serviceDtlMap[service].sessionAffinityMap, affinityDetail.clientIPAddress)
		}
	}
}

//Loop through the valid endpoints and then the endpoints associated with the Load Balancer.
// 	Then remove any session affinity records that are not in both lists.
func updateServiceDetailMap(lb *LoadBalancerRR, service string, validEndpoints []string) {
	allEndpoints := map[string]int{}
	for _, validEndpoint := range validEndpoints {
		allEndpoints[validEndpoint] = 1
	}
	for _, existingEndpoint := range lb.endpointsMap[service] {
		allEndpoints[existingEndpoint] = allEndpoints[existingEndpoint] + 1
	}
	for mKey, mVal := range allEndpoints {
		if mVal == 1 {
			glog.V(3).Infof("Delete endpoint %s for service: %s", mKey, service)
			removeSessionAffinityByEndpoint(lb, service, mKey)
			delete(lb.serviceDtlMap[service].sessionAffinityMap, mKey)
		}
	}
}

// OnUpdate manages the registered service endpoints.
// Registered endpoints are updated if found in the update set or
// unregistered if missing from the update set.
func (lb *LoadBalancerRR) OnUpdate(endpoints []api.Endpoints) {
	registeredEndpoints := make(map[string]bool)
	lb.lock.Lock()
	defer lb.lock.Unlock()
	// Update endpoints for services.
	for _, endpoint := range endpoints {
		existingEndpoints, exists := lb.endpointsMap[endpoint.Name]
		validEndpoints := filterValidEndpoints(endpoint.Endpoints)
		if !exists || !reflect.DeepEqual(slice.SortStrings(slice.CopyStrings(existingEndpoints)), slice.SortStrings(validEndpoints)) {
			glog.V(3).Infof("LoadBalancerRR: Setting endpoints for %s to %+v", endpoint.Name, endpoint.Endpoints)
			updateServiceDetailMap(lb, endpoint.Name, validEndpoints)
			// On update can be called without NewService being called externally.
			// to be safe we will call it here.  A new service will only be created
			// if one does not already exist.
			lb.NewService(endpoint.Name, api.AffinityTypeNone, 0)
			lb.endpointsMap[endpoint.Name] = slice.ShuffleStrings(validEndpoints)

			// Reset the round-robin index.
			lb.rrIndex[endpoint.Name] = 0
		}
		registeredEndpoints[endpoint.Name] = true
	}
	// Remove endpoints missing from the update.
	for k, v := range lb.endpointsMap {
		if _, exists := registeredEndpoints[k]; !exists {
			glog.V(3).Infof("LoadBalancerRR: Removing endpoints for %s -> %+v", k, v)
			delete(lb.endpointsMap, k)
			delete(lb.serviceDtlMap, k)
		}
	}
}

func (lb *LoadBalancerRR) CleanupStaleStickySessions(service string) {
	stickyMaxAgeMinutes := lb.serviceDtlMap[service].stickyMaxAgeMinutes
	for key, affinityDetail := range lb.serviceDtlMap[service].sessionAffinityMap {
		if int(time.Now().Sub(affinityDetail.lastUsedDTTM).Minutes()) >= stickyMaxAgeMinutes {
			glog.V(4).Infof("Removing client: %s from sessionAffinityMap for service: %s.  Last used is greater than %d minutes....", affinityDetail.clientIPAddress, service, stickyMaxAgeMinutes)
			delete(lb.serviceDtlMap[service].sessionAffinityMap, key)
		}
	}
}
