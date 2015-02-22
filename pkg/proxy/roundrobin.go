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

type affinityState struct {
	clientIP string
	//clientProtocol  api.Protocol //not yet used
	//sessionCookie   string       //not yet used
	endpoint string
	lastUsed time.Time
}

type affinityPolicy struct {
	affinityType api.AffinityType
	affinityMap  map[string]*affinityState // map client IP -> affinity info
	ttlMinutes   int
}

// balancerKey is a string that the balancer uses to key stored state.
type balancerKey string

// LoadBalancerRR is a round-robin load balancer.
type LoadBalancerRR struct {
	lock          sync.RWMutex
	endpointsMap  map[balancerKey][]string
	rrIndex       map[balancerKey]int
	serviceDtlMap map[balancerKey]affinityPolicy
}

func newAffinityPolicy(affinityType api.AffinityType, ttlMinutes int) *affinityPolicy {
	return &affinityPolicy{
		affinityType: affinityType,
		affinityMap:  make(map[string]*affinityState),
		ttlMinutes:   ttlMinutes,
	}
}

// NewLoadBalancerRR returns a new LoadBalancerRR.
func NewLoadBalancerRR() *LoadBalancerRR {
	return &LoadBalancerRR{
		endpointsMap:  make(map[balancerKey][]string),
		rrIndex:       make(map[balancerKey]int),
		serviceDtlMap: make(map[balancerKey]affinityPolicy),
	}
}

func (lb *LoadBalancerRR) NewService(service string, affinityType api.AffinityType, ttlMinutes int) error {
	lb.lock.Lock()
	defer lb.lock.Unlock()

	lb.newServiceInternal(service, affinityType, ttlMinutes)
	return nil
}

func (lb *LoadBalancerRR) newServiceInternal(service string, affinityType api.AffinityType, ttlMinutes int) {
	if ttlMinutes == 0 {
		ttlMinutes = 180 //default to 3 hours if not specified.  Should 0 be unlimeted instead????
	}
	if _, exists := lb.serviceDtlMap[balancerKey(service)]; !exists {
		lb.serviceDtlMap[balancerKey(service)] = *newAffinityPolicy(affinityType, ttlMinutes)
		glog.V(4).Infof("NewService.  Service does not exist.  So I created it: %+v", lb.serviceDtlMap[balancerKey(service)])
	}
}

// return true if this service is using some form of session affinity.
func isSessionAffinity(affinity *affinityPolicy) bool {
	//Should never be empty string, but chekcing for it to be safe.
	if affinity.affinityType == "" || affinity.affinityType == api.AffinityTypeNone {
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
	serviceDtls, exists := lb.serviceDtlMap[balancerKey(service)]
	endpoints, _ := lb.endpointsMap[balancerKey(service)]
	index := lb.rrIndex[balancerKey(service)]
	sessionAffinityEnabled := isSessionAffinity(&serviceDtls)

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
		sessionAffinity, exists := serviceDtls.affinityMap[ipaddr]
		glog.V(4).Infof("NextEndpoint.  Key: %s. sessionAffinity: %+v", ipaddr, sessionAffinity)
		if exists && int(time.Now().Sub(sessionAffinity.lastUsed).Minutes()) < serviceDtls.ttlMinutes {
			endpoint := sessionAffinity.endpoint
			sessionAffinity.lastUsed = time.Now()
			glog.V(4).Infof("NextEndpoint.  Key: %s. sessionAffinity: %+v", ipaddr, sessionAffinity)
			return endpoint, nil
		}
	}
	endpoint := endpoints[index]
	lb.lock.Lock()
	lb.rrIndex[balancerKey(service)] = (index + 1) % len(endpoints)

	if sessionAffinityEnabled {
		var affinity *affinityState
		affinity, _ = lb.serviceDtlMap[balancerKey(service)].affinityMap[ipaddr]
		if affinity == nil {
			affinity = new(affinityState) //&affinityState{ipaddr, "TCP", "", endpoint, time.Now()}
			lb.serviceDtlMap[balancerKey(service)].affinityMap[ipaddr] = affinity
		}
		affinity.lastUsed = time.Now()
		affinity.endpoint = endpoint
		affinity.clientIP = ipaddr

		glog.V(4).Infof("NextEndpoint. New Affinity key %s: %+v", ipaddr, lb.serviceDtlMap[balancerKey(service)].affinityMap[ipaddr])
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
func removeSessionAffinityByEndpoint(lb *LoadBalancerRR, service balancerKey, endpoint string) {
	for _, affinity := range lb.serviceDtlMap[service].affinityMap {
		if affinity.endpoint == endpoint {
			glog.V(4).Infof("Removing client: %s from affinityMap for service: %s", affinity.endpoint, service)
			delete(lb.serviceDtlMap[service].affinityMap, affinity.clientIP)
		}
	}
}

// Loop through the valid endpoints and then the endpoints associated with the Load Balancer.
// Then remove any session affinity records that are not in both lists.
// This assumes the lb.lock is held.
func updateAffinityMap(lb *LoadBalancerRR, service balancerKey, newEndpoints []string) {
	allEndpoints := map[string]int{}
	for _, validEndpoint := range newEndpoints {
		allEndpoints[validEndpoint] = 1
	}
	for _, existingEndpoint := range lb.endpointsMap[service] {
		allEndpoints[existingEndpoint] = allEndpoints[existingEndpoint] + 1
	}
	for mKey, mVal := range allEndpoints {
		if mVal == 1 {
			glog.V(3).Infof("Delete endpoint %s for service: %s", mKey, service)
			removeSessionAffinityByEndpoint(lb, service, mKey)
			delete(lb.serviceDtlMap[service].affinityMap, mKey)
		}
	}
}

// OnUpdate manages the registered service endpoints.
// Registered endpoints are updated if found in the update set or
// unregistered if missing from the update set.
func (lb *LoadBalancerRR) OnUpdate(allEndpoints []api.Endpoints) {
	registeredEndpoints := make(map[balancerKey]bool)
	lb.lock.Lock()
	defer lb.lock.Unlock()
	// Update endpoints for services.
	for _, svcEndpoints := range allEndpoints {
		curEndpoints, exists := lb.endpointsMap[balancerKey(svcEndpoints.Name)]
		newEndpoints := filterValidEndpoints(svcEndpoints.Endpoints)
		if !exists || !reflect.DeepEqual(slice.SortStrings(slice.CopyStrings(curEndpoints)), slice.SortStrings(newEndpoints)) {
			glog.V(3).Infof("LoadBalancerRR: Setting endpoints for %s to %+v", svcEndpoints.Name, svcEndpoints.Endpoints)
			updateAffinityMap(lb, balancerKey(svcEndpoints.Name), newEndpoints)
			// On update can be called without NewService being called externally.
			// to be safe we will call it here.  A new service will only be created
			// if one does not already exist.
			lb.newServiceInternal(svcEndpoints.Name, api.AffinityTypeNone, 0)
			lb.endpointsMap[balancerKey(svcEndpoints.Name)] = slice.ShuffleStrings(newEndpoints)

			// Reset the round-robin index.
			lb.rrIndex[balancerKey(svcEndpoints.Name)] = 0
		}
		registeredEndpoints[balancerKey(svcEndpoints.Name)] = true
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
	lb.lock.Lock()
	defer lb.lock.Unlock()

	ttlMinutes := lb.serviceDtlMap[balancerKey(service)].ttlMinutes
	for ip, affinity := range lb.serviceDtlMap[balancerKey(service)].affinityMap {
		if int(time.Now().Sub(affinity.lastUsed).Minutes()) >= ttlMinutes {
			glog.V(4).Infof("Removing client: %s from affinityMap for service: %s.  Last used is greater than %d minutes....", affinity.clientIP, service, ttlMinutes)
			delete(lb.serviceDtlMap[balancerKey(service)].affinityMap, ip)
		}
	}
}
