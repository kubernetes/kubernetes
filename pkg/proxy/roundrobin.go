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
	"fmt"
	"net"
	"reflect"
	"strconv"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
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
type balancerKey types.NamespacedName

// LoadBalancerRR is a round-robin load balancer.
type LoadBalancerRR struct {
	lock     sync.RWMutex
	services map[balancerKey]*balancerState
}

type balancerState struct {
	endpoints []string
	index     int
	affinity  affinityPolicy
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
		services: map[balancerKey]*balancerState{},
	}
}

func (lb *LoadBalancerRR) NewService(service types.NamespacedName, affinityType api.AffinityType, ttlMinutes int) error {
	lb.lock.Lock()
	defer lb.lock.Unlock()

	lb.newServiceInternal(service, affinityType, ttlMinutes)
	return nil
}

// This assumes that lb.lock is already held.
func (lb *LoadBalancerRR) newServiceInternal(service types.NamespacedName, affinityType api.AffinityType, ttlMinutes int) *balancerState {
	if ttlMinutes == 0 {
		ttlMinutes = 180 //default to 3 hours if not specified.  Should 0 be unlimeted instead????
	}

	key := balancerKey(service)
	if _, exists := lb.services[key]; !exists {
		lb.services[key] = &balancerState{affinity: *newAffinityPolicy(affinityType, ttlMinutes)}
		glog.V(4).Infof("LoadBalancerRR service %q did not exist, created", service)
	}
	return lb.services[key]
}

// return true if this service is using some form of session affinity.
func isSessionAffinity(affinity *affinityPolicy) bool {
	// Should never be empty string, but checking for it to be safe.
	if affinity.affinityType == "" || affinity.affinityType == api.AffinityTypeNone {
		return false
	}
	return true
}

// NextEndpoint returns a service endpoint.
// The service endpoint is chosen using the round-robin algorithm.
func (lb *LoadBalancerRR) NextEndpoint(service types.NamespacedName, srcAddr net.Addr) (string, error) {
	// Coarse locking is simple.  We can get more fine-grained if/when we
	// can prove it matters.
	lb.lock.Lock()
	defer lb.lock.Unlock()

	key := balancerKey(service)
	state, exists := lb.services[key]
	if !exists || state == nil {
		return "", ErrMissingServiceEntry
	}
	if len(state.endpoints) == 0 {
		return "", ErrMissingEndpoints
	}
	glog.V(4).Infof("NextEndpoint for service %q, srcAddr=%v: endpoints: %+v", service, srcAddr, state.endpoints)

	sessionAffinityEnabled := isSessionAffinity(&state.affinity)

	var ipaddr string
	if sessionAffinityEnabled {
		// Caution: don't shadow ipaddr
		var err error
		ipaddr, _, err = net.SplitHostPort(srcAddr.String())
		if err != nil {
			return "", fmt.Errorf("malformed source address %q: %v", srcAddr.String(), err)
		}
		sessionAffinity, exists := state.affinity.affinityMap[ipaddr]
		if exists && int(time.Now().Sub(sessionAffinity.lastUsed).Minutes()) < state.affinity.ttlMinutes {
			// Affinity wins.
			endpoint := sessionAffinity.endpoint
			sessionAffinity.lastUsed = time.Now()
			glog.V(4).Infof("NextEndpoint for service %q from IP %s with sessionAffinity %+v: %s", service, ipaddr, sessionAffinity, endpoint)
			return endpoint, nil
		}
	}
	// Take the next endpoint.
	endpoint := state.endpoints[state.index]
	state.index = (state.index + 1) % len(state.endpoints)

	if sessionAffinityEnabled {
		var affinity *affinityState
		affinity = state.affinity.affinityMap[ipaddr]
		if affinity == nil {
			affinity = new(affinityState) //&affinityState{ipaddr, "TCP", "", endpoint, time.Now()}
			state.affinity.affinityMap[ipaddr] = affinity
		}
		affinity.lastUsed = time.Now()
		affinity.endpoint = endpoint
		affinity.clientIP = ipaddr
		glog.V(4).Infof("Updated affinity key %s: %+v", ipaddr, state.affinity.affinityMap[ipaddr])
	}

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

// Remove any session affinity records associated to a particular endpoint (for example when a pod goes down).
func removeSessionAffinityByEndpoint(state *balancerState, service balancerKey, endpoint string) {
	for _, affinity := range state.affinity.affinityMap {
		if affinity.endpoint == endpoint {
			glog.V(4).Infof("Removing client: %s from affinityMap for service %q", affinity.endpoint, service)
			delete(state.affinity.affinityMap, affinity.clientIP)
		}
	}
}

// Loop through the valid endpoints and then the endpoints associated with the Load Balancer.
// Then remove any session affinity records that are not in both lists.
// This assumes the lb.lock is held.
func (lb *LoadBalancerRR) updateAffinityMap(service balancerKey, newEndpoints []string) {
	allEndpoints := map[string]int{}
	for _, newEndpoint := range newEndpoints {
		allEndpoints[newEndpoint] = 1
	}
	state, exists := lb.services[service]
	if !exists {
		return
	}
	for _, existingEndpoint := range state.endpoints {
		allEndpoints[existingEndpoint] = allEndpoints[existingEndpoint] + 1
	}
	for mKey, mVal := range allEndpoints {
		if mVal == 1 {
			glog.V(3).Infof("Delete endpoint %s for service %q", mKey, service)
			removeSessionAffinityByEndpoint(state, service, mKey)
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
		name := types.NewNamespacedName(svcEndpoints.Namespace, svcEndpoints.Name)
		key := balancerKey(*name)
		state, exists := lb.services[key]
		curEndpoints := []string{}
		if state != nil {
			curEndpoints = state.endpoints
		}
		newEndpoints := filterValidEndpoints(svcEndpoints.Endpoints)
		if !exists || state == nil || len(curEndpoints) != len(newEndpoints) || !slicesEquiv(slice.CopyStrings(curEndpoints), newEndpoints) {
			glog.V(3).Infof("LoadBalancerRR: Setting endpoints for %s to %+v", svcEndpoints.Name, svcEndpoints.Endpoints)
			lb.updateAffinityMap(key, newEndpoints)
			// On update can be called without NewService being called externally.
			// To be safe we will call it here.  A new service will only be created
			// if one does not already exist.
			state = lb.newServiceInternal(*name, api.AffinityTypeNone, 0)
			state.endpoints = slice.ShuffleStrings(newEndpoints)

			// Reset the round-robin index.
			state.index = 0
		}
		registeredEndpoints[key] = true
	}
	// Remove endpoints missing from the update.
	for k := range lb.services {
		if _, exists := registeredEndpoints[k]; !exists {
			glog.V(3).Infof("LoadBalancerRR: Removing endpoints for %s", k)
			delete(lb.services, k)
		}
	}
}

// Tests whether two slices are equivalent.  This sorts both slices in-place.
func slicesEquiv(lhs, rhs []string) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	if reflect.DeepEqual(slice.SortStrings(lhs), slice.SortStrings(rhs)) {
		return true
	}
	return false
}

func (lb *LoadBalancerRR) CleanupStaleStickySessions(service types.NamespacedName) {
	lb.lock.Lock()
	defer lb.lock.Unlock()

	key := balancerKey(service)
	state, exists := lb.services[key]
	if !exists {
		glog.Warning("CleanupStaleStickySessions called for non-existent balancer key %q", service)
		return
	}
	for ip, affinity := range state.affinity.affinityMap {
		if int(time.Now().Sub(affinity.lastUsed).Minutes()) >= state.affinity.ttlMinutes {
			glog.V(4).Infof("Removing client %s from affinityMap for service %q", affinity.clientIP, service)
			delete(state.affinity.affinityMap, ip)
		}
	}
}
