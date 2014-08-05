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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

var (
	ErrMissingServiceEntry = errors.New("missing service entry")
	ErrMissingEndpoints    = errors.New("missing endpoints")
)

// LoadBalancerRR is a round-robin load balancer.
type LoadBalancerRR struct {
	lock         sync.RWMutex
	endpointsMap map[string][]string
	rrIndex      map[string]int
}

// NewLoadBalancerRR returns a new LoadBalancerRR.
func NewLoadBalancerRR() *LoadBalancerRR {
	return &LoadBalancerRR{
		endpointsMap: make(map[string][]string),
		rrIndex:      make(map[string]int),
	}
}

// NextEndpoint returns a service endpoint.
// The service endpoint is chosen using the round-robin algorithm.
func (lb LoadBalancerRR) NextEndpoint(service string, srcAddr net.Addr) (string, error) {
	lb.lock.RLock()
	endpoints, exists := lb.endpointsMap[service]
	index := lb.rrIndex[service]
	lb.lock.RUnlock()
	if !exists {
		return "", ErrMissingServiceEntry
	}
	if len(endpoints) == 0 {
		return "", ErrMissingEndpoints
	}
	endpoint := endpoints[index]
	lb.lock.Lock()
	lb.rrIndex[service] = (index + 1) % len(endpoints)
	lb.lock.Unlock()
	return endpoint, nil
}

func (lb LoadBalancerRR) isValid(spec string) bool {
	_, port, err := net.SplitHostPort(spec)
	if err != nil {
		return false
	}
	value, err := strconv.Atoi(port)
	if err != nil {
		return false
	}
	return value > 0
}

func (lb LoadBalancerRR) filterValidEndpoints(endpoints []string) []string {
	var result []string
	for _, spec := range endpoints {
		if lb.isValid(spec) {
			result = append(result, spec)
		}
	}
	return result
}

// OnUpdate manages the registered service endpoints.
// Registered endpoints are updated if found in the update set or
// unregistered if missing from the update set.
func (lb LoadBalancerRR) OnUpdate(endpoints []api.Endpoints) {
	registeredEndpoints := make(map[string]bool)
	lb.lock.Lock()
	defer lb.lock.Unlock()
	// Update endpoints for services.
	for _, endpoint := range endpoints {
		existingEndpoints, exists := lb.endpointsMap[endpoint.ID]
		validEndpoints := lb.filterValidEndpoints(endpoint.Endpoints)
		if !exists || !reflect.DeepEqual(existingEndpoints, validEndpoints) {
			glog.Infof("LoadBalancerRR: Setting endpoints for %s to %+v", endpoint.ID, endpoint.Endpoints)
			lb.endpointsMap[endpoint.ID] = validEndpoints
			// Reset the round-robin index.
			lb.rrIndex[endpoint.ID] = 0
		}
		registeredEndpoints[endpoint.ID] = true
	}
	// Remove endpoints missing from the update.
	for k, v := range lb.endpointsMap {
		if _, exists := registeredEndpoints[k]; !exists {
			glog.Infof("LoadBalancerRR: Removing endpoints for %s -> %+v", k, v)
			delete(lb.endpointsMap, k)
		}
	}
}
