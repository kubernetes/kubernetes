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

// RoundRobin Loadbalancer

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

type LoadBalancerRR struct {
	lock         sync.RWMutex
	endpointsMap map[string][]string
	rrIndex      map[string]int
}

func NewLoadBalancerRR() *LoadBalancerRR {
	return &LoadBalancerRR{endpointsMap: make(map[string][]string), rrIndex: make(map[string]int)}
}

func (impl LoadBalancerRR) LoadBalance(service string, srcAddr net.Addr) (string, error) {
	impl.lock.RLock()
	endpoints, exists := impl.endpointsMap[service]
	index := impl.rrIndex[service]
	impl.lock.RUnlock()
	if exists == false {
		return "", errors.New("no service entry for:" + service)
	}
	if len(endpoints) == 0 {
		return "", errors.New("no endpoints for: " + service)
	}
	endpoint := endpoints[index]
	impl.rrIndex[service] = (index + 1) % len(endpoints)
	return endpoint, nil
}

func (impl LoadBalancerRR) IsValid(spec string) bool {
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

func (impl LoadBalancerRR) FilterValidEndpoints(endpoints []string) []string {
	var result []string
	for _, spec := range endpoints {
		if impl.IsValid(spec) {
			result = append(result, spec)
		}
	}
	return result
}

func (impl LoadBalancerRR) OnUpdate(endpoints []api.Endpoints) {
	tmp := make(map[string]bool)
	impl.lock.Lock()
	defer impl.lock.Unlock()
	// First update / add all new endpoints for services.
	for _, value := range endpoints {
		existingEndpoints, exists := impl.endpointsMap[value.Name]
		validEndpoints := impl.FilterValidEndpoints(value.Endpoints)
		if !exists || !reflect.DeepEqual(existingEndpoints, validEndpoints) {
			glog.Infof("LoadBalancerRR: Setting endpoints for %s to %+v", value.Name, value.Endpoints)
			impl.endpointsMap[value.Name] = validEndpoints
			// Start RR from the beginning if added or updated.
			impl.rrIndex[value.Name] = 0
		}
		tmp[value.Name] = true
	}
	// Then remove any endpoints no longer relevant
	for key, value := range impl.endpointsMap {
		_, exists := tmp[key]
		if !exists {
			glog.Infof("LoadBalancerRR: Removing endpoints for %s -> %+v", key, value)
			delete(impl.endpointsMap, key)
		}
	}
}
