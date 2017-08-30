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

package service

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"
)

const (
	allowAllRange = "0.0.0.0/0"
)

// IsAllowAll checks whether the netsets.IPNet allows traffic from 0.0.0.0/0
func IsAllowAll(ipnets netsets.IPNet) bool {
	for _, s := range ipnets.StringSlice() {
		if s == allowAllRange {
			return true
		}
	}
	return false
}

// GetLoadBalancerSourceRanges first try to parse and verify LoadBalancerSourceRanges field from a service.
// If the field is not specified, returns a default (allow-all) value.
func GetLoadBalancerSourceRanges(service *api.Service) (netsets.IPNet, error) {
	specs := []string{allowAllRange}
	if len(service.Spec.LoadBalancerSourceRanges) > 0 {
		specs = service.Spec.LoadBalancerSourceRanges
	}

	ipnets, err := netsets.ParseIPNets(specs...)
	if err != nil {
		return nil, fmt.Errorf("service.Spec.LoadBalancerSourceRanges: %v is not valid. Expecting a list of IP ranges. For example, 10.0.0.0/24. Error msg: %v", specs, err)
	}
	return ipnets, nil
}

// RequestsOnlyLocalTraffic checks if service requests OnlyLocal traffic.
func RequestsOnlyLocalTraffic(service *api.Service) bool {
	if service.Spec.Type != api.ServiceTypeLoadBalancer &&
		service.Spec.Type != api.ServiceTypeNodePort {
		return false
	}

	return service.Spec.ExternalTrafficPolicy == api.ServiceExternalTrafficPolicyTypeLocal
}

// NeedsHealthCheck checks if service needs health check.
func NeedsHealthCheck(service *api.Service) bool {
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		return false
	}
	return RequestsOnlyLocalTraffic(service)
}
