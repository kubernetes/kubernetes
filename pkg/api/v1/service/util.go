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
	"strings"

	v1 "k8s.io/api/core/v1"
	utilnet "k8s.io/utils/net"
)

const (
	defaultLoadBalancerSourceRanges = "0.0.0.0/0"
)

// IsAllowAll checks whether the utilnet.IPNet allows traffic from 0.0.0.0/0
func IsAllowAll(ipnets utilnet.IPNetSet) bool {
	for _, s := range ipnets.StringSlice() {
		if s == "0.0.0.0/0" {
			return true
		}
	}
	return false
}

// GetLoadBalancerSourceRanges first try to parse and verify LoadBalancerSourceRanges field from a service.
// If the field is not specified, turn to parse and verify the AnnotationLoadBalancerSourceRangesKey annotation from a service,
// extracting the source ranges to allow, and if not present returns a default (allow-all) value.
func GetLoadBalancerSourceRanges(service *v1.Service) (utilnet.IPNetSet, error) {
	var ipnets utilnet.IPNetSet
	var err error
	// if SourceRange field is specified, ignore sourceRange annotation
	if len(service.Spec.LoadBalancerSourceRanges) > 0 {
		specs := service.Spec.LoadBalancerSourceRanges
		ipnets, err = utilnet.ParseIPNets(specs...)

		if err != nil {
			return nil, fmt.Errorf("service.Spec.LoadBalancerSourceRanges: %v is not valid. Expecting a list of IP ranges. For example, 10.0.0.0/24. Error msg: %v", specs, err)
		}
	} else {
		val := service.Annotations[v1.AnnotationLoadBalancerSourceRangesKey]
		val = strings.TrimSpace(val)
		if val == "" {
			val = defaultLoadBalancerSourceRanges
		}
		specs := strings.Split(val, ",")
		ipnets, err = utilnet.ParseIPNets(specs...)
		if err != nil {
			return nil, fmt.Errorf("%s: %s is not valid. Expecting a comma-separated list of source IP ranges. For example, 10.0.0.0/24,192.168.2.0/24", v1.AnnotationLoadBalancerSourceRangesKey, val)
		}
	}
	return ipnets, nil
}

// ExternallyAccessible checks if service is externally accessible.
func ExternallyAccessible(service *v1.Service) bool {
	return service.Spec.Type == v1.ServiceTypeLoadBalancer ||
		service.Spec.Type == v1.ServiceTypeNodePort ||
		(service.Spec.Type == v1.ServiceTypeClusterIP && len(service.Spec.ExternalIPs) > 0)
}

// ExternalPolicyLocal checks if service is externally accessible and has ETP = Local.
func ExternalPolicyLocal(service *v1.Service) bool {
	if !ExternallyAccessible(service) {
		return false
	}
	return service.Spec.ExternalTrafficPolicy == v1.ServiceExternalTrafficPolicyLocal
}

// InternalPolicyLocal checks if service has ITP = Local.
func InternalPolicyLocal(service *v1.Service) bool {
	if service.Spec.InternalTrafficPolicy == nil {
		return false
	}
	return *service.Spec.InternalTrafficPolicy == v1.ServiceInternalTrafficPolicyLocal
}

// NeedsHealthCheck checks if service needs health check.
func NeedsHealthCheck(service *v1.Service) bool {
	if service.Spec.Type != v1.ServiceTypeLoadBalancer {
		return false
	}
	return ExternalPolicyLocal(service)
}
