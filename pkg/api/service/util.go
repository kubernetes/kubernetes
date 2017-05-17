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
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"

	"github.com/golang/glog"
)

const (
	defaultLoadBalancerSourceRanges = "0.0.0.0/0"
)

// IsAllowAll checks whether the netsets.IPNet allows traffic from 0.0.0.0/0
func IsAllowAll(ipnets netsets.IPNet) bool {
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
func GetLoadBalancerSourceRanges(service *api.Service) (netsets.IPNet, error) {
	var ipnets netsets.IPNet
	var err error
	// if SourceRange field is specified, ignore sourceRange annotation
	if len(service.Spec.LoadBalancerSourceRanges) > 0 {
		specs := service.Spec.LoadBalancerSourceRanges
		ipnets, err = netsets.ParseIPNets(specs...)

		if err != nil {
			return nil, fmt.Errorf("service.Spec.LoadBalancerSourceRanges: %v is not valid. Expecting a list of IP ranges. For example, 10.0.0.0/24. Error msg: %v", specs, err)
		}
	} else {
		val := service.Annotations[api.AnnotationLoadBalancerSourceRangesKey]
		val = strings.TrimSpace(val)
		if val == "" {
			val = defaultLoadBalancerSourceRanges
		}
		specs := strings.Split(val, ",")
		ipnets, err = netsets.ParseIPNets(specs...)
		if err != nil {
			return nil, fmt.Errorf("%s: %s is not valid. Expecting a comma-separated list of source IP ranges. For example, 10.0.0.0/24,192.168.2.0/24", api.AnnotationLoadBalancerSourceRangesKey, val)
		}
	}
	return ipnets, nil
}

// RequestsOnlyLocalTraffic checks if service requests OnlyLocal traffic.
func RequestsOnlyLocalTraffic(service *api.Service) bool {
	if service.Spec.Type != api.ServiceTypeLoadBalancer &&
		service.Spec.Type != api.ServiceTypeNodePort {
		return false
	}

	// First check the beta annotation and then the first class field. This is so that
	// existing Services continue to work till the user decides to transition to the
	// first class field.
	if l, ok := service.Annotations[api.BetaAnnotationExternalTraffic]; ok {
		switch l {
		case api.AnnotationValueExternalTrafficLocal:
			return true
		case api.AnnotationValueExternalTrafficGlobal:
			return false
		default:
			glog.Errorf("Invalid value for annotation %v: %v", api.BetaAnnotationExternalTraffic, l)
			return false
		}
	}
	return service.Spec.ExternalTrafficPolicy == api.ServiceExternalTrafficPolicyTypeLocal
}

// NeedsHealthCheck Check if service needs health check.
func NeedsHealthCheck(service *api.Service) bool {
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		return false
	}
	return RequestsOnlyLocalTraffic(service)
}

// GetServiceHealthCheckNodePort Return health check node port for service, if one exists
func GetServiceHealthCheckNodePort(service *api.Service) int32 {
	// First check the beta annotation and then the first class field. This is so that
	// existing Services continue to work till the user decides to transition to the
	// first class field.
	if l, ok := service.Annotations[api.BetaAnnotationHealthCheckNodePort]; ok {
		p, err := strconv.Atoi(l)
		if err != nil {
			glog.Errorf("Failed to parse annotation %v: %v", api.BetaAnnotationHealthCheckNodePort, err)
			return 0
		}
		return int32(p)
	}
	return service.Spec.HealthCheckNodePort
}

// ClearExternalTrafficPolicy resets the ExternalTrafficPolicy field.
func ClearExternalTrafficPolicy(service *api.Service) {
	// First check the beta annotation and then the first class field. This is so that
	// existing Services continue to work till the user decides to transition to the
	// first class field.
	if _, ok := service.Annotations[api.BetaAnnotationExternalTraffic]; ok {
		delete(service.Annotations, api.BetaAnnotationExternalTraffic)
		return
	}
	service.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyType("")
}

// SetServiceHealthCheckNodePort sets the given health check node port on service.
// It does not check whether this service needs healthCheckNodePort.
func SetServiceHealthCheckNodePort(service *api.Service, hcNodePort int32) {
	// First check the beta annotation and then the first class field. This is so that
	// existing Services continue to work till the user decides to transition to the
	// first class field.
	if _, ok := service.Annotations[api.BetaAnnotationExternalTraffic]; ok {
		if hcNodePort == 0 {
			delete(service.Annotations, api.BetaAnnotationHealthCheckNodePort)
		} else {
			service.Annotations[api.BetaAnnotationHealthCheckNodePort] = fmt.Sprintf("%d", hcNodePort)
		}
		return
	}
	service.Spec.HealthCheckNodePort = hcNodePort
}
