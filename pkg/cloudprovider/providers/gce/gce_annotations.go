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

package gce

import (
	"fmt"
	"strings"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
)

type LoadBalancerType string
type NetworkTier string

const (
	// ServiceAnnotationLoadBalancerType is annotated on a service with type LoadBalancer
	// dictates what specific kind of GCP LB should be assembled.
	// Currently, only "internal" is supported.
	ServiceAnnotationLoadBalancerType = "cloud.google.com/load-balancer-type"

	LBTypeInternal LoadBalancerType = "Internal"
	// Deprecating the lowercase spelling of Internal.
	deprecatedTypeInternalLowerCase LoadBalancerType = "internal"

	// ServiceAnnotationInternalBackendShare is annotated on a service with "true" when users
	// want to share GCP Backend Services for a set of internal load balancers.
	// ALPHA feature - this may be removed in a future release.
	ServiceAnnotationILBBackendShare = "alpha.cloud.google.com/load-balancer-backend-share"
	// This annotation did not correctly specify "alpha", so both annotations will be checked.
	deprecatedServiceAnnotationILBBackendShare = "cloud.google.com/load-balancer-backend-share"

	// NetworkTierAnnotationKey is annotated on a Service object to indicate which
	// network tier a GCP LB should use. The valid values are "Standard" and
	// "Premium" (default).
	NetworkTierAnnotationKey      = "cloud.google.com/network-tier"
	NetworkTierAnnotationStandard = "Standard"
	NetworkTierAnnotationPremium  = "Premium"

	NetworkTierStandard NetworkTier = NetworkTierAnnotationStandard
	NetworkTierPremium  NetworkTier = NetworkTierAnnotationPremium
	NetworkTierDefault  NetworkTier = NetworkTierPremium
)

// GetLoadBalancerAnnotationType returns the type of GCP load balancer which should be assembled.
func GetLoadBalancerAnnotationType(service *v1.Service) (LoadBalancerType, bool) {
	v := LoadBalancerType("")
	if service.Spec.Type != v1.ServiceTypeLoadBalancer {
		return v, false
	}

	l, ok := service.Annotations[ServiceAnnotationLoadBalancerType]
	v = LoadBalancerType(l)
	if !ok {
		return v, false
	}

	switch v {
	case LBTypeInternal, deprecatedTypeInternalLowerCase:
		return LBTypeInternal, true
	default:
		return v, false
	}
}

// GetLoadBalancerAnnotationBackendShare returns whether this service's backend service should be
// shared with other load balancers. Health checks and the healthcheck firewall will be shared regardless.
func GetLoadBalancerAnnotationBackendShare(service *v1.Service) bool {
	if l, exists := service.Annotations[ServiceAnnotationILBBackendShare]; exists && l == "true" {
		return true
	}

	// Check for deprecated annotation key
	if l, exists := service.Annotations[deprecatedServiceAnnotationILBBackendShare]; exists && l == "true" {
		glog.Warningf("Annotation %q is deprecated and replaced with an alpha-specific key: %q", deprecatedServiceAnnotationILBBackendShare, ServiceAnnotationILBBackendShare)
		return true
	}

	return false
}

// GetServiceNetworkTier returns the network tier of GCP load balancer
// which should be assembled, and an error if the specified tier is not
// supported.
func GetServiceNetworkTier(service *v1.Service) (NetworkTier, error) {
	l, ok := service.Annotations[NetworkTierAnnotationKey]
	if !ok {
		return NetworkTierDefault, nil
	}

	v := NetworkTier(l)
	switch v {
	case NetworkTierStandard:
		fallthrough
	case NetworkTierPremium:
		return v, nil
	default:
		return NetworkTierDefault, fmt.Errorf("unsupported network tier: %q", v)
	}
}

// ToGCEValue converts NetworkTier to a string that we can populate the
// NetworkTier field of GCE objects.
func (n NetworkTier) ToGCEValue() string {
	return strings.ToUpper(string(n))
}

// NetworkTierGCEValueToType converts the value of the NetworkTier field of a
// GCE object to the NetworkTier type.
func NetworkTierGCEValueToType(s string) NetworkTier {
	switch s {
	case "STANDARD":
		return NetworkTierStandard
	case "PREMIUM":
		return NetworkTierPremium
	default:
		return NetworkTier(s)
	}
}
