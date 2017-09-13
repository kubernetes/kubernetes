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
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
)

type LoadBalancerType string
type NetworkTier string

const (
	AnnotationTrueValue  = "true"
	AnnotationFalseValue = "false"

	// ServiceAnnotationLoadBalancerType is annotated on a service with type LoadBalancer
	// dictates what specific kind of GCP LB should be assembled.
	// Currently, only "internal" is supported.
	ServiceAnnotationLoadBalancerType         = "cloud.google.com/load-balancer-type"
	ServiceAnnotationLoadBalancerTypeInternal = "Internal"
	ServiceAnnotationLoadBalancerTypeExternal = "External"

	LBTypeInternal LoadBalancerType = ServiceAnnotationLoadBalancerTypeInternal
	LBTypeExternal LoadBalancerType = ServiceAnnotationLoadBalancerTypeExternal

	// ServiceAnnotationInternalBackendShare is annotated on a service with "true" when users
	// want to share GCP Backend Services for a set of internal load balancers.
	// ALPHA feature - this may be removed in a future release.
	ServiceAnnotationILBBackendShare = "alpha.cloud.google.com/load-balancer-backend-share"

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

type lbAnnotation struct {
	// values includes all valid values for the annotation key.
	values sets.String
	// defaultValue is the default value for the annotation key.
	defaultValue string
}

var (
	// allServiceLBAnnotations defines all supported load balancer annoations
	// for Service.
	allServiceLBAnnotations = map[string]*lbAnnotation{
		ServiceAnnotationLoadBalancerType: {
			values: sets.NewString(
				ServiceAnnotationLoadBalancerTypeInternal,
				ServiceAnnotationLoadBalancerTypeExternal,
				"internal", // The lower-case value has been deprecated.
			),
			defaultValue: ServiceAnnotationLoadBalancerTypeExternal,
		},
		ServiceAnnotationILBBackendShare: {
			values:       sets.NewString(AnnotationTrueValue, AnnotationFalseValue),
			defaultValue: AnnotationFalseValue,
		},
		NetworkTierAnnotationKey: {
			values:       sets.NewString(NetworkTierAnnotationPremium, NetworkTierAnnotationStandard),
			defaultValue: NetworkTierAnnotationPremium,
		},
	}

	// deprecatedServiceAnnotations maps the deprecated key to its replacement.
	// The deprecation keys are not defined as constants on purpose to avoid
	// accidental uses.
	deprecatedServiceLBAnnotations = map[string]string{
		// This key did not correctly specify "alpha" and has been deprecated.
		"cloud.google.com/load-balancer-backend-share": ServiceAnnotationLoadBalancerType,
	}

	// ILBAnnotationKeys is a set of all supported annotation keys for an
	// internal load balancer.
	ILBAnnotationKeys = sets.NewString(ServiceAnnotationILBBackendShare)
	// ELBAnnotationKeys is a set of all supported annotation keys for an
	// external load balancer.
	ELBAnnotationKeys = sets.NewString(NetworkTierAnnotationKey)
)

// validateServiceLBAnnotations validates LB-related annotations in Service.
func validateServiceLBAnnotations(svc *v1.Service) error {
	if svc.Spec.Type != v1.ServiceTypeLoadBalancer {
		return fmt.Errorf("expeceted service type %v, got %v ", v1.ServiceTypeLoadBalancer, svc.Spec.Type)
	}

	if v, ok := svc.Annotations[ServiceAnnotationLoadBalancerType]; ok {
		if err := validateLBAnnotation(ServiceAnnotationLoadBalancerType, v); err != nil {
			return err
		}
	}

	var allErrs []error
	var incompatibleKeys sets.String
	// Determine incompatible keys based on the LB type.
	lbType := GetLoadBalancerAnnotationType(svc)
	if lbType == LBTypeInternal {
		incompatibleKeys = ELBAnnotationKeys.Difference(ILBAnnotationKeys)
	} else {
		incompatibleKeys = ILBAnnotationKeys.Difference(ELBAnnotationKeys)
	}

	for k, v := range svc.Annotations {
		if incompatibleKeys.Has(k) {
			allErrs = append(allErrs, fmt.Errorf("key %q is compatible with LB type %q", k, lbType))
			continue
		}
		err := validateLBAnnotation(k, v)
		allErrs = append(allErrs, err)
	}
	return utilerrors.NewAggregate(allErrs)
}

func validateLBAnnotation(key, value string) error {
	// If the key has been deprecated, replace it with the new key.
	if newKey, ok := deprecatedServiceLBAnnotations[key]; ok {
		glog.Warning("key %s has been deprecated. Please use %q in the future", key, newKey)
		key = newKey
	}

	r, ok := allServiceLBAnnotations[key]
	if !ok {
		// Ignore unknown keys.
		return nil
	}

	if !r.values.Has(value) {
		return fmt.Errorf("invalid value %q for key %q", value, key)
	}

	return nil
}

// GetServiceLBAnnotationValue returns the value of the given annotation key if
// the key exists. If not, it returs the default value. It is the caller's
// responsibility to ensure that the given key exists in allServiceLBAnnotations.
func GetServiceLBAnnotationValue(svc *v1.Service, key string) string {
	if newKey, ok := deprecatedServiceLBAnnotations[key]; ok {
		key = newKey
	}
	if value, ok := svc.Annotations[key]; ok {
		return value
	}

	if r, ok := allServiceLBAnnotations[key]; ok {
		return r.defaultValue
	}
	return ""
}

// GetLoadBalancerAnnotationType returns the type of GCP load balancer which should be assembled.
func GetLoadBalancerAnnotationType(service *v1.Service) LoadBalancerType {
	lbType := GetServiceLBAnnotationValue(service, ServiceAnnotationLoadBalancerType)
	// The lower-case "internal" has been deprecated.
	if lbType == "internal" {
		return LBTypeInternal
	}
	return LoadBalancerType(lbType)
}

// GetLoadBalancerAnnotationBackendShare returns whether this service's backend service should be
// shared with other load balancers. Health checks and the healthcheck firewall will be shared regardless.
func GetLoadBalancerAnnotationBackendShare(service *v1.Service) bool {
	v := GetServiceLBAnnotationValue(service, ServiceAnnotationILBBackendShare)
	if v == AnnotationTrueValue {
		return true
	}
	return false
}

func GetServiceNetworkTier(service *v1.Service) NetworkTier {
	v := GetServiceLBAnnotationValue(service, NetworkTierAnnotationKey)
	return NetworkTier(v)
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
