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
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
)

const (
	// AnnotationLoadBalancerSourceRangesKey is the key of the annotation on a service to set allowed ingress ranges on their LoadBalancers
	//
	// It should be a comma-separated list of CIDRs, e.g. `0.0.0.0/0` to
	// allow full access (the default) or `18.0.0.0/8,56.0.0.0/8` to allow
	// access only from the CIDRs currently allocated to MIT & the USPS.
	//
	// Not all cloud providers support this annotation, though AWS & GCE do.
	AnnotationLoadBalancerSourceRangesKey = "service.beta.kubernetes.io/load-balancer-source-ranges"

	// AnnotationValueExternalTrafficLocal Value of annotation to specify local endpoints behaviour
	AnnotationValueExternalTrafficLocal = "OnlyLocal"
	// AnnotationValueExternalTrafficGlobal Value of annotation to specify global (legacy) behaviour
	AnnotationValueExternalTrafficGlobal = "Global"

	// TODO: The alpha annotations have been deprecated, remove them when we move this feature to GA.

	// AlphaAnnotationHealthCheckNodePort Annotation specifying the healthcheck nodePort for the service
	// If not specified, annotation is created by the service api backend with the allocated nodePort
	// Will use user-specified nodePort value if specified by the client
	AlphaAnnotationHealthCheckNodePort = "service.alpha.kubernetes.io/healthcheck-nodeport"

	// AlphaAnnotationExternalTraffic An annotation that denotes if this Service desires to route external traffic to local
	// endpoints only. This preserves Source IP and avoids a second hop.
	AlphaAnnotationExternalTraffic = "service.alpha.kubernetes.io/external-traffic"

	// BetaAnnotationHealthCheckNodePort is the beta version of AlphaAnnotationHealthCheckNodePort.
	BetaAnnotationHealthCheckNodePort = "service.beta.kubernetes.io/healthcheck-nodeport"

	// BetaAnnotationExternalTraffic is the beta version of AlphaAnnotationExternalTraffic.
	BetaAnnotationExternalTraffic = "service.beta.kubernetes.io/external-traffic"
)

// NeedsHealthCheck Check service for health check annotations
func NeedsHealthCheck(service *api.Service) bool {
	// First check the alpha annotation and then the beta. This is so existing
	// Services continue to work till the user decides to transition to beta.
	// If they transition to beta, there's no way to go back to alpha without
	// rolling back the cluster.
	for _, annotation := range []string{AlphaAnnotationExternalTraffic, BetaAnnotationExternalTraffic} {
		if l, ok := service.Annotations[annotation]; ok {
			if l == AnnotationValueExternalTrafficLocal {
				return true
			} else if l == AnnotationValueExternalTrafficGlobal {
				return false
			} else {
				glog.Errorf("Invalid value for annotation %v: %v", annotation, l)
			}
		}
	}
	return false
}

// GetServiceHealthCheckNodePort Return health check node port annotation for service, if one exists
func GetServiceHealthCheckNodePort(service *api.Service) int32 {
	if !NeedsHealthCheck(service) {
		return 0
	}
	// First check the alpha annotation and then the beta. This is so existing
	// Services continue to work till the user decides to transition to beta.
	// If they transition to beta, there's no way to go back to alpha without
	// rolling back the cluster.
	for _, annotation := range []string{AlphaAnnotationHealthCheckNodePort, BetaAnnotationHealthCheckNodePort} {
		if l, ok := service.Annotations[annotation]; ok {
			p, err := strconv.Atoi(l)
			if err != nil {
				glog.Errorf("Failed to parse annotation %v: %v", annotation, err)
				continue
			}
			return int32(p)
		}
	}
	return 0
}
