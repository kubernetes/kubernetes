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
	"k8s.io/client-go/1.4/pkg/api"
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

	// AnnotationExternalTraffic An annotation that denotes if this Service desires to route external traffic to local
	// endpoints only. This preserves Source IP and avoids a second hop.
	AnnotationExternalTraffic = "service.alpha.kubernetes.io/external-traffic"
	// AnnotationValueExternalTrafficLocal Value of annotation to specify local endpoints behaviour
	AnnotationValueExternalTrafficLocal = "OnlyLocal"
	// AnnotationValueExternalTrafficGlobal Value of annotation to specify global (legacy) behaviour
	AnnotationValueExternalTrafficGlobal = "Global"
	// AnnotationHealthCheckNodePort Annotation specifying the healthcheck nodePort for the service
	// If not specified, annotation is created by the service api backend with the allocated nodePort
	// Will use user-specified nodePort value if specified by the client
	AnnotationHealthCheckNodePort = "service.alpha.kubernetes.io/healthcheck-nodeport"
)

// NeedsHealthCheck Check service for health check annotations
func NeedsHealthCheck(service *api.Service) bool {
	if l, ok := service.Annotations[AnnotationExternalTraffic]; ok {
		if l == AnnotationValueExternalTrafficLocal {
			return true
		} else if l == AnnotationValueExternalTrafficGlobal {
			return false
		} else {
			glog.Errorf("Invalid value for annotation %v", AnnotationExternalTraffic)
			return false
		}
	}
	return false
}

// GetServiceHealthCheckNodePort Return health check node port annotation for service, if one exists
func GetServiceHealthCheckNodePort(service *api.Service) int32 {
	if NeedsHealthCheck(service) {
		if l, ok := service.Annotations[AnnotationHealthCheckNodePort]; ok {
			p, err := strconv.Atoi(l)
			if err != nil {
				glog.Errorf("Failed to parse annotation %v: %v", AnnotationHealthCheckNodePort, err)
				return 0
			}
			return int32(p)
		}
	}
	return 0
}

// GetServiceHealthCheckPathPort Return the path and nodePort programmed into the Cloud LB Health Check
func GetServiceHealthCheckPathPort(service *api.Service) (string, int32) {
	if !NeedsHealthCheck(service) {
		return "", 0
	}
	port := GetServiceHealthCheckNodePort(service)
	if port == 0 {
		return "", 0
	}
	return "/healthz", port
}
