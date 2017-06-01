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

import "k8s.io/kubernetes/pkg/api/v1"

type LoadBalancerType string

const (
	// ServiceAnnotationLoadBalancerType is the annotation used on a service with type LoadBalancer
	// dictates what specific kind of GCP LB should be assembled.
	// Currently, only "internal" is supported.
	ServiceAnnotationLoadBalancerType = "cloud.google.com/load-balancer-type"

	LBTypeInternal LoadBalancerType = "internal"
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
	case LBTypeInternal:
		return v, true
	default:
		return v, false
	}
}
