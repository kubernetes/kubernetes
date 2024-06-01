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
	api "k8s.io/kubernetes/pkg/apis/core"
)

// ExternallyAccessible checks if service is externally accessible.
func ExternallyAccessible(service *api.Service) bool {
	return service.Spec.Type == api.ServiceTypeLoadBalancer ||
		service.Spec.Type == api.ServiceTypeNodePort ||
		(service.Spec.Type == api.ServiceTypeClusterIP && len(service.Spec.ExternalIPs) > 0)
}

// RequestsOnlyLocalTraffic checks if service requests OnlyLocal traffic.
func RequestsOnlyLocalTraffic(service *api.Service) bool {
	if service.Spec.Type != api.ServiceTypeLoadBalancer &&
		service.Spec.Type != api.ServiceTypeNodePort {
		return false
	}

	return service.Spec.ExternalTrafficPolicy == api.ServiceExternalTrafficPolicyLocal
}

// NeedsHealthCheck checks if service needs health check.
func NeedsHealthCheck(service *api.Service) bool {
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		return false
	}
	return RequestsOnlyLocalTraffic(service)
}
