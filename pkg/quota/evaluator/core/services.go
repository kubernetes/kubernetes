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

package core

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
)

// the name used for object count quota
var serviceObjectCountName = generic.ObjectCountQuotaResourceNameFor(v1.SchemeGroupVersion.WithResource("services").GroupResource())

// serviceResources are the set of resources managed by quota associated with services.
var serviceResources = []api.ResourceName{
	serviceObjectCountName,
	api.ResourceServices,
	api.ResourceServicesNodePorts,
	api.ResourceServicesLoadBalancers,
}

// NewServiceEvaluator returns an evaluator that can evaluate services.
func NewServiceEvaluator(f quota.ListerForResourceFunc) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, v1.SchemeGroupVersion.WithResource("services"))
	serviceEvaluator := &serviceEvaluator{listFuncByNamespace: listFuncByNamespace}
	return serviceEvaluator
}

// serviceEvaluator knows how to measure usage for services.
type serviceEvaluator struct {
	// knows how to list items by namespace
	listFuncByNamespace generic.ListFuncByNamespace
}

// Constraints verifies that all required resources are present on the item
func (p *serviceEvaluator) Constraints(required []api.ResourceName, item runtime.Object) error {
	// this is a no-op for services
	return nil
}

// GroupResource that this evaluator tracks
func (p *serviceEvaluator) GroupResource() schema.GroupResource {
	return v1.SchemeGroupVersion.WithResource("services").GroupResource()
}

// Handles returns true of the evaluator should handle the specified operation.
func (p *serviceEvaluator) Handles(a admission.Attributes) bool {
	operation := a.GetOperation()
	// We handle create and update because a service type can change.
	return admission.Create == operation || admission.Update == operation
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (p *serviceEvaluator) Matches(resourceQuota *api.ResourceQuota, item runtime.Object) (bool, error) {
	return generic.Matches(resourceQuota, item, p.MatchingResources, generic.MatchesNoScopeFunc)
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (p *serviceEvaluator) MatchingResources(input []api.ResourceName) []api.ResourceName {
	return quota.Intersection(input, serviceResources)
}

// convert the input object to an internal service object or error.
func toInternalServiceOrError(obj runtime.Object) (*api.Service, error) {
	svc := &api.Service{}
	switch t := obj.(type) {
	case *v1.Service:
		if err := k8s_api_v1.Convert_v1_Service_To_api_Service(t, svc, nil); err != nil {
			return nil, err
		}
	case *api.Service:
		svc = t
	default:
		return nil, fmt.Errorf("expect *api.Service or *v1.Service, got %v", t)
	}
	return svc, nil
}

// Usage knows how to measure usage associated with services
func (p *serviceEvaluator) Usage(item runtime.Object) (api.ResourceList, error) {
	result := api.ResourceList{}
	svc, err := toInternalServiceOrError(item)
	if err != nil {
		return result, err
	}
	ports := len(svc.Spec.Ports)
	// default service usage
	result[serviceObjectCountName] = *(resource.NewQuantity(1, resource.DecimalSI))
	result[api.ResourceServices] = *(resource.NewQuantity(1, resource.DecimalSI))
	result[api.ResourceServicesLoadBalancers] = resource.Quantity{Format: resource.DecimalSI}
	result[api.ResourceServicesNodePorts] = resource.Quantity{Format: resource.DecimalSI}
	switch svc.Spec.Type {
	case api.ServiceTypeNodePort:
		// node port services need to count node ports
		value := resource.NewQuantity(int64(ports), resource.DecimalSI)
		result[api.ResourceServicesNodePorts] = *value
	case api.ServiceTypeLoadBalancer:
		// load balancer services need to count node ports and load balancers
		value := resource.NewQuantity(int64(ports), resource.DecimalSI)
		result[api.ResourceServicesNodePorts] = *value
		result[api.ResourceServicesLoadBalancers] = *(resource.NewQuantity(1, resource.DecimalSI))
	}
	return result, nil
}

// UsageStats calculates aggregate usage for the object.
func (p *serviceEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	return generic.CalculateUsageStats(options, p.listFuncByNamespace, generic.MatchesNoScopeFunc, p.Usage)
}

var _ quota.Evaluator = &serviceEvaluator{}

// QuotaServiceType returns true if the service type is eligible to track against a quota
func QuotaServiceType(service *v1.Service) bool {
	switch service.Spec.Type {
	case v1.ServiceTypeNodePort, v1.ServiceTypeLoadBalancer:
		return true
	}
	return false
}

//GetQuotaServiceType returns ServiceType if the service type is eligible to track against a quota, nor return ""
func GetQuotaServiceType(service *v1.Service) v1.ServiceType {
	switch service.Spec.Type {
	case v1.ServiceTypeNodePort:
		return v1.ServiceTypeNodePort
	case v1.ServiceTypeLoadBalancer:
		return v1.ServiceTypeLoadBalancer
	}
	return v1.ServiceType("")
}
