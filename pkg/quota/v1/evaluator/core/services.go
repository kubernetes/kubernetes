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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
)

// the name used for object count quota
var serviceObjectCountName = generic.ObjectCountQuotaResourceNameFor(corev1.SchemeGroupVersion.WithResource("services").GroupResource())

// serviceResources are the set of resources managed by quota associated with services.
var serviceResources = []corev1.ResourceName{
	serviceObjectCountName,
	corev1.ResourceServices,
	corev1.ResourceServicesNodePorts,
	corev1.ResourceServicesLoadBalancers,
}

// NewServiceEvaluator returns an evaluator that can evaluate services.
func NewServiceEvaluator(f quota.ListerForResourceFunc) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, corev1.SchemeGroupVersion.WithResource("services"))
	serviceEvaluator := &serviceEvaluator{listFuncByNamespace: listFuncByNamespace}
	return serviceEvaluator
}

// serviceEvaluator knows how to measure usage for services.
type serviceEvaluator struct {
	// knows how to list items by namespace
	listFuncByNamespace generic.ListFuncByNamespace
}

// Constraints verifies that all required resources are present on the item
func (p *serviceEvaluator) Constraints(required []corev1.ResourceName, item runtime.Object) error {
	// this is a no-op for services
	return nil
}

// GroupResource that this evaluator tracks
func (p *serviceEvaluator) GroupResource() schema.GroupResource {
	return corev1.SchemeGroupVersion.WithResource("services").GroupResource()
}

// Handles returns true of the evaluator should handle the specified operation.
func (p *serviceEvaluator) Handles(a admission.Attributes) bool {
	operation := a.GetOperation()
	// We handle create and update because a service type can change.
	return admission.Create == operation || admission.Update == operation
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (p *serviceEvaluator) Matches(resourceQuota *corev1.ResourceQuota, item runtime.Object) (bool, error) {
	return generic.Matches(resourceQuota, item, p.MatchingResources, generic.MatchesNoScopeFunc)
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (p *serviceEvaluator) MatchingResources(input []corev1.ResourceName) []corev1.ResourceName {
	return quota.Intersection(input, serviceResources)
}

// MatchingScopes takes the input specified list of scopes and input object. Returns the set of scopes resource matches.
func (p *serviceEvaluator) MatchingScopes(item runtime.Object, scopes []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error) {
	return []corev1.ScopedResourceSelectorRequirement{}, nil
}

// UncoveredQuotaScopes takes the input matched scopes which are limited by configuration and the matched quota scopes.
// It returns the scopes which are in limited scopes but dont have a corresponding covering quota scope
func (p *serviceEvaluator) UncoveredQuotaScopes(limitedScopes []corev1.ScopedResourceSelectorRequirement, matchedQuotaScopes []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error) {
	return []corev1.ScopedResourceSelectorRequirement{}, nil
}

// convert the input object to an internal service object or error.
func toExternalServiceOrError(obj runtime.Object) (*corev1.Service, error) {
	svc := &corev1.Service{}
	switch t := obj.(type) {
	case *corev1.Service:
		svc = t
	case *api.Service:
		if err := k8s_api_v1.Convert_core_Service_To_v1_Service(t, svc, nil); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("expect *api.Service or *v1.Service, got %v", t)
	}
	return svc, nil
}

// Usage knows how to measure usage associated with services
func (p *serviceEvaluator) Usage(item runtime.Object) (corev1.ResourceList, error) {
	result := corev1.ResourceList{}
	svc, err := toExternalServiceOrError(item)
	if err != nil {
		return result, err
	}
	ports := len(svc.Spec.Ports)
	// default service usage
	result[serviceObjectCountName] = *(resource.NewQuantity(1, resource.DecimalSI))
	result[corev1.ResourceServices] = *(resource.NewQuantity(1, resource.DecimalSI))
	result[corev1.ResourceServicesLoadBalancers] = resource.Quantity{Format: resource.DecimalSI}
	result[corev1.ResourceServicesNodePorts] = resource.Quantity{Format: resource.DecimalSI}
	switch svc.Spec.Type {
	case corev1.ServiceTypeNodePort:
		// node port services need to count node ports
		value := resource.NewQuantity(int64(ports), resource.DecimalSI)
		result[corev1.ResourceServicesNodePorts] = *value
	case corev1.ServiceTypeLoadBalancer:
		// load balancer services need to count node ports and load balancers
		value := resource.NewQuantity(int64(ports), resource.DecimalSI)
		result[corev1.ResourceServicesNodePorts] = *value
		result[corev1.ResourceServicesLoadBalancers] = *(resource.NewQuantity(1, resource.DecimalSI))
	}
	return result, nil
}

// UsageStats calculates aggregate usage for the object.
func (p *serviceEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	return generic.CalculateUsageStats(options, p.listFuncByNamespace, generic.MatchesNoScopeFunc, p.Usage)
}

var _ quota.Evaluator = &serviceEvaluator{}

//GetQuotaServiceType returns ServiceType if the service type is eligible to track against a quota, nor return ""
func GetQuotaServiceType(service *corev1.Service) corev1.ServiceType {
	switch service.Spec.Type {
	case corev1.ServiceTypeNodePort:
		return corev1.ServiceTypeNodePort
	case corev1.ServiceTypeLoadBalancer:
		return corev1.ServiceTypeLoadBalancer
	}
	return corev1.ServiceType("")
}
