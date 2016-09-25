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
	"strings"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// NewServiceEvaluator returns an evaluator that can evaluate service quotas
func NewServiceEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	allResources := []api.ResourceName{
		api.ResourceServices,
		api.ResourceServicesNodePorts,
		api.ResourceServicesLoadBalancers,
	}
	return &generic.GenericEvaluator{
		Name:              "Evaluator.Service",
		InternalGroupKind: api.Kind("Service"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: allResources,
			admission.Update: allResources,
		},
		MatchedResourceNames: allResources,
		MatchesScopeFunc:     generic.MatchesNoScopeFunc,
		ConstraintsFunc:      ServiceConstraintsFunc,
		UsageFunc:            ServiceUsageFunc,
		ListFuncByNamespace: func(namespace string, options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Core().Services(namespace).List(options)
		},
	}
}

// ServiceUsageFunc knows how to measure usage associated with services
func ServiceUsageFunc(object runtime.Object) api.ResourceList {
	result := api.ResourceList{}
	if service, ok := object.(*api.Service); ok {
		// default service usage
		result[api.ResourceServices] = resource.MustParse("1")
		result[api.ResourceServicesLoadBalancers] = resource.MustParse("0")
		result[api.ResourceServicesNodePorts] = resource.MustParse("0")
		switch service.Spec.Type {
		case api.ServiceTypeNodePort:
			// node port services need to count node ports
			value := resource.NewQuantity(int64(len(service.Spec.Ports)), resource.DecimalSI)
			result[api.ResourceServicesNodePorts] = *value
		case api.ServiceTypeLoadBalancer:
			// load balancer services need to count load balancers
			result[api.ResourceServicesLoadBalancers] = resource.MustParse("1")
		}
	}
	return result
}

// QuotaServiceType returns true if the service type is eligible to track against a quota
func QuotaServiceType(service *api.Service) bool {
	switch service.Spec.Type {
	case api.ServiceTypeNodePort, api.ServiceTypeLoadBalancer:
		return true
	}
	return false
}

//GetQuotaServiceType returns ServiceType if the service type is eligible to track against a quota, nor return ""
func GetQuotaServiceType(service *api.Service) api.ServiceType {
	switch service.Spec.Type {
	case api.ServiceTypeNodePort:
		return api.ServiceTypeNodePort
	case api.ServiceTypeLoadBalancer:
		return api.ServiceTypeLoadBalancer
	}
	return api.ServiceType("")
}

// ServiceConstraintsFunc verifies that all required resources are captured in service usage.
func ServiceConstraintsFunc(required []api.ResourceName, object runtime.Object) error {
	service, ok := object.(*api.Service)
	if !ok {
		return fmt.Errorf("unexpected input object %v", object)
	}

	requiredSet := quota.ToSet(required)
	missingSet := sets.NewString()
	serviceUsage := ServiceUsageFunc(service)
	serviceSet := quota.ToSet(quota.ResourceNames(serviceUsage))
	if diff := requiredSet.Difference(serviceSet); len(diff) > 0 {
		missingSet.Insert(diff.List()...)
	}

	if len(missingSet) == 0 {
		return nil
	}
	return fmt.Errorf("must specify %s", strings.Join(missingSet.List(), ","))
}
