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

package internalversion

import (
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api"
)

// ServiceListerExpansion allows custom methods to be added to
// ServiceLister.
type ServiceListerExpansion interface {
	GetPodServices(pod *api.Pod) ([]*api.Service, error)
}

// ServiceNamespaceListerExpansion allows custom methods to be added to
// ServiceNamespaceLister.
type ServiceNamespaceListerExpansion interface{}

// TODO: Move this back to scheduler as a helper function that takes a Store,
// rather than a method of ServiceLister.
func (s *serviceLister) GetPodServices(pod *api.Pod) ([]*api.Service, error) {
	allServices, err := s.Services(pod.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var services []*api.Service
	for i := range allServices {
		service := allServices[i]
		if service.Spec.Selector == nil {
			// services with nil selectors match nothing, not everything.
			continue
		}
		selector := labels.Set(service.Spec.Selector).AsSelectorPreValidated()
		if selector.Matches(labels.Set(pod.Labels)) {
			services = append(services, service)
		}
	}

	return services, nil
}
