/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package algorithm

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

// MinionLister interface represents anything that can list minions for a scheduler.
type MinionLister interface {
	List() (list api.NodeList, err error)
}

// FakeMinionLister implements MinionLister on a []string for test purposes.
type FakeMinionLister api.NodeList

// List returns minions as a []string.
func (f FakeMinionLister) List() (api.NodeList, error) {
	return api.NodeList(f), nil
}

// PodLister interface represents anything that can list pods for a scheduler.
type PodLister interface {
	// TODO: make this exactly the same as client's Pods(ns).List() method, by returning a api.PodList
	List(labels.Selector) ([]*api.Pod, error)
}

// FakePodLister implements PodLister on an []api.Pods for test purposes.
type FakePodLister []*api.Pod

// List returns []*api.Pod matching a query.
func (f FakePodLister) List(s labels.Selector) (selected []*api.Pod, err error) {
	for _, pod := range f {
		if s.Matches(labels.Set(pod.Labels)) {
			selected = append(selected, pod)
		}
	}
	return selected, nil
}

// ServiceLister interface represents anything that can produce a list of services; the list is consumed by a scheduler.
type ServiceLister interface {
	// Lists all the services
	List() (api.ServiceList, error)
	// Gets the services for the given pod
	GetPodServices(*api.Pod) ([]api.Service, error)
}

// FakeServiceLister implements ServiceLister on []api.Service for test purposes.
type FakeServiceLister []api.Service

// List returns api.ServiceList, the list of all services.
func (f FakeServiceLister) List() (api.ServiceList, error) {
	return api.ServiceList{Items: f}, nil
}

// GetPodServices gets the services that have the selector that match the labels on the given pod
func (f FakeServiceLister) GetPodServices(pod *api.Pod) (services []api.Service, err error) {
	var selector labels.Selector

	for _, service := range f {
		// consider only services that are in the same namespace as the pod
		if service.Namespace != pod.Namespace {
			continue
		}
		selector = labels.Set(service.Spec.Selector).AsSelector()
		if selector.Matches(labels.Set(pod.Labels)) {
			services = append(services, service)
		}
	}
	if len(services) == 0 {
		err = fmt.Errorf("Could not find service for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return
}

// ControllerLister interface represents anything that can produce a list of ReplicationController; the list is consumed by a scheduler.
type ControllerLister interface {
	// Lists all the replication controllers
	List() ([]api.ReplicationController, error)
	// Gets the services for the given pod
	GetPodControllers(*api.Pod) ([]api.ReplicationController, error)
}

// FakeControllerLister implements ControllerLister on []api.ReplicationController for test purposes.
type FakeControllerLister []api.ReplicationController

// List returns []api.ReplicationController, the list of all ReplicationControllers.
func (f FakeControllerLister) List() ([]api.ReplicationController, error) {
	return f, nil
}

// GetPodControllers gets the ReplicationControllers that have the selector that match the labels on the given pod
func (f FakeControllerLister) GetPodControllers(pod *api.Pod) (controllers []api.ReplicationController, err error) {
	var selector labels.Selector

	for _, controller := range f {
		if controller.Namespace != pod.Namespace {
			continue
		}
		selector = labels.Set(controller.Spec.Selector).AsSelector()
		if selector.Matches(labels.Set(pod.Labels)) {
			controllers = append(controllers, controller)
		}
	}
	if len(controllers) == 0 {
		err = fmt.Errorf("Could not find Replication Controller for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return
}
