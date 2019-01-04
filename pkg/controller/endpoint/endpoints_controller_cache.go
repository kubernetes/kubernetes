/*
Copyright 2019 The Kubernetes Authors.

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

package endpoint

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller"
)

// EndpointsControllerCache is a util that manages a local copy of the state of pods and services
// for purpose of endpoints object syncing.
//
// The purpose of this util is to create a single source of truth that is thread-safe and avoids
// potential race-conditions in the endpoints-controller. In addition it allows to easily determine
// which object (pod/service) changes resulted in the endpoints object change, which is crucial in
// implementing the network programming latency metric.
//
// The cache should be initialized via pod and service listers and kept up-to-date via the
// (Add|Update|Delete)(Pod|Service) methods called in the EndpointsController's event handlers.
// The Initialize method together with the event handlers method are guarded by the same mutex which
// guarantees that no information is ever missed and all events are processed in the serial order.
//
// The cache doesn't store full objects, it stores only references to minimize the memory
// consumption.
type endpointsControllerCache struct {
	// pods is a map of pods indexed by namespace and name, i.e. namespace -> name -> pod.
	pods map[string]map[string]*v1.Pod
	// pods is a map of services indexed by namespace and name, i.e. namespace -> name -> service.
	services map[string]map[string]*v1.Service

	// Mutex guarding this util.
	mutex sync.Mutex
}

// Creates new instance of the endpointsControllerCache.
func newEndpointsControllerCache() *endpointsControllerCache {
	return &endpointsControllerCache{
		pods:     make(map[string]map[string]*v1.Pod),
		services: make(map[string]map[string]*v1.Service),
	}
}

// Initialize initializes the cache with pods and services obtained from the provided listers.
// Pods and services are listed behind the mutex, which temporarily disables event listener
// callbacks (e.g. AddPod, UpdateService, etc.) - making them (this method and informer events)
// mutually exclusive. Given that and the fact that listers always see everything that has been
// already delivered via informer events, this implementation guarantees that no information will be
// ever lost.
// This method should be called exactly once on the startup, before the first call of the
// GetServiceAndPods method, after the lister have synced.
// TODO(mm4tt): Support initialization natively in the Informer framework and get rid of this method.
//              This will require a new signal (and callback in the event handler) informing
//              that the state has been synced and fully delivered to the event handlers via the
//              OnAdd callbacks.
func (e *endpointsControllerCache) Initialize(serviceLister corelisters.ServiceLister, podLister corelisters.PodLister) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	services, err := serviceLister.List(labels.Everything())
	if err != nil {
		// Error shouldn't happened as we're getting objects from a local cache, so fail if it does.
		return fmt.Errorf("unable to list services due to %v", err)
	}
	// Please note that we're listing pods from a different lister, so it's possible that we will
	// initialize the cache with a pod state newer or older than the service state. This inaccuracy
	// doesn't matter in the long run though. If the service / pod state returned by lister is
	// behind, and doesn't reflect what is stored in etcd, then we're guaranteed to get the
	// service / pod state updates via the (Add|Update|Delete)(Pod|Service) methods which will
	// eventually make the state consistent. In other words we're always guaranteed to reach the
	// state when there are no Informer events to be processed (for the moment) and the state of
	// this cache fully reflects the state stored in etcd.
	pods, err := podLister.List(labels.Everything())
	if err != nil {
		// Error shouldn't happened as we're getting objects from a local cache, so fail if it does.
		return fmt.Errorf("unable to list pods due to %v", err)
	}

	for _, service := range services {
		e.getServices(service.Namespace)[service.Name] = service
	}
	for _, pod := range pods {
		e.getPods(pod.Namespace)[pod.Name] = pod
	}
	return nil
}

// AddPod adds new pod to the cache and returns a list of namespaced names of service / endpoints
// objects that should be synced after this operation has been completed.
func (e *endpointsControllerCache) AddPod(pod *v1.Pod) []string {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	e.getPods(pod.Namespace)[pod.Name] = pod
	return e.getPodServiceMemberships(pod)
}

// UpdatePod updates the pod in the cache and returns a list of namespaced names of
// service / endpoints objects that should be synced after this operation has been completed.
func (e *endpointsControllerCache) UpdatePod(before, after *v1.Pod) []string {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	e.getPods(after.Namespace)[after.Name] = after

	services := sets.NewString()
	services.Insert(e.getPodServiceMemberships(before)...)
	services.Insert(e.getPodServiceMemberships(after)...)
	return services.List()
}

// DeletePod deletes the pod from the cache and returns a list of namespaced names of
// service / endpoints objects that should be synced after this operation has been completed.
func (e *endpointsControllerCache) DeletePod(pod *v1.Pod) []string {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	delete(e.getPods(pod.Namespace), pod.Name)
	return e.getPodServiceMemberships(pod)
}

// AddService adds the service to the cache and returns the namespaced name of the service
// being added.
func (e *endpointsControllerCache) AddService(service *v1.Service) (string, error) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	e.getServices(service.Namespace)[service.Name] = service
	return getNamespacedName(service)
}

// UpdateService updates the service in the cache and returns the namespaced name of the service
// being updated.
func (e *endpointsControllerCache) UpdateService(before, after *v1.Service) (string, error) {
	return e.AddService(after)
}

// DeleteService deletes the service from the cache and returns the namespaced name of the service
// being deleted.
func (e *endpointsControllerCache) DeleteService(service *v1.Service) (string, error) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	delete(e.getServices(service.Namespace), service.Name)
	return getNamespacedName(service)
}

// GetServiceAndPods returns service identified by the provided key (namespaced name) and all the
// pods belonging to the service from the cache. This method will be called in the
// EndpointsController.syncService method.
func (e *endpointsControllerCache) GetServiceAndPods(namespace, name string) (service *v1.Service, pods []*v1.Pod) {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	service, ok := e.getServices(namespace)[name]
	if ok && service.Spec.Selector != nil {
		selector := labels.Set(service.Spec.Selector).AsSelectorPreValidated()
		for _, pod := range e.getPods(namespace) {
			if selector.Matches(labels.Set(pod.Labels)) {
				pods = append(pods, pod)
			}
		}
	}
	return service, pods
}

// getPodServiceMemberships returns slice of strings, where each string represents a namespaced
// service name that the provided pod belongs to.
func (e *endpointsControllerCache) getPodServiceMemberships(pod *v1.Pod) (ret []string) {
	ret = []string{}
	for _, service := range e.getServices(pod.Namespace) {
		if service.Spec.Selector == nil {
			// services with nil selectors match nothing, not everything.
			continue
		}
		selector := labels.Set(service.Spec.Selector).AsSelectorPreValidated()
		if selector.Matches(labels.Set(pod.Labels)) {
			key, err := getNamespacedName(service)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to get key for service %#v", service))
				continue
			}
			ret = append(ret, key)
		}
	}
	return ret
}

// getPods returns pod map (name->pod) for the given namespace. This method takes care of
// initialization if the pod map doesn't exist yet for the given namespace, in other words it will
// never return nil.
func (e *endpointsControllerCache) getPods(namespace string) map[string]*v1.Pod {
	if _, ok := e.pods[namespace]; !ok {
		e.pods[namespace] = make(map[string]*v1.Pod)
	}
	return e.pods[namespace]
}

// getServices returns service map (name->service) for the given namespace. This method takes care
// of the map initialization if the service map doesn't exist yet for the given namespace, in other
// words it will never return nil.
func (e *endpointsControllerCache) getServices(namespace string) map[string]*v1.Service {
	if _, ok := e.services[namespace]; !ok {
		e.services[namespace] = make(map[string]*v1.Service)
	}
	return e.services[namespace]
}

// ----- Util Functions -----

// getNamespacedName returns namespaced name of the provided service.
func getNamespacedName(service *v1.Service) (string, error) {
	return controller.KeyFunc(service)
}
