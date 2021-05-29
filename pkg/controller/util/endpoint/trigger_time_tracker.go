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
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// TriggerTimeTracker is used to compute an EndpointsLastChangeTriggerTime
// annotation. See the documentation for that annotation for more details.
//
// Please note that this util may compute a wrong EndpointsLastChangeTriggerTime
// if the same object changes multiple times between two consecutive syncs.
// We're aware of this limitation but we decided to accept it, as fixing it
// would require a major rewrite of the endpoint(Slice) controller and
// Informer framework. Such situations, i.e. frequent updates of the same object
// in a single sync period, should be relatively rare and therefore this util
// should provide a good approximation of the EndpointsLastChangeTriggerTime.
type TriggerTimeTracker struct {
	// ServiceStates is a map, indexed by Service object key, storing the last
	// known Service object state observed during the most recent call of the
	// ComputeEndpointLastChangeTriggerTime function.
	ServiceStates map[ServiceKey]ServiceState

	// mutex guarding the serviceStates map.
	mutex sync.Mutex
}

// NewTriggerTimeTracker creates a new instance of the TriggerTimeTracker.
func NewTriggerTimeTracker() *TriggerTimeTracker {
	return &TriggerTimeTracker{
		ServiceStates: make(map[ServiceKey]ServiceState),
	}
}

// ServiceKey is a key uniquely identifying a Service.
type ServiceKey struct {
	// namespace, name composing a namespaced name - an unique identifier of every Service.
	Namespace, Name string
}

// ServiceState represents a state of an Service object that is known to this util.
type ServiceState struct {
	// lastServiceTriggerTime is a service trigger time observed most recently.
	lastServiceTriggerTime time.Time
	// lastPodTriggerTimes is a map (Pod name -> time) storing the pod trigger
	// times that were observed during the most recent call of the
	// ComputeEndpointLastChangeTriggerTime function.
	lastPodTriggerTimes map[string]time.Time
}

// ComputeEndpointLastChangeTriggerTime updates the state of the Service/Endpoint
// object being synced and returns the time that should be exported as the
// EndpointsLastChangeTriggerTime annotation.
//
// If the method returns a 'zero' time the EndpointsLastChangeTriggerTime
// annotation shouldn't be exported.
//
// Please note that this function may compute a wrong value if the same object
// (pod/service) changes multiple times between two consecutive syncs.
//
// Important: This method is go-routing safe but only when called for different
// keys. The method shouldn't be called concurrently for the same key! This
// contract is fulfilled in the current implementation of the endpoint(slice)
// controller.
func (t *TriggerTimeTracker) ComputeEndpointLastChangeTriggerTime(
	namespace string, service *v1.Service, pods []*v1.Pod) time.Time {

	key := ServiceKey{Namespace: namespace, Name: service.Name}
	// As there won't be any concurrent calls for the same key, we need to guard
	// access only to the serviceStates map.
	t.mutex.Lock()
	state, wasKnown := t.ServiceStates[key]
	t.mutex.Unlock()

	// Update the state before returning.
	defer func() {
		t.mutex.Lock()
		t.ServiceStates[key] = state
		t.mutex.Unlock()
	}()

	// minChangedTriggerTime is the min trigger time of all trigger times that
	// have changed since the last sync.
	var minChangedTriggerTime time.Time
	podTriggerTimes := make(map[string]time.Time)
	for _, pod := range pods {
		if podTriggerTime := getPodTriggerTime(pod); !podTriggerTime.IsZero() {
			podTriggerTimes[pod.Name] = podTriggerTime
			if podTriggerTime.After(state.lastPodTriggerTimes[pod.Name]) {
				// Pod trigger time has changed since the last sync, update minChangedTriggerTime.
				minChangedTriggerTime = min(minChangedTriggerTime, podTriggerTime)
			}
		}
	}
	serviceTriggerTime := getServiceTriggerTime(service)
	if serviceTriggerTime.After(state.lastServiceTriggerTime) {
		// Service trigger time has changed since the last sync, update minChangedTriggerTime.
		minChangedTriggerTime = min(minChangedTriggerTime, serviceTriggerTime)
	}

	state.lastPodTriggerTimes = podTriggerTimes
	state.lastServiceTriggerTime = serviceTriggerTime

	if !wasKnown {
		// New Service, use Service creationTimestamp.
		return service.CreationTimestamp.Time
	}

	// Regular update of endpoint objects, return min of changed trigger times.
	return minChangedTriggerTime
}

// DeleteService deletes service state stored in this util.
func (t *TriggerTimeTracker) DeleteService(namespace, name string) {
	key := ServiceKey{Namespace: namespace, Name: name}
	t.mutex.Lock()
	defer t.mutex.Unlock()
	delete(t.ServiceStates, key)
}

// getPodTriggerTime returns the time of the pod change (trigger) that resulted
// or will result in the endpoint object change.
func getPodTriggerTime(pod *v1.Pod) (triggerTime time.Time) {
	if readyCondition := podutil.GetPodReadyCondition(pod.Status); readyCondition != nil {
		triggerTime = readyCondition.LastTransitionTime.Time
	}
	return triggerTime
}

// getServiceTriggerTime returns the time of the service change (trigger) that
// resulted or will result in the endpoint change.
func getServiceTriggerTime(service *v1.Service) (triggerTime time.Time) {
	return service.CreationTimestamp.Time
}

// min returns minimum of the currentMin and newValue or newValue if the currentMin is not set.
func min(currentMin, newValue time.Time) time.Time {
	if currentMin.IsZero() || newValue.Before(currentMin) {
		return newValue
	}
	return currentMin
}
