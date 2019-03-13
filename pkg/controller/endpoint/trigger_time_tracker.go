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

	"k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// TriggerTimeTracker is a util used to compute the EndpointsLastChangeTriggerTime annotation which
// is exported in the endpoints controller's sync function.
// See the documentation of the EndpointsLastChangeTriggerTime annotation for more details.
//
// Please note that this util may compute a wrong EndpointsLastChangeTriggerTime if a same object
// changes multiple times between two consecutive syncs. We're aware of this limitation but we
// decided to accept it, as fixing it would require a major rewrite of the endpoints controller and
// Informer framework. Such situations, i.e. frequent updates of the same object in a single sync
// period, should be relatively rare and therefore this util should provide a good approximation of
// the EndpointsLastChangeTriggerTime.
// TODO(mm4tt): Implement a more robust mechanism that is not subject to the above limitations.
type TriggerTimeTracker struct {
	// endpointsStates is a map, indexed by Endpoints object key, storing the last known Endpoints
	// object state observed during the most recent call of the ComputeEndpointsLastChangeTriggerTime
	// function.
	endpointsStates map[endpointsKey]endpointsState

	// mutex guarding the endpointsStates map.
	mutex sync.Mutex
}

// NewTriggerTimeTracker creates a new instance of the TriggerTimeTracker.
func NewTriggerTimeTracker() *TriggerTimeTracker {
	return &TriggerTimeTracker{
		endpointsStates: make(map[endpointsKey]endpointsState),
	}
}

// endpointsKey is a key uniquely identifying an Endpoints object.
type endpointsKey struct {
	// namespace, name composing a namespaced name - an unique identifier of every Endpoints object.
	namespace, name string
}

// endpointsState represents a state of an Endpoints object that is known to this util.
type endpointsState struct {
	// lastServiceTriggerTime is a service trigger time observed most recently.
	lastServiceTriggerTime time.Time
	// lastPodTriggerTimes is a map (Pod name -> time) storing the pod trigger times that were
	// observed during the most recent call of the ComputeEndpointsLastChangeTriggerTime function.
	lastPodTriggerTimes map[string]time.Time
}

// ComputeEndpointsLastChangeTriggerTime updates the state of the Endpoints object being synced
// and returns the time that should be exported as the EndpointsLastChangeTriggerTime annotation.
//
// If the method returns a 'zero' time the EndpointsLastChangeTriggerTime annotation shouldn't be
// exported.
//
// Please note that this function may compute a wrong EndpointsLastChangeTriggerTime value if the
// same object (pod/service) changes multiple times between two consecutive syncs.
//
// Important: This method is go-routing safe but only when called for different keys. The method
// shouldn't be called concurrently for the same key! This contract is fulfilled in the current
// implementation of the endpoints controller.
func (t *TriggerTimeTracker) ComputeEndpointsLastChangeTriggerTime(
	namespace, name string, service *v1.Service, pods []*v1.Pod) time.Time {

	key := endpointsKey{namespace: namespace, name: name}
	// As there won't be any concurrent calls for the same key, we need to guard access only to the
	// endpointsStates map.
	t.mutex.Lock()
	state, wasKnown := t.endpointsStates[key]
	t.mutex.Unlock()

	// Update the state before returning.
	defer func() {
		t.mutex.Lock()
		t.endpointsStates[key] = state
		t.mutex.Unlock()
	}()

	// minChangedTriggerTime is the min trigger time of all trigger times that have changed since the
	// last sync.
	var minChangedTriggerTime time.Time
	// TODO(mm4tt): If memory allocation / GC performance impact of recreating map in every call
	// turns out to be too expensive, we should consider rewriting this to reuse the existing map.
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
		// New Endpoints object / new Service, use Service creationTimestamp.
		return service.CreationTimestamp.Time
	} else {
		// Regular update of the Endpoints object, return min of changed trigger times.
		return minChangedTriggerTime
	}
}

// DeleteEndpoints deletes endpoints state stored in this util.
func (t *TriggerTimeTracker) DeleteEndpoints(namespace, name string) {
	key := endpointsKey{namespace: namespace, name: name}
	t.mutex.Lock()
	defer t.mutex.Unlock()
	delete(t.endpointsStates, key)
}

// getPodTriggerTime returns the time of the pod change (trigger) that resulted or will result in
// the endpoints object change.
func getPodTriggerTime(pod *v1.Pod) (triggerTime time.Time) {
	if readyCondition := podutil.GetPodReadyCondition(pod.Status); readyCondition != nil {
		triggerTime = readyCondition.LastTransitionTime.Time
	}
	// TODO(mm4tt): Implement missing cases: deletionTime set, pod label change
	return triggerTime
}

// getServiceTriggerTime returns the time of the service change (trigger) that resulted or will
// result in the endpoints object change.
func getServiceTriggerTime(service *v1.Service) (triggerTime time.Time) {
	// TODO(mm4tt): Ideally we should look at service.LastUpdateTime, but such thing doesn't exist.
	return service.CreationTimestamp.Time
}

// min returns minimum of the currentMin and newValue or newValue if the currentMin is not set.
func min(currentMin, newValue time.Time) time.Time {
	if currentMin.IsZero() || newValue.Before(currentMin) {
		return newValue
	}
	return currentMin
}
