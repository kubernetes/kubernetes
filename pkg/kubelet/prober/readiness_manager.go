/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package prober

import (
	"sync"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// readinessManager maintains the readiness information(probe results) of
// containers over time to allow for implementation of health thresholds.
// This manager is thread-safe, no locks are necessary for the caller.
type readinessManager struct {
	// guards states
	sync.RWMutex
	states map[kubecontainer.ContainerID]bool
}

// newReadinessManager creates ane returns a readiness manager with empty
// contents.
func newReadinessManager() *readinessManager {
	return &readinessManager{states: make(map[kubecontainer.ContainerID]bool)}
}

// getReadiness returns the readiness value for the container with the given ID.
// If the readiness value is found, returns it.
// If the readiness is not found, returns false.
func (r *readinessManager) getReadiness(id kubecontainer.ContainerID) (ready bool, found bool) {
	r.RLock()
	defer r.RUnlock()
	state, found := r.states[id]
	return state, found
}

// setReadiness sets the readiness value for the container with the given ID.
func (r *readinessManager) setReadiness(id kubecontainer.ContainerID, value bool) {
	r.Lock()
	defer r.Unlock()
	r.states[id] = value
}

// removeReadiness clears the readiness value for the container with the given ID.
func (r *readinessManager) removeReadiness(id kubecontainer.ContainerID) {
	r.Lock()
	defer r.Unlock()
	delete(r.states, id)
}
