/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// ContainerRefManager manages the references for the containers.
// The references are used for reporting events such as creation,
// failure, etc. This manager is thread-safe, no locks are necessary
// for the caller.
type ContainerRefManager struct {
	sync.RWMutex
	// TODO(yifan): To use strong type.
	containerIDToRef map[string]*api.ObjectReference
}

// newContainerRefManager creates and returns a container reference manager
// with empty contents.
func newContainerRefManager() *ContainerRefManager {
	c := ContainerRefManager{}
	c.containerIDToRef = make(map[string]*api.ObjectReference)
	return &c
}

// SetRef stores a reference to a pod's container, associating it with the given container id.
func (c *ContainerRefManager) SetRef(id string, ref *api.ObjectReference) {
	c.Lock()
	defer c.Unlock()
	c.containerIDToRef[id] = ref
}

// ClearRef forgets the given container id and its associated container reference.
// TODO(yifan): This is currently never called. Consider to remove this function,
// or figure out when to clear the references.
func (c *ContainerRefManager) ClearRef(id string) {
	c.Lock()
	defer c.Unlock()
	delete(c.containerIDToRef, id)
}

// GetRef returns the container reference of the given id, or (nil, false) if none is stored.
func (c *ContainerRefManager) GetRef(id string) (ref *api.ObjectReference, ok bool) {
	c.RLock()
	defer c.RUnlock()
	ref, ok = c.containerIDToRef[id]
	return ref, ok
}

// GenerateContainerRef returns an *api.ObjectReference which references the given container within the
// given pod. Returns an error if the reference can't be constructed or the container doesn't
// actually belong to the pod.
// TODO: Pods that came to us by static config or over HTTP have no selfLink set, which makes
// this fail and log an error. Figure out how we want to identify these pods to the rest of the
// system.
// TODO(yifan): Revisit this function later, for current case, it does not need to use ContainerRefManager
// as a receiver, and does not need to be exported.
func (c *ContainerRefManager) GenerateContainerRef(pod *api.Pod, container *api.Container) (*api.ObjectReference, error) {
	fieldPath, err := fieldPath(pod, container)
	if err != nil {
		// TODO: figure out intelligent way to refer to containers that we implicitly
		// start (like the pod infra container). This is not a good way, ugh.
		fieldPath = "implicitly required container " + container.Name
	}
	ref, err := api.GetPartialReference(pod, fieldPath)
	if err != nil {
		return nil, err
	}
	return ref, nil
}
