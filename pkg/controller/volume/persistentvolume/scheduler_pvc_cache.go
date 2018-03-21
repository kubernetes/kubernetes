/*
Copyright 2018 The Kubernetes Authors.

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

package persistentvolume

import (
	"k8s.io/api/core/v1"
)

// PVCMatchingCache store Pod's PVC temporary state of matching PVs if
// none of the PVs have node affinity. Then we can skip PV matching on
// subsequent nodes, and just return the result of the first attempt.
// Because PVs under a given storage class name may change. Pod's PVC matching
// states are only cached in one scheduler pass, should be invalidated when Pod
// is rescheduled again.
// Pod's PVC matching states are also removed when the Pod is deleted or
// updated to no longer be schedulable.
type PVCMatchingCache interface {
	// UpdateMatching
	UpdateMatching(pod *v1.Pod, pvc *v1.PersistentVolumeClaim, state *matching)

	// DeleteMatchings will remove all cached entries for the given pod.
	DeleteMatchings(pod *v1.Pod)

	// GetMatching will return the cached state for the given pod and pvc.
	GetMatching(pod *v1.Pod, pvc *v1.PersistentVolumeClaim) *matching

	// GetMatchings will return the cached states for the given pod.
	GetMatchings(pod *v1.Pod) matchings
}

type matching struct {
	// Indicates success or failure of previously matching PVs.
	err error

	// Proposed PV to bind to this claim.
	pv *v1.PersistentVolume
}

// Key = pvc name
// Value = pointer to matching
type matchings map[string]*matching

// Since the scheduelr is serialized, we don't need lock here.
type pvcMatchingCache struct {
	// Key = pod name
	// Value = matchings
	pvcMatchings map[string]matchings
}

func NewPVCMatchingCache() PVCMatchingCache {
	return &pvcMatchingCache{pvcMatchings: map[string]matchings{}}
}

func (c *pvcMatchingCache) UpdateMatching(pod *v1.Pod, pvc *v1.PersistentVolumeClaim, state *matching) {
	podName := getPodName(pod)
	states, ok := c.pvcMatchings[podName]
	if !ok {
		states = matchings{}
		c.pvcMatchings[podName] = states
	}
	states[getPVCName(pvc)] = state
}

func (c *pvcMatchingCache) DeleteMatchings(pod *v1.Pod) {
	podName := getPodName(pod)
	delete(c.pvcMatchings, podName)
}

func (c *pvcMatchingCache) GetMatching(pod *v1.Pod, pvc *v1.PersistentVolumeClaim) *matching {
	matchings := c.GetMatchings(pod)
	if matchings != nil {
		return matchings[getPVCName(pvc)]
	}
	return nil
}

func (c *pvcMatchingCache) GetMatchings(pod *v1.Pod) matchings {
	podName := getPodName(pod)
	matchings, ok := c.pvcMatchings[podName]
	if !ok {
		return nil
	}
	return matchings
}
