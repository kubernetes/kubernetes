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

package queue

import (
	"sync"
	"time"

	ktypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// PodBackoffMap is a structure that stores backoff related information for pods
type PodBackoffMap struct {
	// lock for performing actions on this PodBackoffMap
	lock  sync.RWMutex
	clock util.Clock
	// initial backoff duration
	initialDuration time.Duration
	// maximal backoff duration
	maxDuration time.Duration
	// map for pod -> number of attempts for this pod
	podAttempts map[ktypes.NamespacedName]int
	// map for pod -> lastUpdateTime pod of this pod
	podLastUpdateTime map[ktypes.NamespacedName]time.Time
}

// NewPodBackoffMap creates a PodBackoffMap with initial duration and max duration.
func NewPodBackoffMap(initialDuration, maxDuration time.Duration, clock util.Clock) *PodBackoffMap {
	return &PodBackoffMap{
		clock:             clock,
		initialDuration:   initialDuration,
		maxDuration:       maxDuration,
		podAttempts:       make(map[ktypes.NamespacedName]int),
		podLastUpdateTime: make(map[ktypes.NamespacedName]time.Time),
	}
}

// GetBackoffTime returns the time that nsPod completes backoff
func (pbm *PodBackoffMap) GetBackoffTime(nsPod ktypes.NamespacedName) (time.Time, bool) {
	pbm.lock.RLock()
	defer pbm.lock.RUnlock()
	if _, found := pbm.podAttempts[nsPod]; found == false {
		return time.Time{}, false
	}
	lastUpdateTime := pbm.podLastUpdateTime[nsPod]
	backoffDuration := pbm.calculateBackoffDuration(nsPod)
	backoffTime := lastUpdateTime.Add(backoffDuration)
	return backoffTime, true
}

// calculateBackoffDuration is a helper function for calculating the backoffDuration
// based on the number of attempts the pod has made.
func (pbm *PodBackoffMap) calculateBackoffDuration(nsPod ktypes.NamespacedName) time.Duration {
	backoffDuration := pbm.initialDuration
	if _, found := pbm.podAttempts[nsPod]; found {
		for i := 1; i < pbm.podAttempts[nsPod]; i++ {
			backoffDuration = backoffDuration * 2
			if backoffDuration > pbm.maxDuration {
				return pbm.maxDuration
			}
		}
	}
	return backoffDuration
}

// clearPodBackoff removes all tracking information for nsPod.
// Lock is supposed to be acquired by caller.
func (pbm *PodBackoffMap) clearPodBackoff(nsPod ktypes.NamespacedName) {
	delete(pbm.podAttempts, nsPod)
	delete(pbm.podLastUpdateTime, nsPod)
}

// ClearPodBackoff is the thread safe version of clearPodBackoff
func (pbm *PodBackoffMap) ClearPodBackoff(nsPod ktypes.NamespacedName) {
	pbm.lock.Lock()
	pbm.clearPodBackoff(nsPod)
	pbm.lock.Unlock()
}

// CleanupPodsCompletesBackingoff execute garbage collection on the pod backoff,
// i.e, it will remove a pod from the PodBackoffMap if
// lastUpdateTime + maxDuration >> timestamp
// We should wait longer than the maxDuration so that the pod gets a chance to
// (1) move to the active queue and (2) get an schedule attempt.
func (pbm *PodBackoffMap) CleanupPodsCompletesBackingoff() {
	pbm.lock.Lock()
	defer pbm.lock.Unlock()
	for pod, value := range pbm.podLastUpdateTime {
		// Here we assume that maxDuration should be enough for a pod to move up the
		// active queue and get an schedule attempt.
		if value.Add(2 * pbm.maxDuration).Before(pbm.clock.Now()) {
			pbm.clearPodBackoff(pod)
		}
	}
}

// BackoffPod updates the lastUpdateTime for an nsPod,
// and increases its numberOfAttempts by 1
func (pbm *PodBackoffMap) BackoffPod(nsPod ktypes.NamespacedName) {
	pbm.lock.Lock()
	pbm.podLastUpdateTime[nsPod] = pbm.clock.Now()
	pbm.podAttempts[nsPod]++
	pbm.lock.Unlock()
}
