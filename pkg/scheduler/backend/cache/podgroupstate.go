/*
Copyright 2025 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"maps"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/utils/ptr"
)

// DefaultSchedulingTimeoutDuration defines how long the gang pods should wait at the
// Permit stage for a quorum before being rejected.
var DefaultSchedulingTimeoutDuration = 5 * time.Minute

// PodGroupKey uniquely identifies a specific instance of a PodGroup.
type PodGroupKey struct {
	namespace    string
	workloadName string
	podGroupName string
	replicaKey   string
}

func (pgk PodGroupKey) GetName() string {
	if pgk.replicaKey == "" {
		return fmt.Sprintf("%s-%s", pgk.workloadName, pgk.podGroupName)
	}
	return fmt.Sprintf("%s-%s-%s", pgk.workloadName, pgk.podGroupName, pgk.replicaKey)
}

func (pgk PodGroupKey) GetNamespace() string {
	return pgk.namespace
}

var _ klog.KMetadata = &PodGroupKey{}

func NewPodGroupKey(namespace string, workloadRef *v1.WorkloadReference) PodGroupKey {
	return PodGroupKey{
		namespace:    namespace,
		workloadName: workloadRef.Name,
		podGroupName: workloadRef.PodGroup,
		replicaKey:   workloadRef.PodGroupReplicaKey,
	}
}

// PodGroupState holds the runtime state of a pod group.
type PodGroupState struct {
	lock sync.RWMutex
	// generation gets bumped whenever a PodGroupState is changed.
	// used to detect PodGroupState changes and avoid unnecessary cloning.
	generation int64
	// allPods tracks all pods belonging to the group that are known to the scheduler.
	allPods map[types.UID]*v1.Pod
	// unscheduledPods tracks all pods that are unscheduled for this group,
	// i.e., are neither assumed nor scheduled.
	unscheduledPods sets.Set[types.UID]
	// assumedPods tracks pods that have reached the Reserve stage and are waiting
	// for the rest of the gang to arrive before being allowed to bind.
	assumedPods sets.Set[types.UID]
	// assignedPods tracks all pods belonging to the group that are assigned (bound).
	assignedPods sets.Set[types.UID]
	// schedulingDeadline stores the time at which the gang will time out.
	// It is initialized when the first pod from the group enters the Permit stage.
	schedulingDeadline *time.Time
}

var _ fwk.PodGroupState = &PodGroupState{}

func NewPodGroupState() *PodGroupState {
	return &PodGroupState{
		allPods:         make(map[types.UID]*v1.Pod),
		unscheduledPods: sets.New[types.UID](),
		assumedPods:     sets.New[types.UID](),
		assignedPods:    sets.New[types.UID](),
	}
}

// AddPod adds the pod to this group.
// Depending on the NodeName, it can insert the pod to assignedPods set.
// It is called under the cache lock.
func (pgs *PodGroupState) AddPod(pod *v1.Pod) {
	pgs.generation++
	pgs.allPods[pod.UID] = pod
	if pod.Spec.NodeName != "" {
		pgs.assignedPods.Insert(pod.UID)
	} else {
		pgs.unscheduledPods.Insert(pod.UID)
	}
}

// UpdatePod updates the pod in this group.
// In case of binding, it moves the pod to assignedPods.
// It is called under the cache lock.
func (pgs *PodGroupState) UpdatePod(oldPod, newPod *v1.Pod) {
	pgs.generation++
	pgs.allPods[newPod.UID] = newPod
	if oldPod.Spec.NodeName == "" && newPod.Spec.NodeName != "" {
		pgs.assignedPods.Insert(newPod.UID)
		// Clear pod from unscheduled and assumed when it is assigned.
		pgs.unscheduledPods.Delete(newPod.UID)
		pgs.assumedPods.Delete(newPod.UID)
	}
}

// Clone returns a deep copy of the live pod group state.
// It is called under the cache lock.
func (pgs *PodGroupState) Clone() *PodGroupStateSnapshot {
	var deadline *time.Time
	if pgs.schedulingDeadline != nil {
		deadline = ptr.To(*pgs.schedulingDeadline)
	}

	return &PodGroupStateSnapshot{
		generation:         pgs.generation,
		allPods:            maps.Clone(pgs.allPods),
		unscheduledPods:    pgs.unscheduledPods.Clone(),
		assumedPods:        pgs.assumedPods.Clone(),
		assignedPods:       pgs.assignedPods.Clone(),
		schedulingDeadline: deadline,
	}
}

// DeletePod completely deletes the pod from this group.
// It is called under the cache lock.
func (pgs *PodGroupState) DeletePod(podUID types.UID) {
	pgs.generation++
	delete(pgs.allPods, podUID)
	pgs.unscheduledPods.Delete(podUID)
	pgs.assumedPods.Delete(podUID)
	pgs.assignedPods.Delete(podUID)
}

// Empty returns true when the group is empty.
// It is called under the cache lock.
func (pgs *PodGroupState) Empty() bool {
	return len(pgs.allPods) == 0
}

// AllPods returns the UIDs of all pods known to the scheduler for this group.
func (pgs *PodGroupState) AllPods() sets.Set[types.UID] {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return sets.KeySet(pgs.allPods)
}

// UnscheduledPods returns all pods that are unscheduled for this group,
// i.e., are neither assumed nor assigned.
// The returned map type corresponds to the argument of the PodActivator.Activate method.
func (pgs *PodGroupState) UnscheduledPods() map[string]*v1.Pod {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	unscheduledPods := make(map[string]*v1.Pod, len(pgs.unscheduledPods))
	for podUID := range pgs.unscheduledPods {
		pod := pgs.allPods[podUID]
		unscheduledPods[pod.Name] = pod
	}
	return unscheduledPods
}

// AssumedPods returns the UIDs of all pods for this group in the assumed state,
// i.e., passed the Reserve gate.
func (pgs *PodGroupState) AssumedPods() sets.Set[types.UID] {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.assumedPods.Clone()
}

// AssignedPods returns the UIDs of all pods already assigned (bound) for this group.
func (pgs *PodGroupState) AssignedPods() sets.Set[types.UID] {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.assignedPods.Clone()
}

// SchedulingTimeout returns the remaining time until the pod group scheduling times out.
// A new deadline is created if one doesn't exist, or if the previous one has expired.
func (pgs *PodGroupState) SchedulingTimeout() time.Duration {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	now := time.Now()
	// A new deadline is set if one doesn't exist, or if the old one has passed.
	// This allows a new attempt to form a gang after a previous attempt timed out.
	if pgs.schedulingDeadline == nil || pgs.schedulingDeadline.Before(now) {
		pgs.schedulingDeadline = ptr.To(now.Add(DefaultSchedulingTimeoutDuration))
	}
	return pgs.schedulingDeadline.Sub(now)
}

// AssumePod marks a pod as having reached the Reserve stage.
// It is called under the cache lock.
func (pgs *PodGroupState) AssumePod(podUID types.UID) {
	pgs.generation++
	pgs.assumedPods.Insert(podUID)
	pgs.unscheduledPods.Delete(podUID)
}

// ForgetPod removes a pod from the assumed state.
// It is called under the cache lock.
func (pgs *PodGroupState) ForgetPod(podUID types.UID) {
	pgs.generation++
	pgs.unscheduledPods.Insert(podUID)
	pgs.assumedPods.Delete(podUID)
}

var _ fwk.PodGroupState = &PodGroupStateSnapshot{}

// PodGroupStateSnapshot is an immutable, point-in-time copy of a PodGroupState.
type PodGroupStateSnapshot struct {
	generation         int64
	allPods            map[types.UID]*v1.Pod
	unscheduledPods    sets.Set[types.UID]
	assumedPods        sets.Set[types.UID]
	assignedPods       sets.Set[types.UID]
	schedulingDeadline *time.Time
}

// AllPods returns the UIDs of all pods known to the scheduler for this group.
func (s *PodGroupStateSnapshot) AllPods() sets.Set[types.UID] {
	return sets.KeySet(s.allPods)
}

// UnscheduledPods returns all pods that are unscheduled for this group.
func (s *PodGroupStateSnapshot) UnscheduledPods() map[string]*v1.Pod {
	unscheduledPods := make(map[string]*v1.Pod, len(s.unscheduledPods))
	for podUID := range s.unscheduledPods {
		pod := s.allPods[podUID]
		unscheduledPods[pod.Name] = pod
	}
	return unscheduledPods
}

// AssumedPods returns the UIDs of all assumed pods for this group.
func (s *PodGroupStateSnapshot) AssumedPods() sets.Set[types.UID] {
	return s.assumedPods
}

// AssignedPods returns the UIDs of all assigned (bound) pods for this group.
func (s *PodGroupStateSnapshot) AssignedPods() sets.Set[types.UID] {
	return s.assignedPods
}

// TODO: look into it.
// SchedulingTimeout returns the remaining time until the pod group scheduling times out.
// Unlike the live PodGroupState, this does not create a new deadline.
func (s *PodGroupStateSnapshot) SchedulingTimeout() time.Duration {
	if s.schedulingDeadline == nil {
		return DefaultSchedulingTimeoutDuration
	}
	remaining := time.Until(*s.schedulingDeadline)
	if remaining < 0 {
		return 0
	}
	return remaining
}
