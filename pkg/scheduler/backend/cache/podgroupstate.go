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
	"maps"
	"sync"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

var generation atomic.Int64

// nextPodGroupGeneration increments generation numbers monotonically for a pod group state (instead of per-instance increment)
// to prevent generation reset or collision when a pod group is deleted and recreated with the same name.
func nextPodGroupGeneration() int64 {
	return generation.Add(1)
}

// podGroupStateData holds data and functionality shared between podGroupState and podGroupStateSnapshot.
// Note that the podGroup field is populated from the observed PodGroup API object,
// while other fields are populated from observed Pod objects. This means podGroupStateData
// can exist without a corresponding PodGroup API object as long as at least one
// Pod references it.
type podGroupStateData struct {
	// generation gets bumped whenever the data is changed.
	// It's used to detect changes and avoid unnecessary cloning when taking a snapshot.
	generation int64
	// allPods tracks all pods belonging to the group that are known to the scheduler.
	allPods map[types.UID]*v1.Pod
	// unscheduledPods tracks all pods that are unscheduled for this group,
	// i.e., are neither assumed nor assigned.
	unscheduledPods sets.Set[types.UID]
	// assumedPods tracks pods that have reached the Reserve stage and are waiting
	// for the rest of the gang to arrive before being allowed to bind.
	assumedPods map[types.UID]*v1.Pod
	// assignedPods tracks all pods belonging to the group that are assigned (bound).
	assignedPods sets.Set[types.UID]
	// podGroup is the cached API object of the PodGroup.
	podGroup *schedulingv1beta1.PodGroup
}

func newPodGroupStateData() podGroupStateData {
	return podGroupStateData{
		allPods:         make(map[types.UID]*v1.Pod),
		unscheduledPods: sets.New[types.UID](),
		assumedPods:     make(map[types.UID]*v1.Pod),
		assignedPods:    sets.New[types.UID](),
	}
}

// addPod adds the pod to this group.
// Depending on the NodeName, it can insert the pod into either assignedPods or unscheduledPods.
func (d *podGroupStateData) addPod(pod *v1.Pod) {
	d.generation = nextPodGroupGeneration()
	d.allPods[pod.UID] = pod
	if pod.Spec.NodeName != "" {
		d.assignedPods.Insert(pod.UID)
		// Clear from unscheduled or assumed in case the pod previously existed in the pod group
		// in a different state, e.g., external binding of a previously-queued pod group member.
		d.unscheduledPods.Delete(pod.UID)
		delete(d.assumedPods, pod.UID)
	} else {
		d.unscheduledPods.Insert(pod.UID)
	}
}

// updatePod updates the pod in this group.
// In case of binding, it moves the pod to assignedPods.
func (d *podGroupStateData) updatePod(oldPod, newPod *v1.Pod) {
	d.generation = nextPodGroupGeneration()
	d.allPods[newPod.UID] = newPod
	if oldPod.Spec.NodeName == "" && newPod.Spec.NodeName != "" {
		d.assignedPods.Insert(newPod.UID)
		// Clear pod from unscheduled and assumed when it is assigned.
		d.unscheduledPods.Delete(newPod.UID)
		delete(d.assumedPods, newPod.UID)
	}
}

// deletePod removes the pod from this pod group state.
func (d *podGroupStateData) deletePod(podUID types.UID) {
	d.generation = nextPodGroupGeneration()
	delete(d.allPods, podUID)
	d.unscheduledPods.Delete(podUID)
	delete(d.assumedPods, podUID)
	d.assignedPods.Delete(podUID)
}

// assumePod marks a pod as assumed within the pod group state.
func (d *podGroupStateData) assumePod(pod *v1.Pod) {
	storedPod, ok := d.allPods[pod.UID]
	// A scheduling pod may be removed from the cluster.
	// In that case, we just ignore it.
	if !ok {
		return
	}

	d.generation = nextPodGroupGeneration()
	// If the pod stored in the state is already assigned, put it into assignedPods.
	// Otherwise put it to assumedPods.
	if storedPod.Spec.NodeName != "" {
		d.assignedPods.Insert(pod.UID)
	} else {
		d.assumedPods[pod.UID] = pod
	}
	d.unscheduledPods.Delete(pod.UID)
}

// forgetPod moves a pod back from the assumed state to unscheduled within the pod group state.
func (d *podGroupStateData) forgetPod(podUID types.UID) {

	pod := d.allPods[podUID]
	// A scheduling pod may be removed from the cluster.
	// In that case, we just ignore it.
	if pod == nil {
		return
	}

	d.generation = nextPodGroupGeneration()

	delete(d.assumedPods, podUID)

	// If the pod is already assigned, put it into assignedPods.
	// Otherwise, put it into unscheduledPods.
	if pod.Spec.NodeName != "" {
		d.assignedPods.Insert(podUID)
	} else {
		d.unscheduledPods.Insert(podUID)
	}
}

// scheduledPods returns the pods that are either assumed or assigned for this pod group.
func (d *podGroupStateData) scheduledPods() []*v1.Pod {
	scheduledPods := make([]*v1.Pod, 0, len(d.assignedPods)+len(d.assumedPods))
	for uid := range d.assignedPods {
		scheduledPods = append(scheduledPods, d.allPods[uid])
	}
	for _, pod := range d.assumedPods {
		scheduledPods = append(scheduledPods, pod)
	}
	return scheduledPods
}

// empty returns true when the pod group state contains no pods.
func (d *podGroupStateData) empty() bool {
	return len(d.allPods) == 0 && d.podGroup == nil
}

// allPodsCount returns the number of all pods known to the scheduler for this group.
func (d *podGroupStateData) allPodsCount() int {
	return len(d.allPods)
}

// scheduledPodsCount returns the number of pods for this group that are either assumed or assigned.
func (d *podGroupStateData) scheduledPodsCount() int {
	return len(d.assumedPods) + len(d.assignedPods)
}

// clone returns a clone of the pod group state data.
// It does not deep copy the inner Pod and PodGroup objects
// as they should not be mutated by the scheduler.
// Cache's and snapshot's objects are read-only from the outside,
// unless mutated explicitly by the methods.
func (d *podGroupStateData) clone() podGroupStateData {
	return podGroupStateData{
		generation:      d.generation,
		allPods:         maps.Clone(d.allPods),
		unscheduledPods: d.unscheduledPods.Clone(),
		assumedPods:     maps.Clone(d.assumedPods),
		assignedPods:    d.assignedPods.Clone(),
		podGroup:        d.podGroup,
	}
}

// setPodGroup sets the PodGroup object.
func (d *podGroupStateData) setPodGroup(podGroup *schedulingv1beta1.PodGroup) {
	d.generation = nextPodGroupGeneration()
	d.podGroup = podGroup
}

// removePodGroup removes the PodGroup object.
func (d *podGroupStateData) removePodGroup() {
	d.generation = nextPodGroupGeneration()
	d.podGroup = nil
}

// unscheduledPodsMap returns all unscheduled pods for this pod group.
func (d *podGroupStateData) unscheduledPodsMap() map[string]*v1.Pod {
	result := make(map[string]*v1.Pod, len(d.unscheduledPods))
	for podUID := range d.unscheduledPods {
		pod := d.allPods[podUID]
		result[pod.Name] = pod
	}
	return result
}

// compositePodGroupStateData holds data and functionality shared between compositePodGroupState and compositePodGroupStateSnapshot.
// Note that the compositePodGroup field is populated from the observed CompositePodGroup API object,
// while other fields are populated from observed child PodGroups and CompositePodGroups. This means compositePodGroupStateData
// can exist without a corresponding CompositePodGroup API object as long as at least one
// child references it.
type compositePodGroupStateData struct {
	// generation gets bumped whenever the data is changed.
	// It's used to detect changes and avoid unnecessary cloning when taking a snapshot.
	generation int64
	// compositePodGroup is the cached API object of the CompositePodGroup.
	compositePodGroup *schedulingv1alpha3.CompositePodGroup
	// children tracks all keys for child pod groups and composite pod groups.
	children sets.Set[fwk.EntityKey]
}

func newCompositePodGroupStateData() compositePodGroupStateData {
	return compositePodGroupStateData{
		children: sets.New[fwk.EntityKey](),
	}
}

// clone returns a clone of the composite pod group state data.
// It does not deep copy the inner CompositePodGroup object
// as it should not be mutated by the scheduler.
// Cache's and snapshot's objects are read-only from the outside,
// unless mutated explicitly by the methods.
func (d *compositePodGroupStateData) clone() compositePodGroupStateData {
	return compositePodGroupStateData{
		generation:        d.generation,
		compositePodGroup: d.compositePodGroup,
		children:          d.children.Clone(),
	}
}

// empty returns true when the composite pod group state contains no composite pod group.
func (d *compositePodGroupStateData) empty() bool {
	return d.compositePodGroup == nil && len(d.children) == 0
}

func (d *compositePodGroupStateData) setCompositePodGroup(compositePodGroup *schedulingv1alpha3.CompositePodGroup) {
	d.generation = nextPodGroupGeneration()
	d.compositePodGroup = compositePodGroup
}

func (d *compositePodGroupStateData) removeCompositePodGroup() {
	d.generation = nextPodGroupGeneration()
	d.compositePodGroup = nil
}

// addChild adds a child group to this group.
func (d *compositePodGroupStateData) addChild(child fwk.EntityKey) {
	d.generation = nextPodGroupGeneration()
	d.children.Insert(child)
}

// removeChild removes a child group from this group.
func (d *compositePodGroupStateData) removeChild(child fwk.EntityKey) {
	d.generation = nextPodGroupGeneration()
	d.children.Delete(child)
}

// getChildren returns all child groups for this group.
func (d *compositePodGroupStateData) getChildren() []fwk.EntityKey {
	var children []fwk.EntityKey
	for child := range d.children {
		children = append(children, child)
	}
	return children
}

// podGroupState holds the runtime state of a pod group.
type podGroupState struct {
	lock sync.RWMutex
	podGroupStateData
}

func newPodGroupState() *podGroupState {
	return &podGroupState{podGroupStateData: newPodGroupStateData()}
}

// snapshot returns a deep copy of the live pod group state as an immutable snapshot.
// It must be called under the cache lock.
func (pgs *podGroupState) snapshot() *podGroupStateSnapshot {
	return &podGroupStateSnapshot{podGroupStateData: pgs.podGroupStateData.clone()}
}

// empty returns true when the group contains no pods and the cached PodGroup object is nil.
// It must be called under the cache lock.
func (pgs *podGroupState) empty() bool {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.podGroupStateData.empty()
}

// addPod adds the pod to this group.
// Depending on the NodeName, it can insert the pod into either assignedPods or unscheduledPods.
// It must be called under the cache lock.
func (pgs *podGroupState) addPod(pod *v1.Pod) {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	pgs.podGroupStateData.addPod(pod)
}

// updatePod updates the pod in this group.
// In case of binding, it moves the pod to assignedPods.
// It must be called under the cache lock.
func (pgs *podGroupState) updatePod(oldPod, newPod *v1.Pod) {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	pgs.podGroupStateData.updatePod(oldPod, newPod)
}

// deletePod removes the pod from this pod group state.
// It must be called under the cache lock.
func (pgs *podGroupState) deletePod(podUID types.UID) {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	pgs.podGroupStateData.deletePod(podUID)
}

// assumePod marks a pod as assumed within the pod group state.
// It must be called under the cache lock.
func (pgs *podGroupState) assumePod(pod *v1.Pod) {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	pgs.podGroupStateData.assumePod(pod)
}

// forgetPod moves a pod back from the assumed state to unscheduled within the pod group state.
// It must be called under the cache lock.
func (pgs *podGroupState) forgetPod(podUID types.UID) {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	pgs.podGroupStateData.forgetPod(podUID)
}

// setPodGroup sets the PodGroup object.
// It must be called under the cache lock.
func (pgs *podGroupState) setPodGroup(podGroup *schedulingv1beta1.PodGroup) {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	pgs.podGroupStateData.setPodGroup(podGroup)
	pgs.podGroupStateData.generation = nextPodGroupGeneration()
}

// removePodGroup removes the PodGroup object.
// It must be called under the cache lock.
func (pgs *podGroupState) removePodGroup() {
	pgs.lock.Lock()
	defer pgs.lock.Unlock()

	pgs.podGroupStateData.removePodGroup()
	pgs.podGroupStateData.generation = nextPodGroupGeneration()
}

// AllPods returns the UIDs of all pods known to the scheduler for this group.
func (pgs *podGroupState) AllPods() sets.Set[types.UID] {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return sets.KeySet(pgs.podGroupStateData.allPods)
}

// AllPodsCount returns the number of all pods known to the scheduler for this group.
func (pgs *podGroupState) AllPodsCount() int {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.podGroupStateData.allPodsCount()
}

// UnscheduledPods returns all pods that are unscheduled for this group,
// i.e., are neither assumed nor assigned.
// The returned map type corresponds to the argument of the PodActivator.Activate method.
func (pgs *podGroupState) UnscheduledPods() map[string]*v1.Pod {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.podGroupStateData.unscheduledPodsMap()
}

// AssumedPods returns the UIDs of all pods for this group in the assumed state,
// i.e., that have passed the Reserve stage.
func (pgs *podGroupState) AssumedPods() sets.Set[types.UID] {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return sets.KeySet(pgs.podGroupStateData.assumedPods)
}

// AssignedPods returns the UIDs of all pods already assigned (bound) for this group.
func (pgs *podGroupState) AssignedPods() sets.Set[types.UID] {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.podGroupStateData.assignedPods.Clone()
}

// ScheduledPods returns the pods that are either assumed or assigned for this pod group.
func (pgs *podGroupState) ScheduledPods() []*v1.Pod {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.podGroupStateData.scheduledPods()
}

// ScheduledPodsCount returns the number of pods for this group that are either assumed or assigned.
func (pgs *podGroupState) ScheduledPodsCount() int {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.podGroupStateData.scheduledPodsCount()
}

// PodGroup returns the PodGroup API object.
func (pgs *podGroupState) PodGroup() *schedulingv1beta1.PodGroup {
	pgs.lock.RLock()
	defer pgs.lock.RUnlock()

	return pgs.podGroupStateData.podGroup
}

// compositePodGroupState holds the runtime state of a composite pod group.
type compositePodGroupState struct {
	lock sync.RWMutex
	compositePodGroupStateData
}

// newCompositePodGroupState creates a new compositePodGroupState.
func newCompositePodGroupState() *compositePodGroupState {
	return &compositePodGroupState{compositePodGroupStateData: newCompositePodGroupStateData()}
}

// snapshot returns a deep copy of the live composite pod group state as an immutable snapshot.
// It must be called under the cache lock.
func (cpgs *compositePodGroupState) snapshot() *compositePodGroupStateSnapshot {
	return &compositePodGroupStateSnapshot{compositePodGroupStateData: cpgs.compositePodGroupStateData.clone()}
}

// empty returns true when the composite pod group state contains no composite pod group and no children.
// It must be called under the cache lock.
func (cpgs *compositePodGroupState) empty() bool {
	cpgs.lock.RLock()
	defer cpgs.lock.RUnlock()

	return cpgs.compositePodGroupStateData.empty()
}

// setCompositePodGroup sets the CompositePodGroup object.
// It must be called under the cache lock.
func (cpgs *compositePodGroupState) setCompositePodGroup(compositePodGroup *schedulingv1alpha3.CompositePodGroup) {
	cpgs.lock.Lock()
	defer cpgs.lock.Unlock()

	cpgs.compositePodGroupStateData.setCompositePodGroup(compositePodGroup)
}

// removeCompositePodGroup removes the CompositePodGroup object.
// It must be called under the cache lock.
func (cpgs *compositePodGroupState) removeCompositePodGroup() {
	cpgs.lock.Lock()
	defer cpgs.lock.Unlock()

	cpgs.compositePodGroupStateData.removeCompositePodGroup()
}

// CompositePodGroup returns the CompositePodGroup API object.
func (cpgs *compositePodGroupState) CompositePodGroup() *schedulingv1alpha3.CompositePodGroup {
	cpgs.lock.RLock()
	defer cpgs.lock.RUnlock()

	return cpgs.compositePodGroupStateData.compositePodGroup
}

// GetChildren returns the keys of child pod groups or composite pod groups.
func (cpgs *compositePodGroupState) GetChildren() []fwk.EntityKey {
	cpgs.lock.RLock()
	defer cpgs.lock.RUnlock()

	return cpgs.compositePodGroupStateData.getChildren()
}

// podGroupStateSnapshot is an immutable, point-in-time copy of a podGroupState.
// It is taken before a pod group scheduling cycle and used to track states of pods
// during the cycle without modifying the live state of pods.
type podGroupStateSnapshot struct {
	podGroupStateData
}

// assumePod marks a pod within the pod group state snapshot as assumed.
func (s *podGroupStateSnapshot) assumePod(pod *v1.Pod) {
	s.podGroupStateData.assumePod(pod)
}

// forgetPod removes a pod from the assumed state within the snapshot.
func (s *podGroupStateSnapshot) forgetPod(podUID types.UID) {
	s.podGroupStateData.forgetPod(podUID)
}

// AllPods returns the UIDs of all pods known to the scheduler for this group.
func (s *podGroupStateSnapshot) AllPods() sets.Set[types.UID] {
	return sets.KeySet(s.podGroupStateData.allPods)
}

// UnscheduledPods returns all pods that are unscheduled for this group.
func (s *podGroupStateSnapshot) UnscheduledPods() map[string]*v1.Pod {
	return s.podGroupStateData.unscheduledPodsMap()
}

// AssumedPods returns the UIDs of all assumed pods for this group.
func (s *podGroupStateSnapshot) AssumedPods() sets.Set[types.UID] {
	return sets.KeySet(s.podGroupStateData.assumedPods)
}

// AssignedPods returns the UIDs of all assigned (bound) pods for this group.
func (s *podGroupStateSnapshot) AssignedPods() sets.Set[types.UID] {
	return s.podGroupStateData.assignedPods
}

// ScheduledPods returns the pods that are either assumed or assigned for this pod group.
func (s *podGroupStateSnapshot) ScheduledPods() []*v1.Pod {
	return s.podGroupStateData.scheduledPods()
}

// AllPodsCount returns the number of all pods known to the scheduler for this group.
func (s *podGroupStateSnapshot) AllPodsCount() int {
	return s.podGroupStateData.allPodsCount()
}

// ScheduledPodsCount returns the number of pods for this group that are either assumed or assigned.
func (s *podGroupStateSnapshot) ScheduledPodsCount() int {
	return s.podGroupStateData.scheduledPodsCount()
}

// Clone returns a pod group state snapshot with cloned podGroupStateData.
func (s *podGroupStateSnapshot) Clone() *podGroupStateSnapshot {
	return &podGroupStateSnapshot{podGroupStateData: s.podGroupStateData.clone()}
}

// compositePodGroupStateSnapshot is an immutable, point-in-time copy of a compositePodGroupState.
// It is taken before a pod group scheduling cycle and used to track states of composite pod groups.
type compositePodGroupStateSnapshot struct {
	compositePodGroupStateData
}

// GetChildren returns the keys of the child groups.
func (s *compositePodGroupStateSnapshot) GetChildren() []fwk.EntityKey {
	return s.compositePodGroupStateData.getChildren()
}
