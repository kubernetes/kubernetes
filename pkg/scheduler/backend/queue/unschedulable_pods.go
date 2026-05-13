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

package queue

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// unschedulablePods holds pods that cannot be scheduled.
type unschedulablePods struct {
	// podInfoMap is a map key by a pod's full-name and the value is a pointer to the QueuedPodInfo.
	podInfoMap map[string]*framework.QueuedPodInfo
	keyFunc    func(*v1.Pod) string
	// unschedulableRecorder and gatedRecorder track the number of pods in the unschedulable queue.
	// unschedulableRecorder tracks standard unschedulable pods, while gatedRecorder tracks pods
	// that are specifically blocked by scheduling gates. These recorders handle
	// increments, decrements, and transitions (Gated <-> Ungated).
	unschedulableRecorder, gatedRecorder metrics.MetricRecorder
	// gatedByGateRecorder tracks per-gate counts of pods currently sitting in the
	// gated state, broken down by their spec.SchedulingGates names. Pods gated by
	// plugins without a corresponding spec entry contribute to gatedRecorder but
	// not here.
	gatedByGateRecorder metrics.GatedPodsByGateRecorder
	// countedGates records, for each pod currently counted on gatedByGateRecorder,
	// the gate names that were incremented. We track this independently of
	// pod.Spec.SchedulingGates because the spec can change (gates can only be
	// removed) between the time a pod is added to the gated state and the time
	// we need to decrement its counters.
	countedGates map[string]sets.Set[string]
}

// newUnschedulablePods initializes a new object of unschedulablePods.
func newUnschedulablePods(unschedulableRecorder, gatedRecorder metrics.MetricRecorder, gatedByGateRecorder metrics.GatedPodsByGateRecorder) *unschedulablePods {
	return &unschedulablePods{
		podInfoMap:            make(map[string]*framework.QueuedPodInfo),
		keyFunc:               util.GetPodFullName,
		unschedulableRecorder: unschedulableRecorder,
		gatedRecorder:         gatedRecorder,
		gatedByGateRecorder:   gatedByGateRecorder,
		countedGates:          make(map[string]sets.Set[string]),
	}
}

// updateMetricsOnStateChange handles the metric accounting when a pod changes
// between Gated and Unschedulable states. Per-gate accounting (gatedByGateRecorder)
// is reconciled separately in addOrUpdate so that gate set changes are picked up
// even when the gated/ungated state does not flip.
func (u *unschedulablePods) updateMetricsOnStateChange(gatedBefore, isGated bool) {
	if gatedBefore == isGated {
		return
	}

	if gatedBefore {
		// Transition: Gated -> Ungated
		u.gatedRecorder.Dec()
		u.unschedulableRecorder.Inc()
	} else {
		// Transition: Ungated -> Gated
		u.gatedRecorder.Inc()
		u.unschedulableRecorder.Dec()
	}
}

// addOrUpdate adds a pod to the unschedulable podInfoMap.
// The event should show which event triggered the addition and is used for the metric recording.
func (u *unschedulablePods) addOrUpdate(pInfo *framework.QueuedPodInfo, gatedBefore bool, event string) {
	podID := u.keyFunc(pInfo.Pod)
	if _, exists := u.podInfoMap[podID]; exists {
		u.updateMetricsOnStateChange(gatedBefore, pInfo.Gated())
	} else {
		if pInfo.Gated() {
			u.gatedRecorder.Inc()
		} else {
			u.unschedulableRecorder.Inc()
		}
		metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", event).Inc()
	}
	if pInfo.Gated() {
		u.reconcileCountedGates(podID, pInfo.Pod)
	} else {
		u.releaseCountedGates(podID)
	}
	u.podInfoMap[podID] = pInfo
}

// delete deletes a pod from the unschedulable podInfoMap.
// The `gated` parameter is used to figure out which metric should be decreased.
func (u *unschedulablePods) delete(pod *v1.Pod, gated bool) {
	podID := u.keyFunc(pod)
	if _, exists := u.podInfoMap[podID]; exists {
		if gated {
			u.gatedRecorder.Dec()
		} else {
			u.unschedulableRecorder.Dec()
		}
	}
	u.releaseCountedGates(podID)
	delete(u.podInfoMap, podID)
}

// get returns the QueuedPodInfo if a pod with the same key as the key of the given "pod"
// is found in the map. It returns nil otherwise.
func (u *unschedulablePods) get(pod *v1.Pod) *framework.QueuedPodInfo {
	podKey := u.keyFunc(pod)
	if pInfo, exists := u.podInfoMap[podKey]; exists {
		return pInfo
	}
	return nil
}

// clear removes all the entries from the unschedulable podInfoMap.
func (u *unschedulablePods) clear() {
	u.podInfoMap = make(map[string]*framework.QueuedPodInfo)
	u.unschedulableRecorder.Clear()
	u.gatedRecorder.Clear()
	u.gatedByGateRecorder.Clear()
	u.countedGates = make(map[string]sets.Set[string])
}

// reconcileCountedGates incs/decs gatedByGateRecorder so that the per-gate
// counters reflect the pod's current spec.SchedulingGates. Called whenever a
// pod is added or updated while in the Gated state.
func (u *unschedulablePods) reconcileCountedGates(podID string, pod *v1.Pod) {
	prev := u.countedGates[podID]
	curr := podSchedulingGateNameSet(pod)

	for g := range prev.Difference(curr) {
		u.gatedByGateRecorder.Dec(g)
	}
	for g := range curr.Difference(prev) {
		u.gatedByGateRecorder.Inc(g)
	}

	if curr.Len() == 0 {
		delete(u.countedGates, podID)
	} else {
		u.countedGates[podID] = curr
	}
}

// releaseCountedGates decrements gatedByGateRecorder for the gate set
// previously counted for this pod (if any) and clears the bookkeeping entry.
func (u *unschedulablePods) releaseCountedGates(podID string) {
	prev, ok := u.countedGates[podID]
	if !ok {
		return
	}
	for g := range prev {
		u.gatedByGateRecorder.Dec(g)
	}
	delete(u.countedGates, podID)
}

// podSchedulingGateNameSet returns the set of gate names declared on the pod's
// spec.SchedulingGates.
func podSchedulingGateNameSet(pod *v1.Pod) sets.Set[string] {
	if len(pod.Spec.SchedulingGates) == 0 {
		return sets.New[string]()
	}
	out := sets.New[string]()
	for _, g := range pod.Spec.SchedulingGates {
		out.Insert(g.Name)
	}
	return out
}
