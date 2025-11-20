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
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// unschedulablePods holds pods that cannot be scheduled.
type unschedulablePods struct {
	// podInfoMap is a map key by a pod's full-name and the value is a pointer to the QueuedPodInfo.
	podInfoMap map[string]*framework.QueuedPodInfo
	keyFunc    func(*v1.Pod) string
	// unschedulableRecorder/gatedRecorder updates the counter when elements of an unschedulablePods
	// get added or removed, and it does nothing if it's nil.
	unschedulableRecorder, gatedRecorder metrics.MetricRecorder
}

// newUnschedulablePods initializes a new object of unschedulablePods.
func newUnschedulablePods(unschedulableRecorder, gatedRecorder metrics.MetricRecorder) *unschedulablePods {
	return &unschedulablePods{
		podInfoMap:            make(map[string]*framework.QueuedPodInfo),
		keyFunc:               util.GetPodFullName,
		unschedulableRecorder: unschedulableRecorder,
		gatedRecorder:         gatedRecorder,
	}
}

// updateMetricsOnStateChange handles the metric accounting when a pod changes
// between Gated and Unschedulable states.
func (u *unschedulablePods) updateMetricsOnStateChange(gatedBefore, isGated bool) {
	if gatedBefore == isGated {
		return
	}

	if gatedBefore {
		// Transition: Gated -> Ungated
		if u.gatedRecorder != nil {
			u.gatedRecorder.Dec()
		}
		if u.unschedulableRecorder != nil {
			u.unschedulableRecorder.Inc()
		}
	} else {
		// Transition: Ungated -> Gated
		if u.unschedulableRecorder != nil {
			u.unschedulableRecorder.Dec()
		}
		if u.gatedRecorder != nil {
			u.gatedRecorder.Inc()
		}
	}
}

// addOrUpdate adds a pod to the unschedulable podInfoMap.
// The event should show which event triggered the addition and is used for the metric recording.
func (u *unschedulablePods) addOrUpdate(pInfo *framework.QueuedPodInfo, gatedBefore bool, event string) {
	podID := u.keyFunc(pInfo.Pod)
	if _, exists := u.podInfoMap[podID]; exists {
		u.updateMetricsOnStateChange(gatedBefore, pInfo.Gated())
	} else {
		if pInfo.Gated() && u.gatedRecorder != nil {
			u.gatedRecorder.Inc()
		} else if !pInfo.Gated() && u.unschedulableRecorder != nil {
			u.unschedulableRecorder.Inc()
		}
		metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", event).Inc()
	}
	u.podInfoMap[podID] = pInfo
}

// delete deletes a pod from the unschedulable podInfoMap.
// The `gated` parameter is used to figure out which metric should be decreased.
func (u *unschedulablePods) delete(pod *v1.Pod, gated bool) {
	podID := u.keyFunc(pod)
	if _, exists := u.podInfoMap[podID]; exists {
		if gated && u.gatedRecorder != nil {
			u.gatedRecorder.Dec()
		} else if !gated && u.unschedulableRecorder != nil {
			u.unschedulableRecorder.Dec()
		}
	}
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
	if u.unschedulableRecorder != nil {
		u.unschedulableRecorder.Clear()
	}
	if u.gatedRecorder != nil {
		u.gatedRecorder.Clear()
	}
}
