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
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// unschedulablePodsQueuer is a wrapper for unschedulableQ related operations.
type unschedulablePodsQueuer interface {
	// add adds the pInfo to unschedulable podInfoMap.
	// The event should show which event triggered the addition and is used for the metric recording.
	add(logger klog.Logger, pInfo *framework.QueuedPodInfo, event string)
	// update updates the QueuedPodInfo in unschedulable podInfoMap.
	update(pInfo *framework.QueuedPodInfo)
	// delete deletes a pod from the unschedulable podInfoMap.
	// The `gated` parameter is used to figure out which metric should be decreased.
	delete(pod *v1.Pod, gated bool)
	// get returns the QueuedPodInfo for a pod with the same key as the given pod in unschedulable podInfoMap.
	// It returns false if the pod does not exist.
	get(pod *v1.Pod) (*framework.QueuedPodInfo, bool)
	// has returns true if a pod with the same key as the given pod exists in unschedulable podInfoMap.
	has(pod *v1.Pod) bool
	// list returns all QueuedPodInfo objects in the unschedulable podInfoMap.
	list() []*framework.QueuedPodInfo
	// listPod returns all Pods in the unschedulable podInfoMap.
	listPod() []*v1.Pod
	// len returns length of the unschedulable podInfoMap.
	len() int
	// clear removes all the entries from the unschedulable podInfoMap.
	clear()
}

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

// add adds the pInfo to unschedulable podInfoMap.
// The event should show which event triggered the addition and is used for the metric recording.
func (u *unschedulablePods) add(logger klog.Logger, pInfo *framework.QueuedPodInfo, event string) {
	podID := u.keyFunc(pInfo.Pod)
	if _, exists := u.podInfoMap[podID]; !exists {
		if pInfo.Gated() && u.gatedRecorder != nil {
			u.gatedRecorder.Inc()
		} else if !pInfo.Gated() && u.unschedulableRecorder != nil {
			u.unschedulableRecorder.Inc()
		}
		metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", event).Inc()
		logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", event, "queue", unschedulableQ)
	}
	u.podInfoMap[podID] = pInfo
}

// update updates the QueuedPodInfo in unschedulable podInfoMap.
func (u *unschedulablePods) update(pInfo *framework.QueuedPodInfo) {
	podID := u.keyFunc(pInfo.Pod)
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

// get returns the QueuedPodInfo for a pod with the same key as the given pod in unschedulable podInfoMap.
// It returns false if the pod does not exist.
func (u *unschedulablePods) get(pod *v1.Pod) (*framework.QueuedPodInfo, bool) {
	podKey := u.keyFunc(pod)
	pInfo, exists := u.podInfoMap[podKey]
	return pInfo, exists
}

// has returns true if a pod with the same key as the given pod exists in unschedulable podInfoMap.
func (u *unschedulablePods) has(pod *v1.Pod) bool {
	podKey := u.keyFunc(pod)
	_, exists := u.podInfoMap[podKey]
	return exists
}

// list returns all QueuedPodInfo objects in the unschedulable podInfoMap.
func (u *unschedulablePods) list() []*framework.QueuedPodInfo {
	var result []*framework.QueuedPodInfo
	for _, pInfo := range u.podInfoMap {
		result = append(result, pInfo)
	}
	return result
}

// listPod returns all Pods in the unschedulable podInfoMap.
func (u *unschedulablePods) listPod() []*v1.Pod {
	var result []*v1.Pod
	for _, pInfo := range u.podInfoMap {
		result = append(result, pInfo.Pod)
	}
	return result
}

// len returns length of the unschedulable podInfoMap.
func (u *unschedulablePods) len() int {
	return len(u.podInfoMap)
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
