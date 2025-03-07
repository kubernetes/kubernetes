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
	"time"

	v1 "k8s.io/api/core/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/backend/heap"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/utils/clock"
)

// backoffQOrderingWindowDuration is a duration of an ordering window in the podBackoffQ.
// In each window, represented as a whole second, pods are ordered by priority.
// It is the same as interval of flushing the pods from the podBackoffQ to the activeQ, to flush the whole windows there.
// This works only if PopFromBackoffQ feature is enabled.
const backoffQOrderingWindowDuration = time.Second

// backoffQueuer is a wrapper for backoffQ related operations.
type backoffQueuer interface {
	// isPodBackingoff returns true if a pod is still waiting for its backoff timer.
	// If this returns true, the pod should not be re-tried.
	isPodBackingoff(podInfo *framework.QueuedPodInfo) bool
	// popEachBackoffCompleted run fn for all pods from podBackoffQ and podErrorBackoffQ that completed backoff while popping them.
	popEachBackoffCompleted(logger klog.Logger, fn func(pInfo *framework.QueuedPodInfo))

	// podInitialBackoffDuration returns initial backoff duration that pod can get.
	podInitialBackoffDuration() time.Duration
	// podMaxBackoffDuration returns maximum backoff duration that pod can get.
	podMaxBackoffDuration() time.Duration
	// waitUntilAlignedWithOrderingWindow waits until the time reaches the multiple of backoffQOrderingWindowDuration.
	// It's important to align the flushing time, because podBackoffQ's ordering is based on the windows
	// and whole windows have to be flushed at one time without a visible overhead.
	waitUntilAlignedWithOrderingWindow(stopCh <-chan struct{})

	// add adds the pInfo to backoffQueue.
	// The event should show which event triggered this addition and is used for the metric recording.
	// It also ensures that pInfo is not in both queues.
	add(logger klog.Logger, pInfo *framework.QueuedPodInfo, event string)
	// update updates the pod in backoffQueue if oldPodInfo is already in the queue.
	// It returns new pod info if updated, nil otherwise.
	update(newPod *v1.Pod, oldPodInfo *framework.QueuedPodInfo) *framework.QueuedPodInfo
	// delete deletes the pInfo from backoffQueue.
	delete(pInfo *framework.QueuedPodInfo)
	// get returns the pInfo matching given pInfoLookup, if exists.
	get(pInfoLookup *framework.QueuedPodInfo) (*framework.QueuedPodInfo, bool)
	// has inform if pInfo exists in the queue.
	has(pInfo *framework.QueuedPodInfo) bool
	// list returns all pods that are in the queue.
	list() []*framework.QueuedPodInfo
	// len returns length of the queue.
	len() int
}

// backoffQueue implements backoffQueuer and wraps two queues inside,
// providing seamless access as if it were one queue.
type backoffQueue struct {
	clock clock.Clock

	// podBackoffQ is a heap ordered by backoff expiry. Pods which have completed backoff
	// are popped from this heap before the scheduler looks at activeQ
	podBackoffQ *heap.Heap[*framework.QueuedPodInfo]
	// podErrorBackoffQ is a heap ordered by error backoff expiry. Pods which have completed backoff
	// are popped from this heap before the scheduler looks at activeQ
	podErrorBackoffQ *heap.Heap[*framework.QueuedPodInfo]

	podInitialBackoff time.Duration
	podMaxBackoff     time.Duration
}

func newBackoffQueue(clock clock.Clock, podInitialBackoffDuration time.Duration, podMaxBackoffDuration time.Duration, popFromBackoffQEnabled bool) *backoffQueue {
	bq := &backoffQueue{
		clock:             clock,
		podInitialBackoff: podInitialBackoffDuration,
		podMaxBackoff:     podMaxBackoffDuration,
	}
	podBackoffQLessFn := bq.lessBackoffCompleted
	if popFromBackoffQEnabled {
		podBackoffQLessFn = bq.lessBackoffCompletedWithPriority
	}
	bq.podBackoffQ = heap.NewWithRecorder(podInfoKeyFunc, podBackoffQLessFn, metrics.NewBackoffPodsRecorder())
	bq.podErrorBackoffQ = heap.NewWithRecorder(podInfoKeyFunc, bq.lessBackoffCompleted, metrics.NewBackoffPodsRecorder())

	return bq
}

// podInitialBackoffDuration returns initial backoff duration that pod can get.
func (bq *backoffQueue) podInitialBackoffDuration() time.Duration {
	return bq.podInitialBackoff
}

// podMaxBackoffDuration returns maximum backoff duration that pod can get.
func (bq *backoffQueue) podMaxBackoffDuration() time.Duration {
	return bq.podMaxBackoff
}

// windowForTime truncates the provided time to the podBackoffQ ordering window.
// It returns the lowest possible timestamp in the window.
func (bq *backoffQueue) windowForTime(t time.Time) time.Time {
	return t.Truncate(backoffQOrderingWindowDuration)
}

// waitUntilAlignedWithOrderingWindow waits until the time reaches the multiple of backoffQOrderingWindowDuration.
// It's important to align the flushing time, because podBackoffQ's ordering is based on the windows
// and whole windows have to be flushed at one time without a visible overhead.
func (bq *backoffQueue) waitUntilAlignedWithOrderingWindow(stopCh <-chan struct{}) {
	now := bq.clock.Now()
	// Wait until the time reaches the multiple of backoffQOrderingWindowDuration.
	durationToNextWindow := bq.windowForTime(now.Add(backoffQOrderingWindowDuration)).Sub(now)
	t := bq.clock.NewTimer(durationToNextWindow)
	select {
	case <-stopCh:
		return
	case <-t.C():
	}
}

// lessBackoffCompletedWithPriority is a less function of podBackoffQ if PopFromBackoffQ feature is enabled.
// It orders the pods by priority in the same BackoffOrderingWindow to improve popping order from backoffQ when activeQ is empty.
func (bq *backoffQueue) lessBackoffCompletedWithPriority(pInfo1, pInfo2 *framework.QueuedPodInfo) bool {
	bo1, bow1 := bq.getBackoffTimeAndWindow(pInfo1)
	bo2, bow2 := bq.getBackoffTimeAndWindow(pInfo2)
	if !bow1.Equal(bow2) {
		return bow1.Before(bow2)
	}

	pod1Priority := corev1helpers.PodPriority(pInfo1.Pod)
	pod2Priority := corev1helpers.PodPriority(pInfo2.Pod)
	if pod1Priority != pod2Priority {
		return pod1Priority > pod2Priority
	}

	return bo1.Before(bo2)
}

// lessBackoffCompleted is a less function of podErrorBackoffQ.
func (bq *backoffQueue) lessBackoffCompleted(pInfo1, pInfo2 *framework.QueuedPodInfo) bool {
	bo1, _ := bq.getBackoffTimeAndWindow(pInfo1)
	bo2, _ := bq.getBackoffTimeAndWindow(pInfo2)
	return bo1.Before(bo2)
}

// isPodBackingoff returns true if a pod is still waiting for its backoff timer.
// If this returns true, the pod should not be re-tried.
func (bq *backoffQueue) isPodBackingoff(podInfo *framework.QueuedPodInfo) bool {
	boTime, _ := bq.getBackoffTimeAndWindow(podInfo)
	return boTime.After(bq.clock.Now())
}

// podCompletedBackoff returns true if a pod completed its backoff
// and should be flushed to the activeQ. This method is different than isPodBackingoff,
// because it has to allow to pop only pods from the previous windows.
func (bq *backoffQueue) podCompletedBackoff(podInfo *framework.QueuedPodInfo) bool {
	_, boWindow := bq.getBackoffTimeAndWindow(podInfo)
	return boWindow.Before(bq.windowForTime(bq.clock.Now()))
}

// getBackoffTime returns the time that podInfo completes backoff.
// It caches the result in podInfo.BackoffExpiration and returns this value in subsequent calls.
// The cache will be cleared when this pod is poped from the scheduling queue again (i.e., at activeQ's pop),
// because of the fact that the backoff time is calculated based on podInfo.Attempts,
// which doesn't get changed until the pod's scheduling is retried.
func (bq *backoffQueue) getBackoffTimeAndWindow(podInfo *framework.QueuedPodInfo) (time.Time, time.Time) {
	if podInfo.BackoffExpiration.IsZero() {
		duration := bq.calculateBackoffDuration(podInfo)
		podInfo.BackoffExpiration = podInfo.Timestamp.Add(duration)
		podInfo.BackoffOrderingWindow = bq.windowForTime(podInfo.BackoffExpiration)
	}
	return podInfo.BackoffExpiration, podInfo.BackoffOrderingWindow
}

// calculateBackoffDuration is a helper function for calculating the backoffDuration
// based on the number of attempts the pod has made.
func (bq *backoffQueue) calculateBackoffDuration(podInfo *framework.QueuedPodInfo) time.Duration {
	if podInfo.Attempts == 0 {
		// When the Pod hasn't experienced any scheduling attempts,
		// they aren't obliged to get a backoff penalty at all.
		return 0
	}

	duration := bq.podInitialBackoff
	for i := 1; i < podInfo.Attempts; i++ {
		// Use subtraction instead of addition or multiplication to avoid overflow.
		if duration > bq.podMaxBackoff-duration {
			return bq.podMaxBackoff
		}
		duration += duration
	}
	return duration
}

func (bq *backoffQueue) popEachBackoffCompletedWithQueue(logger klog.Logger, fn func(pInfo *framework.QueuedPodInfo), queue *heap.Heap[*framework.QueuedPodInfo]) {
	for {
		pInfo, ok := queue.Peek()
		if !ok || pInfo == nil {
			break
		}
		pod := pInfo.Pod
		if !bq.podCompletedBackoff(pInfo) {
			break
		}
		_, err := queue.Pop()
		if err != nil {
			logger.Error(err, "Unable to pop pod from backoff queue despite backoff completion", "pod", klog.KObj(pod))
			break
		}
		if fn != nil {
			fn(pInfo)
		}
	}
}

// popEachBackoffCompleted run fn for all pods from podBackoffQ and podErrorBackoffQ that completed backoff while popping them.
func (bq *backoffQueue) popEachBackoffCompleted(logger klog.Logger, fn func(pInfo *framework.QueuedPodInfo)) {
	// Ensure both queues are called
	bq.popEachBackoffCompletedWithQueue(logger, fn, bq.podBackoffQ)
	bq.popEachBackoffCompletedWithQueue(logger, fn, bq.podErrorBackoffQ)
}

// add adds the pInfo to backoffQueue.
// The event should show which event triggered this addition and is used for the metric recording.
// It also ensures that pInfo is not in both queues.
func (bq *backoffQueue) add(logger klog.Logger, pInfo *framework.QueuedPodInfo, event string) {
	// If pod has empty both unschedulable plugins and pending plugins,
	// it means that it failed because of error and should be moved to podErrorBackoffQ.
	if pInfo.UnschedulablePlugins.Len() == 0 && pInfo.PendingPlugins.Len() == 0 {
		bq.podErrorBackoffQ.AddOrUpdate(pInfo)
		// Ensure the pod is not in the podBackoffQ and report the error if it happens.
		err := bq.podBackoffQ.Delete(pInfo)
		if err == nil {
			logger.Error(nil, "BackoffQueue add() was called with a pod that was already in the podBackoffQ", "pod", klog.KObj(pInfo.Pod))
			return
		}
		metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", event).Inc()
		return
	}
	bq.podBackoffQ.AddOrUpdate(pInfo)
	// Ensure the pod is not in the podErrorBackoffQ and report the error if it happens.
	err := bq.podErrorBackoffQ.Delete(pInfo)
	if err == nil {
		logger.Error(nil, "BackoffQueue add() was called with a pod that was already in the podErrorBackoffQ", "pod", klog.KObj(pInfo.Pod))
		return
	}
	metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", event).Inc()
}

// update updates the pod in backoffQueue if oldPodInfo is already in the queue.
// It returns new pod info if updated, nil otherwise.
func (bq *backoffQueue) update(newPod *v1.Pod, oldPodInfo *framework.QueuedPodInfo) *framework.QueuedPodInfo {
	// If the pod is in the backoff queue, update it there.
	if pInfo, exists := bq.podBackoffQ.Get(oldPodInfo); exists {
		_ = pInfo.Update(newPod)
		bq.podBackoffQ.AddOrUpdate(pInfo)
		return pInfo
	}
	// If the pod is in the error backoff queue, update it there.
	if pInfo, exists := bq.podErrorBackoffQ.Get(oldPodInfo); exists {
		_ = pInfo.Update(newPod)
		bq.podErrorBackoffQ.AddOrUpdate(pInfo)
		return pInfo
	}
	return nil
}

// delete deletes the pInfo from backoffQueue.
func (bq *backoffQueue) delete(pInfo *framework.QueuedPodInfo) {
	_ = bq.podBackoffQ.Delete(pInfo)
	_ = bq.podErrorBackoffQ.Delete(pInfo)
}

// get returns the pInfo matching given pInfoLookup, if exists.
func (bq *backoffQueue) get(pInfoLookup *framework.QueuedPodInfo) (*framework.QueuedPodInfo, bool) {
	pInfo, exists := bq.podBackoffQ.Get(pInfoLookup)
	if exists {
		return pInfo, true
	}
	return bq.podErrorBackoffQ.Get(pInfoLookup)
}

// has inform if pInfo exists in the queue.
func (bq *backoffQueue) has(pInfo *framework.QueuedPodInfo) bool {
	return bq.podBackoffQ.Has(pInfo) || bq.podErrorBackoffQ.Has(pInfo)
}

// list returns all pods that are in the queue.
func (bq *backoffQueue) list() []*framework.QueuedPodInfo {
	return append(bq.podBackoffQ.List(), bq.podErrorBackoffQ.List()...)
}

// len returns length of the queue.
func (bq *backoffQueue) len() int {
	return bq.podBackoffQ.Len() + bq.podErrorBackoffQ.Len()
}
