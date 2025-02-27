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
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/backend/heap"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/utils/clock"
)

// backoffQueuer is a wrapper for backoffQ related operations.
type backoffQueuer interface {
	// isPodBackingoff returns true if a pod is still waiting for its backoff timer.
	// If this returns true, the pod should not be re-tried.
	isPodBackingoff(podInfo *framework.QueuedPodInfo) bool
	// getBackoffTime returns the time that podInfo completes backoff
	getBackoffTime(podInfo *framework.QueuedPodInfo) time.Time
	// popEachBackoffCompleted run fn for all pods from podBackoffQ and podErrorBackoffQ that completed backoff while popping them.
	popEachBackoffCompleted(logger klog.Logger, fn func(pInfo *framework.QueuedPodInfo))

	// podInitialBackoffDuration returns initial backoff duration that pod can get.
	podInitialBackoffDuration() time.Duration
	// podMaxBackoffDuration returns maximum backoff duration that pod can get.
	podMaxBackoffDuration() time.Duration

	// add adds the pInfo to backoffQueue.
	// It also ensures that pInfo is not in both queues.
	add(logger klog.Logger, pInfo *framework.QueuedPodInfo)
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

func newBackoffQueue(clock clock.Clock, podInitialBackoffDuration time.Duration, podMaxBackoffDuration time.Duration) *backoffQueue {
	bq := &backoffQueue{
		clock:             clock,
		podInitialBackoff: podInitialBackoffDuration,
		podMaxBackoff:     podMaxBackoffDuration,
	}
	bq.podBackoffQ = heap.NewWithRecorder(podInfoKeyFunc, bq.lessBackoffCompleted, metrics.NewBackoffPodsRecorder())
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

// lessBackoffCompleted is a less function of podBackoffQ and podErrorBackoffQ.
func (bq *backoffQueue) lessBackoffCompleted(pInfo1, pInfo2 *framework.QueuedPodInfo) bool {
	bo1 := bq.getBackoffTime(pInfo1)
	bo2 := bq.getBackoffTime(pInfo2)
	return bo1.Before(bo2)
}

// isPodBackingoff returns true if a pod is still waiting for its backoff timer.
// If this returns true, the pod should not be re-tried.
func (bq *backoffQueue) isPodBackingoff(podInfo *framework.QueuedPodInfo) bool {
	boTime := bq.getBackoffTime(podInfo)
	return boTime.After(bq.clock.Now())
}

// getBackoffTime returns the time that podInfo completes backoff
func (bq *backoffQueue) getBackoffTime(podInfo *framework.QueuedPodInfo) time.Time {
	duration := bq.calculateBackoffDuration(podInfo)
	backoffTime := podInfo.Timestamp.Add(duration)
	return backoffTime
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
		if bq.isPodBackingoff(pInfo) {
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
// It also ensures that pInfo is not in both queues.
func (bq *backoffQueue) add(logger klog.Logger, pInfo *framework.QueuedPodInfo) {
	// If pod has empty both unschedulable plugins and pending plugins,
	// it means that it failed because of error and should be moved to podErrorBackoffQ.
	if pInfo.UnschedulablePlugins.Len() == 0 && pInfo.PendingPlugins.Len() == 0 {
		bq.podErrorBackoffQ.AddOrUpdate(pInfo)
		// Ensure the pod is not in the podBackoffQ and report the error if it happens.
		err := bq.podBackoffQ.Delete(pInfo)
		if err == nil {
			logger.Error(nil, "BackoffQueue add() was called with a pod that was already in the podBackoffQ", "pod", klog.KObj(pInfo.Pod))
		}
		return
	}
	bq.podBackoffQ.AddOrUpdate(pInfo)
	// Ensure the pod is not in the podErrorBackoffQ and report the error if it happens.
	err := bq.podErrorBackoffQ.Delete(pInfo)
	if err == nil {
		logger.Error(nil, "BackoffQueue add() was called with a pod that was already in the podErrorBackoffQ", "pod", klog.KObj(pInfo.Pod))
	}
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
