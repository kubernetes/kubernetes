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
	"math"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/backend/heap"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/utils/clock"
)

// backoffQOrderingWindowDuration is a duration of an ordering window in the entityBackoffQ.
// In each window, represented as a whole second, pods are ordered by priority.
// It is the same as interval of flushing the pods from the entityBackoffQ to the activeQ, to flush the whole windows there.
// This works only if PopFromBackoffQ feature is enabled.
// See the KEP-5142 (http://kep.k8s.io/5142) for rationale.
const backoffQOrderingWindowDuration = time.Second

// backoffQueuer is a wrapper for backoffQ related operations.
// Its methods that relies on the queues, take the lock inside.
type backoffQueuer interface {
	// isEntityBackingoff returns true if an entity is still waiting for its backoff timer.
	// If this returns true, the entity should not be re-tried.
	// If the entity backoff time is in the actual ordering window, it should still be backing off.
	isEntityBackingoff(entity framework.QueuedEntityInfo) bool
	// popAllBackoffCompleted pops all entities from entityBackoffQ and entityErrorBackoffQ that completed backoff.
	popAllBackoffCompleted(logger klog.Logger) []framework.QueuedEntityInfo

	// podInitialBackoffDuration returns initial backoff duration that pod can get.
	podInitialBackoffDuration() time.Duration
	// podMaxBackoffDuration returns maximum backoff duration that pod can get.
	podMaxBackoffDuration() time.Duration
	// waitUntilAlignedWithOrderingWindow waits until the time reaches a multiple of backoffQOrderingWindowDuration.
	// It then runs the f function at the backoffQOrderingWindowDuration interval using a ticker.
	// It's important to align the flushing time, because entityBackoffQ's ordering is based on the windows
	// and whole windows have to be flushed at one time without a visible latency.
	waitUntilAlignedWithOrderingWindow(f func(), stopCh <-chan struct{})

	// add adds the entity to backoffQueue.
	// The event should show which event triggered this addition and is used for the metric recording.
	// It also ensures that entity is not in both queues.
	add(logger klog.Logger, entity framework.QueuedEntityInfo, event string)
	// update updates the pod in backoffQueue if oldEntity is already in the queue and the pod is present there.
	// It returns new pod info if updated, nil otherwise.
	update(newPod *v1.Pod, oldEntity framework.QueuedEntityInfo) *framework.QueuedPodInfo
	// delete deletes the entity from backoffQueue.
	// It returns the removed entity if found, nil otherwise.
	delete(entityLookup framework.QueuedEntityInfo) framework.QueuedEntityInfo
	// get returns the entity matching given entityLookup, if exists.
	get(entityLookup framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool)
	// has inform if entity exists in the queue.
	has(entityLookup framework.QueuedEntityInfo) bool
	// list returns all pods that are in the queue.
	list() []*v1.Pod
	// len returns length of the queue.
	len() int
}

// backoffQueue implements backoffQueuer and wraps two queues inside,
// providing seamless access as if it were one queue.
type backoffQueue struct {
	// lock synchronizes all operations related to backoffQ.
	// It protects both entityBackoffQ and entityErrorBackoffQ.
	// Caution: DO NOT take "SchedulingQueue.lock" or "activeQueue.lock" after taking "lock".
	// You should always take "SchedulingQueue.lock" and "activeQueue.lock" first, otherwise the queue could end up in deadlock.
	// "lock" should not be taken after taking "nominator.nLock".
	// Correct locking order is: SchedulingQueue.lock > activeQueue.lock > lock > nominator.nLock.
	lock sync.RWMutex

	clock clock.WithTicker

	// entityBackoffQ is a heap ordered by backoff expiry. Pods which have completed backoff
	// are popped from this heap before the scheduler looks at activeQ
	entityBackoffQ *heap.Heap[framework.QueuedEntityInfo]
	// entityErrorBackoffQ is a heap ordered by error backoff expiry. Pods which have completed backoff
	// are popped from this heap before the scheduler looks at activeQ
	entityErrorBackoffQ *heap.Heap[framework.QueuedEntityInfo]

	podInitialBackoff time.Duration
	podMaxBackoff     time.Duration
	// activeQLessFn is used as an eventual less function if two backoff times are equal,
	// when the SchedulerPopFromBackoffQ feature is enabled.
	activeQLessFn func(entity1, entity2 framework.QueuedEntityInfo) bool

	// isPopFromBackoffQEnabled indicates whether the feature gate SchedulerPopFromBackoffQ is enabled.
	isPopFromBackoffQEnabled bool
}

func newBackoffQueue(clock clock.WithTicker, podInitialBackoffDuration time.Duration, podMaxBackoffDuration time.Duration, activeQLessFn func(entity1, entity2 framework.QueuedEntityInfo) bool, popFromBackoffQEnabled bool) *backoffQueue {
	bq := &backoffQueue{
		clock:                    clock,
		podInitialBackoff:        podInitialBackoffDuration,
		podMaxBackoff:            podMaxBackoffDuration,
		isPopFromBackoffQEnabled: popFromBackoffQEnabled,
		activeQLessFn:            activeQLessFn,
	}
	entityBackoffQLessFn := bq.lessBackoffCompleted
	if popFromBackoffQEnabled {
		entityBackoffQLessFn = bq.lessBackoffCompletedWithPriority
	}
	bq.entityBackoffQ = heap.NewWithRecorder(queuedEntityKeyFunc, entityBackoffQLessFn, metrics.NewBackoffPodsRecorder())
	bq.entityErrorBackoffQ = heap.NewWithRecorder(queuedEntityKeyFunc, bq.lessBackoffCompleted, metrics.NewBackoffPodsRecorder())

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

// alignToWindow truncates the provided time to the entityBackoffQ ordering window.
// It returns the lowest possible timestamp in the window.
func (bq *backoffQueue) alignToWindow(t time.Time) time.Time {
	if !bq.isPopFromBackoffQEnabled {
		return t
	}
	return t.Truncate(backoffQOrderingWindowDuration)
}

// waitUntilAlignedWithOrderingWindow waits until the time reaches a multiple of backoffQOrderingWindowDuration.
// It then runs the f function at the backoffQOrderingWindowDuration interval using a ticker.
// It's important to align the flushing time, because entityBackoffQ's ordering is based on the windows
// and whole windows have to be flushed at one time without a visible latency.
func (bq *backoffQueue) waitUntilAlignedWithOrderingWindow(f func(), stopCh <-chan struct{}) {
	now := bq.clock.Now()
	// Wait until the time reaches the multiple of backoffQOrderingWindowDuration.
	durationToNextWindow := bq.alignToWindow(now.Add(backoffQOrderingWindowDuration)).Sub(now)
	timer := bq.clock.NewTimer(durationToNextWindow)
	select {
	case <-stopCh:
		timer.Stop()
		return
	case <-timer.C():
	}

	// Run a ticker to make sure the invocations of f function
	// are aligned with the backoffQ's ordering window.
	ticker := bq.clock.NewTicker(backoffQOrderingWindowDuration)
	for {
		select {
		case <-stopCh:
			return
		default:
		}

		f()

		// NOTE: b/c there is no priority selection in golang
		// it is possible for this to race, meaning we could
		// trigger ticker.C and stopCh, and ticker.C select falls through.
		// In order to mitigate we re-check stopCh at the beginning
		// of every loop to prevent extra executions of f().
		select {
		case <-stopCh:
			ticker.Stop()
			return
		case <-ticker.C():
		}
	}
}

// lessBackoffCompletedWithPriority is a less function of entityBackoffQ if PopFromBackoffQ feature is enabled.
// It orders the entities in the same BackoffOrderingWindow the same as the activeQ will do to improve popping order from backoffQ when activeQ is empty.
func (bq *backoffQueue) lessBackoffCompletedWithPriority(entity1, entity2 framework.QueuedEntityInfo) bool {
	bo1 := bq.getBackoffTime(entity1)
	bo2 := bq.getBackoffTime(entity2)
	if !bo1.Equal(bo2) {
		return bo1.Before(bo2)
	}
	// If the backoff time is the same, sort the entity in the same manner as activeQ does.
	return bq.activeQLessFn(entity1, entity2)
}

// lessBackoffCompleted is a less function of entityErrorBackoffQ.
func (bq *backoffQueue) lessBackoffCompleted(entity1, entity2 framework.QueuedEntityInfo) bool {
	bo1 := bq.getBackoffTime(entity1)
	bo2 := bq.getBackoffTime(entity2)
	return bo1.Before(bo2)
}

// isEntityBackingoff returns true if an entity is still waiting for its backoff timer.
// If this returns true, the entity should not be re-tried.
// If the entity backoff time is in the actual ordering window, it should still be backing off.
func (bq *backoffQueue) isEntityBackingoff(entity framework.QueuedEntityInfo) bool {
	boTime := bq.getBackoffTime(entity)
	// Don't use After, because in case of windows equality we want to return true.
	return !boTime.Before(bq.alignToWindow(bq.clock.Now()))
}

// getBackoffTime returns the time that entity completes backoff.
func (bq *backoffQueue) getBackoffTime(entity framework.QueuedEntityInfo) time.Time {
	if bq.podMaxBackoff == 0 {
		// If podMaxBackoff is set to 0, the backoff should be disabled completely.
		return time.Time{}
	}
	count := entity.GetUnschedulableCount()
	if entity.GetConsecutiveErrorsCount() > 0 {
		// This entity has experienced an error status at the last scheduling cycle,
		// and we should consider the error count for the backoff duration.
		count = entity.GetConsecutiveErrorsCount()
	}

	if count == 0 {
		// When the entity hasn't experienced any scheduling attempts,
		// they don't have to get a backoff.
		return time.Time{}
	}

	if entity.GetBackoffExpiration().IsZero() {
		duration := bq.calculateBackoffDuration(count, entity.Size())
		entity.SetBackoffExpiration(bq.alignToWindow(entity.GetTimestamp().Add(duration)))
	}

	return entity.GetBackoffExpiration()
}

// calculateBackoffDuration is a helper function for calculating the backoffDuration
// based on the number of attempts the item has made.
// The maximum backoff duration is multiplied by the size of the entity.
func (bq *backoffQueue) calculateBackoffDuration(count int, entitySize int) time.Duration {
	if count == 0 {
		return 0
	}
	maxBackoff := bq.podMaxBackoff
	if entitySize > 1 {
		// Multiply the maximum backoff duration by the square root of number of pods in the entity.
		// This makes the max backoff time longer than for individual containers, while still being
		// reasonably long.
		maxBackoff = time.Duration(float64(maxBackoff) * math.Sqrt(float64(entitySize)))
	}

	shift := count - 1
	if bq.podInitialBackoff > maxBackoff>>shift {
		return maxBackoff
	}
	return time.Duration(bq.podInitialBackoff << shift)
}

func (bq *backoffQueue) popAllBackoffCompletedWithQueue(logger klog.Logger, queue *heap.Heap[framework.QueuedEntityInfo]) []framework.QueuedEntityInfo {
	var poppedEntities []framework.QueuedEntityInfo
	for {
		entity, ok := queue.Peek()
		if !ok || entity == nil {
			break
		}
		if bq.isEntityBackingoff(entity) {
			break
		}
		_, err := queue.Pop()
		if err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Unable to pop entity from backoff queue despite backoff completion", "type", entity.Type(), "entity", klog.KObj(entity))
			break
		}
		poppedEntities = append(poppedEntities, entity)
	}
	return poppedEntities
}

// popAllBackoffCompleted pops all entities from entityBackoffQ and entityErrorBackoffQ that completed backoff.
func (bq *backoffQueue) popAllBackoffCompleted(logger klog.Logger) []framework.QueuedEntityInfo {
	bq.lock.Lock()
	defer bq.lock.Unlock()

	// Ensure both queues are called
	return append(bq.popAllBackoffCompletedWithQueue(logger, bq.entityBackoffQ), bq.popAllBackoffCompletedWithQueue(logger, bq.entityErrorBackoffQ)...)
}

// add adds the entity to backoffQueue.
// The event should show which event triggered this addition and is used for the metric recording.
// It also ensures that entity is not in both queues.
func (bq *backoffQueue) add(logger klog.Logger, entity framework.QueuedEntityInfo, event string) {
	bq.lock.Lock()
	defer bq.lock.Unlock()

	// If entity has empty both unschedulable plugins and pending plugins,
	// it means that it failed because of error and should be moved to entityErrorBackoffQ.
	if entity.GetUnschedulablePlugins().Len() == 0 && entity.GetPendingPlugins().Len() == 0 {
		bq.entityErrorBackoffQ.AddOrUpdate(entity)
		// Ensure the entity is not in the entityBackoffQ and report the error if it happens.
		if deletedEntity := bq.entityBackoffQ.Delete(entity); deletedEntity != nil {
			logger.Error(nil, "BackoffQueue add() was called with an entity that was already in the entityBackoffQ", "type", entity.Type(), "entity", klog.KObj(entity))
			return
		}
		metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", event).Add(float64(entity.Size()))
		logger.V(5).Info("Entity moved to an internal scheduling queue", "type", entity.Type(), "entity", klog.KObj(entity), "event", event, "queue", backoffQ)
		return
	}
	bq.entityBackoffQ.AddOrUpdate(entity)
	// Ensure the entity is not in the entityErrorBackoffQ and report the error if it happens.
	if deletedEntity := bq.entityErrorBackoffQ.Delete(entity); deletedEntity != nil {
		logger.Error(nil, "BackoffQueue add() was called with an entity that was already in the entityErrorBackoffQ", "type", entity.Type(), "entity", klog.KObj(entity))
		return
	}
	metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", event).Add(float64(entity.Size()))
	logger.V(5).Info("Entity moved to an internal scheduling queue", "type", entity.Type(), "entity", klog.KObj(entity), "event", event, "queue", backoffQ)
}

// update updates the pod in backoffQueue if oldEntity is already in the queue and the pod is present there.
// It returns new pod info if updated, nil otherwise.
func (bq *backoffQueue) update(newPod *v1.Pod, oldEntity framework.QueuedEntityInfo) *framework.QueuedPodInfo {
	bq.lock.Lock()
	defer bq.lock.Unlock()

	// If the entity is in the backoff queue, update the pod there.
	if entity, exists := bq.entityBackoffQ.Get(oldEntity); exists {
		podInfo, err := entity.Update(newPod)
		if err != nil {
			return nil
		}
		bq.entityBackoffQ.AddOrUpdate(entity)
		return podInfo
	}
	// If the entity is in the error backoff queue, update the pod there.
	if entity, exists := bq.entityErrorBackoffQ.Get(oldEntity); exists {
		podInfo, err := entity.Update(newPod)
		if err != nil {
			return nil
		}
		bq.entityErrorBackoffQ.AddOrUpdate(entity)
		return podInfo
	}
	return nil
}

// delete deletes the entity from backoffQueue.
// It returns the removed entity if found, nil otherwise.
func (bq *backoffQueue) delete(entityLookup framework.QueuedEntityInfo) framework.QueuedEntityInfo {
	bq.lock.Lock()
	defer bq.lock.Unlock()

	if entity := bq.entityBackoffQ.Delete(entityLookup); entity != nil {
		return entity
	}
	return bq.entityErrorBackoffQ.Delete(entityLookup)
}

// popBackoff pops the entity from the entityBackoffQ.
// It returns error if the queue is empty.
// This doesn't pop the entities from the entityErrorBackoffQ.
func (bq *backoffQueue) popBackoff() (framework.QueuedEntityInfo, error) {
	bq.lock.Lock()
	defer bq.lock.Unlock()

	return bq.entityBackoffQ.Pop()
}

// get returns the entity matching given entityLookup, if exists.
func (bq *backoffQueue) get(entityLookup framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool) {
	bq.lock.RLock()
	defer bq.lock.RUnlock()

	entity, exists := bq.entityBackoffQ.Get(entityLookup)
	if exists {
		return entity, true
	}
	return bq.entityErrorBackoffQ.Get(entityLookup)
}

// has inform if entity exists in the queue.
func (bq *backoffQueue) has(entityLookup framework.QueuedEntityInfo) bool {
	bq.lock.RLock()
	defer bq.lock.RUnlock()

	return bq.entityBackoffQ.Has(entityLookup) || bq.entityErrorBackoffQ.Has(entityLookup)
}

// list returns all pods that are in the queue.
func (bq *backoffQueue) list() []*v1.Pod {
	bq.lock.RLock()
	defer bq.lock.RUnlock()

	var result []*v1.Pod
	for _, entity := range bq.entityBackoffQ.List() {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			result = append(result, pInfo.Pod)
			return true
		})
	}
	for _, entity := range bq.entityErrorBackoffQ.List() {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			result = append(result, pInfo.Pod)
			return true
		})
	}
	return result
}

// len returns length of the queue.
func (bq *backoffQueue) len() int {
	bq.lock.RLock()
	defer bq.lock.RUnlock()

	return bq.entityBackoffQ.Len() + bq.entityErrorBackoffQ.Len()
}

// lenBackoff returns length of the entityBackoffQ.
func (bq *backoffQueue) lenBackoff() int {
	bq.lock.RLock()
	defer bq.lock.RUnlock()

	return bq.entityBackoffQ.Len()
}
