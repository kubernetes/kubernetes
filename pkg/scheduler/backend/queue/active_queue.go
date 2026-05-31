/*
Copyright 2024 The Kubernetes Authors.

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
	"container/list"
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/backend/heap"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// activeQueuer is a wrapper for activeQ related operations.
// Its methods take the lock inside.
type activeQueuer interface {
	underLock(func(unlockedActiveQ unlockedActiveQueuer))
	underRLock(func(unlockedActiveQ unlockedActiveQueueReader))

	// delete removes the entity from the activeQ.
	// It returns the entity if it was removed, nil otherwise.
	delete(entityLookup framework.QueuedEntityInfo) framework.QueuedEntityInfo
	pop(logger klog.Logger) (framework.QueuedEntityInfo, error)
	list() []*v1.Pod
	len() int
	has(entityLookup framework.QueuedEntityInfo) bool
	// add adds a new entity to the activeQ.
	// The event should show which event triggered this addition and is used for the metric recording.
	// Note: it does not signal the pop() method to wake up,
	// so the caller is responsible for calling broadcast() after executing this method.
	add(logger klog.Logger, entity framework.QueuedEntityInfo, event string)
	// get returns the entity matching entity inside the activeQ.
	get(entityLookup framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool)

	listInFlightEvents() []interface{}
	listInFlightPods() []*v1.Pod
	clusterEventsForPod(logger klog.Logger, pInfo *framework.QueuedPodInfo) ([]*clusterEvent, error)
	addEventsIfPodInFlight(oldPod, newPod *v1.Pod, events []fwk.ClusterEvent) bool
	addEventIfAnyInFlight(oldObj, newObj interface{}, event fwk.ClusterEvent) bool

	isLastPoppedEntity(entityLookup framework.QueuedEntityInfo) bool
	clearPoppedEntity()

	schedulingCycle() int64
	done(pod types.UID)
	close()
	broadcast()
}

// unlockedActiveQueuer defines activeQ methods that are not protected by the lock itself.
// underLock() method should be used to protect these methods.
type unlockedActiveQueuer interface {
	unlockedActiveQueueReader
	// update updates the pod in activeQ if oldEntity is already in the queue and the pod is present there.
	// It returns new pod info if updated, nil otherwise.
	update(newPod *v1.Pod, oldEntity framework.QueuedEntityInfo) *framework.QueuedPodInfo
	// addEventsIfPodInFlight adds events to inFlightEvents if the newPod is in inFlightPods.
	// It returns true if pushed the event to the inFlightEvents.
	addEventsIfPodInFlight(oldPod, newPod *v1.Pod, events []fwk.ClusterEvent) bool
}

// unlockedActiveQueueReader defines activeQ read-only methods that are not protected by the lock itself.
// underLock() or underRLock() method should be used to protect these methods.
type unlockedActiveQueueReader interface {
	// get returns the entity matching entity inside the activeQ.
	// Returns false if the entity doesn't exist in the queue.
	// This method should be called in activeQueue.underLock() or activeQueue.underRLock().
	get(entityLookup framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool)
}

// unlockedActiveQueue defines activeQ methods that are not protected by the lock itself.
// activeQueue.underLock() or activeQueue.underRLock() method should be used to protect these methods.
type unlockedActiveQueue struct {
	queue           *heap.Heap[framework.QueuedEntityInfo]
	inFlightPods    map[types.UID]*list.Element
	inFlightEvents  *list.List
	metricsRecorder MetricAsyncRecorder
}

func newUnlockedActiveQueue(queue *heap.Heap[framework.QueuedEntityInfo], inFlightPods map[types.UID]*list.Element, inFlightEvents *list.List, metricsRecorder MetricAsyncRecorder) *unlockedActiveQueue {
	return &unlockedActiveQueue{
		queue:           queue,
		inFlightPods:    inFlightPods,
		inFlightEvents:  inFlightEvents,
		metricsRecorder: metricsRecorder,
	}
}

// update updates the pod in activeQ if oldEntity is already in the queue and the pod is present there.
// It returns new pod info if updated, nil otherwise.
func (uaq *unlockedActiveQueue) update(newPod *v1.Pod, oldEntity framework.QueuedEntityInfo) *framework.QueuedPodInfo {
	if entity, exists := uaq.queue.Get(oldEntity); exists {
		podInfo, err := entity.Update(newPod)
		if err != nil {
			return nil
		}
		uaq.queue.AddOrUpdate(entity)
		return podInfo
	}
	return nil
}

// addEventsIfPodInFlight adds events to inFlightEvents if the newPod is in inFlightPods.
// It returns true if pushed the event to the inFlightEvents.
func (uaq *unlockedActiveQueue) addEventsIfPodInFlight(oldPod, newPod *v1.Pod, events []fwk.ClusterEvent) bool {
	_, ok := uaq.inFlightPods[newPod.UID]
	if ok {
		for _, event := range events {
			uaq.metricsRecorder.ObserveInFlightEventsAsync(event.Label(), 1, false)
			uaq.inFlightEvents.PushBack(&clusterEvent{
				event:  event,
				oldObj: oldPod,
				newObj: newPod,
			})
		}
	}
	return ok
}

// get returns the entity matching entity inside the activeQ.
// Returns false if the entity doesn't exist in the queue.
// This method should be called in activeQueue.underLock() or activeQueue.underRLock().
func (uaq *unlockedActiveQueue) get(entityLookup framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool) {
	return uaq.queue.Get(entityLookup)
}

// backoffQPopper defines method that is used to pop from the backoffQ when the activeQ is empty.
type backoffQPopper interface {
	// popBackoff pops the item from the podBackoffQ.
	popBackoff() (framework.QueuedEntityInfo, error)
	// len returns length of the podBackoffQ queue.
	lenBackoff() int
}

// activeQueue implements activeQueuer. All of the fields have to be protected using the lock.
type activeQueue struct {
	// lock synchronizes all operations related to activeQ.
	// It protects activeQ, inFlightPods, inFlightEvents, schedulingCycle and closed fields.
	// Caution: DO NOT take "SchedulingQueue.lock" after taking "lock".
	// You should always take "SchedulingQueue.lock" first, otherwise the queue could end up in deadlock.
	// "lock" should not be taken after taking "backoffQueue.lock" or "nominator.nLock".
	// Correct locking order is: SchedulingQueue.lock > lock > backoffQueue.lock > nominator.nLock.
	lock sync.RWMutex

	// activeQ is heap structure that scheduler actively looks at to find pods to
	// schedule. Head of heap is the highest priority pod.
	queue *heap.Heap[framework.QueuedEntityInfo]

	// unlockedQueue is a wrapper of queue providing methods that are not locked themselves
	// and can be used in the underLock() or underRLock().
	unlockedQueue *unlockedActiveQueue

	// cond is a condition that is notified when the pod is added to activeQ.
	// When SchedulerPopFromBackoffQ feature is enabled,
	// condition is also notified when the pod is added to backoffQ.
	// It is used with lock.
	cond sync.Cond

	// inFlightPods holds the UID of all pods which have been popped out for which Done
	// hasn't been called yet - in other words, all pods that are currently being
	// processed (being scheduled, in permit, or in the binding cycle).
	//
	// The values in the map are the entry of each pod in the inFlightEvents list.
	// The value of that entry is the *v1.Pod at the time that scheduling of that
	// pod started, which can be useful for logging or debugging.
	inFlightPods map[types.UID]*list.Element

	// inFlightEvents holds the events received by the scheduling queue
	// (entry value is clusterEvent) together with in-flight pods (entry
	// value is *v1.Pod). Entries get added at the end while the mutex is
	// locked, so they get serialized.
	//
	// The pod entries are added in Pop and used to track which events
	// occurred after the pod scheduling attempt for that pod started.
	// They get removed when the scheduling attempt is done, at which
	// point all events that occurred in the meantime are processed.
	//
	// After removal of a pod, events at the start of the list are no
	// longer needed because all of the other in-flight pods started
	// later. Those events can be removed.
	inFlightEvents *list.List

	// schedCycle represents sequence number of scheduling cycle and is incremented
	// when a pod is popped.
	schedCycle int64

	// closed indicates that the queue is closed.
	// It is mainly used to let Pop() exit its control loop while waiting for an item.
	closed bool

	metricsRecorder MetricAsyncRecorder

	// backoffQPopper is used to pop from backoffQ when activeQ is empty.
	// It is non-nil only when SchedulerPopFromBackoffQ feature is enabled.
	backoffQPopper backoffQPopper

	// lastPoppedEntityKey is the latest entity that was popped from the activeQ.
	// It's used to check if the scheduling cycle is pending for a PodGroup matching the newly added pod.
	// It should be cleared when the entity is re-added to the scheduling queue.
	lastPoppedEntityKey string
}

func newActiveQueue(queue *heap.Heap[framework.QueuedEntityInfo], metricRecorder MetricAsyncRecorder, backoffQPopper backoffQPopper) *activeQueue {
	aq := &activeQueue{
		queue:           queue,
		inFlightPods:    make(map[types.UID]*list.Element),
		inFlightEvents:  list.New(),
		metricsRecorder: metricRecorder,
		backoffQPopper:  backoffQPopper,
	}
	aq.cond.L = &aq.lock
	aq.unlockedQueue = newUnlockedActiveQueue(queue, aq.inFlightPods, aq.inFlightEvents, metricRecorder)

	return aq
}

// underLock runs the fn function under the lock.Lock.
// fn can run unlockedActiveQueuer methods but should NOT run any other activeQueue method,
// as it would end up in deadlock.
func (aq *activeQueue) underLock(fn func(unlockedActiveQ unlockedActiveQueuer)) {
	aq.lock.Lock()
	defer aq.lock.Unlock()
	fn(aq.unlockedQueue)
}

// underLock runs the fn function under the lock.RLock.
// fn can run unlockedActiveQueueReader methods but should NOT run any other activeQueue method,
// as it would end up in deadlock.
func (aq *activeQueue) underRLock(fn func(unlockedActiveQ unlockedActiveQueueReader)) {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	fn(aq.unlockedQueue)
}

// delete removes the entity from the activeQ.
// It returns the entity if it was removed, nil otherwise.
func (aq *activeQueue) delete(entityLookup framework.QueuedEntityInfo) framework.QueuedEntityInfo {
	aq.lock.Lock()
	defer aq.lock.Unlock()

	return aq.queue.Delete(entityLookup)
}

// unlockedMoveEntityToInFlight moves the entity to the in-flight state.
// This method should be called under the lock.
func (aq *activeQueue) unlockedMoveEntityToInFlight(logger klog.Logger, entity framework.QueuedEntityInfo) error {
	entity.IncAttempts()
	var podsToDiscard []*framework.QueuedPodInfo
	entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		// If the pod is already in the map, we shouldn't overwrite the inFlightPods otherwise it'd lead to a memory leak.
		// https://github.com/kubernetes/kubernetes/pull/127016
		if _, ok := aq.inFlightPods[pInfo.Pod.UID]; ok {
			podsToDiscard = append(podsToDiscard, pInfo)
			return true
		}

		aq.metricsRecorder.ObserveInFlightEventsAsync(metrics.PodPoppedInFlightEvent, 1, false)
		aq.inFlightPods[pInfo.Pod.UID] = aq.inFlightEvents.PushBack(pInfo.Pod)
		return true
	})
	if len(podsToDiscard) > 0 {
		switch specificEntity := entity.(type) {
		case *framework.QueuedPodInfo:
			return fmt.Errorf("the same pod is tracked in multiple places in the scheduler: %s", klog.KObj(podsToDiscard[0]))
		case *framework.QueuedPodGroupInfo:
			for _, pInfo := range podsToDiscard {
				// Pod should always exist in the pod group, ignoring the return value.
				_ = specificEntity.RemovePod(pInfo.Pod)
				logger.Error(nil, "Discarding the popped pod group member. It's tracked in multiple places in the scheduler", "pod", klog.KObj(pInfo), "podGroup", klog.KObj(specificEntity))
			}
			if len(specificEntity.QueuedPodInfos) == 0 {
				return fmt.Errorf("all pods from the pod group are tracked in multiple places in the scheduler: %s", klog.KObj(specificEntity))
			}
		default:
			return fmt.Errorf("unexpected entity type: %T", entity)
		}
	}
	aq.schedCycle++

	// Update metrics for unschedulable plugins.
	entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		for plugin := range pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins) {
			metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Dec()
		}
		return true
	})
	return nil
}

// pop removes the head of the queue and returns it.
// It blocks if the queue is empty and waits until a new entity is added to the queue.
// It increments scheduling cycle when a pod is popped.
func (aq *activeQueue) pop(logger klog.Logger) (framework.QueuedEntityInfo, error) {
	aq.lock.Lock()
	defer aq.lock.Unlock()

	return aq.unlockedPop(logger)
}

func (aq *activeQueue) unlockedPop(logger klog.Logger) (framework.QueuedEntityInfo, error) {
	var entity framework.QueuedEntityInfo
	for aq.queue.Len() == 0 {
		// backoffQPopper is non-nil only if SchedulerPopFromBackoffQ feature is enabled.
		// In case of non-empty backoffQ, try popping from there.
		if aq.backoffQPopper != nil && aq.backoffQPopper.lenBackoff() != 0 {
			break
		}
		// When the queue is empty, invocation of Pop() is blocked until new entity is enqueued.
		// When Close() is called, the p.closed is set and the condition is broadcast,
		// which causes this loop to continue and return from the Pop().
		if aq.closed {
			logger.V(2).Info("Scheduling queue is closed")
			return nil, nil
		}
		aq.cond.Wait()
	}
	entity, err := aq.queue.Pop()
	if err != nil {
		if aq.backoffQPopper == nil {
			return nil, err
		}
		// Try to pop from backoffQ when activeQ is empty.
		entity, err = aq.backoffQPopper.popBackoff()
		if err != nil {
			return nil, err
		}
		metrics.SchedulerQueueIncomingPods.WithLabelValues("active", framework.PopFromBackoffQ).Add(float64(entity.Size()))
	}
	err = aq.unlockedMoveEntityToInFlight(logger, entity)
	if err != nil {
		// Just report it as an error, but no need to stop the scheduler
		// because it likely doesn't cause any visible issues from the scheduling perspective.
		utilruntime.HandleErrorWithLogger(logger, err, "Discarding the popped entity", "type", entity.Type(), "entity", klog.KObj(entity))
		// Just ignore/discard this duplicated entity and try to pop the next one.
		return aq.unlockedPop(logger)
	}
	aq.lastPoppedEntityKey = queuedEntityKeyFunc(entity)

	return entity, nil
}

// isLastPoppedEntity checks if the last popped entity is the given entity.
func (aq *activeQueue) isLastPoppedEntity(entityLookup framework.QueuedEntityInfo) bool {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	return aq.lastPoppedEntityKey == queuedEntityKeyFunc(entityLookup)
}

// clearPoppedEntity clears the last popped entity.
func (aq *activeQueue) clearPoppedEntity() {
	aq.lock.Lock()
	defer aq.lock.Unlock()
	aq.lastPoppedEntityKey = ""
}

// list returns all pods that are in the queue.
func (aq *activeQueue) list() []*v1.Pod {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	var result []*v1.Pod
	for _, entity := range aq.queue.List() {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			result = append(result, pInfo.Pod)
			return true
		})
	}
	return result
}

// len returns length of the queue.
func (aq *activeQueue) len() int {
	return aq.queue.Len()
}

// has inform if entity exists in the queue.
func (aq *activeQueue) has(entityLookup framework.QueuedEntityInfo) bool {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	return aq.queue.Has(entityLookup)
}

// add adds a new entity to the activeQ.
// The event should show which event triggered this addition and is used for the metric recording.
// Note: it does not signal the pop() method to wake up,
// so the caller is responsible for calling broadcast() after executing this method.
func (aq *activeQueue) add(logger klog.Logger, entity framework.QueuedEntityInfo, event string) {
	aq.lock.Lock()
	defer aq.lock.Unlock()

	aq.queue.AddOrUpdate(entity)
	metrics.SchedulerQueueIncomingPods.WithLabelValues("active", event).Add(float64(entity.Size()))
	logger.V(5).Info("Entity moved to an internal scheduling queue", "type", entity.Type(), "entity", klog.KObj(entity), "event", event, "queue", activeQ)
}

// get returns the entity matching entity inside the activeQ.
func (aq *activeQueue) get(entityLookup framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool) {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	return aq.unlockedQueue.get(entityLookup)
}

// listInFlightEvents returns all inFlightEvents.
func (aq *activeQueue) listInFlightEvents() []interface{} {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	var values []interface{}
	for event := aq.inFlightEvents.Front(); event != nil; event = event.Next() {
		values = append(values, event.Value)
	}
	return values
}

// listInFlightPods returns all inFlightPods.
func (aq *activeQueue) listInFlightPods() []*v1.Pod {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	var pods []*v1.Pod
	for _, obj := range aq.inFlightPods {
		pods = append(pods, obj.Value.(*v1.Pod))
	}
	return pods
}

// clusterEventsForPod gets all cluster events that have happened during pod for entity is being scheduled.
func (aq *activeQueue) clusterEventsForPod(logger klog.Logger, pInfo *framework.QueuedPodInfo) ([]*clusterEvent, error) {
	aq.lock.RLock()
	defer aq.lock.RUnlock()

	logger.V(5).Info("Checking events for in-flight pod", "pod", klog.KObj(pInfo.Pod), "unschedulablePlugins", pInfo.UnschedulablePlugins, "inFlightEventsSize", aq.inFlightEvents.Len(), "inFlightPodsSize", len(aq.inFlightPods))

	// AddUnschedulablePodIfNotPresent is called with the Pod at the end of scheduling or binding.
	// So, given entity should have been Pop()ed before,
	// we can assume entity must be recorded in inFlightPods and thus inFlightEvents.
	inFlightPod, ok := aq.inFlightPods[pInfo.Pod.UID]
	if !ok {
		return nil, fmt.Errorf("in flight entity isn't found in the scheduling queue. If you see this error log, it's likely a bug in the scheduler")
	}

	var events []*clusterEvent
	for event := inFlightPod.Next(); event != nil; event = event.Next() {
		e, ok := event.Value.(*clusterEvent)
		if !ok {
			// Must be another in-flight Pod (*v1.Pod). Can be ignored.
			continue
		}
		events = append(events, e)
	}
	return events, nil
}

// addEventsIfPodInFlight adds events to inFlightEvents if the newPod is in inFlightPods.
// It returns true if pushed the event to the inFlightEvents.
func (aq *activeQueue) addEventsIfPodInFlight(oldPod, newPod *v1.Pod, events []fwk.ClusterEvent) bool {
	aq.lock.Lock()
	defer aq.lock.Unlock()

	return aq.unlockedQueue.addEventsIfPodInFlight(oldPod, newPod, events)
}

// addEventIfAnyInFlight adds clusterEvent to inFlightEvents if any pod is in inFlightPods.
// It returns true if pushed the event to the inFlightEvents.
func (aq *activeQueue) addEventIfAnyInFlight(oldObj, newObj interface{}, event fwk.ClusterEvent) bool {
	aq.lock.Lock()
	defer aq.lock.Unlock()

	if len(aq.inFlightPods) != 0 {
		aq.metricsRecorder.ObserveInFlightEventsAsync(event.Label(), 1, false)
		aq.inFlightEvents.PushBack(&clusterEvent{
			event:  event,
			oldObj: oldObj,
			newObj: newObj,
		})
		return true
	}
	return false
}

func (aq *activeQueue) schedulingCycle() int64 {
	aq.lock.RLock()
	defer aq.lock.RUnlock()
	return aq.schedCycle
}

// done must be called for pod returned by Pop. This allows the queue to
// keep track of which pods are currently being processed.
func (aq *activeQueue) done(podUID types.UID) {
	aq.lock.Lock()
	defer aq.lock.Unlock()

	aq.unlockedDone(podUID)
}

// unlockedDone is used by the activeQueue internally and doesn't take the lock itself.
// It assumes the lock is already taken outside before the method is called.
func (aq *activeQueue) unlockedDone(podUID types.UID) {
	inFlightEntity, ok := aq.inFlightPods[podUID]
	if !ok {
		// This entity is already done()ed.
		return
	}
	delete(aq.inFlightPods, podUID)

	// Remove the entity from the list.
	aq.inFlightEvents.Remove(inFlightEntity)

	aggrMetricsCounter := map[string]int{}
	// Remove events which are only referred to by this Pod
	// so that the inFlightEvents list doesn't grow infinitely.
	// If the pod was at the head of the list, then all
	// events between it and the next pod are no longer needed
	// and can be removed.
	for {
		e := aq.inFlightEvents.Front()
		if e == nil {
			// Empty list.
			break
		}
		ev, ok := e.Value.(*clusterEvent)
		if !ok {
			// A pod, must stop pruning.
			break
		}
		aq.inFlightEvents.Remove(e)
		aggrMetricsCounter[ev.event.Label()]--
	}

	for evLabel, count := range aggrMetricsCounter {
		aq.metricsRecorder.ObserveInFlightEventsAsync(evLabel, float64(count), false)
	}

	aq.metricsRecorder.ObserveInFlightEventsAsync(metrics.PodPoppedInFlightEvent, -1,
		// If it's the last Pod in inFlightPods, we should force-flush the metrics.
		// Otherwise, especially in small clusters, which don't get a new Pod frequently,
		// the metrics might not be flushed for a long time.
		len(aq.inFlightPods) == 0)
}

// close closes the activeQueue.
func (aq *activeQueue) close() {
	aq.lock.Lock()
	defer aq.lock.Unlock()
	// We should call done() for all in-flight pods to clean up the inFlightEvents metrics.
	// It's safe even if the binding cycle running asynchronously calls done() afterwards
	// done() will just be a no-op.
	for pod := range aq.inFlightPods {
		aq.unlockedDone(pod)
	}
	aq.closed = true
}

// broadcast notifies the pop() operation that new pod(s) was added to the activeQueue.
func (aq *activeQueue) broadcast() {
	aq.cond.Broadcast()
}
