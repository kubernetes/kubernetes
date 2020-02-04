/*
Copyright 2017 The Kubernetes Authors.

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

// This file contains structures that implement scheduling queue types.
// Scheduling queues hold pods waiting to be scheduled. This file implements a
// priority queue which has two sub queues. One sub-queue holds pods that are
// being considered for scheduling. This is called activeQ. Another queue holds
// pods that are already tried and are determined to be unschedulable. The latter
// is called unschedulableQ.

package queue

import (
	"fmt"
	"reflect"
	"sync"
	"time"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ktypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

var (
	queueClosed = "scheduling queue is closed"
)

// If the pod stays in unschedulableQ longer than the unschedulableQTimeInterval,
// the pod will be moved from unschedulableQ to activeQ.
const unschedulableQTimeInterval = 60 * time.Second

// SchedulingQueue is an interface for a queue to store pods waiting to be scheduled.
// The interface follows a pattern similar to cache.FIFO and cache.Heap and
// makes it easy to use those data structures as a SchedulingQueue.
type SchedulingQueue interface {
	Add(pod *v1.Pod) error
	AddIfNotPresent(pod *v1.Pod) error
	// AddUnschedulableIfNotPresent adds an unschedulable pod back to scheduling queue.
	// The podSchedulingCycle represents the current scheduling cycle number which can be
	// returned by calling SchedulingCycle().
	AddUnschedulableIfNotPresent(pod *v1.Pod, podSchedulingCycle int64) error
	// SchedulingCycle returns the current number of scheduling cycle which is
	// cached by scheduling queue. Normally, incrementing this number whenever
	// a pod is popped (e.g. called Pop()) is enough.
	SchedulingCycle() int64
	// Pop removes the head of the queue and returns it. It blocks if the
	// queue is empty and waits until a new item is added to the queue.
	Pop() (*v1.Pod, error)
	Update(oldPod, newPod *v1.Pod) error
	Delete(pod *v1.Pod) error
	MoveAllToActiveQueue()
	AssignedPodAdded(pod *v1.Pod)
	AssignedPodUpdated(pod *v1.Pod)
	NominatedPodsForNode(nodeName string) []*v1.Pod
	PendingPods() []*v1.Pod
	// Close closes the SchedulingQueue so that the goroutine which is
	// waiting to pop items can exit gracefully.
	Close()
	// UpdateNominatedPodForNode adds the given pod to the nominated pod map or
	// updates it if it already exists.
	UpdateNominatedPodForNode(pod *v1.Pod, nodeName string)
	// DeleteNominatedPodIfExists deletes nominatedPod from internal cache
	DeleteNominatedPodIfExists(pod *v1.Pod)
	// NumUnschedulablePods returns the number of unschedulable pods exist in the SchedulingQueue.
	NumUnschedulablePods() int
}

// NewSchedulingQueue initializes a priority queue as a new scheduling queue.
func NewSchedulingQueue(stop <-chan struct{}, fwk framework.Framework) SchedulingQueue {
	return NewPriorityQueue(stop, fwk)
}

// NominatedNodeName returns nominated node name of a Pod.
func NominatedNodeName(pod *v1.Pod) string {
	return pod.Status.NominatedNodeName
}

// PriorityQueue implements a scheduling queue.
// The head of PriorityQueue is the highest priority pending pod. This structure
// has three sub queues. One sub-queue holds pods that are being considered for
// scheduling. This is called activeQ and is a Heap. Another queue holds
// pods that are already tried and are determined to be unschedulable. The latter
// is called unschedulableQ. The third queue holds pods that are moved from
// unschedulable queues and will be moved to active queue when backoff are completed.
type PriorityQueue struct {
	stop  <-chan struct{}
	clock util.Clock
	// podBackoff tracks backoff for pods attempting to be rescheduled
	podBackoff *PodBackoffMap

	lock sync.RWMutex
	cond sync.Cond

	// activeQ is heap structure that scheduler actively looks at to find pods to
	// schedule. Head of heap is the highest priority pod.
	activeQ *util.Heap
	// podBackoffQ is a heap ordered by backoff expiry. Pods which have completed backoff
	// are popped from this heap before the scheduler looks at activeQ
	podBackoffQ *util.Heap
	// unschedulableQ holds pods that have been tried and determined unschedulable.
	unschedulableQ *UnschedulablePodsMap
	// nominatedPods is a structures that stores pods which are nominated to run
	// on nodes.
	nominatedPods *nominatedPodMap
	// schedulingCycle represents sequence number of scheduling cycle and is incremented
	// when a pod is popped.
	schedulingCycle int64
	// moveRequestCycle caches the sequence number of scheduling cycle when we
	// received a move request. Unscheduable pods in and before this scheduling
	// cycle will be put back to activeQueue if we were trying to schedule them
	// when we received move request.
	moveRequestCycle int64

	// closed indicates that the queue is closed.
	// It is mainly used to let Pop() exit its control loop while waiting for an item.
	closed bool
}

// Making sure that PriorityQueue implements SchedulingQueue.
var _ = SchedulingQueue(&PriorityQueue{})

// newPodInfoNoTimestamp builds a PodInfo object without timestamp.
func newPodInfoNoTimestamp(pod *v1.Pod) *framework.PodInfo {
	return &framework.PodInfo{
		Pod: pod,
	}
}

// activeQComp is the function used by the activeQ heap algorithm to sort pods.
// It sorts pods based on their priority. When priorities are equal, it uses
// PodInfo.timestamp.
func activeQComp(podInfo1, podInfo2 interface{}) bool {
	pInfo1 := podInfo1.(*framework.PodInfo)
	pInfo2 := podInfo2.(*framework.PodInfo)
	prio1 := util.GetPodPriority(pInfo1.Pod)
	prio2 := util.GetPodPriority(pInfo2.Pod)
	return (prio1 > prio2) || (prio1 == prio2 && pInfo1.Timestamp.Before(pInfo2.Timestamp))
}

// NewPriorityQueue creates a PriorityQueue object.
func NewPriorityQueue(stop <-chan struct{}, fwk framework.Framework) *PriorityQueue {
	return NewPriorityQueueWithClock(stop, util.RealClock{}, fwk)
}

// NewPriorityQueueWithClock creates a PriorityQueue which uses the passed clock for time.
func NewPriorityQueueWithClock(stop <-chan struct{}, clock util.Clock, fwk framework.Framework) *PriorityQueue {
	comp := activeQComp
	if fwk != nil {
		if queueSortFunc := fwk.QueueSortFunc(); queueSortFunc != nil {
			comp = func(podInfo1, podInfo2 interface{}) bool {
				pInfo1 := podInfo1.(*framework.PodInfo)
				pInfo2 := podInfo2.(*framework.PodInfo)

				return queueSortFunc(pInfo1, pInfo2)
			}
		}
	}

	pq := &PriorityQueue{
		clock:            clock,
		stop:             stop,
		podBackoff:       NewPodBackoffMap(1*time.Second, 10*time.Second, clock),
		activeQ:          util.NewHeapWithRecorder(podInfoKeyFunc, comp, metrics.NewActivePodsRecorder()),
		unschedulableQ:   newUnschedulablePodsMap(metrics.NewUnschedulablePodsRecorder()),
		nominatedPods:    newNominatedPodMap(),
		moveRequestCycle: -1,
	}
	pq.cond.L = &pq.lock
	pq.podBackoffQ = util.NewHeapWithRecorder(podInfoKeyFunc, pq.podsCompareBackoffCompleted, metrics.NewBackoffPodsRecorder())

	pq.run()

	return pq
}

// run starts the goroutine to pump from podBackoffQ to activeQ
func (p *PriorityQueue) run() {
	go wait.Until(p.flushBackoffQCompleted, 1.0*time.Second, p.stop)
	go wait.Until(p.flushUnschedulableQLeftover, 30*time.Second, p.stop)
}

// Add adds a pod to the active queue. It should be called only when a new pod
// is added so there is no chance the pod is already in active/unschedulable/backoff queues
func (p *PriorityQueue) Add(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	pInfo := p.newPodInfo(pod)
	if err := p.activeQ.Add(pInfo); err != nil {
		klog.Errorf("Error adding pod %v/%v to the scheduling queue: %v", pod.Namespace, pod.Name, err)
		return err
	}
	if p.unschedulableQ.get(pod) != nil {
		klog.Errorf("Error: pod %v/%v is already in the unschedulable queue.", pod.Namespace, pod.Name)
		p.unschedulableQ.delete(pod)
	}
	// Delete pod from backoffQ if it is backing off
	if err := p.podBackoffQ.Delete(pInfo); err == nil {
		klog.Errorf("Error: pod %v/%v is already in the podBackoff queue.", pod.Namespace, pod.Name)
	}
	p.nominatedPods.add(pod, "")
	p.cond.Broadcast()

	return nil
}

// AddIfNotPresent adds a pod to the active queue if it is not present in any of
// the queues. If it is present in any, it doesn't do any thing.
func (p *PriorityQueue) AddIfNotPresent(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	if p.unschedulableQ.get(pod) != nil {
		return nil
	}

	pInfo := p.newPodInfo(pod)
	if _, exists, _ := p.activeQ.Get(pInfo); exists {
		return nil
	}
	if _, exists, _ := p.podBackoffQ.Get(pInfo); exists {
		return nil
	}
	err := p.activeQ.Add(pInfo)
	if err != nil {
		klog.Errorf("Error adding pod %v/%v to the scheduling queue: %v", pod.Namespace, pod.Name, err)
	} else {
		p.nominatedPods.add(pod, "")
		p.cond.Broadcast()
	}
	return err
}

// nsNameForPod returns a namespacedname for a pod
func nsNameForPod(pod *v1.Pod) ktypes.NamespacedName {
	return ktypes.NamespacedName{
		Namespace: pod.Namespace,
		Name:      pod.Name,
	}
}

// clearPodBackoff clears all backoff state for a pod (resets expiry)
func (p *PriorityQueue) clearPodBackoff(pod *v1.Pod) {
	p.podBackoff.ClearPodBackoff(nsNameForPod(pod))
}

// isPodBackingOff returns true if a pod is still waiting for its backoff timer.
// If this returns true, the pod should not be re-tried.
func (p *PriorityQueue) isPodBackingOff(pod *v1.Pod) bool {
	boTime, exists := p.podBackoff.GetBackoffTime(nsNameForPod(pod))
	if !exists {
		return false
	}
	return boTime.After(p.clock.Now())
}

// backoffPod checks if pod is currently undergoing backoff. If it is not it updates the backoff
// timeout otherwise it does nothing.
func (p *PriorityQueue) backoffPod(pod *v1.Pod) {
	p.podBackoff.CleanupPodsCompletesBackingoff()

	podID := nsNameForPod(pod)
	boTime, found := p.podBackoff.GetBackoffTime(podID)
	if !found || boTime.Before(p.clock.Now()) {
		p.podBackoff.BackoffPod(podID)
	}
}

// SchedulingCycle returns current scheduling cycle.
func (p *PriorityQueue) SchedulingCycle() int64 {
	p.lock.RLock()
	defer p.lock.RUnlock()
	return p.schedulingCycle
}

// AddUnschedulableIfNotPresent inserts a pod that cannot be scheduled into
// the queue, unless it is already in the queue. Normally, PriorityQueue puts
// unschedulable pods in `unschedulableQ`. But if there has been a recent move
// request, then the pod is put in `podBackoffQ`.
func (p *PriorityQueue) AddUnschedulableIfNotPresent(pod *v1.Pod, podSchedulingCycle int64) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	if p.unschedulableQ.get(pod) != nil {
		return fmt.Errorf("pod is already present in unschedulableQ")
	}

	pInfo := p.newPodInfo(pod)
	if _, exists, _ := p.activeQ.Get(pInfo); exists {
		return fmt.Errorf("pod is already present in the activeQ")
	}
	if _, exists, _ := p.podBackoffQ.Get(pInfo); exists {
		return fmt.Errorf("pod is already present in the backoffQ")
	}

	// Every unschedulable pod is subject to backoff timers.
	p.backoffPod(pod)

	// If a move request has been received, move it to the BackoffQ, otherwise move
	// it to unschedulableQ.
	if p.moveRequestCycle >= podSchedulingCycle {
		if err := p.podBackoffQ.Add(pInfo); err != nil {
			return fmt.Errorf("error adding pod %v to the backoff queue: %v", pod.Name, err)
		}
	} else {
		p.unschedulableQ.addOrUpdate(pInfo)
	}

	p.nominatedPods.add(pod, "")
	return nil

}

// flushBackoffQCompleted Moves all pods from backoffQ which have completed backoff in to activeQ
func (p *PriorityQueue) flushBackoffQCompleted() {
	p.lock.Lock()
	defer p.lock.Unlock()

	for {
		rawPodInfo := p.podBackoffQ.Peek()
		if rawPodInfo == nil {
			return
		}
		pod := rawPodInfo.(*framework.PodInfo).Pod
		boTime, found := p.podBackoff.GetBackoffTime(nsNameForPod(pod))
		if !found {
			klog.Errorf("Unable to find backoff value for pod %v in backoffQ", nsNameForPod(pod))
			p.podBackoffQ.Pop()
			p.activeQ.Add(rawPodInfo)
			defer p.cond.Broadcast()
			continue
		}

		if boTime.After(p.clock.Now()) {
			return
		}
		_, err := p.podBackoffQ.Pop()
		if err != nil {
			klog.Errorf("Unable to pop pod %v from backoffQ despite backoff completion.", nsNameForPod(pod))
			return
		}
		p.activeQ.Add(rawPodInfo)
		defer p.cond.Broadcast()
	}
}

// flushUnschedulableQLeftover moves pod which stays in unschedulableQ longer than the durationStayUnschedulableQ
// to activeQ.
func (p *PriorityQueue) flushUnschedulableQLeftover() {
	p.lock.Lock()
	defer p.lock.Unlock()

	var podsToMove []*framework.PodInfo
	currentTime := p.clock.Now()
	for _, pInfo := range p.unschedulableQ.podInfoMap {
		lastScheduleTime := pInfo.Timestamp
		if currentTime.Sub(lastScheduleTime) > unschedulableQTimeInterval {
			podsToMove = append(podsToMove, pInfo)
		}
	}

	if len(podsToMove) > 0 {
		p.movePodsToActiveQueue(podsToMove)
	}
}

// Pop removes the head of the active queue and returns it. It blocks if the
// activeQ is empty and waits until a new item is added to the queue. It
// increments scheduling cycle when a pod is popped.
func (p *PriorityQueue) Pop() (*v1.Pod, error) {
	p.lock.Lock()
	defer p.lock.Unlock()
	for p.activeQ.Len() == 0 {
		// When the queue is empty, invocation of Pop() is blocked until new item is enqueued.
		// When Close() is called, the p.closed is set and the condition is broadcast,
		// which causes this loop to continue and return from the Pop().
		if p.closed {
			return nil, fmt.Errorf(queueClosed)
		}
		p.cond.Wait()
	}
	obj, err := p.activeQ.Pop()
	if err != nil {
		return nil, err
	}
	pInfo := obj.(*framework.PodInfo)
	p.schedulingCycle++
	return pInfo.Pod, err
}

// isPodUpdated checks if the pod is updated in a way that it may have become
// schedulable. It drops status of the pod and compares it with old version.
func isPodUpdated(oldPod, newPod *v1.Pod) bool {
	strip := func(pod *v1.Pod) *v1.Pod {
		p := pod.DeepCopy()
		p.ResourceVersion = ""
		p.Generation = 0
		p.Status = v1.PodStatus{}
		return p
	}
	return !reflect.DeepEqual(strip(oldPod), strip(newPod))
}

// Update updates a pod in the active or backoff queue if present. Otherwise, it removes
// the item from the unschedulable queue if pod is updated in a way that it may
// become schedulable and adds the updated one to the active queue.
// If pod is not present in any of the queues, it is added to the active queue.
func (p *PriorityQueue) Update(oldPod, newPod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()

	if oldPod != nil {
		oldPodInfo := newPodInfoNoTimestamp(oldPod)
		// If the pod is already in the active queue, just update it there.
		if oldPodInfo, exists, _ := p.activeQ.Get(oldPodInfo); exists {
			p.nominatedPods.update(oldPod, newPod)
			newPodInfo := newPodInfoNoTimestamp(newPod)
			newPodInfo.Timestamp = oldPodInfo.(*framework.PodInfo).Timestamp
			err := p.activeQ.Update(newPodInfo)
			return err
		}

		// If the pod is in the backoff queue, update it there.
		if oldPodInfo, exists, _ := p.podBackoffQ.Get(oldPodInfo); exists {
			p.nominatedPods.update(oldPod, newPod)
			p.podBackoffQ.Delete(newPodInfoNoTimestamp(oldPod))
			newPodInfo := newPodInfoNoTimestamp(newPod)
			newPodInfo.Timestamp = oldPodInfo.(*framework.PodInfo).Timestamp
			err := p.activeQ.Add(newPodInfo)
			if err == nil {
				p.cond.Broadcast()
			}
			return err
		}
	}

	// If the pod is in the unschedulable queue, updating it may make it schedulable.
	if usPodInfo := p.unschedulableQ.get(newPod); usPodInfo != nil {
		p.nominatedPods.update(oldPod, newPod)
		newPodInfo := newPodInfoNoTimestamp(newPod)
		newPodInfo.Timestamp = usPodInfo.Timestamp
		if isPodUpdated(oldPod, newPod) {
			// If the pod is updated reset backoff
			p.clearPodBackoff(newPod)
			p.unschedulableQ.delete(usPodInfo.Pod)
			err := p.activeQ.Add(newPodInfo)
			if err == nil {
				p.cond.Broadcast()
			}
			return err
		}
		// Pod is already in unschedulable queue and hasnt updated, no need to backoff again
		p.unschedulableQ.addOrUpdate(newPodInfo)
		return nil
	}
	// If pod is not in any of the queues, we put it in the active queue.
	err := p.activeQ.Add(p.newPodInfo(newPod))
	if err == nil {
		p.nominatedPods.add(newPod, "")
		p.cond.Broadcast()
	}
	return err
}

// Delete deletes the item from either of the two queues. It assumes the pod is
// only in one queue.
func (p *PriorityQueue) Delete(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.nominatedPods.delete(pod)
	err := p.activeQ.Delete(newPodInfoNoTimestamp(pod))
	if err != nil { // The item was probably not found in the activeQ.
		p.clearPodBackoff(pod)
		p.podBackoffQ.Delete(newPodInfoNoTimestamp(pod))
		p.unschedulableQ.delete(pod)
	}
	return nil
}

// AssignedPodAdded is called when a bound pod is added. Creation of this pod
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodAdded(pod *v1.Pod) {
	p.lock.Lock()
	p.movePodsToActiveQueue(p.getUnschedulablePodsWithMatchingAffinityTerm(pod))
	p.lock.Unlock()
}

// AssignedPodUpdated is called when a bound pod is updated. Change of labels
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodUpdated(pod *v1.Pod) {
	p.lock.Lock()
	p.movePodsToActiveQueue(p.getUnschedulablePodsWithMatchingAffinityTerm(pod))
	p.lock.Unlock()
}

// MoveAllToActiveQueue moves all pods from unschedulableQ to activeQ. This
// function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an item, it receives it after all the pods are in the
// queue and the head is the highest priority pod.
func (p *PriorityQueue) MoveAllToActiveQueue() {
	p.lock.Lock()
	defer p.lock.Unlock()

	// There is a chance of errors when adding pods to other queues,
	// we make a temporary slice to store the pods,
	// since the probability is low, we set its len to 0
	addErrorPods := make([]*framework.PodInfo, 0)

	for _, pInfo := range p.unschedulableQ.podInfoMap {
		pod := pInfo.Pod
		if p.isPodBackingOff(pod) {
			if err := p.podBackoffQ.Add(pInfo); err != nil {
				klog.Errorf("Error adding pod %v to the backoff queue: %v", pod.Name, err)
				addErrorPods = append(addErrorPods, pInfo)
			}
		} else {
			if err := p.activeQ.Add(pInfo); err != nil {
				klog.Errorf("Error adding pod %v to the scheduling queue: %v", pod.Name, err)
				addErrorPods = append(addErrorPods, pInfo)
			}
		}
	}
	p.unschedulableQ.clear()
	// Adding pods that we could not move to Active queue or Backoff queue back to the Unschedulable queue
	for _, podInfo := range addErrorPods {
		p.unschedulableQ.addOrUpdate(podInfo)
	}
	p.moveRequestCycle = p.schedulingCycle
	p.cond.Broadcast()
}

// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) movePodsToActiveQueue(podInfoList []*framework.PodInfo) {
	for _, pInfo := range podInfoList {
		pod := pInfo.Pod
		if p.isPodBackingOff(pod) {
			if err := p.podBackoffQ.Add(pInfo); err != nil {
				klog.Errorf("Error adding pod %v to the backoff queue: %v", pod.Name, err)
			} else {
				p.unschedulableQ.delete(pod)
			}
		} else {
			if err := p.activeQ.Add(pInfo); err != nil {
				klog.Errorf("Error adding pod %v to the scheduling queue: %v", pod.Name, err)
			} else {
				p.unschedulableQ.delete(pod)
			}
		}
	}
	p.moveRequestCycle = p.schedulingCycle
	p.cond.Broadcast()
}

// getUnschedulablePodsWithMatchingAffinityTerm returns unschedulable pods which have
// any affinity term that matches "pod".
// NOTE: this function assumes lock has been acquired in caller.
func (p *PriorityQueue) getUnschedulablePodsWithMatchingAffinityTerm(pod *v1.Pod) []*framework.PodInfo {
	var podsToMove []*framework.PodInfo
	for _, pInfo := range p.unschedulableQ.podInfoMap {
		up := pInfo.Pod
		affinity := up.Spec.Affinity
		if affinity != nil && affinity.PodAffinity != nil {
			terms := predicates.GetPodAffinityTerms(affinity.PodAffinity)
			for _, term := range terms {
				namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(up, &term)
				selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
				if err != nil {
					klog.Errorf("Error getting label selectors for pod: %v.", up.Name)
				}
				if priorityutil.PodMatchesTermsNamespaceAndSelector(pod, namespaces, selector) {
					podsToMove = append(podsToMove, pInfo)
					break
				}
			}
		}
	}
	return podsToMove
}

// NominatedPodsForNode returns pods that are nominated to run on the given node,
// but they are waiting for other pods to be removed from the node before they
// can be actually scheduled.
func (p *PriorityQueue) NominatedPodsForNode(nodeName string) []*v1.Pod {
	p.lock.RLock()
	defer p.lock.RUnlock()
	return p.nominatedPods.podsForNode(nodeName)
}

// PendingPods returns all the pending pods in the queue. This function is
// used for debugging purposes in the scheduler cache dumper and comparer.
func (p *PriorityQueue) PendingPods() []*v1.Pod {
	p.lock.RLock()
	defer p.lock.RUnlock()
	result := []*v1.Pod{}
	for _, pInfo := range p.activeQ.List() {
		result = append(result, pInfo.(*framework.PodInfo).Pod)
	}
	for _, pInfo := range p.podBackoffQ.List() {
		result = append(result, pInfo.(*framework.PodInfo).Pod)
	}
	for _, pInfo := range p.unschedulableQ.podInfoMap {
		result = append(result, pInfo.Pod)
	}
	return result
}

// Close closes the priority queue.
func (p *PriorityQueue) Close() {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.closed = true
	p.cond.Broadcast()
}

// DeleteNominatedPodIfExists deletes pod nominatedPods.
func (p *PriorityQueue) DeleteNominatedPodIfExists(pod *v1.Pod) {
	p.lock.Lock()
	p.nominatedPods.delete(pod)
	p.lock.Unlock()
}

// UpdateNominatedPodForNode adds a pod to the nominated pods of the given node.
// This is called during the preemption process after a node is nominated to run
// the pod. We update the structure before sending a request to update the pod
// object to avoid races with the following scheduling cycles.
func (p *PriorityQueue) UpdateNominatedPodForNode(pod *v1.Pod, nodeName string) {
	p.lock.Lock()
	p.nominatedPods.add(pod, nodeName)
	p.lock.Unlock()
}

func (p *PriorityQueue) podsCompareBackoffCompleted(podInfo1, podInfo2 interface{}) bool {
	pInfo1 := podInfo1.(*framework.PodInfo)
	pInfo2 := podInfo2.(*framework.PodInfo)
	bo1, _ := p.podBackoff.GetBackoffTime(nsNameForPod(pInfo1.Pod))
	bo2, _ := p.podBackoff.GetBackoffTime(nsNameForPod(pInfo2.Pod))
	return bo1.Before(bo2)
}

// NumUnschedulablePods returns the number of unschedulable pods exist in the SchedulingQueue.
func (p *PriorityQueue) NumUnschedulablePods() int {
	p.lock.RLock()
	defer p.lock.RUnlock()
	return len(p.unschedulableQ.podInfoMap)
}

// newPodInfo builds a PodInfo object.
func (p *PriorityQueue) newPodInfo(pod *v1.Pod) *framework.PodInfo {
	if p.clock == nil {
		return &framework.PodInfo{
			Pod: pod,
		}
	}

	return &framework.PodInfo{
		Pod:       pod,
		Timestamp: p.clock.Now(),
	}
}

// UnschedulablePodsMap holds pods that cannot be scheduled. This data structure
// is used to implement unschedulableQ.
type UnschedulablePodsMap struct {
	// podInfoMap is a map key by a pod's full-name and the value is a pointer to the PodInfo.
	podInfoMap map[string]*framework.PodInfo
	keyFunc    func(*v1.Pod) string
	// metricRecorder updates the counter when elements of an unschedulablePodsMap
	// get added or removed, and it does nothing if it's nil
	metricRecorder metrics.MetricRecorder
}

// Add adds a pod to the unschedulable podInfoMap.
func (u *UnschedulablePodsMap) addOrUpdate(pInfo *framework.PodInfo) {
	podID := u.keyFunc(pInfo.Pod)
	if _, exists := u.podInfoMap[podID]; !exists && u.metricRecorder != nil {
		u.metricRecorder.Inc()
	}
	u.podInfoMap[podID] = pInfo
}

// Delete deletes a pod from the unschedulable podInfoMap.
func (u *UnschedulablePodsMap) delete(pod *v1.Pod) {
	podID := u.keyFunc(pod)
	if _, exists := u.podInfoMap[podID]; exists && u.metricRecorder != nil {
		u.metricRecorder.Dec()
	}
	delete(u.podInfoMap, podID)
}

// Get returns the PodInfo if a pod with the same key as the key of the given "pod"
// is found in the map. It returns nil otherwise.
func (u *UnschedulablePodsMap) get(pod *v1.Pod) *framework.PodInfo {
	podKey := u.keyFunc(pod)
	if pInfo, exists := u.podInfoMap[podKey]; exists {
		return pInfo
	}
	return nil
}

// Clear removes all the entries from the unschedulable podInfoMap.
func (u *UnschedulablePodsMap) clear() {
	u.podInfoMap = make(map[string]*framework.PodInfo)
	if u.metricRecorder != nil {
		u.metricRecorder.Clear()
	}
}

// newUnschedulablePodsMap initializes a new object of UnschedulablePodsMap.
func newUnschedulablePodsMap(metricRecorder metrics.MetricRecorder) *UnschedulablePodsMap {
	return &UnschedulablePodsMap{
		podInfoMap:     make(map[string]*framework.PodInfo),
		keyFunc:        util.GetPodFullName,
		metricRecorder: metricRecorder,
	}
}

// nominatedPodMap is a structure that stores pods nominated to run on nodes.
// It exists because nominatedNodeName of pod objects stored in the structure
// may be different than what scheduler has here. We should be able to find pods
// by their UID and update/delete them.
type nominatedPodMap struct {
	// nominatedPods is a map keyed by a node name and the value is a list of
	// pods which are nominated to run on the node. These are pods which can be in
	// the activeQ or unschedulableQ.
	nominatedPods map[string][]*v1.Pod
	// nominatedPodToNode is map keyed by a Pod UID to the node name where it is
	// nominated.
	nominatedPodToNode map[ktypes.UID]string
}

func (npm *nominatedPodMap) add(p *v1.Pod, nodeName string) {
	// always delete the pod if it already exist, to ensure we never store more than
	// one instance of the pod.
	npm.delete(p)

	nnn := nodeName
	if len(nnn) == 0 {
		nnn = NominatedNodeName(p)
		if len(nnn) == 0 {
			return
		}
	}
	npm.nominatedPodToNode[p.UID] = nnn
	for _, np := range npm.nominatedPods[nnn] {
		if np.UID == p.UID {
			klog.V(4).Infof("Pod %v/%v already exists in the nominated map!", p.Namespace, p.Name)
			return
		}
	}
	npm.nominatedPods[nnn] = append(npm.nominatedPods[nnn], p)
}

func (npm *nominatedPodMap) delete(p *v1.Pod) {
	nnn, ok := npm.nominatedPodToNode[p.UID]
	if !ok {
		return
	}
	for i, np := range npm.nominatedPods[nnn] {
		if np.UID == p.UID {
			npm.nominatedPods[nnn] = append(npm.nominatedPods[nnn][:i], npm.nominatedPods[nnn][i+1:]...)
			if len(npm.nominatedPods[nnn]) == 0 {
				delete(npm.nominatedPods, nnn)
			}
			break
		}
	}
	delete(npm.nominatedPodToNode, p.UID)
}

func (npm *nominatedPodMap) update(oldPod, newPod *v1.Pod) {
	// In some cases, an Update event with no "NominatedNode" present is received right
	// after a node("NominatedNode") is reserved for this pod in memory.
	// In this case, we need to keep reserving the NominatedNode when updating the pod pointer.
	nodeName := ""
	// We won't fall into below `if` block if the Update event represents:
	// (1) NominatedNode info is added
	// (2) NominatedNode info is updated
	// (3) NominatedNode info is removed
	if NominatedNodeName(oldPod) == "" && NominatedNodeName(newPod) == "" {
		if nnn, ok := npm.nominatedPodToNode[oldPod.UID]; ok {
			// This is the only case we should continue reserving the NominatedNode
			nodeName = nnn
		}
	}
	// We update irrespective of the nominatedNodeName changed or not, to ensure
	// that pod pointer is updated.
	npm.delete(oldPod)
	npm.add(newPod, nodeName)
}

func (npm *nominatedPodMap) podsForNode(nodeName string) []*v1.Pod {
	if list, ok := npm.nominatedPods[nodeName]; ok {
		return list
	}
	return nil
}

func newNominatedPodMap() *nominatedPodMap {
	return &nominatedPodMap{
		nominatedPods:      make(map[string][]*v1.Pod),
		nominatedPodToNode: make(map[ktypes.UID]string),
	}
}

// MakeNextPodFunc returns a function to retrieve the next pod from a given
// scheduling queue
func MakeNextPodFunc(queue SchedulingQueue) func() *v1.Pod {
	return func() *v1.Pod {
		pod, err := queue.Pop()
		if err == nil {
			klog.V(4).Infof("About to try and schedule pod %v/%v", pod.Namespace, pod.Name)
			return pod
		}
		klog.Errorf("Error while retrieving next pod from scheduling queue: %v", err)
		return nil
	}
}

func podInfoKeyFunc(obj interface{}) (string, error) {
	return cache.MetaNamespaceKeyFunc(obj.(*framework.PodInfo).Pod)
}
