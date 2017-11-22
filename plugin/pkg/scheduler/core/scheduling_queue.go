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
// Scheduling queues hold pods waiting to be scheduled. This file has two types
// of scheduling queue: 1) a FIFO, which is mostly the same as cache.FIFO, 2) a
// priority queue which has two sub queues. One sub-queue holds pods that are
// being considered for scheduling. This is called activeQ. Another queue holds
// pods that are already tried and are determined to be unschedulable. The latter
// is called unschedulableQ.
// FIFO is here for flag-gating purposes and allows us to use the traditional
// scheduling queue when util.PodPriorityEnabled() returns false.

package core

import (
	"container/heap"
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/util"

	"github.com/golang/glog"
	"reflect"
)

// SchedulingQueue is an interface for a queue to store pods waiting to be scheduled.
// The interface follows a pattern similar to cache.FIFO and cache.Heap and
// makes it easy to use those data structures as a SchedulingQueue.
type SchedulingQueue interface {
	Add(pod *v1.Pod) error
	AddIfNotPresent(pod *v1.Pod) error
	AddUnschedulableIfNotPresent(pod *v1.Pod) error
	Pop() (*v1.Pod, error)
	Update(pod *v1.Pod) error
	Delete(pod *v1.Pod) error
	MoveAllToActiveQueue()
	AssignedPodAdded(pod *v1.Pod)
	AssignedPodUpdated(pod *v1.Pod)
	WaitingPodsForNode(nodeName string) []*v1.Pod
}

// NewSchedulingQueue initializes a new scheduling queue. If pod priority is
// enabled a priority queue is returned. If it is disabled, a FIFO is returned.
func NewSchedulingQueue() SchedulingQueue {
	if util.PodPriorityEnabled() {
		return NewPriorityQueue()
	}
	return NewFIFO()
}

// FIFO is basically a simple wrapper around cache.FIFO to make it compatible
// with the SchedulingQueue interface.
type FIFO struct {
	*cache.FIFO
}

var _ = SchedulingQueue(&FIFO{}) // Making sure that FIFO implements SchedulingQueue.

func (f *FIFO) Add(pod *v1.Pod) error {
	return f.FIFO.Add(pod)
}

func (f *FIFO) AddIfNotPresent(pod *v1.Pod) error {
	return f.FIFO.AddIfNotPresent(pod)
}

// AddUnschedulableIfNotPresent adds an unschedulable pod back to the queue. In
// FIFO it is added to the end of the queue.
func (f *FIFO) AddUnschedulableIfNotPresent(pod *v1.Pod) error {
	return f.FIFO.AddIfNotPresent(pod)
}

func (f *FIFO) Update(pod *v1.Pod) error {
	return f.FIFO.Update(pod)
}

func (f *FIFO) Delete(pod *v1.Pod) error {
	return f.FIFO.Delete(pod)
}

// Pop removes the head of FIFO and returns it.
// This is just a copy/paste of cache.Pop(queue Queue) from fifo.go that scheduler
// has always been using. There is a comment in that file saying that this method
// shouldn't be used in production code, but scheduler has always been using it.
// This function does minimal error checking.
func (f *FIFO) Pop() (*v1.Pod, error) {
	var result interface{}
	f.FIFO.Pop(func(obj interface{}) error {
		result = obj
		return nil
	})
	return result.(*v1.Pod), nil
}

// FIFO does not need to react to events, as all pods are always in the active
// scheduling queue anyway.
func (f *FIFO) AssignedPodAdded(pod *v1.Pod)   {}
func (f *FIFO) AssignedPodUpdated(pod *v1.Pod) {}

// MoveAllToActiveQueue does nothing in FIFO as all pods are always in the active queue.
func (f *FIFO) MoveAllToActiveQueue() {}

// WaitingPodsForNode returns pods that are nominated to run on the given node,
// but FIFO does not support it.
func (f *FIFO) WaitingPodsForNode(nodeName string) []*v1.Pod {
	return nil
}

func NewFIFO() *FIFO {
	return &FIFO{FIFO: cache.NewFIFO(cache.MetaNamespaceKeyFunc)}
}

// UnschedulablePods is an interface for a queue that is used to keep unschedulable
// pods. These pods are not actively reevaluated for scheduling. They are moved
// to the active scheduling queue on certain events, such as termination of a pod
// in the cluster, addition of nodes, etc.
type UnschedulablePods interface {
	Add(pod *v1.Pod)
	Delete(pod *v1.Pod)
	Update(pod *v1.Pod)
	GetPodsWaitingForNode(nodeName string) []*v1.Pod
	Get(pod *v1.Pod) *v1.Pod
	Clear()
}

// PriorityQueue implements a scheduling queue. It is an alternative to FIFO.
// The head of PriorityQueue is the highest priority pending pod. This structure
// has two sub queues. One sub-queue holds pods that are being considered for
// scheduling. This is called activeQ and is a Heap. Another queue holds
// pods that are already tried and are determined to be unschedulable. The latter
// is called unschedulableQ.
// Heap is already thread safe, but we need to acquire another lock here to ensure
// atomicity of operations on the two data structures..
type PriorityQueue struct {
	lock sync.RWMutex
	cond sync.Cond

	// activeQ is heap structure that scheduler actively looks at to find pods to
	// schedule. Head of heap is the highest priority pod.
	activeQ *Heap
	// unschedulableQ holds pods that have been tried and determined unschedulable.
	unschedulableQ *UnschedulablePodsMap
	// receivedMoveRequest is set to true whenever we receive a request to move a
	// pod from the unschedulableQ to the activeQ, and is set to false, when we pop
	// a pod from the activeQ. It indicates if we received a move request when a
	// pod was in flight (we were trying to schedule it). In such a case, we put
	// the pod back into the activeQ if it is determined unschedulable.
	receivedMoveRequest bool
}

// Making sure that PriorityQueue implements SchedulingQueue.
var _ = SchedulingQueue(&PriorityQueue{})

func NewPriorityQueue() *PriorityQueue {
	pq := &PriorityQueue{
		activeQ:        newHeap(cache.MetaNamespaceKeyFunc, util.HigherPriorityPod),
		unschedulableQ: newUnschedulablePodsMap(),
	}
	pq.cond.L = &pq.lock
	return pq
}

// Add adds a pod to the active queue. It should be called only when a new pod
// is added so there is no chance the pod is already in either queue.
func (p *PriorityQueue) Add(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	err := p.activeQ.Add(pod)
	if err != nil {
		glog.Errorf("Error adding pod %v to the scheduling queue: %v", pod.Name, err)
	} else {
		if p.unschedulableQ.Get(pod) != nil {
			glog.Errorf("Error: pod %v is already in the unschedulable queue.", pod.Name)
			p.unschedulableQ.Delete(pod)
		}
		p.cond.Broadcast()
	}
	return err
}

// AddIfNotPresent adds a pod to the active queue if it is not present in any of
// the two queues. If it is present in any, it doesn't do any thing.
func (p *PriorityQueue) AddIfNotPresent(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	if p.unschedulableQ.Get(pod) != nil {
		return nil
	}
	if _, exists, _ := p.activeQ.Get(pod); exists {
		return nil
	}
	err := p.activeQ.Add(pod)
	if err != nil {
		glog.Errorf("Error adding pod %v to the scheduling queue: %v", pod.Name, err)
	} else {
		p.cond.Broadcast()
	}
	return err
}

func isPodUnschedulable(pod *v1.Pod) bool {
	_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
	return cond != nil && cond.Status == v1.ConditionFalse && cond.Reason == v1.PodReasonUnschedulable
}

// AddUnschedulableIfNotPresent does nothing if the pod is present in either
// queue. Otherwise it adds the pod to the unschedulable queue if
// p.receivedMoveRequest is false, and to the activeQ if p.receivedMoveRequest is true.
func (p *PriorityQueue) AddUnschedulableIfNotPresent(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	if p.unschedulableQ.Get(pod) != nil {
		return fmt.Errorf("pod is already present in unschedulableQ")
	}
	if _, exists, _ := p.activeQ.Get(pod); exists {
		return fmt.Errorf("pod is already present in the activeQ")
	}
	if !p.receivedMoveRequest && isPodUnschedulable(pod) {
		p.unschedulableQ.Add(pod)
		return nil
	}
	err := p.activeQ.Add(pod)
	if err == nil {
		p.cond.Broadcast()
	}
	return err
}

// Pop removes the head of the active queue and returns it. It blocks if the
// activeQ is empty and waits until a new item is added to the queue. It also
// clears receivedMoveRequest to mark the beginning of a new scheduling cycle.
func (p *PriorityQueue) Pop() (*v1.Pod, error) {
	p.lock.Lock()
	defer p.lock.Unlock()
	for len(p.activeQ.data.queue) == 0 {
		p.cond.Wait()
	}
	obj, err := p.activeQ.Pop()
	if err != nil {
		return nil, err
	}
	p.receivedMoveRequest = false
	return obj.(*v1.Pod), err
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

// Update updates a pod in the active queue if present. Otherwise, it removes
// the item from the unschedulable queue and adds the updated one to the active
// queue.
func (p *PriorityQueue) Update(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	// If the pod is already in the active queue, just update it there.
	if _, exists, _ := p.activeQ.Get(pod); exists {
		err := p.activeQ.Update(pod)
		return err
	}
	// If the pod is in the unschedulable queue, updating it may make it schedulable.
	if oldPod := p.unschedulableQ.Get(pod); oldPod != nil {
		if isPodUpdated(oldPod, pod) {
			p.unschedulableQ.Delete(oldPod)
			err := p.activeQ.Add(pod)
			if err == nil {
				p.cond.Broadcast()
			}
			return err
		} else {
			p.unschedulableQ.Update(pod)
			return nil
		}
	}
	// If pod is not in any of the two queue, we put it in the active queue.
	err := p.activeQ.Add(pod)
	if err == nil {
		p.cond.Broadcast()
	}
	return err
}

// Delete deletes the item from either of the two queues. It assumes the pod is
// only in one queue.
func (p *PriorityQueue) Delete(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	if _, exists, _ := p.activeQ.Get(pod); exists {
		return p.activeQ.Delete(pod)
	}
	p.unschedulableQ.Delete(pod)
	return nil
}

// AssignedPodAdded is called when a bound pod is added. Creation of this pod
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodAdded(pod *v1.Pod) {
	p.movePodsToActiveQueue(p.getUnschedulablePodsWithMatchingAffinityTerm(pod))
}

// AssignedPodUpdated is called when a bound pod is updated. Change of labels
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodUpdated(pod *v1.Pod) {
	p.movePodsToActiveQueue(p.getUnschedulablePodsWithMatchingAffinityTerm(pod))
}

// MoveAllToActiveQueue moves all pods from unschedulableQ to activeQ. This
// function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an item, it receives it after all the pods are in the
// queue and the head is the highest priority pod.
// TODO(bsalamat): We should add a back-off mechanism here so that a high priority
// pod which is unschedulable does not go to the head of the queue frequently. For
// example in a cluster where a lot of pods being deleted, such a high priority
// pod can deprive other pods from getting scheduled.
func (p *PriorityQueue) MoveAllToActiveQueue() {
	p.lock.Lock()
	defer p.lock.Unlock()
	var unschedulablePods []interface{}
	for _, pod := range p.unschedulableQ.pods {
		unschedulablePods = append(unschedulablePods, pod)
	}
	p.activeQ.BulkAdd(unschedulablePods)
	p.unschedulableQ.Clear()
	p.receivedMoveRequest = true
	p.cond.Broadcast()
}

func (p *PriorityQueue) movePodsToActiveQueue(pods []*v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()
	for _, pod := range pods {
		p.activeQ.Add(pod)
		p.unschedulableQ.Delete(pod)
	}
	p.receivedMoveRequest = true
	p.cond.Broadcast()
}

// getUnschedulablePodsWithMatchingAffinityTerm returns unschedulable pods which have
// any affinity term that matches "pod".
func (p *PriorityQueue) getUnschedulablePodsWithMatchingAffinityTerm(pod *v1.Pod) []*v1.Pod {
	p.lock.RLock()
	defer p.lock.RUnlock()
	podsToMove := []*v1.Pod{}
	for _, up := range p.unschedulableQ.pods {
		affinity := up.Spec.Affinity
		if affinity != nil && affinity.PodAffinity != nil {
			terms := predicates.GetPodAffinityTerms(affinity.PodAffinity)
			for _, term := range terms {
				namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(up, &term)
				selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
				if err != nil {
					glog.Errorf("Error getting label selectors for pod: %v.", up.Name)
				}
				if priorityutil.PodMatchesTermsNamespaceAndSelector(pod, namespaces, selector) {
					podsToMove = append(podsToMove, up)
				}
			}
		}
	}
	return podsToMove
}

// WaitingPodsForNode returns pods that are nominated to run on the given node,
// but they are waiting for other pods to be removed from the node before they
// can be actually scheduled.
func (p *PriorityQueue) WaitingPodsForNode(nodeName string) []*v1.Pod {
	p.lock.RLock()
	defer p.lock.RUnlock()
	pods := p.unschedulableQ.GetPodsWaitingForNode(nodeName)
	for _, obj := range p.activeQ.List() {
		pod := obj.(*v1.Pod)
		if pod.Annotations != nil {
			if n, ok := pod.Annotations[NominatedNodeAnnotationKey]; ok && n == nodeName {
				pods = append(pods, pod)
			}
		}
	}
	return pods
}

// UnschedulablePodsMap holds pods that cannot be scheduled. This data structure
// is used to implement unschedulableQ.
type UnschedulablePodsMap struct {
	// pods is a map key by a pod's full-name and the value is a pointer to the pod.
	pods map[string]*v1.Pod
	// nominatedPods is a map keyed by a node name and the value is a list of
	// pods' full-names which are nominated to run on the node.
	nominatedPods map[string][]string
	keyFunc       func(*v1.Pod) string
}

var _ = UnschedulablePods(&UnschedulablePodsMap{})

func NominatedNodeName(pod *v1.Pod) string {
	nominatedNodeName, ok := pod.Annotations[NominatedNodeAnnotationKey]
	if !ok {
		return ""
	}
	return nominatedNodeName
}

// Add adds a pod to the unschedulable pods.
func (u *UnschedulablePodsMap) Add(pod *v1.Pod) {
	podKey := u.keyFunc(pod)
	if _, exists := u.pods[podKey]; !exists {
		u.pods[podKey] = pod
		nominatedNodeName := NominatedNodeName(pod)
		if len(nominatedNodeName) > 0 {
			u.nominatedPods[nominatedNodeName] = append(u.nominatedPods[nominatedNodeName], podKey)
		}
	}
}

func (u *UnschedulablePodsMap) deleteFromNominated(pod *v1.Pod) {
	nominatedNodeName := NominatedNodeName(pod)
	if len(nominatedNodeName) > 0 {
		podKey := u.keyFunc(pod)
		nps := u.nominatedPods[nominatedNodeName]
		for i, np := range nps {
			if np == podKey {
				u.nominatedPods[nominatedNodeName] = append(nps[:i], nps[i+1:]...)
				if len(u.nominatedPods[nominatedNodeName]) == 0 {
					delete(u.nominatedPods, nominatedNodeName)
				}
				break
			}
		}
	}
}

// Delete deletes a pod from the unschedulable pods.
func (u *UnschedulablePodsMap) Delete(pod *v1.Pod) {
	podKey := u.keyFunc(pod)
	if p, exists := u.pods[podKey]; exists {
		u.deleteFromNominated(p)
		delete(u.pods, podKey)
	}
}

// Update updates a pod in the unschedulable pods.
func (u *UnschedulablePodsMap) Update(pod *v1.Pod) {
	podKey := u.keyFunc(pod)
	oldPod, exists := u.pods[podKey]
	if !exists {
		u.Add(pod)
		return
	}
	u.pods[podKey] = pod
	oldNominateNodeName := NominatedNodeName(oldPod)
	nominatedNodeName := NominatedNodeName(pod)
	if oldNominateNodeName != nominatedNodeName {
		u.deleteFromNominated(oldPod)
		if len(nominatedNodeName) > 0 {
			u.nominatedPods[nominatedNodeName] = append(u.nominatedPods[nominatedNodeName], podKey)
		}
	}
}

// Get returns the pod if a pod with the same key as the key of the given "pod"
// is found in the map. It returns nil otherwise.
func (u *UnschedulablePodsMap) Get(pod *v1.Pod) *v1.Pod {
	podKey := u.keyFunc(pod)
	if p, exists := u.pods[podKey]; exists {
		return p
	}
	return nil
}

// GetPodsWaitingForNode returns a list of unschedulable pods whose NominatedNodeNames
// are equal to the given nodeName.
func (u *UnschedulablePodsMap) GetPodsWaitingForNode(nodeName string) []*v1.Pod {
	var pods []*v1.Pod
	for _, key := range u.nominatedPods[nodeName] {
		pods = append(pods, u.pods[key])
	}
	return pods
}

// Clear removes all the entries from the unschedulable maps.
func (u *UnschedulablePodsMap) Clear() {
	u.pods = make(map[string]*v1.Pod)
	u.nominatedPods = make(map[string][]string)
}

// newUnschedulablePodsMap initializes a new object of UnschedulablePodsMap.
func newUnschedulablePodsMap() *UnschedulablePodsMap {
	return &UnschedulablePodsMap{
		pods:          make(map[string]*v1.Pod),
		nominatedPods: make(map[string][]string),
		keyFunc:       util.GetPodFullName,
	}
}

// Below is the implementation of the a heap. The logic is pretty much the same
// as cache.heap, however, this heap does not perform synchronization. It leaves
// synchronization to the SchedulingQueue.

type LessFunc func(interface{}, interface{}) bool
type KeyFunc func(obj interface{}) (string, error)

type heapItem struct {
	obj   interface{} // The object which is stored in the heap.
	index int         // The index of the object's key in the Heap.queue.
}

type itemKeyValue struct {
	key string
	obj interface{}
}

// heapData is an internal struct that implements the standard heap interface
// and keeps the data stored in the heap.
type heapData struct {
	// items is a map from key of the objects to the objects and their index.
	// We depend on the property that items in the map are in the queue and vice versa.
	items map[string]*heapItem
	// queue implements a heap data structure and keeps the order of elements
	// according to the heap invariant. The queue keeps the keys of objects stored
	// in "items".
	queue []string

	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc
	// lessFunc is used to compare two objects in the heap.
	lessFunc LessFunc
}

var (
	_ = heap.Interface(&heapData{}) // heapData is a standard heap
)

// Less compares two objects and returns true if the first one should go
// in front of the second one in the heap.
func (h *heapData) Less(i, j int) bool {
	if i > len(h.queue) || j > len(h.queue) {
		return false
	}
	itemi, ok := h.items[h.queue[i]]
	if !ok {
		return false
	}
	itemj, ok := h.items[h.queue[j]]
	if !ok {
		return false
	}
	return h.lessFunc(itemi.obj, itemj.obj)
}

// Len returns the number of items in the Heap.
func (h *heapData) Len() int { return len(h.queue) }

// Swap implements swapping of two elements in the heap. This is a part of standard
// heap interface and should never be called directly.
func (h *heapData) Swap(i, j int) {
	h.queue[i], h.queue[j] = h.queue[j], h.queue[i]
	item := h.items[h.queue[i]]
	item.index = i
	item = h.items[h.queue[j]]
	item.index = j
}

// Push is supposed to be called by heap.Push only.
func (h *heapData) Push(kv interface{}) {
	keyValue := kv.(*itemKeyValue)
	n := len(h.queue)
	h.items[keyValue.key] = &heapItem{keyValue.obj, n}
	h.queue = append(h.queue, keyValue.key)
}

// Pop is supposed to be called by heap.Pop only.
func (h *heapData) Pop() interface{} {
	key := h.queue[len(h.queue)-1]
	h.queue = h.queue[0 : len(h.queue)-1]
	item, ok := h.items[key]
	if !ok {
		// This is an error
		return nil
	}
	delete(h.items, key)
	return item.obj
}

// Heap is a thread-safe producer/consumer queue that implements a heap data structure.
// It can be used to implement priority queues and similar data structures.
type Heap struct {
	// data stores objects and has a queue that keeps their ordering according
	// to the heap invariant.
	data *heapData
}

// Add inserts an item, and puts it in the queue. The item is updated if it
// already exists.
func (h *Heap) Add(obj interface{}) error {
	key, err := h.data.keyFunc(obj)
	if err != nil {
		return cache.KeyError{Obj: obj, Err: err}
	}
	if _, exists := h.data.items[key]; exists {
		h.data.items[key].obj = obj
		heap.Fix(h.data, h.data.items[key].index)
	} else {
		heap.Push(h.data, &itemKeyValue{key, obj})
	}
	return nil
}

// BulkAdd adds all the items in the list to the queue.
func (h *Heap) BulkAdd(list []interface{}) error {
	for _, obj := range list {
		key, err := h.data.keyFunc(obj)
		if err != nil {
			return cache.KeyError{Obj: obj, Err: err}
		}
		if _, exists := h.data.items[key]; exists {
			h.data.items[key].obj = obj
			heap.Fix(h.data, h.data.items[key].index)
		} else {
			heap.Push(h.data, &itemKeyValue{key, obj})
		}
	}
	return nil
}

// AddIfNotPresent inserts an item, and puts it in the queue. If an item with
// the key is present in the map, no changes is made to the item.
func (h *Heap) AddIfNotPresent(obj interface{}) error {
	key, err := h.data.keyFunc(obj)
	if err != nil {
		return cache.KeyError{Obj: obj, Err: err}
	}
	if _, exists := h.data.items[key]; !exists {
		heap.Push(h.data, &itemKeyValue{key, obj})
	}
	return nil
}

// Update is the same as Add in this implementation. When the item does not
// exist, it is added.
func (h *Heap) Update(obj interface{}) error {
	return h.Add(obj)
}

// Delete removes an item.
func (h *Heap) Delete(obj interface{}) error {
	key, err := h.data.keyFunc(obj)
	if err != nil {
		return cache.KeyError{Obj: obj, Err: err}
	}
	if item, ok := h.data.items[key]; ok {
		heap.Remove(h.data, item.index)
		return nil
	}
	return fmt.Errorf("object not found")
}

// Pop returns the head of the heap.
func (h *Heap) Pop() (interface{}, error) {
	obj := heap.Pop(h.data)
	if obj != nil {
		return obj, nil
	} else {
		return nil, fmt.Errorf("object was removed from heap data")
	}
}

// Get returns the requested item, or sets exists=false.
func (h *Heap) Get(obj interface{}) (interface{}, bool, error) {
	key, err := h.data.keyFunc(obj)
	if err != nil {
		return nil, false, cache.KeyError{Obj: obj, Err: err}
	}
	return h.GetByKey(key)
}

// GetByKey returns the requested item, or sets exists=false.
func (h *Heap) GetByKey(key string) (interface{}, bool, error) {
	item, exists := h.data.items[key]
	if !exists {
		return nil, false, nil
	}
	return item.obj, true, nil
}

// List returns a list of all the items.
func (h *Heap) List() []interface{} {
	list := make([]interface{}, 0, len(h.data.items))
	for _, item := range h.data.items {
		list = append(list, item.obj)
	}
	return list
}

// newHeap returns a Heap which can be used to queue up items to process.
func newHeap(keyFn KeyFunc, lessFn LessFunc) *Heap {
	return &Heap{
		data: &heapData{
			items:    map[string]*heapItem{},
			queue:    []string{},
			keyFunc:  keyFn,
			lessFunc: lessFn,
		},
	}
}
