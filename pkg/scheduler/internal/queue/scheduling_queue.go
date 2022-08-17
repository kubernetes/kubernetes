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
// priority queue which has two sub queues and a additional data structure,
// namely: activeQ, backoffQ and unschedulablePods.
// - activeQ holds pods that are being considered for scheduling.
// - backoffQ holds pods that moved from unschedulablePods and will move to
//   activeQ when their backoff periods complete.
// - unschedulablePods holds pods that were already attempted for scheduling and
//   are currently determined to be unschedulable.

package queue

import (
	"fmt"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/internal/heap"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/clock"
)

const (
	// DefaultPodMaxInUnschedulablePodsDuration is the default value for the maximum
	// time a pod can stay in unschedulablePods. If a pod stays in unschedulablePods
	// for longer than this value, the pod will be moved from unschedulablePods to
	// backoffQ or activeQ. If this value is empty, the default value (5min)
	// will be used.
	DefaultPodMaxInUnschedulablePodsDuration time.Duration = 5 * time.Minute

	queueClosed = "scheduling queue is closed"

	// Scheduling queue names
	activeQName       = "Active"
	backoffQName      = "Backoff"
	unschedulablePods = "Unschedulable"
)

const (
	// DefaultPodInitialBackoffDuration is the default value for the initial backoff duration
	// for unschedulable pods. To change the default podInitialBackoffDurationSeconds used by the
	// scheduler, update the ComponentConfig value in defaults.go
	DefaultPodInitialBackoffDuration time.Duration = 1 * time.Second
	// DefaultPodMaxBackoffDuration is the default value for the max backoff duration
	// for unschedulable pods. To change the default podMaxBackoffDurationSeconds used by the
	// scheduler, update the ComponentConfig value in defaults.go
	DefaultPodMaxBackoffDuration time.Duration = 10 * time.Second
)

// PreEnqueueCheck is a function type. It's used to build functions that
// run against a Pod and the caller can choose to enqueue or skip the Pod
// by the checking result.
type PreEnqueueCheck func(pod *v1.Pod) bool

// SchedulingQueue is an interface for a queue to store pods waiting to be scheduled.
// The interface follows a pattern similar to cache.FIFO and cache.Heap and
// makes it easy to use those data structures as a SchedulingQueue.
type SchedulingQueue interface {
	framework.PodNominator
	Add(pod *v1.Pod) error
	// Activate moves the given pods to activeQ iff they're in unschedulablePods or backoffQ.
	// The passed-in pods are originally compiled from plugins that want to activate Pods,
	// by injecting the pods through a reserved CycleState struct (PodsToActivate).
	Activate(pods map[string]*v1.Pod)
	// AddUnschedulableIfNotPresent adds an unschedulable pod back to scheduling queue.
	// The podSchedulingCycle represents the current scheduling cycle number which can be
	// returned by calling SchedulingCycle().
	AddUnschedulableIfNotPresent(pod *framework.QueuedPodInfo, podSchedulingCycle int64) error
	// SchedulingCycle returns the current number of scheduling cycle which is
	// cached by scheduling queue. Normally, incrementing this number whenever
	// a pod is popped (e.g. called Pop()) is enough.
	SchedulingCycle() int64
	// Pop removes the head of the queue and returns it. It blocks if the
	// queue is empty and waits until a new item is added to the queue.
	Pop() (*framework.QueuedPodInfo, error)
	Update(oldPod, newPod *v1.Pod) error
	Delete(pod *v1.Pod) error
	MoveAllToActiveOrBackoffQueue(event framework.ClusterEvent, preCheck PreEnqueueCheck)
	AssignedPodAdded(pod *v1.Pod)
	AssignedPodUpdated(pod *v1.Pod)
	PendingPods() ([]*v1.Pod, string)
	// Close closes the SchedulingQueue so that the goroutine which is
	// waiting to pop items can exit gracefully.
	Close()
	// Run starts the goroutines managing the queue.
	Run()
}

// NewSchedulingQueue initializes a priority queue as a new scheduling queue.
func NewSchedulingQueue(
	lessFn framework.LessFunc,
	informerFactory informers.SharedInformerFactory,
	opts ...Option) SchedulingQueue {
	return NewPriorityQueue(lessFn, informerFactory, opts...)
}

// NominatedNodeName returns nominated node name of a Pod.
func NominatedNodeName(pod *v1.Pod) string {
	return pod.Status.NominatedNodeName
}

// PriorityQueue implements a scheduling queue.
// The head of PriorityQueue is the highest priority pending pod. This structure
// has two sub queues and a additional data structure, namely: activeQ,
// backoffQ and unschedulablePods.
//   - activeQ holds pods that are being considered for scheduling.
//   - backoffQ holds pods that moved from unschedulablePods and will move to
//     activeQ when their backoff periods complete.
//   - unschedulablePods holds pods that were already attempted for scheduling and
//     are currently determined to be unschedulable.
type PriorityQueue struct {
	// PodNominator abstracts the operations to maintain nominated Pods.
	framework.PodNominator

	stop  chan struct{}
	clock clock.Clock

	// pod initial backoff duration.
	podInitialBackoffDuration time.Duration
	// pod maximum backoff duration.
	podMaxBackoffDuration time.Duration
	// the maximum time a pod can stay in the unschedulablePods.
	podMaxInUnschedulablePodsDuration time.Duration

	lock sync.RWMutex
	cond sync.Cond

	// activeQ is heap structure that scheduler actively looks at to find pods to
	// schedule. Head of heap is the highest priority pod.
	activeQ *heap.Heap
	// podBackoffQ is a heap ordered by backoff expiry. Pods which have completed backoff
	// are popped from this heap before the scheduler looks at activeQ
	podBackoffQ *heap.Heap
	// unschedulablePods holds pods that have been tried and determined unschedulable.
	unschedulablePods *UnschedulablePods
	// schedulingCycle represents sequence number of scheduling cycle and is incremented
	// when a pod is popped.
	schedulingCycle int64
	// moveRequestCycle caches the sequence number of scheduling cycle when we
	// received a move request. Unschedulable pods in and before this scheduling
	// cycle will be put back to activeQueue if we were trying to schedule them
	// when we received move request.
	moveRequestCycle int64

	clusterEventMap map[framework.ClusterEvent]sets.String

	// closed indicates that the queue is closed.
	// It is mainly used to let Pop() exit its control loop while waiting for an item.
	closed bool

	nsLister listersv1.NamespaceLister
}

type priorityQueueOptions struct {
	clock                             clock.Clock
	podInitialBackoffDuration         time.Duration
	podMaxBackoffDuration             time.Duration
	podMaxInUnschedulablePodsDuration time.Duration
	podNominator                      framework.PodNominator
	clusterEventMap                   map[framework.ClusterEvent]sets.String
}

// Option configures a PriorityQueue
type Option func(*priorityQueueOptions)

// WithClock sets clock for PriorityQueue, the default clock is clock.RealClock.
func WithClock(clock clock.Clock) Option {
	return func(o *priorityQueueOptions) {
		o.clock = clock
	}
}

// WithPodInitialBackoffDuration sets pod initial backoff duration for PriorityQueue.
func WithPodInitialBackoffDuration(duration time.Duration) Option {
	return func(o *priorityQueueOptions) {
		o.podInitialBackoffDuration = duration
	}
}

// WithPodMaxBackoffDuration sets pod max backoff duration for PriorityQueue.
func WithPodMaxBackoffDuration(duration time.Duration) Option {
	return func(o *priorityQueueOptions) {
		o.podMaxBackoffDuration = duration
	}
}

// WithPodNominator sets pod nominator for PriorityQueue.
func WithPodNominator(pn framework.PodNominator) Option {
	return func(o *priorityQueueOptions) {
		o.podNominator = pn
	}
}

// WithClusterEventMap sets clusterEventMap for PriorityQueue.
func WithClusterEventMap(m map[framework.ClusterEvent]sets.String) Option {
	return func(o *priorityQueueOptions) {
		o.clusterEventMap = m
	}
}

// WithPodMaxInUnschedulablePodsDuration sets podMaxInUnschedulablePodsDuration for PriorityQueue.
func WithPodMaxInUnschedulablePodsDuration(duration time.Duration) Option {
	return func(o *priorityQueueOptions) {
		o.podMaxInUnschedulablePodsDuration = duration
	}
}

var defaultPriorityQueueOptions = priorityQueueOptions{
	clock:                             clock.RealClock{},
	podInitialBackoffDuration:         DefaultPodInitialBackoffDuration,
	podMaxBackoffDuration:             DefaultPodMaxBackoffDuration,
	podMaxInUnschedulablePodsDuration: DefaultPodMaxInUnschedulablePodsDuration,
}

// Making sure that PriorityQueue implements SchedulingQueue.
var _ SchedulingQueue = &PriorityQueue{}

// newQueuedPodInfoForLookup builds a QueuedPodInfo object for a lookup in the queue.
func newQueuedPodInfoForLookup(pod *v1.Pod, plugins ...string) *framework.QueuedPodInfo {
	// Since this is only used for a lookup in the queue, we only need to set the Pod,
	// and so we avoid creating a full PodInfo, which is expensive to instantiate frequently.
	return &framework.QueuedPodInfo{
		PodInfo:              &framework.PodInfo{Pod: pod},
		UnschedulablePlugins: sets.NewString(plugins...),
	}
}

// NewPriorityQueue creates a PriorityQueue object.
func NewPriorityQueue(
	lessFn framework.LessFunc,
	informerFactory informers.SharedInformerFactory,
	opts ...Option,
) *PriorityQueue {
	options := defaultPriorityQueueOptions
	for _, opt := range opts {
		opt(&options)
	}

	comp := func(podInfo1, podInfo2 interface{}) bool {
		pInfo1 := podInfo1.(*framework.QueuedPodInfo)
		pInfo2 := podInfo2.(*framework.QueuedPodInfo)
		return lessFn(pInfo1, pInfo2)
	}

	if options.podNominator == nil {
		options.podNominator = NewPodNominator(informerFactory.Core().V1().Pods().Lister())
	}

	pq := &PriorityQueue{
		PodNominator:                      options.podNominator,
		clock:                             options.clock,
		stop:                              make(chan struct{}),
		podInitialBackoffDuration:         options.podInitialBackoffDuration,
		podMaxBackoffDuration:             options.podMaxBackoffDuration,
		podMaxInUnschedulablePodsDuration: options.podMaxInUnschedulablePodsDuration,
		activeQ:                           heap.NewWithRecorder(podInfoKeyFunc, comp, metrics.NewActivePodsRecorder()),
		unschedulablePods:                 newUnschedulablePods(metrics.NewUnschedulablePodsRecorder()),
		moveRequestCycle:                  -1,
		clusterEventMap:                   options.clusterEventMap,
	}
	pq.cond.L = &pq.lock
	pq.podBackoffQ = heap.NewWithRecorder(podInfoKeyFunc, pq.podsCompareBackoffCompleted, metrics.NewBackoffPodsRecorder())
	pq.nsLister = informerFactory.Core().V1().Namespaces().Lister()

	return pq
}

// Run starts the goroutine to pump from podBackoffQ to activeQ
func (p *PriorityQueue) Run() {
	go wait.Until(p.flushBackoffQCompleted, 1.0*time.Second, p.stop)
	go wait.Until(p.flushUnschedulablePodsLeftover, 30*time.Second, p.stop)
}

// Add adds a pod to the active queue. It should be called only when a new pod
// is added so there is no chance the pod is already in active/unschedulable/backoff queues
func (p *PriorityQueue) Add(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	pInfo := p.newQueuedPodInfo(pod)
	if err := p.activeQ.Add(pInfo); err != nil {
		klog.ErrorS(err, "Error adding pod to the active queue", "pod", klog.KObj(pod))
		return err
	}
	if p.unschedulablePods.get(pod) != nil {
		klog.ErrorS(nil, "Error: pod is already in the unschedulable queue", "pod", klog.KObj(pod))
		p.unschedulablePods.delete(pod)
	}
	// Delete pod from backoffQ if it is backing off
	if err := p.podBackoffQ.Delete(pInfo); err == nil {
		klog.ErrorS(nil, "Error: pod is already in the podBackoff queue", "pod", klog.KObj(pod))
	}
	klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", PodAdd, "queue", activeQName)
	metrics.SchedulerQueueIncomingPods.WithLabelValues("active", PodAdd).Inc()
	p.PodNominator.AddNominatedPod(pInfo.PodInfo, nil)
	p.cond.Broadcast()

	return nil
}

// Activate moves the given pods to activeQ iff they're in unschedulablePods or backoffQ.
func (p *PriorityQueue) Activate(pods map[string]*v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	activated := false
	for _, pod := range pods {
		if p.activate(pod) {
			activated = true
		}
	}

	if activated {
		p.cond.Broadcast()
	}
}

func (p *PriorityQueue) activate(pod *v1.Pod) bool {
	// Verify if the pod is present in activeQ.
	if _, exists, _ := p.activeQ.Get(newQueuedPodInfoForLookup(pod)); exists {
		// No need to activate if it's already present in activeQ.
		return false
	}
	var pInfo *framework.QueuedPodInfo
	// Verify if the pod is present in unschedulablePods or backoffQ.
	if pInfo = p.unschedulablePods.get(pod); pInfo == nil {
		// If the pod doesn't belong to unschedulablePods or backoffQ, don't activate it.
		if obj, exists, _ := p.podBackoffQ.Get(newQueuedPodInfoForLookup(pod)); !exists {
			klog.ErrorS(nil, "To-activate pod does not exist in unschedulablePods or backoffQ", "pod", klog.KObj(pod))
			return false
		} else {
			pInfo = obj.(*framework.QueuedPodInfo)
		}
	}

	if pInfo == nil {
		// Redundant safe check. We shouldn't reach here.
		klog.ErrorS(nil, "Internal error: cannot obtain pInfo")
		return false
	}

	if err := p.activeQ.Add(pInfo); err != nil {
		klog.ErrorS(err, "Error adding pod to the scheduling queue", "pod", klog.KObj(pod))
		return false
	}
	p.unschedulablePods.delete(pod)
	p.podBackoffQ.Delete(pInfo)
	metrics.SchedulerQueueIncomingPods.WithLabelValues("active", ForceActivate).Inc()
	p.PodNominator.AddNominatedPod(pInfo.PodInfo, nil)
	return true
}

// isPodBackingoff returns true if a pod is still waiting for its backoff timer.
// If this returns true, the pod should not be re-tried.
func (p *PriorityQueue) isPodBackingoff(podInfo *framework.QueuedPodInfo) bool {
	boTime := p.getBackoffTime(podInfo)
	return boTime.After(p.clock.Now())
}

// SchedulingCycle returns current scheduling cycle.
func (p *PriorityQueue) SchedulingCycle() int64 {
	p.lock.RLock()
	defer p.lock.RUnlock()
	return p.schedulingCycle
}

// AddUnschedulableIfNotPresent inserts a pod that cannot be scheduled into
// the queue, unless it is already in the queue. Normally, PriorityQueue puts
// unschedulable pods in `unschedulablePods`. But if there has been a recent move
// request, then the pod is put in `podBackoffQ`.
func (p *PriorityQueue) AddUnschedulableIfNotPresent(pInfo *framework.QueuedPodInfo, podSchedulingCycle int64) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	pod := pInfo.Pod
	if p.unschedulablePods.get(pod) != nil {
		return fmt.Errorf("Pod %v is already present in unschedulable queue", klog.KObj(pod))
	}

	if _, exists, _ := p.activeQ.Get(pInfo); exists {
		return fmt.Errorf("Pod %v is already present in the active queue", klog.KObj(pod))
	}
	if _, exists, _ := p.podBackoffQ.Get(pInfo); exists {
		return fmt.Errorf("Pod %v is already present in the backoff queue", klog.KObj(pod))
	}

	// Refresh the timestamp since the pod is re-added.
	pInfo.Timestamp = p.clock.Now()

	// If a move request has been received, move it to the BackoffQ, otherwise move
	// it to unschedulablePods.
	for plugin := range pInfo.UnschedulablePlugins {
		metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Inc()
	}
	if p.moveRequestCycle >= podSchedulingCycle {
		if err := p.podBackoffQ.Add(pInfo); err != nil {
			return fmt.Errorf("error adding pod %v to the backoff queue: %v", klog.KObj(pod), err)
		}
		klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", ScheduleAttemptFailure, "queue", backoffQName)
		metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", ScheduleAttemptFailure).Inc()
	} else {
		p.unschedulablePods.addOrUpdate(pInfo)
		klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", ScheduleAttemptFailure, "queue", unschedulablePods)
		metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", ScheduleAttemptFailure).Inc()

	}

	p.PodNominator.AddNominatedPod(pInfo.PodInfo, nil)
	return nil
}

// flushBackoffQCompleted Moves all pods from backoffQ which have completed backoff in to activeQ
func (p *PriorityQueue) flushBackoffQCompleted() {
	p.lock.Lock()
	defer p.lock.Unlock()
	activated := false
	for {
		rawPodInfo := p.podBackoffQ.Peek()
		if rawPodInfo == nil {
			break
		}
		pod := rawPodInfo.(*framework.QueuedPodInfo).Pod
		boTime := p.getBackoffTime(rawPodInfo.(*framework.QueuedPodInfo))
		if boTime.After(p.clock.Now()) {
			break
		}
		_, err := p.podBackoffQ.Pop()
		if err != nil {
			klog.ErrorS(err, "Unable to pop pod from backoff queue despite backoff completion", "pod", klog.KObj(pod))
			break
		}
		p.activeQ.Add(rawPodInfo)
		klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", BackoffComplete, "queue", activeQName)
		metrics.SchedulerQueueIncomingPods.WithLabelValues("active", BackoffComplete).Inc()
		activated = true
	}

	if activated {
		p.cond.Broadcast()
	}
}

// flushUnschedulablePodsLeftover moves pods which stay in unschedulablePods
// longer than podMaxInUnschedulablePodsDuration to backoffQ or activeQ.
func (p *PriorityQueue) flushUnschedulablePodsLeftover() {
	p.lock.Lock()
	defer p.lock.Unlock()

	var podsToMove []*framework.QueuedPodInfo
	currentTime := p.clock.Now()
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		lastScheduleTime := pInfo.Timestamp
		if currentTime.Sub(lastScheduleTime) > p.podMaxInUnschedulablePodsDuration {
			podsToMove = append(podsToMove, pInfo)
		}
	}

	if len(podsToMove) > 0 {
		p.movePodsToActiveOrBackoffQueue(podsToMove, UnschedulableTimeout)
	}
}

// Pop removes the head of the active queue and returns it. It blocks if the
// activeQ is empty and waits until a new item is added to the queue. It
// increments scheduling cycle when a pod is popped.
func (p *PriorityQueue) Pop() (*framework.QueuedPodInfo, error) {
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
	pInfo := obj.(*framework.QueuedPodInfo)
	pInfo.Attempts++
	p.schedulingCycle++
	return pInfo, nil
}

// isPodUpdated checks if the pod is updated in a way that it may have become
// schedulable. It drops status of the pod and compares it with old version.
func isPodUpdated(oldPod, newPod *v1.Pod) bool {
	strip := func(pod *v1.Pod) *v1.Pod {
		p := pod.DeepCopy()
		p.ResourceVersion = ""
		p.Generation = 0
		p.Status = v1.PodStatus{}
		p.ManagedFields = nil
		p.Finalizers = nil
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
		oldPodInfo := newQueuedPodInfoForLookup(oldPod)
		// If the pod is already in the active queue, just update it there.
		if oldPodInfo, exists, _ := p.activeQ.Get(oldPodInfo); exists {
			pInfo := updatePod(oldPodInfo, newPod)
			p.PodNominator.UpdateNominatedPod(oldPod, pInfo.PodInfo)
			return p.activeQ.Update(pInfo)
		}

		// If the pod is in the backoff queue, update it there.
		if oldPodInfo, exists, _ := p.podBackoffQ.Get(oldPodInfo); exists {
			pInfo := updatePod(oldPodInfo, newPod)
			p.PodNominator.UpdateNominatedPod(oldPod, pInfo.PodInfo)
			return p.podBackoffQ.Update(pInfo)
		}
	}

	// If the pod is in the unschedulable queue, updating it may make it schedulable.
	if usPodInfo := p.unschedulablePods.get(newPod); usPodInfo != nil {
		pInfo := updatePod(usPodInfo, newPod)
		p.PodNominator.UpdateNominatedPod(oldPod, pInfo.PodInfo)
		if isPodUpdated(oldPod, newPod) {
			if p.isPodBackingoff(usPodInfo) {
				if err := p.podBackoffQ.Add(pInfo); err != nil {
					return err
				}
				p.unschedulablePods.delete(usPodInfo.Pod)
				klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", PodUpdate, "queue", backoffQName)
			} else {
				if err := p.activeQ.Add(pInfo); err != nil {
					return err
				}
				p.unschedulablePods.delete(usPodInfo.Pod)
				klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", BackoffComplete, "queue", activeQName)
				p.cond.Broadcast()
			}
		} else {
			// Pod update didn't make it schedulable, keep it in the unschedulable queue.
			p.unschedulablePods.addOrUpdate(pInfo)
		}

		return nil
	}
	// If pod is not in any of the queues, we put it in the active queue.
	pInfo := p.newQueuedPodInfo(newPod)
	if err := p.activeQ.Add(pInfo); err != nil {
		return err
	}
	p.PodNominator.AddNominatedPod(pInfo.PodInfo, nil)
	klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", PodUpdate, "queue", activeQName)
	p.cond.Broadcast()
	return nil
}

// Delete deletes the item from either of the two queues. It assumes the pod is
// only in one queue.
func (p *PriorityQueue) Delete(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.PodNominator.DeleteNominatedPodIfExists(pod)
	if err := p.activeQ.Delete(newQueuedPodInfoForLookup(pod)); err != nil {
		// The item was probably not found in the activeQ.
		p.podBackoffQ.Delete(newQueuedPodInfoForLookup(pod))
		p.unschedulablePods.delete(pod)
	}
	return nil
}

// AssignedPodAdded is called when a bound pod is added. Creation of this pod
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodAdded(pod *v1.Pod) {
	p.lock.Lock()
	p.movePodsToActiveOrBackoffQueue(p.getUnschedulablePodsWithMatchingAffinityTerm(pod), AssignedPodAdd)
	p.lock.Unlock()
}

// AssignedPodUpdated is called when a bound pod is updated. Change of labels
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodUpdated(pod *v1.Pod) {
	p.lock.Lock()
	p.movePodsToActiveOrBackoffQueue(p.getUnschedulablePodsWithMatchingAffinityTerm(pod), AssignedPodUpdate)
	p.lock.Unlock()
}

// MoveAllToActiveOrBackoffQueue moves all pods from unschedulablePods to activeQ or backoffQ.
// This function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an item, it receives the signal after all the pods are in the
// queue and the head is the highest priority pod.
func (p *PriorityQueue) MoveAllToActiveOrBackoffQueue(event framework.ClusterEvent, preCheck PreEnqueueCheck) {
	p.lock.Lock()
	defer p.lock.Unlock()
	unschedulablePods := make([]*framework.QueuedPodInfo, 0, len(p.unschedulablePods.podInfoMap))
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		if preCheck == nil || preCheck(pInfo.Pod) {
			unschedulablePods = append(unschedulablePods, pInfo)
		}
	}
	p.movePodsToActiveOrBackoffQueue(unschedulablePods, event)
}

// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) movePodsToActiveOrBackoffQueue(podInfoList []*framework.QueuedPodInfo, event framework.ClusterEvent) {
	activated := false
	for _, pInfo := range podInfoList {
		// If the event doesn't help making the Pod schedulable, continue.
		// Note: we don't run the check if pInfo.UnschedulablePlugins is nil, which denotes
		// either there is some abnormal error, or scheduling the pod failed by plugins other than PreFilter, Filter and Permit.
		// In that case, it's desired to move it anyways.
		if len(pInfo.UnschedulablePlugins) != 0 && !p.podMatchesEvent(pInfo, event) {
			continue
		}
		pod := pInfo.Pod
		if p.isPodBackingoff(pInfo) {
			if err := p.podBackoffQ.Add(pInfo); err != nil {
				klog.ErrorS(err, "Error adding pod to the backoff queue", "pod", klog.KObj(pod))
			} else {
				klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", event, "queue", backoffQName)
				metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", event.Label).Inc()
				p.unschedulablePods.delete(pod)
			}
		} else {
			if err := p.activeQ.Add(pInfo); err != nil {
				klog.ErrorS(err, "Error adding pod to the scheduling queue", "pod", klog.KObj(pod))
			} else {
				klog.V(5).InfoS("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", event, "queue", activeQName)
				activated = true
				metrics.SchedulerQueueIncomingPods.WithLabelValues("active", event.Label).Inc()
				p.unschedulablePods.delete(pod)
			}
		}
	}
	p.moveRequestCycle = p.schedulingCycle
	if activated {
		p.cond.Broadcast()
	}
}

// getUnschedulablePodsWithMatchingAffinityTerm returns unschedulable pods which have
// any affinity term that matches "pod".
// NOTE: this function assumes lock has been acquired in caller.
func (p *PriorityQueue) getUnschedulablePodsWithMatchingAffinityTerm(pod *v1.Pod) []*framework.QueuedPodInfo {
	var nsLabels labels.Set
	nsLabels = interpodaffinity.GetNamespaceLabelsSnapshot(pod.Namespace, p.nsLister)

	var podsToMove []*framework.QueuedPodInfo
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		for _, term := range pInfo.RequiredAffinityTerms {
			if term.Matches(pod, nsLabels) {
				podsToMove = append(podsToMove, pInfo)
				break
			}
		}

	}
	return podsToMove
}

var pendingPodsSummary = "activeQ:%v; backoffQ:%v; unschedulablePods:%v"

// PendingPods returns all the pending pods in the queue; accompanied by a debugging string
// recording showing the number of pods in each queue respectively.
// This function is used for debugging purposes in the scheduler cache dumper and comparer.
func (p *PriorityQueue) PendingPods() ([]*v1.Pod, string) {
	p.lock.RLock()
	defer p.lock.RUnlock()
	var result []*v1.Pod
	for _, pInfo := range p.activeQ.List() {
		result = append(result, pInfo.(*framework.QueuedPodInfo).Pod)
	}
	for _, pInfo := range p.podBackoffQ.List() {
		result = append(result, pInfo.(*framework.QueuedPodInfo).Pod)
	}
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		result = append(result, pInfo.Pod)
	}
	return result, fmt.Sprintf(pendingPodsSummary, p.activeQ.Len(), p.podBackoffQ.Len(), len(p.unschedulablePods.podInfoMap))
}

// Close closes the priority queue.
func (p *PriorityQueue) Close() {
	p.lock.Lock()
	defer p.lock.Unlock()
	close(p.stop)
	p.closed = true
	p.cond.Broadcast()
}

// DeleteNominatedPodIfExists deletes <pod> from nominatedPods.
func (npm *nominator) DeleteNominatedPodIfExists(pod *v1.Pod) {
	npm.Lock()
	npm.delete(pod)
	npm.Unlock()
}

// AddNominatedPod adds a pod to the nominated pods of the given node.
// This is called during the preemption process after a node is nominated to run
// the pod. We update the structure before sending a request to update the pod
// object to avoid races with the following scheduling cycles.
func (npm *nominator) AddNominatedPod(pi *framework.PodInfo, nominatingInfo *framework.NominatingInfo) {
	npm.Lock()
	npm.add(pi, nominatingInfo)
	npm.Unlock()
}

// NominatedPodsForNode returns a copy of pods that are nominated to run on the given node,
// but they are waiting for other pods to be removed from the node.
func (npm *nominator) NominatedPodsForNode(nodeName string) []*framework.PodInfo {
	npm.RLock()
	defer npm.RUnlock()
	// Make a copy of the nominated Pods so the caller can mutate safely.
	pods := make([]*framework.PodInfo, len(npm.nominatedPods[nodeName]))
	for i := 0; i < len(pods); i++ {
		pods[i] = npm.nominatedPods[nodeName][i].DeepCopy()
	}
	return pods
}

func (p *PriorityQueue) podsCompareBackoffCompleted(podInfo1, podInfo2 interface{}) bool {
	pInfo1 := podInfo1.(*framework.QueuedPodInfo)
	pInfo2 := podInfo2.(*framework.QueuedPodInfo)
	bo1 := p.getBackoffTime(pInfo1)
	bo2 := p.getBackoffTime(pInfo2)
	return bo1.Before(bo2)
}

// newQueuedPodInfo builds a QueuedPodInfo object.
func (p *PriorityQueue) newQueuedPodInfo(pod *v1.Pod, plugins ...string) *framework.QueuedPodInfo {
	now := p.clock.Now()
	return &framework.QueuedPodInfo{
		PodInfo:                 framework.NewPodInfo(pod),
		Timestamp:               now,
		InitialAttemptTimestamp: now,
		UnschedulablePlugins:    sets.NewString(plugins...),
	}
}

// getBackoffTime returns the time that podInfo completes backoff
func (p *PriorityQueue) getBackoffTime(podInfo *framework.QueuedPodInfo) time.Time {
	duration := p.calculateBackoffDuration(podInfo)
	backoffTime := podInfo.Timestamp.Add(duration)
	return backoffTime
}

// calculateBackoffDuration is a helper function for calculating the backoffDuration
// based on the number of attempts the pod has made.
func (p *PriorityQueue) calculateBackoffDuration(podInfo *framework.QueuedPodInfo) time.Duration {
	duration := p.podInitialBackoffDuration
	for i := 1; i < podInfo.Attempts; i++ {
		// Use subtraction instead of addition or multiplication to avoid overflow.
		if duration > p.podMaxBackoffDuration-duration {
			return p.podMaxBackoffDuration
		}
		duration += duration
	}
	return duration
}

func updatePod(oldPodInfo interface{}, newPod *v1.Pod) *framework.QueuedPodInfo {
	pInfo := oldPodInfo.(*framework.QueuedPodInfo)
	pInfo.Update(newPod)
	return pInfo
}

// UnschedulablePods holds pods that cannot be scheduled. This data structure
// is used to implement unschedulablePods.
type UnschedulablePods struct {
	// podInfoMap is a map key by a pod's full-name and the value is a pointer to the QueuedPodInfo.
	podInfoMap map[string]*framework.QueuedPodInfo
	keyFunc    func(*v1.Pod) string
	// metricRecorder updates the counter when elements of an unschedulablePodsMap
	// get added or removed, and it does nothing if it's nil
	metricRecorder metrics.MetricRecorder
}

// Add adds a pod to the unschedulable podInfoMap.
func (u *UnschedulablePods) addOrUpdate(pInfo *framework.QueuedPodInfo) {
	podID := u.keyFunc(pInfo.Pod)
	if _, exists := u.podInfoMap[podID]; !exists && u.metricRecorder != nil {
		u.metricRecorder.Inc()
	}
	u.podInfoMap[podID] = pInfo
}

// Delete deletes a pod from the unschedulable podInfoMap.
func (u *UnschedulablePods) delete(pod *v1.Pod) {
	podID := u.keyFunc(pod)
	if _, exists := u.podInfoMap[podID]; exists && u.metricRecorder != nil {
		u.metricRecorder.Dec()
	}
	delete(u.podInfoMap, podID)
}

// Get returns the QueuedPodInfo if a pod with the same key as the key of the given "pod"
// is found in the map. It returns nil otherwise.
func (u *UnschedulablePods) get(pod *v1.Pod) *framework.QueuedPodInfo {
	podKey := u.keyFunc(pod)
	if pInfo, exists := u.podInfoMap[podKey]; exists {
		return pInfo
	}
	return nil
}

// Clear removes all the entries from the unschedulable podInfoMap.
func (u *UnschedulablePods) clear() {
	u.podInfoMap = make(map[string]*framework.QueuedPodInfo)
	if u.metricRecorder != nil {
		u.metricRecorder.Clear()
	}
}

// newUnschedulablePods initializes a new object of UnschedulablePods.
func newUnschedulablePods(metricRecorder metrics.MetricRecorder) *UnschedulablePods {
	return &UnschedulablePods{
		podInfoMap:     make(map[string]*framework.QueuedPodInfo),
		keyFunc:        util.GetPodFullName,
		metricRecorder: metricRecorder,
	}
}

// nominator is a structure that stores pods nominated to run on nodes.
// It exists because nominatedNodeName of pod objects stored in the structure
// may be different than what scheduler has here. We should be able to find pods
// by their UID and update/delete them.
type nominator struct {
	// podLister is used to verify if the given pod is alive.
	podLister listersv1.PodLister
	// nominatedPods is a map keyed by a node name and the value is a list of
	// pods which are nominated to run on the node. These are pods which can be in
	// the activeQ or unschedulablePods.
	nominatedPods map[string][]*framework.PodInfo
	// nominatedPodToNode is map keyed by a Pod UID to the node name where it is
	// nominated.
	nominatedPodToNode map[types.UID]string

	sync.RWMutex
}

func (npm *nominator) add(pi *framework.PodInfo, nominatingInfo *framework.NominatingInfo) {
	// Always delete the pod if it already exists, to ensure we never store more than
	// one instance of the pod.
	npm.delete(pi.Pod)

	var nodeName string
	if nominatingInfo.Mode() == framework.ModeOverride {
		nodeName = nominatingInfo.NominatedNodeName
	} else if nominatingInfo.Mode() == framework.ModeNoop {
		if pi.Pod.Status.NominatedNodeName == "" {
			return
		}
		nodeName = pi.Pod.Status.NominatedNodeName
	}

	if npm.podLister != nil {
		// If the pod was removed or if it was already scheduled, don't nominate it.
		updatedPod, err := npm.podLister.Pods(pi.Pod.Namespace).Get(pi.Pod.Name)
		if err != nil {
			klog.V(4).InfoS("Pod doesn't exist in podLister, aborted adding it to the nominator", "pod", klog.KObj(pi.Pod))
			return
		}
		if updatedPod.Spec.NodeName != "" {
			klog.V(4).InfoS("Pod is already scheduled to a node, aborted adding it to the nominator", "pod", klog.KObj(pi.Pod), "node", updatedPod.Spec.NodeName)
			return
		}
	}

	npm.nominatedPodToNode[pi.Pod.UID] = nodeName
	for _, npi := range npm.nominatedPods[nodeName] {
		if npi.Pod.UID == pi.Pod.UID {
			klog.V(4).InfoS("Pod already exists in the nominator", "pod", klog.KObj(npi.Pod))
			return
		}
	}
	npm.nominatedPods[nodeName] = append(npm.nominatedPods[nodeName], pi)
}

func (npm *nominator) delete(p *v1.Pod) {
	nnn, ok := npm.nominatedPodToNode[p.UID]
	if !ok {
		return
	}
	for i, np := range npm.nominatedPods[nnn] {
		if np.Pod.UID == p.UID {
			npm.nominatedPods[nnn] = append(npm.nominatedPods[nnn][:i], npm.nominatedPods[nnn][i+1:]...)
			if len(npm.nominatedPods[nnn]) == 0 {
				delete(npm.nominatedPods, nnn)
			}
			break
		}
	}
	delete(npm.nominatedPodToNode, p.UID)
}

// UpdateNominatedPod updates the <oldPod> with <newPod>.
func (npm *nominator) UpdateNominatedPod(oldPod *v1.Pod, newPodInfo *framework.PodInfo) {
	npm.Lock()
	defer npm.Unlock()
	// In some cases, an Update event with no "NominatedNode" present is received right
	// after a node("NominatedNode") is reserved for this pod in memory.
	// In this case, we need to keep reserving the NominatedNode when updating the pod pointer.
	var nominatingInfo *framework.NominatingInfo
	// We won't fall into below `if` block if the Update event represents:
	// (1) NominatedNode info is added
	// (2) NominatedNode info is updated
	// (3) NominatedNode info is removed
	if NominatedNodeName(oldPod) == "" && NominatedNodeName(newPodInfo.Pod) == "" {
		if nnn, ok := npm.nominatedPodToNode[oldPod.UID]; ok {
			// This is the only case we should continue reserving the NominatedNode
			nominatingInfo = &framework.NominatingInfo{
				NominatingMode:    framework.ModeOverride,
				NominatedNodeName: nnn,
			}
		}
	}
	// We update irrespective of the nominatedNodeName changed or not, to ensure
	// that pod pointer is updated.
	npm.delete(oldPod)
	npm.add(newPodInfo, nominatingInfo)
}

// NewPodNominator creates a nominator as a backing of framework.PodNominator.
// A podLister is passed in so as to check if the pod exists
// before adding its nominatedNode info.
func NewPodNominator(podLister listersv1.PodLister) framework.PodNominator {
	return &nominator{
		podLister:          podLister,
		nominatedPods:      make(map[string][]*framework.PodInfo),
		nominatedPodToNode: make(map[types.UID]string),
	}
}

// MakeNextPodFunc returns a function to retrieve the next pod from a given
// scheduling queue
func MakeNextPodFunc(queue SchedulingQueue) func() *framework.QueuedPodInfo {
	return func() *framework.QueuedPodInfo {
		podInfo, err := queue.Pop()
		if err == nil {
			klog.V(4).InfoS("About to try and schedule pod", "pod", klog.KObj(podInfo.Pod))
			for plugin := range podInfo.UnschedulablePlugins {
				metrics.UnschedulableReason(plugin, podInfo.Pod.Spec.SchedulerName).Dec()
			}
			return podInfo
		}
		klog.ErrorS(err, "Error while retrieving next pod from scheduling queue")
		return nil
	}
}

func podInfoKeyFunc(obj interface{}) (string, error) {
	return cache.MetaNamespaceKeyFunc(obj.(*framework.QueuedPodInfo).Pod)
}

// Checks if the Pod may become schedulable upon the event.
// This is achieved by looking up the global clusterEventMap registry.
func (p *PriorityQueue) podMatchesEvent(podInfo *framework.QueuedPodInfo, clusterEvent framework.ClusterEvent) bool {
	if clusterEvent.IsWildCard() {
		return true
	}

	for evt, nameSet := range p.clusterEventMap {
		// Firstly verify if the two ClusterEvents match:
		// - either the registered event from plugin side is a WildCardEvent,
		// - or the two events have identical Resource fields and *compatible* ActionType.
		//   Note the ActionTypes don't need to be *identical*. We check if the ANDed value
		//   is zero or not. In this way, it's easy to tell Update&Delete is not compatible,
		//   but Update&All is.
		evtMatch := evt.IsWildCard() ||
			(evt.Resource == clusterEvent.Resource && evt.ActionType&clusterEvent.ActionType != 0)

		// Secondly verify the plugin name matches.
		// Note that if it doesn't match, we shouldn't continue to search.
		if evtMatch && intersect(nameSet, podInfo.UnschedulablePlugins) {
			return true
		}
	}

	return false
}

func intersect(x, y sets.String) bool {
	if len(x) > len(y) {
		x, y = y, x
	}
	for v := range x {
		if y.Has(v) {
			return true
		}
	}
	return false
}
