/*
Copyright 2019 The Kubernetes Authors.

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

package queueset

import (
	"context"
	"math"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/runtime"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise/lockingpromise"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/apiserver/pkg/util/shufflesharding"
	"k8s.io/klog"
)

const nsTimeFmt = "2006-01-02 15:04:05.000000000"

// queueSetFactory implements the QueueSetFactory interface
// queueSetFactory makes QueueSet objects.
type queueSetFactory struct {
	counter counter.GoRoutineCounter
	clock   clock.PassiveClock
}

// NewQueueSetFactory creates a new QueueSetFactory object
func NewQueueSetFactory(c clock.PassiveClock, counter counter.GoRoutineCounter) fq.QueueSetFactory {
	return &queueSetFactory{
		counter: counter,
		clock:   c,
	}
}

// queueSet implements the Fair Queuing for Server Requests technique
// described in this package's doc, and a pointer to one implements
// the QueueSet interface.  The clock, GoRoutineCounter, and estimated
// service time should not be changed; the fields listed after the
// lock must be accessed only while holding the lock.
// This is not yet designed to support limiting concurrency without
// queuing (this will need to be added soon).
type queueSet struct {
	clock                clock.PassiveClock
	counter              counter.GoRoutineCounter
	estimatedServiceTime float64

	lock sync.Mutex

	// config holds the current configuration.  Its DesiredNumQueues
	// may be less than the current number of queues.  If its
	// DesiredNumQueues is zero then its other queuing parameters
	// retain the settings they had when DesiredNumQueues was last
	// non-zero (if ever).
	config fq.QueueSetConfig

	// queues may be longer than the desired number, while the excess
	// queues are still draining.
	queues []*queue

	// virtualTime is the number of virtual seconds since process startup
	virtualTime float64

	// lastRealTime is what `clock.Now()` yielded when `virtualTime` was last updated
	lastRealTime time.Time

	// robinIndex is the index of the last queue dispatched
	robinIndex int

	// totRequestsWaiting is the sum, over all the queues, of the
	// number of requests waiting in that queue
	totRequestsWaiting int

	// totRequestsExecuting is the total number of requests of this
	// queueSet that are currently executing.  That is the same as the
	// sum, over all the queues, of the number of requests executing
	// from that queue.
	totRequestsExecuting int

	emptyHandler fq.EmptyHandler
	dealer       *shufflesharding.Dealer
}

// NewQueueSet creates a new QueueSet object.
// There is a new QueueSet created for each priority level.
func (qsf queueSetFactory) NewQueueSet(config fq.QueueSetConfig) (fq.QueueSet, error) {
	fq := &queueSet{
		clock:                qsf.clock,
		counter:              qsf.counter,
		estimatedServiceTime: 60,
		config:               config,
		lastRealTime:         qsf.clock.Now(),
	}
	err := fq.SetConfiguration(config)
	if err != nil {
		return nil, err
	}
	return fq, nil
}

// createQueues is a helper method for initializing an array of n queues
func createQueues(n, baseIndex int) []*queue {
	fqqueues := make([]*queue, n)
	for i := 0; i < n; i++ {
		fqqueues[i] = &queue{index: baseIndex + i, requests: make([]*request, 0)}
	}
	return fqqueues
}

// SetConfiguration is used to set the configuration for a queueSet
// update handling for when fields are updated is handled here as well -
// eg: if DesiredNum is increased, SetConfiguration reconciles by
// adding more queues.
func (qs *queueSet) SetConfiguration(config fq.QueueSetConfig) error {
	qs.lockAndSyncTime()
	defer qs.lock.Unlock()
	var dealer *shufflesharding.Dealer

	if config.DesiredNumQueues > 0 {
		var err error
		dealer, err = shufflesharding.NewDealer(config.DesiredNumQueues, config.HandSize)
		if err != nil {
			return errors.Wrap(err, "shuffle sharding dealer creation failed")
		}
		// Adding queues is the only thing that requires immediate action
		// Removing queues is handled by omitting indexes >DesiredNum from
		// chooseQueueIndexLocked
		numQueues := len(qs.queues)
		if config.DesiredNumQueues > numQueues {
			qs.queues = append(qs.queues,
				createQueues(config.DesiredNumQueues-numQueues, len(qs.queues))...)
		}
	} else {
		config.QueueLengthLimit = qs.config.QueueLengthLimit
		config.HandSize = qs.config.HandSize
		config.RequestWaitLimit = qs.config.RequestWaitLimit
	}

	qs.config = config
	qs.dealer = dealer

	qs.dispatchAsMuchAsPossibleLocked()
	return nil
}

// Quiesce controls whether the QueueSet is operating normally or is quiescing.
// A quiescing QueueSet drains as normal but does not admit any
// new requests. Passing a non-nil handler means the system should
// be quiescing, a nil handler means the system should operate
// normally. A call to Wait while the system is quiescing
// will be rebuffed by returning tryAnother=true. If all the
// queues have no requests waiting nor executing while the system
// is quiescing then the handler will eventually be called with no
// locks held (even if the system becomes non-quiescing between the
// triggering state and the required call).
func (qs *queueSet) Quiesce(eh fq.EmptyHandler) {
	qs.lock.Lock()
	defer qs.lock.Unlock()
	qs.emptyHandler = eh
	if eh == nil {
		return
	}
	// Here we check whether there are any requests queued or executing and
	// if not then fork an invocation of the EmptyHandler.
	qs.maybeForkEmptyHandlerLocked()
}

// A decision about a request
type requestDecision int

// Values passed through a request's decision
const (
	decisionExecute requestDecision = iota
	decisionReject
	decisionCancel
	decisionTryAnother
)

// Wait uses the given hashValue as the source of entropy as it
// shuffle-shards a request into a queue and waits for a decision on
// what to do with that request.  The descr1 and descr2 values play no
// role in the logic but appear in log messages; we use two because
// the main client characterizes a request by two items that, if
// bundled together in a larger data structure, would lose interesting
// details when formatted.  If tryAnother==true at return then the
// QueueSet has become undesirable and the client should try to find a
// different QueueSet to use; execute and afterExecution are
// irrelevant in this case.  Otherwise, if execute then the client
// should start executing the request and, once the request finishes
// execution or is canceled, call afterExecution().  Otherwise the
// client should not execute the request and afterExecution is
// irrelevant.
func (qs *queueSet) Wait(ctx context.Context, hashValue uint64, descr1, descr2 interface{}) (tryAnother, execute bool, afterExecution func()) {
	var req *request
	decision := func() requestDecision {
		qs.lockAndSyncTime()
		defer qs.lock.Unlock()
		// A call to Wait while the system is quiescing will be rebuffed by
		// returning `tryAnother=true`.
		if qs.emptyHandler != nil {
			klog.V(5).Infof("QS(%s): rebuffing request %#+v %#+v with TryAnother", qs.config.Name, descr1, descr2)
			return decisionTryAnother
		}

		// ========================================================================
		// Step 0:
		// Apply only concurrency limit, if zero queues desired
		if qs.config.DesiredNumQueues < 1 {
			if qs.totRequestsExecuting >= qs.config.ConcurrencyLimit {
				klog.V(5).Infof("QS(%s): rejecting request %#+v %#+v because %d are executing and the limit is %d", qs.config.Name, descr1, descr2, qs.totRequestsExecuting, qs.config.ConcurrencyLimit)
				return decisionReject
			}
			req = qs.dispatchSansQueue(descr1, descr2)
			return decisionExecute
		}

		// ========================================================================
		// Step 1:
		// 1) Start with shuffle sharding, to pick a queue.
		// 2) Reject old requests that have been waiting too long
		// 3) Reject current request if there is not enough concurrency shares and
		// we are at max queue length
		// 4) If not rejected, create a request and enqueue
		req = qs.timeoutOldRequestsAndRejectOrEnqueueLocked(hashValue, descr1, descr2)
		// req == nil means that the request was rejected - no remaining
		// concurrency shares and at max queue length already
		if req == nil {
			klog.V(5).Infof("QS(%s): rejecting request %#+v %#+v due to queue full", qs.config.Name, descr1, descr2)
			metrics.AddReject(qs.config.Name, "queue-full")
			return decisionReject
		}

		// ========================================================================
		// Step 2:
		// The next step is to invoke the method that dequeues as much
		// as possible.
		// This method runs a loop, as long as there are non-empty
		// queues and the number currently executing is less than the
		// assured concurrency value.  The body of the loop uses the
		// fair queuing technique to pick a queue and dispatch a
		// request from that queue.
		qs.dispatchAsMuchAsPossibleLocked()

		// ========================================================================
		// Step 3:

		// Set up a relay from the context's Done channel to the world
		// of well-counted goroutines. We Are Told that every
		// request's context's Done channel gets closed by the time
		// the request is done being processed.
		doneCh := ctx.Done()
		if doneCh != nil {
			qs.preCreateOrUnblockGoroutine()
			go func() {
				defer runtime.HandleCrash()
				qs.goroutineDoneOrBlocked()
				select {
				case <-doneCh:
					klog.V(6).Infof("QS(%s): Context of request %#+v %#+v is Done", qs.config.Name, descr1, descr2)
					req.decision.Set(decisionCancel)
				}
				qs.goroutineDoneOrBlocked()
			}()
		}

		// ========================================================================
		// Step 4:
		// The final step in Wait is to wait on a decision from
		// somewhere and then act on it.
		decisionAny := req.decision.GetLocked()
		var decision requestDecision
		switch dec := decisionAny.(type) {
		case requestDecision:
			decision = dec
		default:
			klog.Errorf("QS(%s): Impossible decision %#+v (of type %T) for request %#+v %#+v", qs.config.Name, decisionAny, decisionAny, descr1, descr2)
			decision = decisionExecute
		}
		switch decision {
		case decisionReject:
			klog.V(5).Infof("QS(%s): request %#+v %#+v timed out after being enqueued\n", qs.config.Name, descr1, descr2)
			metrics.AddReject(qs.config.Name, "time-out")
		case decisionCancel:
			qs.syncTimeLocked()
			// TODO(aaron-prindle) add metrics to these two cases
			if req.isWaiting {
				klog.V(5).Infof("QS(%s): Ejecting request %#+v %#+v from its queue", qs.config.Name, descr1, descr2)
				// remove the request from the queue as it has timed out
				for i := range req.queue.requests {
					if req == req.queue.requests[i] {
						// remove the request
						req.queue.requests = append(req.queue.requests[:i],
							req.queue.requests[i+1:]...)
						break
					}
				}
				// At this point, if the qs is quiescing,
				// has zero requests executing, and has zero requests enqueued
				// then a call to the EmptyHandler should be forked.
				qs.maybeForkEmptyHandlerLocked()
			} else {
				klog.V(5).Infof("QS(%s): request %#+v %#+v canceled shortly after dispatch", qs.config.Name, descr1, descr2)
			}
		}
		return decision
	}()
	switch decision {
	case decisionTryAnother:
		return true, false, func() {}
	case decisionReject, decisionCancel:
		return false, false, func() {}
	default:
		if decision != decisionExecute {
			klog.Errorf("Impossible decision %q", decision)
		}
		return false, true, func() {
			qs.finishRequestAndDispatchAsMuchAsPossible(req)
		}
	}
}

// lockAndSyncTime acquires the lock and updates the virtual time.
// Doing them together avoids the mistake of modify some queue state
// before calling syncTimeLocked.
func (qs *queueSet) lockAndSyncTime() {
	qs.lock.Lock()
	qs.syncTimeLocked()
}

// syncTimeLocked updates the virtual time based on the assumption
// that the current state of the queues has been in effect since
// `qs.lastRealTime`.  Thus, it should be invoked after acquiring the
// lock and before modifying the state of any queue.
func (qs *queueSet) syncTimeLocked() {
	realNow := qs.clock.Now()
	timeSinceLast := realNow.Sub(qs.lastRealTime).Seconds()
	qs.lastRealTime = realNow
	qs.virtualTime += timeSinceLast * qs.getVirtualTimeRatio()
}

// getVirtualTimeRatio calculates the rate at which virtual time has
// been advancing, according to the logic in `doc.go`.
func (qs *queueSet) getVirtualTimeRatio() float64 {
	activeQueues := 0
	reqs := 0
	for _, queue := range qs.queues {
		reqs += queue.requestsExecuting
		if len(queue.requests) > 0 || queue.requestsExecuting > 0 {
			activeQueues++
		}
	}
	if activeQueues == 0 {
		return 0
	}
	return math.Min(float64(reqs), float64(qs.config.ConcurrencyLimit)) / float64(activeQueues)
}

// timeoutOldRequestsAndRejectOrEnqueueLocked encapsulates the logic required
// to validate and enqueue a request for the queueSet/QueueSet:
// 1) Start with shuffle sharding, to pick a queue.
// 2) Reject old requests that have been waiting too long
// 3) Reject current request if there is not enough concurrency shares and
// we are at max queue length
// 4) If not rejected, create a request and enqueue
// returns the enqueud request on a successful enqueue
// returns nil in the case that there is no available concurrency or
// the queuelengthlimit has been reached
func (qs *queueSet) timeoutOldRequestsAndRejectOrEnqueueLocked(hashValue uint64, descr1, descr2 interface{}) *request {
	//	Start with the shuffle sharding, to pick a queue.
	queueIdx := qs.chooseQueueIndexLocked(hashValue, descr1, descr2)
	queue := qs.queues[queueIdx]
	// The next step is the logic to reject requests that have been waiting too long
	qs.removeTimedOutRequestsFromQueueLocked(queue)
	// NOTE: currently timeout is only checked for each new request.  This means that there can be
	// requests that are in the queue longer than the timeout if there are no new requests
	// We prefer the simplicity over the promptness, at least for now.

	// Create a request and enqueue
	req := &request{
		decision:    lockingpromise.NewLockingPromise(&qs.lock, qs.counter),
		arrivalTime: qs.clock.Now(),
		queue:       queue,
		descr1:      descr1,
		descr2:      descr2,
	}
	if ok := qs.rejectOrEnqueueLocked(req); !ok {
		return nil
	}
	metrics.ObserveQueueLength(qs.config.Name, len(queue.requests))
	return req
}

// chooseQueueIndexLocked uses shuffle sharding to select a queue index
// using the given hashValue and the shuffle sharding parameters of the queueSet.
func (qs *queueSet) chooseQueueIndexLocked(hashValue uint64, descr1, descr2 interface{}) int {
	bestQueueIdx := -1
	bestQueueLen := int(math.MaxInt32)
	// the dealer uses the current desired number of queues, which is no larger than the number in `qs.queues`.
	qs.dealer.Deal(hashValue, func(queueIdx int) {
		thisLen := len(qs.queues[queueIdx].requests)
		klog.V(7).Infof("QS(%s): For request %#+v %#+v considering queue %d of length %d", qs.config.Name, descr1, descr2, queueIdx, thisLen)
		if thisLen < bestQueueLen {
			bestQueueIdx, bestQueueLen = queueIdx, thisLen
		}
	})
	klog.V(6).Infof("QS(%s): For request %#+v %#+v chose queue %d, had %d waiting & %d executing", qs.config.Name, descr1, descr2, bestQueueIdx, bestQueueLen, qs.queues[bestQueueIdx].requestsExecuting)
	return bestQueueIdx
}

// removeTimedOutRequestsFromQueueLocked rejects old requests that have been enqueued
// past the requestWaitLimit
func (qs *queueSet) removeTimedOutRequestsFromQueueLocked(queue *queue) {
	timeoutIdx := -1
	now := qs.clock.Now()
	reqs := queue.requests
	// reqs are sorted oldest -> newest
	// can short circuit loop (break) if oldest requests are not timing out
	// as newer requests also will not have timed out

	// now - requestWaitLimit = waitLimit
	waitLimit := now.Add(-qs.config.RequestWaitLimit)
	for i, req := range reqs {
		if waitLimit.After(req.arrivalTime) {
			req.decision.SetLocked(decisionReject)
			// get index for timed out requests
			timeoutIdx = i
		} else {
			break
		}
	}
	// remove timed out requests from queue
	if timeoutIdx != -1 {
		// timeoutIdx + 1 to remove the last timeout req
		removeIdx := timeoutIdx + 1
		// remove all the timeout requests
		queue.requests = reqs[removeIdx:]
		// decrement the # of requestsEnqueued
		qs.totRequestsWaiting -= removeIdx
	}
}

// rejectOrEnqueueLocked rejects or enqueues the newly arrived request if
// resource criteria isn't met
func (qs *queueSet) rejectOrEnqueueLocked(request *request) bool {
	queue := request.queue
	curQueueLength := len(queue.requests)
	// rejects the newly arrived request if resource criteria not met
	if qs.totRequestsExecuting >= qs.config.ConcurrencyLimit &&
		curQueueLength >= qs.config.QueueLengthLimit {
		return false
	}

	qs.enqueueLocked(request)
	return true
}

// enqueues a request into an queueSet
func (qs *queueSet) enqueueLocked(request *request) {
	queue := request.queue
	if len(queue.requests) == 0 && queue.requestsExecuting == 0 {
		// the queue’s virtual start time is set to the virtual time.
		queue.virtualStart = qs.virtualTime
		if klog.V(6) {
			klog.Infof("QS(%s) at r=%s v=%.9fs: initialized queue %d virtual start time due to request %#+v %#+v", qs.config.Name, qs.clock.Now().Format(nsTimeFmt), queue.virtualStart, queue.index, request.descr1, request.descr2)
		}
	}
	queue.Enqueue(request)
	qs.totRequestsWaiting++
	metrics.UpdateFlowControlRequestsInQueue(qs.config.Name, qs.totRequestsWaiting)
}

// dispatchAsMuchAsPossibleLocked runs a loop, as long as there
// are non-empty queues and the number currently executing is less than the
// assured concurrency value.  The body of the loop uses the fair queuing
// technique to pick a queue, dequeue the request at the head of that
// queue, increment the count of the number executing, and send true
// to the request's channel.
func (qs *queueSet) dispatchAsMuchAsPossibleLocked() {
	for qs.totRequestsWaiting != 0 && qs.totRequestsExecuting < qs.config.ConcurrencyLimit {
		ok := qs.dispatchLocked()
		if !ok {
			break
		}
	}
}

func (qs *queueSet) dispatchSansQueue(descr1, descr2 interface{}) *request {
	now := qs.clock.Now()
	req := &request{
		startTime:   now,
		arrivalTime: now,
		descr1:      descr1,
		descr2:      descr2,
	}
	qs.totRequestsExecuting++
	if klog.V(5) {
		klog.Infof("QS(%s) at r=%s v=%.9fs: immediate dispatch of request %#+v %#+v, qs will have %d executing", qs.config.Name, now.Format(nsTimeFmt), qs.virtualTime, descr1, descr2, qs.totRequestsExecuting)
	}
	metrics.UpdateFlowControlRequestsExecuting(qs.config.Name, qs.totRequestsExecuting)
	return req
}

// dispatchLocked uses the Fair Queuing for Server Requests method to
// select a queue and dispatch the oldest request in that queue.  The
// return value indicates whether a request was dispatched; this will
// be false when there are no requests waiting in any queue.
func (qs *queueSet) dispatchLocked() bool {
	queue := qs.selectQueueLocked()
	if queue == nil {
		return false
	}
	request, ok := queue.Dequeue()
	if !ok { // This should never happen.  But if it does...
		return false
	}
	request.startTime = qs.clock.Now()
	// request dequeued, service has started
	qs.totRequestsWaiting--
	qs.totRequestsExecuting++
	queue.requestsExecuting++
	if klog.V(6) {
		klog.Infof("QS(%s) at r=%s v=%.9fs: dispatching request %#+v %#+v from queue %d with virtual start time %.9fs, queue will have %d waiting & %d executing", qs.config.Name, request.startTime.Format(nsTimeFmt), qs.virtualTime, request.descr1, request.descr2, queue.index, queue.virtualStart, len(queue.requests), queue.requestsExecuting)
	}
	// When a request is dequeued for service -> qs.virtualStart += G
	queue.virtualStart += qs.estimatedServiceTime
	metrics.UpdateFlowControlRequestsExecuting(qs.config.Name, qs.totRequestsExecuting)
	request.decision.SetLocked(decisionExecute)
	return ok
}

// selectQueueLocked examines the queues in round robin order and
// returns the first one of those for which the virtual finish time of
// the oldest waiting request is minimal.
func (qs *queueSet) selectQueueLocked() *queue {
	minVirtualFinish := math.Inf(1)
	var minQueue *queue
	var minIndex int
	nq := len(qs.queues)
	for range qs.queues {
		qs.robinIndex = (qs.robinIndex + 1) % nq
		queue := qs.queues[qs.robinIndex]
		if len(queue.requests) != 0 {
			currentVirtualFinish := queue.GetVirtualFinish(0, qs.estimatedServiceTime)
			if currentVirtualFinish < minVirtualFinish {
				minVirtualFinish = currentVirtualFinish
				minQueue = queue
				minIndex = qs.robinIndex
			}
		}
	}
	// we set the round robin indexing to start at the chose queue
	// for the next round.  This way the non-selected queues
	// win in the case that the virtual finish times are the same
	qs.robinIndex = minIndex
	return minQueue
}

// finishRequestAndDispatchAsMuchAsPossible is a convenience method
// which calls finishRequest for a given request and then dispatches
// as many requests as possible.  This is all of what needs to be done
// once a request finishes execution or is canceled.
func (qs *queueSet) finishRequestAndDispatchAsMuchAsPossible(req *request) {
	qs.lockAndSyncTime()
	defer qs.lock.Unlock()

	qs.finishRequestLocked(req)
	qs.dispatchAsMuchAsPossibleLocked()
}

// finishRequestLocked is a callback that should be used when a
// previously dispatched request has completed it's service.  This
// callback updates important state in the queueSet
func (qs *queueSet) finishRequestLocked(r *request) {
	qs.totRequestsExecuting--
	metrics.UpdateFlowControlRequestsExecuting(qs.config.Name, qs.totRequestsExecuting)

	if r.queue == nil {
		if klog.V(6) {
			klog.Infof("QS(%s) at r=%s v=%.9fs: request %#+v %#+v finished, qs will have %d executing", qs.config.Name, qs.clock.Now().Format(nsTimeFmt), qs.virtualTime, r.descr1, r.descr2, qs.totRequestsExecuting)
		}
		return
	}

	S := qs.clock.Since(r.startTime).Seconds()

	// When a request finishes being served, and the actual service time was S,
	// the queue’s virtual start time is decremented by G - S.
	r.queue.virtualStart -= qs.estimatedServiceTime - S

	// request has finished, remove from requests executing
	r.queue.requestsExecuting--

	if klog.V(6) {
		klog.Infof("QS(%s) at r=%s v=%.9fs: request %#+v %#+v finished, adjusted queue %d virtual start time to %.9fs due to service time %.9fs, queue will have %d waiting & %d executing", qs.config.Name, qs.clock.Now().Format(nsTimeFmt), qs.virtualTime, r.descr1, r.descr2, r.queue.index, r.queue.virtualStart, S, len(r.queue.requests), r.queue.requestsExecuting)
	}

	// If there are more queues than desired and this one has no
	// requests then remove it
	if len(qs.queues) > qs.config.DesiredNumQueues &&
		len(r.queue.requests) == 0 &&
		r.queue.requestsExecuting == 0 {
		qs.queues = removeQueueAndUpdateIndexes(qs.queues, r.queue.index)

		// decrement here to maintain the invariant that (qs.robinIndex+1) % numQueues
		// is the index of the next queue after the one last dispatched from
		if qs.robinIndex >= r.queue.index {
			qs.robinIndex--
		}

		// At this point, if the qs is quiescing,
		// has zero requests executing, and has zero requests enqueued
		// then a call to the EmptyHandler should be forked.
		qs.maybeForkEmptyHandlerLocked()
	}
}

// removeQueueAndUpdateIndexes uses reslicing to remove an index from a slice
// and then updates the 'index' field of the queues to be correct
func removeQueueAndUpdateIndexes(queues []*queue, index int) []*queue {
	keptQueues := append(queues[:index], queues[index+1:]...)
	for i := index; i < len(keptQueues); i++ {
		keptQueues[i].index--
	}
	return keptQueues
}

func (qs *queueSet) maybeForkEmptyHandlerLocked() {
	if qs.emptyHandler != nil && qs.totRequestsWaiting == 0 &&
		qs.totRequestsExecuting == 0 {
		qs.preCreateOrUnblockGoroutine()
		go func(eh fq.EmptyHandler) {
			defer runtime.HandleCrash()
			defer qs.goroutineDoneOrBlocked()
			eh.HandleEmpty()
		}(qs.emptyHandler)
	}
}

// preCreateOrUnblockGoroutine needs to be called before creating a
// goroutine associated with this queueSet or unblocking a blocked
// one, to properly update the accounting used in testing.
func (qs *queueSet) preCreateOrUnblockGoroutine() {
	qs.counter.Add(1)
}

// goroutineDoneOrBlocked needs to be called at the end of every
// goroutine associated with this queueSet or when such a goroutine is
// about to wait on some other goroutine to do something; this is to
// properly update the accounting used in testing.
func (qs *queueSet) goroutineDoneOrBlocked() {
	qs.counter.Add(-1)
}
