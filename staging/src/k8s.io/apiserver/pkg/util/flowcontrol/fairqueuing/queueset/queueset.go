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
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/apiserver/pkg/util/promise/lockingpromise"
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
type queueSet struct {
	clock                clock.PassiveClock
	counter              counter.GoRoutineCounter
	estimatedServiceTime float64

	lock   sync.Mutex
	config fq.QueueSetConfig

	// queues may be longer than the desired number, while the excess
	// queues are still draining.
	queues       []*queue
	virtualTime  float64
	lastRealTime time.Time

	// robinIndex is the index of the last queue dispatched
	robinIndex int

	// numRequestsEnqueued is the number of requests currently waiting
	// in a queue (eg: incremeneted on Enqueue, decremented on Dequue)
	numRequestsEnqueued int

	emptyHandler fq.EmptyHandler
	dealer       *shufflesharding.Dealer
}

// NewQueueSet creates a new QueueSet object
// There is a new QueueSet created for each priority level.
func (qsf queueSetFactory) NewQueueSet(config fq.QueueSetConfig) (fq.QueueSet, error) {
	dealer, err := shufflesharding.NewDealer(config.DesiredNumQueues, config.HandSize)
	if err != nil {
		return nil, errors.Wrap(err, "shuffle sharding dealer creation failed")
	}

	fq := &queueSet{
		config:               config,
		counter:              qsf.counter,
		queues:               createQueues(config.DesiredNumQueues, 0),
		clock:                qsf.clock,
		virtualTime:          0,
		estimatedServiceTime: 60,
		lastRealTime:         qsf.clock.Now(),
		dealer:               dealer,
	}
	return fq, nil
}

// createQueues is a helper method for initializing an array of n queues
func createQueues(n, baseIndex int) []*queue {
	fqqueues := make([]*queue, n)
	for i := 0; i < n; i++ {
		fqqueues[i] = &queue{Index: baseIndex + i, Requests: make([]*request, 0)}
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

	dealer, err := shufflesharding.NewDealer(config.DesiredNumQueues, config.HandSize)
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

// Values passed through a request's Decision
const (
	DecisionExecute    = "execute"
	DecisionReject     = "reject"
	DecisionCancel     = "cancel"
	DecisionTryAnother = "tryAnother"
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
	decision := func() string {
		qs.lockAndSyncTime()
		defer qs.lock.Unlock()
		// A call to Wait while the system is quiescing will be rebuffed by
		// returning `tryAnother=true`.
		if qs.emptyHandler != nil {
			klog.V(5).Infof("QS(%s): rebuffing request %#+v %#+v with TryAnother", qs.config.Name, descr1, descr2)
			return DecisionTryAnother
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
			return DecisionReject
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
					req.Decision.Set(DecisionCancel)
				}
				qs.goroutineDoneOrBlocked()
			}()
		}

		// ========================================================================
		// Step 4:
		// The final step in Wait is to wait on a decision from
		// somewhere and then act on it.
		decisionAny := req.Decision.GetLocked()
		var decisionStr string
		switch d := decisionAny.(type) {
		case string:
			decisionStr = d
		default:
			klog.Errorf("QS(%s): Impossible decision %#+v (of type %T) for request %#+v %#+v", qs.config.Name, decisionAny, decisionAny, descr1, descr2)
			decisionStr = DecisionExecute
		}
		switch decisionStr {
		case DecisionReject:
			klog.V(5).Infof("QS(%s): request %#+v %#+v timed out after being enqueued\n", qs.config.Name, descr1, descr2)
			metrics.AddReject(qs.config.Name, "time-out")
		case DecisionCancel:
			qs.syncTimeLocked()
			// TODO(aaron-prindle) add metrics to these two cases
			if req.IsWaiting {
				klog.V(5).Infof("QS(%s): Ejecting request %#+v %#+v from its queue", qs.config.Name, descr1, descr2)
				// remove the request from the queue as it has timed out
				for i := range req.Queue.Requests {
					if req == req.Queue.Requests[i] {
						// remove the request
						req.Queue.Requests = append(req.Queue.Requests[:i],
							req.Queue.Requests[i+1:]...)
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
		return decisionStr
	}()
	switch decision {
	case DecisionTryAnother:
		return true, false, func() {}
	case DecisionReject:
		return false, false, func() {}
	case DecisionCancel:
		return false, false, func() {}
	default:
		if decision != DecisionExecute {
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
	timesincelast := realNow.Sub(qs.lastRealTime).Seconds()
	qs.lastRealTime = realNow
	qs.virtualTime += timesincelast * qs.getVirtualTimeRatio()
}

// getVirtualTimeRatio calculates the rate at which virtual time has
// been advancing, according to the logic in `doc.go`.
func (qs *queueSet) getVirtualTimeRatio() float64 {
	activeQueues := 0
	reqs := 0
	for _, queue := range qs.queues {
		reqs += queue.RequestsExecuting
		if len(queue.Requests) > 0 || queue.RequestsExecuting > 0 {
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
		Decision:    lockingpromise.NewLockingPromise(&qs.lock, qs.counter),
		ArrivalTime: qs.clock.Now(),
		Queue:       queue,
		descr1:      descr1,
		descr2:      descr2,
	}
	if ok := qs.rejectOrEnqueueLocked(req); !ok {
		return nil
	}
	metrics.ObserveQueueLength(qs.config.Name, len(queue.Requests))
	return req
}

// chooseQueueIndexLocked uses shuffle sharding to select a queue index
// using the given hashValue and the shuffle sharding parameters of the queueSet.
func (qs *queueSet) chooseQueueIndexLocked(hashValue uint64, descr1, descr2 interface{}) int {
	bestQueueIdx := -1
	bestQueueLen := int(math.MaxInt32)
	// the dealer uses the current desired number of queues, which is no larger than the number in `qs.queues`.
	qs.dealer.Deal(hashValue, func(queueIdx int) {
		thisLen := len(qs.queues[queueIdx].Requests)
		klog.V(7).Infof("QS(%s): For request %#+v %#+v considering queue %d of length %d", qs.config.Name, descr1, descr2, queueIdx, thisLen)
		if thisLen < bestQueueLen {
			bestQueueIdx, bestQueueLen = queueIdx, thisLen
		}
	})
	klog.V(6).Infof("QS(%s): For request %#+v %#+v chose queue %d, had %d waiting & %d executing", qs.config.Name, descr1, descr2, bestQueueIdx, bestQueueLen, qs.queues[bestQueueIdx].RequestsExecuting)
	return bestQueueIdx
}

// removeTimedOutRequestsFromQueueLocked rejects old requests that have been enqueued
// past the requestWaitLimit
func (qs *queueSet) removeTimedOutRequestsFromQueueLocked(queue *queue) {
	timeoutIdx := -1
	now := qs.clock.Now()
	reqs := queue.Requests
	// reqs are sorted oldest -> newest
	// can short circuit loop (break) if oldest requests are not timing out
	// as newer requests also will not have timed out

	// now - requestWaitLimit = waitLimit
	waitLimit := now.Add(-qs.config.RequestWaitLimit)
	for i, req := range reqs {
		if waitLimit.After(req.ArrivalTime) {
			req.Decision.SetLocked(DecisionReject)
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
		queue.Requests = reqs[removeIdx:]
		// decrement the # of requestsEnqueued
		qs.numRequestsEnqueued -= removeIdx
	}
}

// rejectOrEnqueueLocked rejects or enqueues the newly arrived request if
// resource criteria isn't met
func (qs *queueSet) rejectOrEnqueueLocked(request *request) bool {
	queue := request.Queue
	curQueueLength := len(queue.Requests)
	// rejects the newly arrived request if resource criteria not met
	if qs.getRequestsExecutingLocked() >= qs.config.ConcurrencyLimit &&
		curQueueLength >= qs.config.QueueLengthLimit {
		return false
	}

	qs.enqueueLocked(request)
	return true
}

// enqueues a request into an queueSet
func (qs *queueSet) enqueueLocked(request *request) {
	queue := request.Queue
	if len(queue.Requests) == 0 && queue.RequestsExecuting == 0 {
		// the queue’s virtual start time is set to the virtual time.
		queue.VirtualStart = qs.virtualTime
		if klog.V(6) {
			klog.Infof("QS(%s) at r=%s v=%.9fs: initialized queue %d virtual start time due to request %#+v %#+v", qs.config.Name, qs.clock.Now().Format(nsTimeFmt), queue.VirtualStart, queue.Index, request.descr1, request.descr2)
		}
	}
	queue.Enqueue(request)
	qs.numRequestsEnqueued++
	metrics.UpdateFlowControlRequestsInQueue(qs.config.Name, qs.numRequestsEnqueued)
}

// getRequestsExecutingLocked gets the # of requests which are "executing":
// this is the # of requests which have been dispatched but have not
// finished (via the finishRequestLocked method invoked after service)
func (qs *queueSet) getRequestsExecutingLocked() int {
	total := 0
	for _, queue := range qs.queues {
		total += queue.RequestsExecuting
	}
	return total
}

// dispatchAsMuchAsPossibleLocked runs a loop, as long as there
// are non-empty queues and the number currently executing is less than the
// assured concurrency value.  The body of the loop uses the fair queuing
// technique to pick a queue, dequeue the request at the head of that
// queue, increment the count of the number executing, and send true
// to the request's channel.
func (qs *queueSet) dispatchAsMuchAsPossibleLocked() {
	for qs.numRequestsEnqueued != 0 && qs.getRequestsExecutingLocked() < qs.config.ConcurrencyLimit {
		_, ok := qs.dispatchLocked()
		if !ok {
			break
		}
	}
}

// dispatchLocked is a convenience method for dequeueing requests that
// require a message to be sent through the requests channel
// this is a required pattern for the QueueSet the queueSet supports
func (qs *queueSet) dispatchLocked() (*request, bool) {
	queue := qs.selectQueueLocked()
	if queue == nil {
		return nil, false
	}
	request, ok := queue.Dequeue()
	if !ok {
		return nil, false
	}
	request.StartTime = qs.clock.Now()
	// request dequeued, service has started
	queue.RequestsExecuting++
	qs.numRequestsEnqueued--
	if klog.V(6) {
		klog.Infof("QS(%s) at r=%s v=%.9fs: dispatching request %#+v %#+v from queue %d with virtual start time %.9fs, queue will have %d waiting & %d executing", qs.config.Name, request.StartTime.Format(nsTimeFmt), qs.virtualTime, request.descr1, request.descr2, queue.Index, queue.VirtualStart, len(queue.Requests), queue.RequestsExecuting)
	}
	// When a request is dequeued for service -> qs.VirtualStart += G
	queue.VirtualStart += qs.estimatedServiceTime
	metrics.UpdateFlowControlRequestsExecuting(qs.config.Name, queue.RequestsExecuting)
	request.Decision.SetLocked(DecisionExecute)
	return request, ok
}

/// selectQueueLocked selects the minimum virtualFinish time from the set of queues
// the starting queue is selected via roundrobin
func (qs *queueSet) selectQueueLocked() *queue {
	minVirtualFinish := math.Inf(1)
	var minQueue *queue
	var minIndex int
	for range qs.queues {
		qs.robinIndex = (qs.robinIndex + 1) % len(qs.queues)
		queue := qs.queues[qs.robinIndex]
		if len(queue.Requests) != 0 {
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
	S := qs.clock.Since(r.StartTime).Seconds()

	// When a request finishes being served, and the actual service time was S,
	// the queue’s virtual start time is decremented by G - S.
	r.Queue.VirtualStart -= qs.estimatedServiceTime - S

	// request has finished, remove from requests executing
	r.Queue.RequestsExecuting--

	if klog.V(6) {
		klog.Infof("QS(%s) at r=%s v=%.9fs: request %#+v %#+v finished, adjusted queue %d virtual start time to %.9fs due to service time %.9fs, queue will have %d waiting & %d executing", qs.config.Name, qs.clock.Now().Format(nsTimeFmt), qs.virtualTime, r.descr1, r.descr2, r.Queue.Index, r.Queue.VirtualStart, S, len(r.Queue.Requests), r.Queue.RequestsExecuting)
	}

	// Logic to remove quiesced queues
	// >= as Index=25 is out of bounds for DesiredNum=25 [0...24]
	if r.Queue.Index >= qs.config.DesiredNumQueues &&
		len(r.Queue.Requests) == 0 &&
		r.Queue.RequestsExecuting == 0 {
		qs.queues = removeQueueAndUpdateIndexes(qs.queues, r.Queue.Index)

		// decrement here to maintain the invariant that (qs.robinIndex+1) % numQueues
		// is the index of the next queue after the one last dispatched from
		if qs.robinIndex >= r.Queue.Index {
			qs.robinIndex--
		}

		// At this point, if the qs is quiescing,
		// has zero requests executing, and has zero requests enqueued
		// then a call to the EmptyHandler should be forked.
		qs.maybeForkEmptyHandlerLocked()
	}
}

// removeQueueAndUpdateIndexes uses reslicing to remove an index from a slice
// and then updates the 'Index' field of the queues to be correct
func removeQueueAndUpdateIndexes(queues []*queue, index int) []*queue {
	keptQueues := append(queues[:index], queues[index+1:]...)
	for i := index; i < len(keptQueues); i++ {
		keptQueues[i].Index--
	}
	return keptQueues
}

func (qs *queueSet) maybeForkEmptyHandlerLocked() {
	if qs.emptyHandler != nil && qs.numRequestsEnqueued == 0 &&
		qs.getRequestsExecutingLocked() == 0 {
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
