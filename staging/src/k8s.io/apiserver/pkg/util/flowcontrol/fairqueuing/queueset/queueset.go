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
	"k8s.io/apiserver/pkg/util/shufflesharding"
	"k8s.io/klog"
)

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

// NewQueueSet creates a new QueueSet object
// There is a new QueueSet created for each priority level.
func (qsf queueSetFactory) NewQueueSet(config fq.QueueSetConfig) (fq.QueueSet, error) {
	return newQueueSet(config, qsf.clock, qsf.counter)
}

// queueSet is a fair queuing implementation designed with three major differences:
// 1) dispatches requests to be served rather than requests to be transmitted
// 2) serves multiple requests at once
// 3) a request's service time is not known until it finishes
// implementation of:
// https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md
type queueSet struct {
	lock                 sync.Mutex
	config               fq.QueueSetConfig
	counter              counter.GoRoutineCounter
	clock                clock.PassiveClock
	queues               []*fq.Queue
	virtualTime          float64
	estimatedServiceTime float64
	lastRealTime         time.Time
	robinIndex           int
	// numRequestsEnqueued is the number of requests currently enqueued
	// (eg: incremeneted on Enqueue, decremented on Dequue)
	numRequestsEnqueued int
	emptyHandler        fq.EmptyHandler
	dealer              *shufflesharding.Dealer
}

// initQueues is a helper method for initializing an array of n queues
func initQueues(n, baseIndex int) []*fq.Queue {
	fqqueues := make([]*fq.Queue, n)
	for i := 0; i < n; i++ {
		fqqueues[i] = &fq.Queue{Index: baseIndex + i, Requests: make([]*fq.Request, 0)}
	}
	return fqqueues
}

// newQueueSet creates a new queueSet from passed in parameters
func newQueueSet(config fq.QueueSetConfig, c clock.PassiveClock, counter counter.GoRoutineCounter) (*queueSet, error) {
	dealer, err := shufflesharding.NewDealer(config.DesiredNumQueues, config.HandSize)
	if err != nil {
		return nil, errors.Wrap(err, "shuffle sharding dealer creation failed")
	}

	fq := &queueSet{
		config:       config,
		counter:      counter,
		queues:       initQueues(config.DesiredNumQueues, 0),
		clock:        c,
		virtualTime:  0,
		lastRealTime: c.Now(),
		dealer:       dealer,
	}
	return fq, nil
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
			initQueues(config.DesiredNumQueues-numQueues, len(qs.queues))...)
	}

	qs.config = config
	qs.dealer = dealer

	qs.dequeueWithChannelAsMuchAsPossible()
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
	if eh == nil {
		qs.emptyHandler = eh
		return
	}
	// Here we check whether there are any requests queued or executing and
	// if not then fork an invocation of the EmptyHandler.
	qs.maybeForkEmptyHandlerLocked()

	qs.emptyHandler = eh
}

// Wait uses the given hashValue as the source of entropy
// as it shuffle-shards a request into a queue and waits for
// a decision on what to do with that request.  If tryAnother==true
// at return then the QueueSet has become undesirable and the client
// should try to find a different QueueSet to use; execute and
// afterExecution are irrelevant in this case.  Otherwise, if execute
// then the client should start executing the request and, once the
// request finishes execution or is canceled, call afterExecution().
// Otherwise the client should not execute the
// request and afterExecution is irrelevant.
func (qs *queueSet) Wait(ctx context.Context, hashValue uint64) (tryAnother, execute bool, afterExecution func()) {
	var req *fq.Request
	shouldReturn, tryAnother, execute, afterExecution := func() (
		shouldReturn, tryAnother, execute bool, afterExecution func()) {

		qs.lockAndSyncTime()
		defer qs.lock.Unlock()
		// A call to Wait while the system is quiescing will be rebuffed by
		// returning `tryAnother=true`.
		if qs.emptyHandler != nil {
			return true, true, false, nil
		}

		// ========================================================================
		// Step 1:
		// 1) Start with shuffle sharding, to pick a queue.
		// 2) Reject old requests that have been waiting too long
		// 3) Reject current request if there is not enough concurrency shares and
		// we are at max queue length
		// 4) If not rejected, create a request and enqueue
		req = qs.timeoutOldRequestsAndRejectOrEnqueueLocked(hashValue)
		// req == nil means that the request was rejected - no remaining
		// concurrency shares and at max queue length already
		if req == nil {
			metrics.AddReject(qs.config.Name, "queue-full")
			return true, false, false, func() {}
		}

		// ========================================================================
		// Step 2:
		// 1) The next step is to invoke the method that dequeues as much as possible.

		// This method runs a loop, as long as there
		// are non-empty queues and the number currently executing is less than the
		// assured concurrency value.  The body of the loop uses the fair queuing
		// technique to pick a queue, dequeue the request at the head of that
		// queue, increment the count of the number executing, and send true to
		// the request's channel.
		qs.dequeueWithChannelAsMuchAsPossible()
		return false, false, false, func() {}
	}()
	if shouldReturn {
		return tryAnother, execute, afterExecution
	}

	// ========================================================================
	// Step 3:
	// After that method finishes its loop and returns, the final step in Wait
	// is to `select` (wait) on a message from the enqueud request's channel
	// and return appropriately.  While waiting this thread does no additional
	// work so we decrement the go routine counter
	qs.counter.Add(-1)

	select {
	case execute := <-req.DequeueChannel:
		if execute {
			// execute the request
			return false, true, func() {
				qs.finishRequestAndDequeueWithChannelAsMuchAsPossible(req)
			}
		}
		klog.V(5).Infof("request timed out after being enqueued\n")
		metrics.AddReject(qs.config.Name, "time-out")
		return false, false, func() {}
	case <-ctx.Done():
		klog.V(5).Infof("request cancelled\n")
		func() {
			qs.lockAndSyncTime()
			defer qs.lock.Unlock()

			// TODO(aaron-prindle) add metrics to these two cases
			if req.Enqueued {
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
				// At this point we know that req was in its queue earlier and another
				// goroutine has removed req from its queue and called qs.counter.Add(1)
				// in anticipation of unblocking this goroutine through the other arm of this
				// select. In this case we need to decrement the counter because this goroutine
				// was actually unblocked through a different code path.
				qs.counter.Add(-1)
			}
		}()
		return false, false, func() {}
	}
}

// syncTimeLocked is used to sync the time of the queueSet by looking at the elapsed
// time since the last sync and this value based on the 'virtualtime ratio'
// which scales inversely to the # of active flows
func (qs *queueSet) syncTimeLocked() {
	realNow := qs.clock.Now()
	timesincelast := realNow.Sub(qs.lastRealTime).Seconds()
	qs.lastRealTime = realNow
	var virtualTimeRatio float64

	activeQueues := 0
	reqs := 0
	for _, queue := range qs.queues {
		reqs += queue.RequestsExecuting

		if len(queue.Requests) > 0 || queue.RequestsExecuting > 0 {
			activeQueues++
		}
	}
	if activeQueues != 0 {
		// TODO(aaron-prindle) document the math.Min usage
		virtualTimeRatio = math.Min(float64(reqs), float64(qs.config.ConcurrencyLimit)) / float64(activeQueues)
	}

	qs.virtualTime += timesincelast * virtualTimeRatio
}

func (qs *queueSet) lockAndSyncTime() {
	qs.lock.Lock()
	qs.syncTimeLocked()
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
func (qs *queueSet) timeoutOldRequestsAndRejectOrEnqueueLocked(hashValue uint64) *fq.Request {
	//	Start with the shuffle sharding, to pick a queue.
	queueIdx := qs.chooseQueueIndexLocked(hashValue)
	queue := qs.queues[queueIdx]
	// The next step is the logic to reject requests that have been waiting too long
	qs.removeTimedOutRequestsFromQueueLocked(queue)
	// NOTE: currently timeout is only checked for each new request.  This means that there can be
	// requests that are in the queue longer than the timeout if there are no new requests
	// We prefer the simplicity over the promptness, at least for now.

	// Create a request and enqueue
	req := &fq.Request{
		DequeueChannel:  make(chan bool, 1),
		RealEnqueueTime: qs.clock.Now(),
		Queue:           queue,
	}
	if ok := qs.rejectOrEnqueueLocked(req); !ok {
		return nil
	}
	metrics.ObserveQueueLength(qs.config.Name, len(queue.Requests))
	return req
}

// removeTimedOutRequestsFromQueueLocked rejects old requests that have been enqueued
// past the requestWaitLimit
func (qs *queueSet) removeTimedOutRequestsFromQueueLocked(queue *fq.Queue) {
	timeoutIdx := -1
	now := qs.clock.Now()
	reqs := queue.Requests
	// reqs are sorted oldest -> newest
	// can short circuit loop (break) if oldest requests are not timing out
	// as newer requests also will not have timed out

	// now - requestWaitLimit = waitLimit
	waitLimit := now.Add(-qs.config.RequestWaitLimit)
	for i, req := range reqs {
		if waitLimit.After(req.RealEnqueueTime) {
			qs.counter.Add(1)
			req.DequeueChannel <- false
			close(req.DequeueChannel)
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

// getRequestsExecutingLocked gets the # of requests which are "executing":
// this is the# of requests/requests which have been dequeued but have not had
// finished (via the FinishRequest method invoked after service)
func (qs *queueSet) getRequestsExecutingLocked() int {
	total := 0
	for _, queue := range qs.queues {
		total += queue.RequestsExecuting
	}
	return total
}

// chooseQueueIndexLocked uses shuffle sharding to select a queue index
// using the given hashValue and the shuffle sharding parameters of the queueSet.
func (qs *queueSet) chooseQueueIndexLocked(hashValue uint64) int {
	bestQueueIdx := -1
	bestQueueLen := int(math.MaxInt32)
	// DesiredNum is used here instead of numQueues to omit quiescing queues
	qs.dealer.Deal(hashValue, func(queueIdx int) {
		thisLen := len(qs.queues[queueIdx].Requests)
		if thisLen < bestQueueLen {
			bestQueueIdx, bestQueueLen = queueIdx, thisLen
		}
	})
	return bestQueueIdx
}

// updateQueueVirtualStartTime updates the virtual start time for a queue
// this is done when a new request is enqueued.  For more info see:
// https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md#dispatching
func (qs *queueSet) updateQueueVirtualStartTime(request *fq.Request, queue *fq.Queue) {
	// When a request arrives to an empty queue with no requests executing:
	// len(queue.Requests) == 1 as enqueue has just happened prior (vs  == 0)
	if len(queue.Requests) == 1 && queue.RequestsExecuting == 0 {
		// the queue’s virtual start time is set to the virtual time.
		queue.VirtualStart = qs.virtualTime
	}
}

// enqueues a request into an queueSet
func (qs *queueSet) enqueue(request *fq.Request) {
	queue := request.Queue
	queue.Enqueue(request)
	qs.updateQueueVirtualStartTime(request, queue)
	qs.numRequestsEnqueued++

	metrics.UpdateFlowControlRequestsInQueue(qs.config.Name, qs.numRequestsEnqueued)
}

// rejectOrEnqueueLocked rejects or enqueues the newly arrived request if
// resource criteria isn't met
func (qs *queueSet) rejectOrEnqueueLocked(request *fq.Request) bool {
	queue := request.Queue
	curQueueLength := len(queue.Requests)
	// rejects the newly arrived request if resource criteria not met
	if qs.getRequestsExecutingLocked() >= qs.config.ConcurrencyLimit &&
		curQueueLength >= qs.config.QueueLengthLimit {
		return false
	}

	qs.enqueue(request)
	return true
}

// selectQueue selects the minimum virtualFinish time from the set of queues
// the starting queue is selected via roundrobin
func (qs *queueSet) selectQueue() *fq.Queue {
	minVirtualFinish := math.Inf(1)
	var minQueue *fq.Queue
	var minIndex int
	for range qs.queues {
		queue := qs.queues[qs.robinIndex]
		if len(queue.Requests) != 0 {
			currentVirtualFinish := queue.GetVirtualFinish(0, qs.estimatedServiceTime)
			if currentVirtualFinish < minVirtualFinish {
				minVirtualFinish = currentVirtualFinish
				minQueue = queue
				minIndex = qs.robinIndex
			}
		}
		qs.robinIndex = (qs.robinIndex + 1) % len(qs.queues)
	}
	// we set the round robin indexing to start at the chose queue
	// for the next round.  This way the non-selected queues
	// win in the case that the virtual finish times are the same
	qs.robinIndex = minIndex
	return minQueue
}

// dequeue dequeues a request from the queueSet
func (qs *queueSet) dequeueLocked() (*fq.Request, bool) {
	queue := qs.selectQueue()
	if queue == nil {
		return nil, false
	}
	request, ok := queue.Dequeue()
	if !ok {
		return nil, false
	}
	// When a request is dequeued for service -> qs.VirtualStart += G
	queue.VirtualStart += qs.estimatedServiceTime
	request.StartTime = qs.clock.Now()
	// request dequeued, service has started
	queue.RequestsExecuting++
	metrics.UpdateFlowControlRequestsExecuting(qs.config.Name, queue.RequestsExecuting)
	qs.numRequestsEnqueued--
	return request, ok
}

// dequeueWithChannelAsMuchAsPossible runs a loop, as long as there
// are non-empty queues and the number currently executing is less than the
// assured concurrency value.  The body of the loop uses the fair queuing
// technique to pick a queue, dequeue the request at the head of that
// queue, increment the count of the number executing, and send true
// to the request's channel.
func (qs *queueSet) dequeueWithChannelAsMuchAsPossible() {
	for qs.numRequestsEnqueued != 0 && qs.getRequestsExecutingLocked() < qs.config.ConcurrencyLimit {
		_, ok := qs.dequeueWithChannel()
		if !ok {
			break
		}
	}
}

// dequeueWithChannel is a convenience method for dequeueing requests that
// require a message to be sent through the requests channel
// this is a required pattern for the QueueSet the queueSet supports
func (qs *queueSet) dequeueWithChannel() (*fq.Request, bool) {
	req, ok := qs.dequeueLocked()
	if !ok {
		return nil, false
	}
	qs.counter.Add(1)
	req.DequeueChannel <- true
	close(req.DequeueChannel)
	return req, ok
}

// removeQueueAndUpdateIndexes uses reslicing to remove an index from a slice
// and then updates the 'Index' field of the queues to be correct
func removeQueueAndUpdateIndexes(queues []*fq.Queue, index int) []*fq.Queue {
	keptQueues := append(queues[:index], queues[index+1:]...)
	for i := index; i < len(keptQueues); i++ {
		keptQueues[i].Index--
	}
	return keptQueues
}

// finishRequestLocked is a callback that should be used when a previously dequeued request
// has completed it's service.  This callback updates important state in the
// queueSet
func (qs *queueSet) finishRequestLocked(r *fq.Request) {
	S := qs.clock.Since(r.StartTime).Seconds()

	// When a request finishes being served, and the actual service time was S,
	// the queue’s virtual start time is decremented by G - S.
	r.Queue.VirtualStart -= qs.estimatedServiceTime - S

	// request has finished, remove from requests executing
	r.Queue.RequestsExecuting--

	// Logic to remove quiesced queues
	// >= as QueueIdx=25 is out of bounds for DesiredNum=25 [0...24]
	if r.Queue.Index >= qs.config.DesiredNumQueues &&
		len(r.Queue.Requests) == 0 &&
		r.Queue.RequestsExecuting == 0 {
		qs.queues = removeQueueAndUpdateIndexes(qs.queues, r.Queue.Index)

		// decrement here to maintain the invariant that (qs.robinIndex+1) % numQueues
		// is the index of the next queue after the one last dispatched from
		if qs.robinIndex >= -r.Queue.Index {
			qs.robinIndex--
		}

		// At this point, if the qs is quiescing,
		// has zero requests executing, and has zero requests enqueued
		// then a call to the EmptyHandler should be forked.
		qs.maybeForkEmptyHandlerLocked()
	}
}

func (qs *queueSet) maybeForkEmptyHandlerLocked() {
	if qs.emptyHandler != nil && qs.numRequestsEnqueued == 0 &&
		qs.getRequestsExecutingLocked() == 0 {
		qs.counter.Add(1)
		go func(eh fq.EmptyHandler) {
			defer runtime.HandleCrash()
			defer qs.counter.Add(-1)
			eh.HandleEmpty()
		}(qs.emptyHandler)
	}
}

// finishRequestAndDequeueWithChannelAsMuchAsPossible is a convenience method which calls finishRequest
// for a given request and then dequeues as many requests as possible
// and updates that request's channel signifying it is is dequeued
// this is a callback used for the filter that the queueSet supports
func (qs *queueSet) finishRequestAndDequeueWithChannelAsMuchAsPossible(req *fq.Request) {
	qs.lockAndSyncTime()
	defer qs.lock.Unlock()

	qs.finishRequestLocked(req)
	qs.dequeueWithChannelAsMuchAsPossible()
}
