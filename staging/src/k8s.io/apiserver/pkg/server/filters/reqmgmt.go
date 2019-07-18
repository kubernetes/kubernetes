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

package filters

import (
	"fmt"
	"hash/fnv"
	"math"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/filters/fq"
	"k8s.io/apiserver/pkg/server/filters/reqmgmt/inflight"
	restclient "k8s.io/client-go/rest"
	cache "k8s.io/client-go/tools/cache"
	workqueue "k8s.io/client-go/util/workqueue"

	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"

	"k8s.io/klog"
)

// This request filter implements https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md

// FairQueuingFactory knows how to make FairQueueingSystem objects.
// This filter makes a FairQueuingSystem for each priority level.
type FairQueuingFactory interface {
	NewFairQueuingSystem(concurrencyLimit, numQueues, queueLengthLimit int, requestWaitLimit time.Duration, clk clock.Clock) FairQueuingSystem
}

// FairQueuingSystem is the abstraction for the queuing and
// dispatching functionality of one non-exempt priority level.  It
// covers the functionality described in the "Assignment to a Queue",
// "Queuing", and "Dispatching" sections of
// https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md
// .  Some day we may have connections between priority levels, but
// today is not that day.
type FairQueuingSystem interface {
	// The following methods modify or query the configuration

	SetConcurrencyLimit(int)
	SetDesiredNumQueues(int)
	GetDesiredNumQueues() int
	GetActualNumQueues() int
	SetQueueLengthLimit(int)
	SetRequestWaitLimit(time.Duration)

	// DoOnEmpty records a function to be called when all the queues
	// have no requests waiting nor executing.  This is setting a
	// notification handler.  A call to `DoOnEmpty(fn)` requires the
	// implementation to guarantee that if `fn != nil` then eventually
	// either (1) there is another call to `DoOnEmpty` or (2) if all
	// the queues have no requests waiting nor executing then
	// eventually there will be a call to `fn()` with no mutexes
	// locked.  Note that (2) effectively has two levels of
	// "eventually" (eventually empty, then eventually the call to
	// fn), and that (1) is about how the client can effectively
	// remove the promise of (2).
	//
	// The filter uses this for a priority level that has become
	// undesired, setting a handler that will cause the priority level
	// to eventually be removed from the filter if the filter still
	// wants that.  If the filter later changes its mind and wants to
	// preserve the priority level then the filter can use this to
	// cancel the handler registration.
	DoOnEmpty(func())

	// Wait, in the happy case, shuffle shards the given request into
	// a queue and eventually dispatches the request from that queue.
	// Dispatching means to return with `execute==true`.  In the
	// unhappy cases the request is eventually rejected, which means
	// to return with `execute=false`.  In the happy case the caller
	// is required to invoke the returned `afterExecution` after the
	// request is done executing.  The hash value and hand size are
	// used to do the shuffle sharding.
	Wait(hashValue uint64, handSize int32) (execute bool, afterExecution func())
}

type FairQueuingSystemImpl struct {
	lock sync.Mutex
	// queues                   []*inflight.Queue
	fqq                      *inflight.FQScheduler
	concurrencyLimit         int
	assuredConcurrencyShares int
	actualnumqueues          int // TODO(aaron-prindle) currently not updated/used...
	numqueues                int
	queueLengthLimit         int
	requestWaitLimit         time.Duration
}

// initQueues is a convenience method for initializing an array of n queues
func initQueues(numQueues int) []fq.FQQueue {
	queues := make([]*fq.Queue, 0, numQueues)
	fqqueues := make([]fq.FQQueue, numQueues, numQueues)

	for i := 0; i < numQueues; i++ {
		queues = append(queues, &fq.Queue{})
		packets := []*fq.Packet{}
		fqpackets := make([]fq.FQPacket, len(packets), len(packets))
		queues[i].Packets = fqpackets

		fqqueues[i] = queues[i]
	}

	return fqqueues
}

func NewFairQueuingSystem(concurrencyLimit, numQueues, queueLengthLimit int, requestWaitLimit time.Duration, clk clock.Clock) FairQueuingSystem {
	return &FairQueuingSystemImpl{
		// queues: queues,
		fqq: inflight.NewFQScheduler(initQueues(numQueues), clk),
		// TODO(aaron-prindle) concurrency limit is updated dynamically
		concurrencyLimit: concurrencyLimit,
		numqueues:        numQueues,
		queueLengthLimit: queueLengthLimit,
		requestWaitLimit: requestWaitLimit,
	}
}

func (s *FairQueuingSystemImpl) SetConcurrencyLimit(concurrencyLimit int) {
	s.concurrencyLimit = concurrencyLimit
}

func (s *FairQueuingSystemImpl) SetDesiredNumQueues(numqueues int) {
	s.numqueues = numqueues
}

func (s *FairQueuingSystemImpl) GetDesiredNumQueues() int {
	return s.numqueues
}

func (s *FairQueuingSystemImpl) GetActualNumQueues() int {
	return s.actualnumqueues
}

func (s *FairQueuingSystemImpl) SetQueueLengthLimit(queueLengthLimit int) {
	s.queueLengthLimit = queueLengthLimit
}

func (s *FairQueuingSystemImpl) SetRequestWaitLimit(requestWaitLimit time.Duration) {
	s.requestWaitLimit = requestWaitLimit
}

func (s *FairQueuingSystemImpl) DoOnEmpty(func()) {
	// When an undesired queue becomes empty it is
	// deleted and the fair queuing round-robin pointer is advanced if it was
	// pointing to that queue.

	// DoOnEmpty is part of how a priority level gets shut down.  Read the PR
	// I have open that adds a section to the KEP discussing how the filter
	// handles changes in the configuration objects. (edited)

	panic("not implemented")
}

// TODO(aaron-prindle) DEBUG - remove seq
var seq = 1

func (s *FairQueuingSystemImpl) Wait(hashValue uint64, handSize int32) (execute bool, afterExecution func()) {
	// Wait, in the happy case, shuffle shards the given request into
	// a queue and eventually dispatches the request from that queue.
	// Dispatching means to return with `execute==true`.  In the
	// unhappy cases the request is eventually rejected, which means
	// to return with `execute=false`.  In the happy case the caller
	// is required to invoke the returned `afterExecution` after the
	// request is done executing.  The hash value and hand size are
	// used to do the shuffle sharding.

	// TODO(aaron-prindle) verify what should/shouldn't be locked!!!!

	//	So that starts with the shuffle sharding, to pick a queue.
	queueIdx := s.fqq.ChooseQueueIdx(hashValue, int(handSize))

	// The next
	// step is the logic to reject requests that have been waiting too long.  I
	// imagine that each request has a channel of `struct {execute bool,
	// afterExecution func()}`.
	// --
	// NOTE: currently timeout is only checked for each new request.  This means that there can be
	// requests that are in the queue longer than the timeout if there are no new requests
	// We think this is a fine tradeoff

	func() {
		// TODO(aaron-prindle) verify if this actually needs to be locked...
		s.lock.Lock()
		defer s.lock.Unlock()

		timeoutIdx := -1
		now := time.Now()
		queues := s.fqq.GetQueues()
		fmt.Printf("s.requestWaitLimit: %v\n", s.requestWaitLimit)
		fmt.Printf("queueIdx: %v\n", queueIdx)
		queue := queues[queueIdx]
		pkts := queue.GetPackets()
		// pkts are sorted oldest -> newest
		// can short circuit loop (break) if oldest packets are not timing out
		//  as newer packets also will not have timed out
		for i, pkt := range pkts {
			channelPkt := pkt.(*inflight.Packet)
			limit := channelPkt.EnqueueTime.Add(s.requestWaitLimit)
			if now.After(limit) {
				fmt.Println("timeout!!!")
				channelPkt.DequeueChannel <- false
				close(channelPkt.DequeueChannel)

				// // TODO(aaron-prindle) verify this makes sense here
				// get idx for timed out packets
				timeoutIdx = i

			} else {
				break
			}
		}
		fmt.Printf("timeoutIdx: %d\n", timeoutIdx)
		// remove timed out packets from queue
		// TODO(aaron-prindle) BAD, currently copies slice for deletion
		if timeoutIdx != -1 {
			// timeoutIdx + 1 to remove the last timeout pkt
			fmt.Println("HERE!")
			removeIdx := timeoutIdx + 1
			queue.SetPackets(pkts[removeIdx:])
			// queue.SetRequestsExecuting(
			// 	queue.GetRequestsExecuting() - removeIdx)
			s.fqq.DecrementPackets(removeIdx)
		}

	}()

	// The next step is to reject the newly
	// arrived request if the number executing is at least the priority
	// level's assured concurrency value (I forgot to mention this part
	// in the KEP; I hope it is obvious why this condition is needed)
	// AND the chosen queue is at or exceeding its current length
	// limit.
	shouldReject := false
	var pkt fq.FQPacket
	func() {
		// lock for enqueue for queue length check and increment
		s.lock.Lock()
		defer s.lock.Unlock()

		queues := s.fqq.GetQueues()
		curQueueLength := len(queues[queueIdx].GetPackets())
		fmt.Printf("curQueueLength: %v\n", curQueueLength)
		fmt.Printf("s.fqq.QueueLengthLimit: %v\n", s.queueLengthLimit)

		if s.fqq.GetRequestsExecuting() >= s.concurrencyLimit && curQueueLength >= s.fqq.QueueLengthLimit {
			shouldReject = true
			return
		}

		// ENQUEUE - Assuming the newly arrived request has not been rejected,
		// the next step is to put it in the queue.
		pkt = fq.FQPacket(&inflight.Packet{
			DequeueChannel: make(chan bool, 1),
			Seq:            seq,
			EnqueueTime:    time.Now(),
		})
		// can't set nested struct val in initializer
		pkt.SetQueueIdx(queueIdx)
		// TODO(aaron-prindle) REMOVE - DEBUG - remove seq
		seq++

		// TODO(aaron-prindle) make it so enqueue fails if the queues are full?
		// --
		// I think this is taken care of by the check above, need to verify
		s.fqq.Enqueue(pkt)

		channelPkt := pkt.(*inflight.Packet)
		// TODO(aaron-prindle) REMOVE - DEBUG
		fmt.Printf("ENQUEUED: %d\n", channelPkt.Seq)
	}()
	if shouldReject {
		return false, func() {}
	}
	// The next step is to invoke the method that
	// dequeues as much as possible.  This method runs a loop, as long as there
	// are non-empty queues and the number currently executing is less than the
	// assured concurrency value.  The body of the loop uses the fair queuing
	// technique to pick a queue, dequeue the request at the head of that
	// queue, increment the count of the number executing, and send `{true,
	// handleCompletion(that dequeued request)}` to the request's channel.

	// TODO(aaron-prindle) ERROR - with timeout remove from queueu
	// this is looping forever..
	for !s.fqq.IsEmpty() && s.fqq.GetRequestsExecuting() < s.concurrencyLimit {
		fmt.Printf("s.fqq.GetRequestsExecuting(): %v\n", s.fqq.GetRequestsExecuting())
		fmt.Printf("s.concurrencyLimit(): %v\n", s.concurrencyLimit)
		s.fqq.Dequeue()
	}

	// After that method finishes its loop and returns, the final step in Wait
	// is to `select` on either request timeout or receipt of a record on the
	// newly arrived request's channel, and return appropriately.  If a record
	// has been sent to the request's channel then this `select` will
	// immediately complete (this is what I meant in my earlier remark about
	// waiting only if the concurrency limit is reached).
	channelPkt := pkt.(*inflight.Packet)
	// I am glossing over the
	// details of getting the request's channel closed properly; I assume you
	// can work that out.
	//--
	// see staging/src/k8s.io/apiserver/pkg/server/filters/reqmgmt.go:144

	select {
	case execute := <-channelPkt.DequeueChannel:
		if execute {
			// execute
			return true, func() { s.fqq.FinishPacketAndDequeueNextPacket(pkt) }
		}
		// timed out
		fmt.Printf("channelPkt.DequeueChannel timed out\n")
		// klog.V(5).Infof("channelPkt.DequeueChannel timed out\n")
		return false, func() {}
	}

}

// RMState is the variable state that this filter is working with at a
// given point in time.
type RMState struct {
	serverConcurrencyLimit int

	// flowSchemas holds the flow schema objects, sorted by increasing
	// numerical (decreasing logical) matching precedence
	flowSchemas FlowSchemaSeq

	// priorityLevelStates maps the PriorityLevelConfiguration object
	// name to the state for that level
	priorityLevelStates map[string]*PriorityLevelState
}

func getACV(serverConcurrencyLimit int, plcs []*rmtypesv1a1.PriorityLevelConfiguration, origPLC *rmtypesv1a1.PriorityLevelConfiguration) int {
	var denom float64
	for _, plc := range plcs {
		denom += float64(plc.Spec.AssuredConcurrencyShares)
	}
	return int(math.Ceil(float64(serverConcurrencyLimit) * float64(origPLC.Spec.AssuredConcurrencyShares) / denom))
}

func (r *RMState) updateACV(ps *PriorityLevelState) {
	var denom float64
	for _, pl := range r.priorityLevelStates {
		denom += float64(pl.config.AssuredConcurrencyShares)
	}
	ps.concurrencyLimit = int(math.Ceil(float64(r.serverConcurrencyLimit) * float64(ps.config.AssuredConcurrencyShares) / denom))
}

// FlowSchemaSeq holds sorted set of pointers to FlowSchema objects.
// FLowSchemaSeq implements `sort.Interface` (TODO: implement this).
type FlowSchemaSeq []*rmtypesv1a1.FlowSchema

// PriorityLevelState holds the state specific to a priority level.
// golint requires that I write something here,
// even if I can not think of something better than a tautology.
type PriorityLevelState struct {
	// config holds the configuration after defaulting logic has been applied
	config rmtypesv1a1.PriorityLevelConfigurationSpec

	// concurrencyLimit is the limit on number executing
	concurrencyLimit int

	fqs FairQueuingSystem
}

// requestManagement holds all the state and infrastructure of this
// filter
type requestManagement struct {
	fairQueuingFactory FairQueuingFactory

	// configQueue holds TypedConfigObjectReference values, identifying
	// config objects that need to be processed
	configQueue workqueue.RateLimitingInterface

	// plInformer is the informer for priority level config objects
	plInformer cache.SharedIndexInformer

	// fsInformer is the informer for flow schema config objects
	fsInformer cache.SharedIndexInformer

	// plLister belongs here too
	// plLister
	// fsLister belongs here too
	// fsLister

	// serverConcurrencyLimit is the limit on the server's total
	// number of non-exempt requests being served at once.  This comes
	// from server configuration.
	serverConcurrencyLimit int

	// requestWaitLimit comes from server configuration.
	requestWaitLimit time.Duration

	// curState holds a pointer to the current RMState.  That is,
	// `Load()` produces a `*RMState`.  When a config work queue worker
	// processes a configuration change, it stores a new pointer here ---
	// it does NOT side-effect the old `RMState` value.  The new `RMState`
	// has a freshly constructed slice of FlowSchema pointers and a
	// freshly constructed map of priority level states.  But the new
	// `RMState.priorityLevelStates` includes in its range at least all the
	// `*PriorityLevelState` values of the old `RMState.priorityLevelStates`.
	// Consequently the filter can load a `*RMState` and work with it
	// without concern for concurrent updates.  When a priority level is
	// finally deleted, this will also involve storing a new `*RMState`
	// pointer here, but in this case the range of the
	// `RMState.priorityLevels` will be reduced --- by removal of the
	// priority level that is no longer in use.
	curState atomic.Value
}

// TypedConfigObjectReference is a reference to a relevant config API object.
// No namespace is needed because none of these objects is namespaced.
type TypedConfigObjectReference struct {
	Kind string
	Name string
}

func (tr *TypedConfigObjectReference) String() string {
	return tr.Kind + "/" + tr.Name
}

// rmSetup is invoked at startup to create the infrastructure of this filter
func rmSetup(loopbackClientConfig *restclient.Config, serverConcurrencyLimit int, requestWaitLimit time.Duration) *requestManagement {
	reqMgmt := &requestManagement{
		configQueue:            workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(200*time.Millisecond, 8*time.Hour), "req_mgmt_config_queue"),
		serverConcurrencyLimit: serverConcurrencyLimit,
		requestWaitLimit:       requestWaitLimit,
	}
	// TODO: finish implementation

	// TODO(aaron-prindle) figure out what needs to be done...
	// fetch related k8s objects - FlowSchemas, RequestPriorityLevels
	return reqMgmt
}

func initPriorityLevelStates(serverConcurrencyLimit int, plcs []*rmtypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration) map[string]*PriorityLevelState {
	pls := map[string]*PriorityLevelState{}

	// TODO(aaron-prindle) change this to a test method and change clock to be
	// clk := clock.RealClock{}
	clk := clock.RealClock{}

	for _, plc := range plcs {
		concurrencyLimit := getACV(serverConcurrencyLimit, plcs, plc)

		pls[plc.Name] = &PriorityLevelState{
			config: plc.Spec,
			fqs: NewFairQueuingSystem(concurrencyLimit, int(plc.Spec.Queues),
				int(plc.Spec.QueueLengthLimit), requestWaitLimit, clk),
		}
	}
	return pls
}

// WithRequestManagement limits the number of in-flight requests in a fine-grained way
func WithRequestManagement(
	handler http.Handler,
	// loopbackClientConfig *restclient.Config,
	// -- added, REMOVE
	flowSchemas []*rmtypesv1a1.FlowSchema,
	priorityLevelConfigurations []*rmtypesv1a1.PriorityLevelConfiguration,
	// --
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
) http.Handler {
	// reqMgmt := rmSetup(loopbackClientConfig, serverConcurrencyLimit, requestWaitLimit)
	// --
	// TODO(aaron-prindle) REMOVE - TEST - make dynamic
	pls := initPriorityLevelStates(serverConcurrencyLimit, priorityLevelConfigurations, requestWaitLimit)

	// rmState.updateACV(ps) // TODO(aaron-prindle) rename updateACV?
	// ps.fqs.SetConcurrencyLimit(ps.concurrencyLimit)

	rmState := RMState{
		flowSchemas:            flowSchemas,
		priorityLevelStates:    pls,
		serverConcurrencyLimit: serverConcurrencyLimit,
	}

	// --
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if !ok {
			handleError(w, r, fmt.Errorf("no RequestInfo found in context, handler chain must be wrong"))
			return
		}

		// Skip tracking long running events.
		if longRunningRequestCheck != nil && longRunningRequestCheck(r, requestInfo) {
			handler.ServeHTTP(w, r)
			return
		}

		// --ORIG-- START
		// TODO(aaron-prindle) CHANGE removed for early testing

		// get state of regMgmnt so we know all the obj/config values
		// rmState := reqMgmt.curState.Load().(*RMState)

		// analyze request to get flowSchema (so can get it's info - PriorityState, FlowDistinguisher)
		// fs := reqMgmt.pickFlowSchema(r, rmState.flowSchemas, rmState.priorityLevelStates)

		// get the PriorityState, if exempt pass through
		// ps := reqMgmt.requestPriorityState(r, fs, rmState.priorityLevelStates)

		// calculate the flow distinguisher for the flowSchema
		// flowDistinguisher := reqMgmt.computeFlowDistinguisher(r, fs)

		// hash the flowDistinguisher (for shuffle sharding
		// hashValue := reqMgmt.hashFlowID(fs.Name, flowDistinguisher)
		// -- ORIG -- END

		// -- TEST -- START
		fs := pickFlowSchema(r, rmState.flowSchemas, rmState.priorityLevelStates)

		ps := requestPriorityState(r, fs, rmState.priorityLevelStates)
		if ps.config.Exempt {
			klog.V(5).Infof("Serving %v without delay\n", r)
			handler.ServeHTTP(w, r)
			return
		}

		flowDistinguisher := computeFlowDistinguisher(r, fs)

		hashValue := hashFlowID(fs.Name, flowDistinguisher)
		// -- TEST -- END

		// TODO(aaron-prindle) verify this is needed?
		// rmState.updateACV(ps) // TODO(aaron-prindle) rename updateACV?
		// ps.fqs.SetConcurrencyLimit(ps.concurrencyLimit)

		// attempt to enqueue and dispatch the package
		execute, afterExecute := ps.fqs.Wait(hashValue, ps.config.HandSize)

		if execute {
			klog.V(5).Infof("Serving %v after queuing\n", r)
			timedOut := ctx.Done()
			finished := make(chan struct{})
			go func() {
				handler.ServeHTTP(w, r)
				close(finished)
			}()
			select {
			case <-timedOut:
				klog.V(5).Infof("Timed out waiting for %v to finish\n", r)
			case <-finished:
				// TODO(aaron-prindle) update this with anything that should be done
			}
			afterExecute()
		} else {
			klog.V(5).Infof("Rejecting %v\n", r)
			tooManyRequests(r, w)
		}
		return
	})

}

func hash(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// func (requestManagement) computeFlowDistinguisher(r *http.Request, fs *rmtypesv1a1.FlowSchema) string {
// 	// TODO: implement
// 	// TODO(aaron-prindle) CHANGE replace w/ proper implementation
// 	return fs.Name
// }

// func (requestManagement) hashFlowID(fsName, fDistinguisher string) uint64 {
// 	// TODO: implement
// 	// TODO(aaron-prindle) verify implementation
// 	return hash(fmt.Sprintf("%s,%s", fsName, fDistinguisher))
// }

// func (requestManagement) pickFlowSchema(r *http.Request, flowSchemas FlowSchemaSeq, priorityLevelStates map[string]*PriorityLevelState) *rmtypesv1a1.FlowSchema {
// 	// TODO(aaron-prindle) CHANGE replace w/ proper implementation
// 	priority := r.Header.Get("PRIORITY")
// 	idx, err := strconv.Atoi(priority)
// 	if err != nil {
// 		panic("strconv.Atoi(priority) errored")
// 	}
// 	// TODO(aaron-prindle) can also use MatchingPrecedence for dummy method
// 	return flowSchemas[idx]
// }

// func (requestManagement) requestPriorityState(r *http.Request, fs *rmtypesv1a1.FlowSchema, priorityLevelStates map[string]*PriorityLevelState) *PriorityLevelState {
// 	// TODO: implement
// 	out, ok := priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]
// 	if !ok {
// 		panic("NOT OK!!")
// 	}
// 	return out
// }

func computeFlowDistinguisher(r *http.Request, fs *rmtypesv1a1.FlowSchema) string {
	// TODO: implement
	// TODO(aaron-prindle) CHANGE replace w/ proper implementation
	return fs.Name
}

func hashFlowID(fsName, fDistinguisher string) uint64 {
	// TODO: implement
	// TODO(aaron-prindle) verify implementation
	return hash(fmt.Sprintf("%s,%s", fsName, fDistinguisher))
}

func pickFlowSchema(r *http.Request, flowSchemas FlowSchemaSeq, priorityLevelStates map[string]*PriorityLevelState) *rmtypesv1a1.FlowSchema {
	// TODO(aaron-prindle) CHANGE replace w/ proper implementation
	priority := r.Header.Get("PRIORITY")
	idx, err := strconv.Atoi(priority)
	if err != nil {
		panic("strconv.Atoi(priority) errored")
	}
	// TODO(aaron-prindle) can also use MatchingPrecedence for dummy method
	return flowSchemas[idx]
}

func requestPriorityState(r *http.Request, fs *rmtypesv1a1.FlowSchema, priorityLevelStates map[string]*PriorityLevelState) *PriorityLevelState {
	// TODO: implement
	out, ok := priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]

	if !ok {
		panic("NOT OK!!")
	}
	return out
}
