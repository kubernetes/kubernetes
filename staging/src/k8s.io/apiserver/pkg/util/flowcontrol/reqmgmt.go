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
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	rmtypesv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	rmlistersv1alpha1 "k8s.io/client-go/listers/flowcontrol/v1alpha1"
)

// Interface defines how the request-management filter interacts with the underlying system.
type Interface interface {
	GetCurrentState() *RequestManagementState
	Run(stopCh <-chan struct{}) error
}

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
	// SetConfiguration updates the configuration
	SetConfiguration(concurrencyLimit, desiredNumQueues, queueLengthLimit int, requestWaitLimit time.Duration)

	// Quiesce controls whether this system is quiescing.  Passing a
	// non-nil handler means the system should become quiescent, a nil
	// handler means the system should become non-quiescent.  A call
	// to Wait while the system is quiescent will be rebuffed by
	// returning `quiescent=true`.  If all the queues have no requests
	// waiting nor executing while the system is quiescent then the
	// handler will eventually be called with no locks held (even if
	// the system becomes non-quiescent between the triggering state
	// and the required call).
	//
	// The filter uses this for a priority level that has become
	// undesired, setting a handler that will cause the priority level
	// to eventually be removed from the filter if the filter still
	// wants that.  If the filter later changes its mind and wants to
	// preserve the priority level then the filter can use this to
	// cancel the handler registration.
	Quiesce(EmptyHandler)

	// Wait, in the happy case, shuffle shards the given request into
	// a queue and eventually dispatches the request from that queue.
	// Dispatching means to return with `quiescent==false` and
	// `execute==true`.  In one unhappy case the request is
	// immediately rebuffed with `quiescent==true` (which tells the
	// filter that there has been a timing splinter and the filter
	// re-calcuates the priority level to use); in all other cases
	// `quiescent` will be returned `false` (even if the system is
	// quiescent by then).  In the non-quiescent unhappy cases the
	// request is eventually rejected, which means to return with
	// `execute=false`.  In the happy case the caller is required to
	// invoke the returned `afterExecution` after the request is done
	// executing.  The hash value and hand size are used to do the
	// shuffle sharding.
	Wait(hashValue uint64, handSize int32) (quiescent, execute bool, afterExecution func())
}

// EmptyHandler can be notified that something is empty
type EmptyHandler interface {
	// HandleEmpty is called to deliver the notification
	HandleEmpty()
}

// RequestManagementState is the variable state that this filter is working with at a
// given point in time.
type RequestManagementState struct {
	// flowSchemas holds the flow schema objects, sorted by increasing
	// numerical (decreasing logical) matching precedence
	flowSchemas FlowSchemaSequence

	// priorityLevelStates maps the PriorityLevelConfiguration object
	// name to the state for that level
	priorityLevelStates map[string]*PriorityLevelState
}

// GetFlowSchemas returns an array of latest observed flow-schemas by the system
func (s *RequestManagementState) GetFlowSchemas() []*rmtypesv1alpha1.FlowSchema {
	return s.flowSchemas
}

// GetPriorityLevelStates returns a map of latest states of the existing priority-levels
func (s *RequestManagementState) GetPriorityLevelStates() map[string]*PriorityLevelState {
	return s.priorityLevelStates
}

// FlowSchemaSequence holds sorted set of pointers to FlowSchema objects.
// FlowSchemaSequence implements `sort.Interface` (TODO: implement this).
type FlowSchemaSequence []*rmtypesv1alpha1.FlowSchema

// PriorityLevelState holds the state specific to a priority level.
// golint requires that I write something here,
// even if I can not think of something better than a tautology.
type PriorityLevelState struct {
	// config holds the configuration after defaulting logic has been applied
	config rmtypesv1alpha1.PriorityLevelConfigurationSpec

	// concurrencyLimit is the limit on number executing
	concurrencyLimit int

	fairQueuingSystem FairQueuingSystem
}

// IsExempt returns if the priority-level is exempt
func (s *PriorityLevelState) IsExempt() bool {
	return s.config.Exempt
}

// GetHandSize returns the hand-size of the priority-level
func (s *PriorityLevelState) GetHandSize() int32 {
	return s.config.HandSize
}

// requestManagement holds all the state and infrastructure of this
// filter
type requestManagementSystem struct {
	clk clock.Clock

	fairQueuingFactory FairQueuingFactory

	// configQueue holds TypedConfigObjectReference values, identifying
	// config objects that need to be processed
	configQueue workqueue.RateLimitingInterface

	// plInformer is the informer for priority level config objects
	plInformer cache.SharedIndexInformer

	plLister rmlistersv1alpha1.PriorityLevelConfigurationLister

	// fsInformer is the informer for flow schema config objects
	fsInformer cache.SharedIndexInformer

	fsLister rmlistersv1alpha1.FlowSchemaLister

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

// NewRequestManagementSystem creates a new instance of request-management system
func NewRequestManagementSystem(
	informerFactory kubeinformers.SharedInformerFactory,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	clk clock.Clock,
) Interface {
	reqMgmt := &requestManagementSystem{
		clk:                    clk,
		configQueue:            workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(200*time.Millisecond, 8*time.Hour), "req_mgmt_config_queue"),
		plLister:               informerFactory.Flowcontrol().V1alpha1().PriorityLevelConfigurations().Lister(),
		fsLister:               informerFactory.Flowcontrol().V1alpha1().FlowSchemas().Lister(),
		serverConcurrencyLimit: serverConcurrencyLimit,
		requestWaitLimit:       requestWaitLimit,
	}
	// TODO: finish implementation
	return reqMgmt
}

// Run bootstraps the request-management system
func (r *requestManagementSystem) Run(stopCh <-chan struct{}) error {
	// TODO: implement
	return nil
}

// GetCurrentState returns the current state of the request-management system
func (r *requestManagementSystem) GetCurrentState() *RequestManagementState {
	return r.curState.Load().(*RequestManagementState)
}

// GetFairQueuingSystem returns the fair-queuing system of the priority-level
func (s *PriorityLevelState) GetFairQueuingSystem() FairQueuingSystem {
	return s.fairQueuingSystem
}

// RequestDigest holds necessary info from request for flow-control
type RequestDigest struct {
	RequestInfo *request.RequestInfo
	User        user.Info
}

// PickFlowSchema returns a best-matching flow-schema according to the request
func PickFlowSchema(digest RequestDigest, flowSchemas FlowSchemaSequence, priorityLevelStates map[string]*PriorityLevelState) *rmtypesv1alpha1.FlowSchema {
	// TODO: implement
	return nil
}

// ComputeFlowDistinguisher computes flow-distinguisher according to request information for a flow-schema
func ComputeFlowDistinguisher(digest RequestDigest, flowSchema *rmtypesv1alpha1.FlowSchema) uint64 {
	var fDistinguisher string
	// TODO: implement
	return hashFlowID(flowSchema.Name, fDistinguisher)
}

// RequestPriorityState requests for the current state of a priority-level
func RequestPriorityState(digest RequestDigest, fs *rmtypesv1alpha1.FlowSchema, priorityLevelStates map[string]*PriorityLevelState) *PriorityLevelState {
	// TODO: implement
	return nil
}

// HashFlowID hashes the inputs into 64-bits
func hashFlowID(fsName, fDistinguisher string) uint64 {
	// TODO: implement
	return 0
}
