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
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	rmtypesv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	rmlistersv1alpha1 "k8s.io/client-go/listers/flowcontrol/v1alpha1"
)

// Interface defines how the request-management filter interacts with the underlying system.
type Interface interface {
	// Wait decides what to do about the request with the given digest
	// and, if appropriate, enqueues that request and waits for it to be
	// dequeued before returning.  If `execute == false` then the request
	// is being rejected.  If `execute == true` then the caller should
	// handle the request and then call `afterExecute()`.
	Wait(requestDigest RequestDigest) (execute bool, afterExecute func())

	// Run monitors config objects from the main apiservers and causes
	// any needed changes to local behavior
	Run(stopCh <-chan struct{}) error
}

// This request filter implements https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md

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

	queues fq.QueueSet
}

// requestManagement holds all the state and infrastructure of this
// filter
type requestManagementSystem struct {
	clk clock.Clock

	fairQueuingFactory fq.QueueSetFactory

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

// Run runs the controller that deals with config API objects
func (reqMgmt *requestManagementSystem) Run(stopCh <-chan struct{}) error {
	// TODO: implement
	return nil
}

// RequestDigest holds necessary info from request for flow-control
type RequestDigest struct {
	RequestInfo *request.RequestInfo
	User        user.Info
}

func (reqMgmt *requestManagementSystem) Wait(requestDigest RequestDigest) (execute bool, afterExecute func()) {
	for {
		rmState := reqMgmt.curState.Load().(*RequestManagementState)
		fs := rmState.pickFlowSchema(requestDigest)
		// RequestManagementState is constructed to guarantee that the following succeeds
		ps := rmState.priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]
		if ps.config.Exempt {
			klog.V(7).Infof("Serving %v without delay", requestDigest)
			return true, func() {}
		}
		flowDistinguisher := requestDigest.ComputeFlowDistinguisher(fs.Spec.DistinguisherMethod)
		hashValue := hashFlowID(fs.Name, flowDistinguisher)
		quiescent, execute, afterExecute := ps.queues.Wait(hashValue, ps.config.HandSize)
		if quiescent {
			klog.V(5).Infof("Request %v landed in timing splinter, re-classifying", requestDigest)
			continue
		}
		return execute, afterExecute
	}
}

func (rmState *RequestManagementState) pickFlowSchema(rd RequestDigest) *rmtypesv1alpha1.FlowSchema {
	return nil
	// TODO: implement
}

// ComputeFlowDistinguisher extracts the flow distinguisher according to the given method
func (rd RequestDigest) ComputeFlowDistinguisher(method *rmtypesv1alpha1.FlowDistinguisherMethod) string {
	// TODO: implement
	return ""
}

// HashFlowID hashes the inputs into 64-bits
func hashFlowID(fsName, fDistinguisher string) uint64 {
	// TODO: implement
	return 0
}
