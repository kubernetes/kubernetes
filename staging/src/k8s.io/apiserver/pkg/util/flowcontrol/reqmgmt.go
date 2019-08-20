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

package flowcontrol

import (
	"hash/crc64"
	"sync"
	"sync/atomic"
	"time"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	rmtypesv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	rmclientv1alpha1 "k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1"
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

// requestManagementState is the variable state that this filter is
// working with at a given point in time.
type requestManagementState struct {
	// flowSchemas holds the flow schema objects, sorted by increasing
	// numerical (decreasing logical) matching precedence.  Every
	// FlowSchema in this slice is immutable.
	flowSchemas rmtypesv1alpha1.FlowSchemaSequence

	// priorityLevelStates maps the PriorityLevelConfiguration object
	// name to the state for that level.  Every field of every
	// priorityLevelState in here is immutable.  Every name referenced
	// from a member of `flowSchemas` has an entry here.
	priorityLevelStates map[string]*priorityLevelState
}

// priorityLevelState holds the state specific to a priority level.
type priorityLevelState struct {
	// config holds the configuration after defaulting logic has been applied
	config rmtypesv1alpha1.PriorityLevelConfigurationSpec

	// concurrencyLimit is the limit on number executing
	concurrencyLimit int

	queues fq.QueueSet

	emptyHandler *emptyRelay
}

// requestManagementSystem holds all the state and infrastructure of
// this filter
type requestManagementSystem struct {
	// wg is kept informed of when goroutines start or stop or begin or end waiting
	wg fq.OptionalWaitGroup

	queueSetFactory fq.QueueSetFactory

	// configQueue holds TypedConfigObjectReference values, identifying
	// config objects that need to be processed
	configQueue workqueue.RateLimitingInterface

	plLister         rmlistersv1alpha1.PriorityLevelConfigurationLister
	plInformerSynced cache.InformerSynced

	fsInformerSynced cache.InformerSynced
	fsLister         rmlistersv1alpha1.FlowSchemaLister

	flowcontrolClient rmclientv1alpha1.FlowcontrolV1alpha1Interface

	readyFunc func() bool

	// serverConcurrencyLimit is the limit on the server's total
	// number of non-exempt requests being served at once.  This comes
	// from server configuration.
	serverConcurrencyLimit int

	// requestWaitLimit comes from server configuration.
	requestWaitLimit time.Duration

	// curState holds a pointer to the current requestManagementState.
	// That is, `Load()` produces a `*requestManagementState`.  When a
	// config work queue worker processes a configuration change, it
	// stores a new pointer here --- it does NOT side-effect the old
	// `requestManagementState` value.  The new
	// `requestManagementState` has a freshly constructed slice of
	// FlowSchema pointers and a freshly constructed map of priority
	// level states.
	curState atomic.Value
}

// NewRequestManagementSystem creates a new instance of request-management system
func NewRequestManagementSystem(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient rmclientv1alpha1.FlowcontrolV1alpha1Interface,
	queueSetFactory fq.QueueSetFactory,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	waitGroup *sync.WaitGroup,
) Interface {
	return NewRequestManagementSystemWithPreservation(
		informerFactory,
		flowcontrolClient,
		queueSetFactory,
		serverConcurrencyLimit,
		requestWaitLimit,
		waitGroup,
		nil, nil)
}

// NewRequestManagementSystemWithPreservation creates a new instance
// of request-management system with preservation.  The WaitGroup is
// optional and, if supplied, is kept informed of whenever a goroutine
// is started or stopped or begins or finishes waiting --- except that
// the configuration controller is not fully plumbed yet.
func NewRequestManagementSystemWithPreservation(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient rmclientv1alpha1.FlowcontrolV1alpha1Interface,
	queueSetFactory fq.QueueSetFactory,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	waitGroup *sync.WaitGroup,
	preservingFlowSchemas []*rmtypesv1alpha1.FlowSchema,
	preservingPriorityLevels []*rmtypesv1alpha1.PriorityLevelConfiguration,
) Interface {
	reqMgmt := &requestManagementSystem{
		wg:                     fq.WrapWaitGroupPointer(waitGroup),
		queueSetFactory:        queueSetFactory,
		serverConcurrencyLimit: serverConcurrencyLimit,
		requestWaitLimit:       requestWaitLimit,
		flowcontrolClient:      flowcontrolClient,
	}
	klog.V(2).Infof("NewRequestManagementSystem with serverConcurrencyLimit=%d, requestWaitLimit=%s", serverConcurrencyLimit, requestWaitLimit)
	reqMgmt.initializeConfigController(informerFactory)
	emptyRMState := &requestManagementState{
		priorityLevelStates: make(map[string]*priorityLevelState),
	}
	reqMgmt.curState.Store(emptyRMState)
	if len(preservingFlowSchemas) > 0 || len(preservingPriorityLevels) > 0 {
		reqMgmt.digestConfigObjects(
			preservingPriorityLevels,
			preservingFlowSchemas,
		)
	}
	reqMgmt.readyFunc = func() bool {
		existingFSNames, existingPLNames := sets.NewString(), sets.NewString()
		existingFlowSchemas, err := reqMgmt.fsLister.List(labels.Everything())
		if err != nil {
			klog.Errorf("failed to list flow-schemas: %v", err)
			return false
		}
		existingPriorityLevels, err := reqMgmt.plLister.List(labels.Everything())
		if err != nil {
			klog.Errorf("failed to list priority-levels: %v", err)
			return false
		}

		for _, fs := range existingFlowSchemas {
			existingFSNames.Insert(fs.Name)
		}
		for _, fs := range preservingFlowSchemas {
			if !existingFSNames.Has(fs.Name) {
				klog.V(5).Infof("waiting for preserved flow-schema %s to be ready", fs.Name)
				return false
			}
		}
		for _, pl := range existingPriorityLevels {
			existingPLNames.Insert(pl.Name)
		}
		for _, pl := range preservingPriorityLevels {
			if !existingPLNames.Has(pl.Name) {
				klog.V(5).Infof("waiting for preserved priority-level %s to be ready", pl.Name)
				return false
			}
		}
		return true
	}
	return reqMgmt
}

// RequestDigest holds necessary info from request for flow-control
type RequestDigest struct {
	RequestInfo *request.RequestInfo
	User        user.Info
}

func (reqMgmt *requestManagementSystem) Wait(requestDigest RequestDigest) (bool, func()) {
	for {
		rmState := reqMgmt.curState.Load().(*requestManagementState)

		var matchingflowSchemaName string
		var matchingpriorityLevelName string
		var matchingFlowDistinguisherMethod *rmtypesv1alpha1.FlowDistinguisherMethod

		// 1. computing flow
		fs := rmState.pickFlowSchema(requestDigest)
		switch {
		case fs != nil: // successfully matched a flow-schema
			matchingflowSchemaName = fs.Name
			matchingpriorityLevelName = fs.Spec.PriorityLevelConfiguration.Name
			matchingFlowDistinguisherMethod = fs.Spec.DistinguisherMethod
		default: // reject
			return false, func() {}
		}

		// 2. computing hash
		flowDistinguisher := requestDigest.ComputeFlowDistinguisher(matchingFlowDistinguisherMethod)
		hashValue := hashFlowID(matchingflowSchemaName, flowDistinguisher)

		// 3. executing
		ps := rmState.priorityLevelStates[matchingpriorityLevelName]
		if ps.config.Exempt {
			klog.V(7).Infof("Serving %v without delay", requestDigest)
			return true, func() {}
		}
		quiescent, execute, afterExecute := ps.queues.Wait(hashValue, ps.config.HandSize)
		if quiescent {
			klog.V(5).Infof("Request %v landed in timing splinter, re-classifying", requestDigest)
			continue
		}
		return execute, afterExecute
	}
}

func (rmState *requestManagementState) pickFlowSchema(rd RequestDigest) *rmtypesv1alpha1.FlowSchema {
	for _, flowSchema := range rmState.flowSchemas {
		if matchesFlowSchema(rd, flowSchema) {
			return flowSchema
		}
	}
	return nil
}

// ComputeFlowDistinguisher extracts the flow distinguisher according to the given method
func (rd RequestDigest) ComputeFlowDistinguisher(method *rmtypesv1alpha1.FlowDistinguisherMethod) string {
	if method == nil {
		return ""
	}
	switch method.Type {
	case rmtypesv1alpha1.FlowDistinguisherMethodByUserType:
		return rd.User.GetName()
	case rmtypesv1alpha1.FlowDistinguisherMethodByNamespaceType:
		return rd.RequestInfo.Namespace
	default:
		// this line shall never reach
		panic("invalid flow-distinguisher method")
	}
}

// HashFlowID hashes the inputs into 64-bits
func hashFlowID(fsName, fDistinguisher string) uint64 {
	return crc64.Checksum([]byte(fsName+fDistinguisher), crc64.MakeTable(crc64.ECMA))
}
