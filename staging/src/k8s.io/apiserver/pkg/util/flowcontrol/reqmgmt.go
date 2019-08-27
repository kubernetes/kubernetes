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
	"strconv"
	"sync/atomic"
	"time"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/waitgroup"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	rmtypesv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	fcbootstrap "k8s.io/apiserver/pkg/util/flowcontrol/bootstrap"
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

// requestManagerState is the variable state that this filter is
// working with at a given point in time.
type requestManagerState struct {
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

// requestManager holds all the state and infrastructure of
// this filter
type requestManager struct {
	// wg is kept informed of when goroutines start or stop or begin or end waiting
	wg waitgroup.OptionalWaitGroup

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

	// curState holds a pointer to the current requestManagerState.
	// That is, `Load()` produces a `*requestManagerState`.  When a
	// config work queue worker processes a configuration change, it
	// stores a new pointer here --- it does NOT side-effect the old
	// `requestManagerState` value.  The new
	// `requestManagerState` has a freshly constructed slice of
	// FlowSchema pointers and a freshly constructed map of priority
	// level states.
	curState atomic.Value
}

// NewRequestManager creates a new instance to implement API priority and fairness
func NewRequestManager(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient rmclientv1alpha1.FlowcontrolV1alpha1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	waitForAllPredefined bool,
) Interface {
	wg := waitgroup.NoWaitGroup()
	return NewRequestManagerTestable(
		informerFactory,
		flowcontrolClient,
		serverConcurrencyLimit,
		requestWaitLimit,
		waitForAllPredefined,
		wg,
		fq.NewQueueSetFactory(&clock.RealClock{}, wg),
	)
}

// NewRequestManagerTestable is extra flexible to facilitate testing
func NewRequestManagerTestable(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient rmclientv1alpha1.FlowcontrolV1alpha1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	waitForAllPredefined bool,
	wg waitgroup.OptionalWaitGroup,
	queueSetFactory fq.QueueSetFactory,
) Interface {
	initialPLCs := fcbootstrap.InitialPriorityLevelConfigurations
	initialFSs := fcbootstrap.InitialFlowSchemas
	waitPLCs := initialPLCs
	waitFSs := initialFSs
	if waitForAllPredefined {
		waitPLCs = fcbootstrap.PredefinedPriorityLevelConfigurations()
		waitFSs = fcbootstrap.PredefinedFlowSchemas()
	}

	reqMgr := &requestManager{
		wg:                     wg,
		queueSetFactory:        queueSetFactory,
		serverConcurrencyLimit: serverConcurrencyLimit,
		requestWaitLimit:       requestWaitLimit,
		flowcontrolClient:      flowcontrolClient,
	}
	klog.V(2).Infof("NewRequestManagementSystem with serverConcurrencyLimit=%d, requestWaitLimit=%s", serverConcurrencyLimit, requestWaitLimit)
	reqMgr.initializeConfigController(informerFactory)
	emptyRMState := &requestManagerState{
		priorityLevelStates: make(map[string]*priorityLevelState),
	}
	reqMgr.curState.Store(emptyRMState)
	reqMgr.digestConfigObjects(
		initialPLCs,
		initialFSs,
	)
	reqMgr.readyFunc = func() bool {
		existingFSNames, existingPLNames := sets.NewString(), sets.NewString()
		existingFlowSchemas, err := reqMgr.fsLister.List(labels.Everything())
		if err != nil {
			klog.Errorf("failed to list flow-schemas: %v", err)
			return false
		}
		existingPriorityLevels, err := reqMgr.plLister.List(labels.Everything())
		if err != nil {
			klog.Errorf("failed to list priority-levels: %v", err)
			return false
		}

		for _, fs := range existingFlowSchemas {
			existingFSNames.Insert(fs.Name)
		}
		for _, fs := range waitFSs {
			if !existingFSNames.Has(fs.Name) {
				klog.V(5).Infof("waiting for flow-schema %s to be ready", fs.Name)
				return false
			}
		}
		for _, pl := range existingPriorityLevels {
			existingPLNames.Insert(pl.Name)
		}
		for _, pl := range waitPLCs {
			if !existingPLNames.Has(pl.Name) {
				klog.V(5).Infof("waiting for priority-level %s to be ready", pl.Name)
				return false
			}
		}
		return true
	}
	return reqMgr
}

// RequestDigest holds necessary info from request for flow-control
type RequestDigest struct {
	RequestInfo *request.RequestInfo
	User        user.Info
}

func (reqMgr *requestManager) Wait(requestDigest RequestDigest) (bool, func()) {
	startWaitingTime := time.Now()
	for {
		rmState := reqMgr.curState.Load().(*requestManagerState)

		// 1. figure out which flow schema applies
		fs := rmState.pickFlowSchema(requestDigest)
		if fs == nil { // reject
			metrics.AddReject("<none>", "non-match")
			klog.V(7).Infof("Rejecting requestInfo=%#+v, userInfo=%#+v because no FlowSchema matched", requestDigest.RequestInfo, requestDigest.User)
			return false, func() {}
		}
		plName := fs.Spec.PriorityLevelConfiguration.Name

		// 2. early out for exempt
		ps := rmState.priorityLevelStates[plName]
		if ps.config.Exempt {
			klog.V(7).Infof("Serving requestInfo=%#+v, userInfo=%#+v, fs=%s, pl=%s without delay", requestDigest.RequestInfo, requestDigest.User, fs.Name, plName)
			startExecutionTime := time.Now()
			return true, func() {
				metrics.ObserveExecutionDuration(plName, fs.Name, time.Now().Sub(startExecutionTime))
			}
		}

		// 3. computing hash
		flowDistinguisher := requestDigest.ComputeFlowDistinguisher(fs.Spec.DistinguisherMethod)
		hashValue := hashFlowID(fs.Name, flowDistinguisher)

		// 4. queuing
		quiescent, execute, afterExecute := ps.queues.Wait(hashValue, ps.config.HandSize)
		if quiescent {
			klog.V(5).Infof("Request requestInfo=%#+v, userInfo=%#+v, fs=%s, pl=%s landed in timing splinter, re-classifying", requestDigest.RequestInfo, requestDigest.User, fs.Name, plName)
			continue
		}

		// 5. execute or reject
		metrics.ObserveWaitingDuration(plName, fs.Name, strconv.FormatBool(execute), time.Now().Sub(startWaitingTime))
		if !execute {
			klog.V(7).Infof("Rejecting requestInfo=%#+v, userInfo=%#+v, fs=%s, pl=%s after fair queuing", requestDigest.RequestInfo, requestDigest.User, fs.Name, plName)
			return false, func() {}
		}
		klog.V(7).Infof("Serving requestInfo=%#+v, userInfo=%#+v, fs=%s, pl=%s after fair queuing", requestDigest.RequestInfo, requestDigest.User, fs.Name, plName)
		startExecutionTime := time.Now()
		return execute, func() {
			metrics.ObserveExecutionDuration(plName, fs.Name, time.Now().Sub(startExecutionTime))
			afterExecute()
		}
	}
}

func (rmState *requestManagerState) pickFlowSchema(rd RequestDigest) *rmtypesv1alpha1.FlowSchema {
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
