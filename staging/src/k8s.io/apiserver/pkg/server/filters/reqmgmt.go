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
	"net/http"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	cache "k8s.io/client-go/tools/cache"
	workqueue "k8s.io/client-go/util/workqueue"

	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	rmlisterv1a1 "k8s.io/client-go/listers/flowcontrol/v1alpha1"

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

// RMState is the variable state that this filter is working with at a
// given point in time.
type RMState struct {
	// flowSchemas holds the flow schema objects, sorted by increasing
	// numerical (decreasing logical) matching precedence
	flowSchemas FlowSchemaSeq

	// priorityLevelStates maps the PriorityLevelConfiguration object
	// name to the state for that level
	priorityLevelStates map[string]*PriorityLevelState
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
	clk clock.Clock

	fairQueuingFactory FairQueuingFactory

	// configQueue holds TypedConfigObjectReference values, identifying
	// config objects that need to be processed
	configQueue workqueue.RateLimitingInterface

	// plInformer is the informer for priority level config objects
	plInformer cache.SharedIndexInformer

	plLister rmlisterv1a1.PriorityLevelConfigurationLister

	// fsInformer is the informer for flow schema config objects
	fsInformer cache.SharedIndexInformer

	fsLister rmlisterv1a1.FlowSchemaLister

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
func rmSetup(kubeClient kubernetes.Interface, serverConcurrencyLimit int, requestWaitLimit time.Duration, clk clock.Clock) *requestManagement {
	kubeInformerFactory := kubeinformers.NewSharedInformerFactory(kubeClient, 0)
	fci := kubeInformerFactory.Flowcontrol().V1alpha1()
	pli := fci.PriorityLevelConfigurations()
	fsi := fci.FlowSchemas()
	reqMgmt := &requestManagement{
		clk:                    clk,
		configQueue:            workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(200*time.Millisecond, 8*time.Hour), "req_mgmt_config_queue"),
		plInformer:             pli.Informer(),
		plLister:               pli.Lister(),
		fsInformer:             fsi.Informer(),
		fsLister:               fsi.Lister(),
		serverConcurrencyLimit: serverConcurrencyLimit,
		requestWaitLimit:       requestWaitLimit,
	}
	// TODO: finish implementation
	return reqMgmt
}

// WithRequestManagement limits the number of in-flight requests in a fine-grained way
func WithRequestManagement(
	handler http.Handler,
	clientConfig *restclient.Config,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
) http.Handler {
	kubeClient, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		klog.Errorf("Failed to construct Kubernetes client (%s), skipping prioritization and fairness filter", err.Error())
		return handler
	}
	return WithRequestManagementByClient(handler, kubeClient, serverConcurrencyLimit, requestWaitLimit, longRunningRequestCheck, clock.RealClock{})
}

// WithRequestManagementByClient limits the number of in-flight
// requests in a fine-grained way and is more appropriate than
// WithRequestManagement for testing
func WithRequestManagementByClient(
	handler http.Handler,
	kubeClient kubernetes.Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
	clk clock.Clock,
) http.Handler {
	reqMgmt := rmSetup(kubeClient, serverConcurrencyLimit, requestWaitLimit, clk)

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

		for {
			rmState := reqMgmt.curState.Load().(*RMState)
			fs := reqMgmt.pickFlowSchema(r, rmState.flowSchemas, rmState.priorityLevelStates)
			ps := reqMgmt.requestPriorityState(r, fs, rmState.priorityLevelStates)
			if ps.config.Exempt {
				klog.V(5).Infof("Serving %v without delay\n", r)
				handler.ServeHTTP(w, r)
				return
			}
			flowDistinguisher := reqMgmt.computeFlowDistinguisher(r, fs)
			hashValue := reqMgmt.hashFlowID(fs.Name, flowDistinguisher)
			quiescent, execute, afterExecute := ps.fqs.Wait(hashValue, ps.config.HandSize)
			if quiescent {
				klog.V(3).Infof("Request %v landed in timing splinter, re-classifying", r)
				continue
			}
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
				}
				afterExecute()
			} else {
				klog.V(5).Infof("Rejecting %v\n", r)

				tooManyRequests(r, w)
			}
		}

		return
	})
}

func (requestManagement) computeFlowDistinguisher(r *http.Request, fs *rmtypesv1a1.FlowSchema) string {
	// TODO: implement
	return ""
}

func (requestManagement) hashFlowID(fsName, fDistinguisher string) uint64 {
	// TODO: implement
	return 0
}

func (requestManagement) pickFlowSchema(r *http.Request, flowSchemas FlowSchemaSeq, priorityLevelStates map[string]*PriorityLevelState) *rmtypesv1a1.FlowSchema {
	// TODO: implement
	return nil
}

func (requestManagement) requestPriorityState(r *http.Request, fs *rmtypesv1a1.FlowSchema, priorityLevelStates map[string]*PriorityLevelState) *PriorityLevelState {
	// TODO: implement
	return nil
}
