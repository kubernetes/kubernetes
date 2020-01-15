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
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/util/apihelpers"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/apiserver/pkg/util/shufflesharding"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// This file contains a simple local (to the apiserver) controller
// that digests API Priority and Fairness config objects (FlowSchema
// and PriorityLevelConfiguration) into the data structure that the
// filter uses.  At this first level of development this controller
// takes the simplest possible approach: whenever notified of any
// change to any config object, all them are read and processed as a
// whole.

// initializeConfigController sets up the controller that processes
// config API objects.
func (reqMgr *requestManager) initializeConfigController(informerFactory kubeinformers.SharedInformerFactory) {
	reqMgr.configQueue = workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(200*time.Millisecond, 8*time.Hour), "req_mgmt_config_queue")
	fci := informerFactory.Flowcontrol().V1alpha1()
	pli := fci.PriorityLevelConfigurations()
	fsi := fci.FlowSchemas()
	reqMgr.plLister = pli.Lister()
	reqMgr.plInformerSynced = pli.Informer().HasSynced
	reqMgr.fsLister = fsi.Lister()
	reqMgr.fsInformerSynced = fsi.Informer().HasSynced
	pli.Informer().AddEventHandler(reqMgr)
	fsi.Informer().AddEventHandler(reqMgr)
}

func (reqMgr *requestManager) triggerReload() {
	klog.V(7).Info("triggered request-management system reloading")
	reqMgr.configQueue.Add(0)
}

// OnAdd handles notification from an informer of object creation
func (reqMgr *requestManager) OnAdd(obj interface{}) {
	reqMgr.triggerReload()
}

// OnUpdate handles notification from an informer of object update
func (reqMgr *requestManager) OnUpdate(oldObj, newObj interface{}) {
	reqMgr.triggerReload()
}

// OnDelete handles notification from an informer of object deletion
func (reqMgr *requestManager) OnDelete(obj interface{}) {
	reqMgr.triggerReload()
}

func (reqMgr *requestManager) Run(stopCh <-chan struct{}) error {
	defer reqMgr.configQueue.ShutDown()
	klog.Info("Starting reqmgmt config controller")
	if ok := cache.WaitForCacheSync(stopCh, reqMgr.plInformerSynced, reqMgr.fsInformerSynced); !ok {
		return fmt.Errorf("Never achieved initial sync")
	}
	klog.Info("Running reqmgmt config worker")
	wait.Until(reqMgr.runWorker, time.Second, stopCh)
	klog.Info("Shutting down reqmgmt config worker")
	return nil
}

func (reqMgr *requestManager) runWorker() {
	for reqMgr.processNextWorkItem() {
	}
}

func (reqMgr *requestManager) processNextWorkItem() bool {
	obj, shutdown := reqMgr.configQueue.Get()
	if shutdown {
		return false
	}

	func(obj interface{}) {
		defer reqMgr.configQueue.Done(obj)
		if !reqMgr.syncOne() {
			reqMgr.configQueue.AddRateLimited(obj)
		} else {
			reqMgr.configQueue.Forget(obj)
		}
	}(obj)

	return true
}

// syncOne attempts to sync all the config for the reqmgmt filter.  It
// either succeeds and returns `true` or logs an error and returns
// `false`.
func (reqMgr *requestManager) syncOne() bool {
	all := labels.Everything()
	newPLs, err := reqMgr.plLister.List(all)
	if err != nil {
		klog.Errorf("Unable to list PriorityLevelConfiguration objects: %s", err.Error())
		return false
	}
	newFSs, err := reqMgr.fsLister.List(all)
	if err != nil {
		klog.Errorf("Unable to list FlowSchema objects: %s", err.Error())
		return false
	}
	reqMgr.digestConfigObjects(newPLs, newFSs)
	return true
}

// digestConfigObjects is given all the API objects that configure
// reqMgr and writes its consequent new requestManagerState.  This
// function has three loops over priority levels: one to digest the
// given objects, one to handle old objects, and one to divide up the
// server's total concurrency limit among the surviving priority
// levels.
func (reqMgr *requestManager) digestConfigObjects(newPLs []*rmtypesv1a1.PriorityLevelConfiguration, newFSs []*rmtypesv1a1.FlowSchema) {
	oldRMState := reqMgr.curState.Load().(*requestManagerState)
	var shareSum float64 // accumulated in first two loops over priority levels
	newRMState := &requestManagerState{
		priorityLevelStates: make(map[string]*priorityLevelState),
	}

	// Buffer of priority levels that need a call to QueueSet::Quiesce.
	// See the explanation later for why these calls are held until the end.
	newlyQuiescent := make([]*priorityLevelState, 0)

	// Keep track of which mandatory objects have been digested.
	var haveExemptPL, haveCatchAllPL, haveExemptFS, haveCatchAllFS bool

	// Digest each given PriorityLevelConfiguration.
	// Pretend broken ones do not exist.
	for _, pl := range newPLs {
		state := oldRMState.priorityLevelStates[pl.Name]
		if state == nil {
			state = &priorityLevelState{
				config: pl.Spec,
			}
		} else {
			oState := *state
			state = &oState
			state.config = pl.Spec
			if state.emptyHandler != nil { // it was undesired, but no longer
				klog.V(3).Infof("Priority level %q was undesired and has become desired again", pl.Name)
				state.emptyHandler = nil
				state.queues.Quiesce(nil)
			}
		}
		if state.config.Limited != nil {
			shareSum += float64(state.config.Limited.AssuredConcurrencyShares)
			qsConfig, err := qscOfPL(pl, reqMgr.requestWaitLimit)
			if err != nil {
				klog.Warningf(err.Error())
				continue
			}
			state.qsConfig = qsConfig
		}
		haveExemptPL = haveExemptPL || pl.Name == rmtypesv1a1.PriorityLevelConfigurationNameExempt
		haveCatchAllPL = haveCatchAllPL || pl.Name == rmtypesv1a1.PriorityLevelConfigurationNameCatchAll
		newRMState.priorityLevelStates[pl.Name] = state
	}

	// Digest the given FlowSchema objects.  Ones that reference a
	// missing or broken priority level are not to be passed on to the
	// filter for use.  We do this before holding over old priority
	// levels so that requests stop going to those levels and
	// FlowSchemaStatus values reflect this.
	fsSeq := make(apihelpers.FlowSchemaSequence, 0, len(newFSs))
	for i, fs := range newFSs {
		_, goodPriorityRef := newRMState.priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]

		// Ensure the object's status reflects whether its priority
		// level reference is broken.
		//
		// TODO: consider
		// k8s.io/apimachinery/pkg/util/errors.NewAggregate
		// errors from all of these and return it at the end.
		//
		// TODO: consider not even trying if server is not handling
		// requests yet.
		reqMgr.syncFlowSchemaStatus(fs, !goodPriorityRef)

		if !goodPriorityRef {
			continue
		}
		fsSeq = append(fsSeq, newFSs[i])
		haveExemptFS = haveExemptFS || fs.Name == rmtypesv1a1.FlowSchemaNameExempt
		haveCatchAllFS = haveCatchAllFS || fs.Name == rmtypesv1a1.FlowSchemaNameCatchAll
	}
	// sort into the order to be used for matching
	sort.Sort(fsSeq)

	// Supply missing mandatory FlowSchemas, in correct position
	if !haveExemptFS {
		fsSeq = append(apihelpers.FlowSchemaSequence{fcboot.MandatoryFlowSchemaExempt}, fsSeq...)
	}
	if !haveCatchAllFS {
		fsSeq = append(fsSeq, fcboot.MandatoryFlowSchemaCatchAll)
	}

	newRMState.flowSchemas = fsSeq
	if klog.V(5) {
		for _, fs := range fsSeq {
			klog.Infof("Using FlowSchema %s: %#+v", fs.Name, fs.Spec)
		}
	}

	// Consider all the priority levels in the previous configuration.
	// Keep the ones that are in the new config, supply mandatory
	// behavior, or still have non-empty queues; for the rest: drop it
	// if it has no queues, otherwise start the quiescing process if
	// that has not already been started.
	for plName, plState := range oldRMState.priorityLevelStates {
		if newRMState.priorityLevelStates[plName] != nil {
			// Still desired and already updated
			continue
		}
		newState := *plState
		plState = &newState
		if plState.emptyHandler != nil && plState.emptyHandler.IsEmpty() {
			// The queues are empty and will never get any more requests
			plState.queues = nil
			plState.emptyHandler = nil
			klog.V(3).Infof("Retired queues for undesired quiescing priority level %q", plName)
		}
		if plName == rmtypesv1a1.PriorityLevelConfigurationNameExempt && !haveExemptPL || plName == rmtypesv1a1.PriorityLevelConfigurationNameCatchAll && !haveCatchAllPL {
			klog.V(3).Infof("Retaining old priority level %q with Type=%v because of lack of replacement", plName, plState.config.Type)
		} else {
			if plState.queues == nil {
				klog.V(3).Infof("Removing undesired priority level %q, Type=%v", plName, plState.config.Type)
				continue
			}
			if plState.emptyHandler == nil {
				klog.V(3).Infof("Priority level %q became undesired", plName)
				plState.emptyHandler = &emptyRelay{reqMgr: reqMgr}
				newlyQuiescent = append(newlyQuiescent, plState)
			}
		}
		if plState.config.Limited != nil {
			shareSum += float64(plState.config.Limited.AssuredConcurrencyShares)
		}
		haveExemptPL = haveExemptPL || plName == rmtypesv1a1.PriorityLevelConfigurationNameExempt
		haveCatchAllPL = haveCatchAllPL || plName == rmtypesv1a1.PriorityLevelConfigurationNameCatchAll
		newRMState.priorityLevelStates[plName] = plState
	}

	// Supply missing mandatory objects
	if !haveExemptPL {
		newRMState.imaginePL(fcboot.MandatoryPriorityLevelConfigurationExempt, reqMgr.requestWaitLimit, &shareSum)
	}
	if !haveCatchAllPL {
		newRMState.imaginePL(fcboot.MandatoryPriorityLevelConfigurationCatchAll, reqMgr.requestWaitLimit, &shareSum)
	}

	// For all the priority levels of the new config, divide up the
	// server's total concurrency limit among them and create/update
	// their QueueSets.
	for plName, plState := range newRMState.priorityLevelStates {
		if plState.config.Limited == nil {
			klog.V(5).Infof("Using exempt priority level %q: quiescent=%v", plName, plState.emptyHandler != nil)
			continue
		}

		plState.qsConfig.ConcurrencyLimit = int(math.Ceil(float64(reqMgr.serverConcurrencyLimit) * float64(plState.config.Limited.AssuredConcurrencyShares) / shareSum))
		metrics.UpdateSharedConcurrencyLimit(plName, plState.qsConfig.ConcurrencyLimit)

		if plState.queues == nil {
			klog.V(5).Infof("Introducing queues for priority level %q: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, plState.qsConfig.ConcurrencyLimit, plState.emptyHandler != nil, plState.config.Limited.AssuredConcurrencyShares, shareSum)
			plState.queues = reqMgr.queueSetFactory.NewQueueSet(plState.qsConfig)
		} else {
			klog.V(5).Infof("Retaining queues for priority level %q: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, plState.qsConfig.ConcurrencyLimit, plState.emptyHandler != nil, plState.config.Limited.AssuredConcurrencyShares, shareSum)
			plState.queues.SetConfiguration(plState.qsConfig)
		}
	}

	// The new config has been constructed, pass to filter for use.
	reqMgr.curState.Store(newRMState)
	klog.V(5).Infof("Switched to new RequestManagementState")

	// We delay the calls to QueueSet::Quiesce so that the following
	// proof works.
	//
	// 1. For any QueueSet S: if a call S.Wait() returns with
	//    tryAnother==true then a call to S.Quiesce with a non-nil
	//    handler happened before that return.  This is from the
	//    contract of QueueSet.
	//
	// 2. Every S.Quiesce call with a non-nil handler happens after a
	//    call to `reqMgr.curState.Store(X)` for which S is in
	//    newlyQuiescent and S's priority level is not referenced from
	//    any FlowSchema in X.  This is established by the text of
	//    this function and the immutability of each FlowSchemaSpec
	//    and of each plState after completion of the loop iteration
	//    that constructs it.
	//
	// 3. If a call to S.Wait that returns with tryAnother==true
	//    happens before a call to `curState.Load()` that returns a
	//    value Y then either (3a) Y sends no traffic to S or (3b) Y
	//    is a value stored after X.  This is the contract of the
	//    `curState` field.  Chaining together the "happens before"
	//    relationships of (2), (1), and (3) implies that
	//    `curState.Store(X)` happens before `curState.Load()` returns
	//    Y.  The fact that this function stores a fresh pointer in
	//    curState each time, together with the contract of
	//    `atomic.Value`, implies that either Y is X (which, in turn,
	//    implies 3a) or Y is a value stored later (which is 3b).
	for _, plState := range newlyQuiescent {
		plState.queues.Quiesce(plState.emptyHandler)
	}
}

func qscOfPL(pl *rmtypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration) (fq.QueueSetConfig, error) {
	qsConfig := fq.QueueSetConfig{Name: pl.Name,
		RequestWaitLimit: requestWaitLimit}
	var err error
	if qc := pl.Spec.Limited.LimitResponse.Queuing; qc != nil {
		dealer, e1 := shufflesharding.NewDealer(int(qc.Queues), int(qc.HandSize))
		if e1 != nil {
			err = errors.Wrap(e1, fmt.Sprintf("priority level %q has QueuingConfiguration %#+v, which caused shufflesharding.NewDealer to fail", pl.Name, *qc))
		}
		qsConfig.DesiredNumQueues = int(qc.Queues)
		qsConfig.QueueLengthLimit = int(qc.QueueLengthLimit)
		qsConfig.Dealer = dealer
	}
	return qsConfig, err
}

func (reqMgr *requestManager) syncFlowSchemaStatus(fs *rmtypesv1a1.FlowSchema, isDangling bool) {
	danglingCondition := apihelpers.GetFlowSchemaConditionByType(fs, rmtypesv1a1.FlowSchemaConditionDangling)
	if danglingCondition == nil {
		danglingCondition = &rmtypesv1a1.FlowSchemaCondition{
			Type: rmtypesv1a1.FlowSchemaConditionDangling,
		}
	}

	switch {
	case isDangling && danglingCondition.Status != rmtypesv1a1.ConditionTrue:
		danglingCondition.Status = rmtypesv1a1.ConditionTrue
		danglingCondition.LastTransitionTime = metav1.Now()
	case !isDangling && danglingCondition.Status != rmtypesv1a1.ConditionFalse:
		danglingCondition.Status = rmtypesv1a1.ConditionFalse
		danglingCondition.LastTransitionTime = metav1.Now()
	default:
		// the dangling status is already in sync, skip updating
		return
	}

	apihelpers.SetFlowSchemaCondition(fs, *danglingCondition)

	_, err := reqMgr.flowcontrolClient.FlowSchemas().UpdateStatus(fs)
	if err != nil {
		klog.Warningf("failed updating condition for flow-schema %s", fs.Name)
	}
}

func (newRMState *requestManagerState) imaginePL(proto *rmtypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration, shareSum *float64) {
	klog.Warningf("No %s PriorityLevelConfiguration found, imagining one", proto.Name)
	qsConfig, err := qscOfPL(proto, requestWaitLimit)
	if err != nil {
		klog.Errorf(err.Error())
	}
	newRMState.priorityLevelStates[proto.Name] = &priorityLevelState{
		config:   proto.Spec,
		qsConfig: qsConfig,
	}
	if proto.Spec.Limited != nil {
		*shareSum += float64(proto.Spec.Limited.AssuredConcurrencyShares)
	}
	return
}

type emptyRelay struct {
	sync.RWMutex
	reqMgr *requestManager
	empty  bool
}

var _ fq.EmptyHandler = &emptyRelay{}

func (er *emptyRelay) HandleEmpty() {
	er.Lock()
	er.empty = true
	// TODO: to support testing of the config controller, extend
	// goroutine tracking to the config queue and worker
	er.reqMgr.configQueue.Add(0)
	er.Unlock()
	er.reqMgr.wg.Add(-1)
}

func (er *emptyRelay) IsEmpty() bool {
	er.RLock()
	defer func() { er.RUnlock() }()
	return er.empty
}
