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
	"k8s.io/apiserver/pkg/util/apihelpers"
	fcboot "k8s.io/apiserver/pkg/util/flowcontrol/bootstrap"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/apiserver/pkg/util/shufflesharding"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
)

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

func (reqMgr *requestManager) digestConfigObjects(newPLs []*rmtypesv1a1.PriorityLevelConfiguration, newFSs []*rmtypesv1a1.FlowSchema) {
	oldRMState := reqMgr.curState.Load().(*requestManagerState)
	var shareSum float64
	newRMState := &requestManagerState{
		priorityLevelStates: make(map[string]*priorityLevelState),
	}
	newlyQuiescent := make([]*priorityLevelState, 0)
	var haveExemptPL, haveCatchAllPL, haveExemptFS, haveCatchAllFS bool
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
		} else {
			haveExemptPL = true
		}
		haveCatchAllPL = haveCatchAllPL || pl.Name == rmtypesv1a1.PriorityLevelConfigurationNameCatchAll
		newRMState.priorityLevelStates[pl.Name] = state
	}

	fsSeq := make(apihelpers.FlowSchemaSequence, 0, len(newFSs))
	for i, fs := range newFSs {
		_, flowSchemaExists := newRMState.priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]
		reqMgr.syncFlowSchemaStatus(fs, !flowSchemaExists)
		if flowSchemaExists {
			fsSeq = append(fsSeq, newFSs[i])
			haveExemptFS = haveExemptFS || fs.Name == rmtypesv1a1.FlowSchemaNameExempt
			haveCatchAllFS = haveCatchAllFS || fs.Name == rmtypesv1a1.FlowSchemaNameCatchAll
		}
	}
	sort.Sort(fsSeq)

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
		if plState.config.Limited == nil && !haveExemptPL || plName == rmtypesv1a1.PriorityLevelConfigurationNameCatchAll && !haveCatchAllPL {
			klog.V(3).Infof("Retaining old priority level %q with Type=%v because of lack of replacement", plName, plState.config.Type)
		} else {
			if plState.queues == nil { // should never happen; but if it does:
				klog.V(3).Infof("Immediately removing undesired priority level %q, Type=%v", plName, plState.config.Type)
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
		} else {
			haveExemptPL = true
		}
		haveCatchAllPL = haveCatchAllPL || plName == rmtypesv1a1.PriorityLevelConfigurationNameCatchAll
		newRMState.priorityLevelStates[plName] = plState
	}

	if !haveExemptPL {
		newRMState.imaginePL(0, reqMgr.requestWaitLimit, &shareSum)
	}
	if !haveCatchAllPL {
		newRMState.imaginePL(1, reqMgr.requestWaitLimit, &shareSum)
	}
	if !haveExemptFS {
		fsSeq = append(apihelpers.FlowSchemaSequence{fcboot.NewFSAllGroups(rmtypesv1a1.FlowSchemaNameExempt, rmtypesv1a1.PriorityLevelConfigurationNameExempt, 0, "", user.SystemPrivilegedGroup)}, fsSeq...)
	}
	if !haveCatchAllFS {
		fsSeq = append(fsSeq, fcboot.NewFSAllGroups(rmtypesv1a1.FlowSchemaNameCatchAll, rmtypesv1a1.PriorityLevelConfigurationNameCatchAll, math.MaxInt32, rmtypesv1a1.FlowDistinguisherMethodByUserType, user.AllAuthenticated, user.AllUnauthenticated))
	}

	newRMState.flowSchemas = fsSeq
	if klog.V(5) {
		for _, fs := range fsSeq {
			klog.Infof("Using FlowSchema %s: %#+v", fs.Name, fs.Spec)
		}
	}
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
			plState.queues.SetConfiguration(plState.qsConfig) // TODO: make sure that retains other config if zero desired queues
		}
	}
	reqMgr.curState.Store(newRMState)
	klog.V(5).Infof("Switched to new RequestManagementState")
	// We do the following only after updating curState to guarantee
	// that if Wait returns `tryAnother==true` then a fresh load from
	// curState will yield an requestManagerState that is at least
	// as up-to-date as the data here.
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

func (newRMState *requestManagerState) imaginePL(protoIdx int, requestWaitLimit time.Duration, shareSum *float64) {
	proto := fcboot.InitialPriorityLevelConfigurations[protoIdx]
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
