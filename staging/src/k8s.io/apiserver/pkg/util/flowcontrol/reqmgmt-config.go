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

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	corev1 "k8s.io/api/core/v1"
	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	if ok := cache.WaitForCacheSync(stopCh, reqMgr.readyFunc, reqMgr.plInformerSynced, reqMgr.fsInformerSynced); !ok {
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
	plByName := map[string]*rmtypesv1a1.PriorityLevelConfiguration{}
	for i, pl := range newPLs {
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
				klog.V(3).Infof("Priority level %s was undesired and has become desired again", pl.Name)
				state.emptyHandler = nil
				state.queues.Quiesce(nil)
			}
		}
		if !pl.Spec.Exempt {
			shareSum += float64(state.config.AssuredConcurrencyShares)
		}
		newRMState.priorityLevelStates[pl.Name] = state
		plByName[pl.Name] = newPLs[i]
	}

	fsSeq := make(rmtypesv1a1.FlowSchemaSequence, 0, len(newFSs))
	for i, fs := range newFSs {
		_, flowSchemaExists := newRMState.priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]
		reqMgr.syncFlowSchemaStatus(fs, !flowSchemaExists)
		if flowSchemaExists {
			fsSeq = append(fsSeq, newFSs[i])
		}
	}
	sort.Sort(fsSeq)
	newRMState.flowSchemas = fsSeq

	if klog.V(5) {
		for _, fs := range fsSeq {
			klog.Infof("Using FlowSchema %s: %#+v", fs.Name, fs.Spec)
		}
	}
	for plName, plState := range oldRMState.priorityLevelStates {
		if newRMState.priorityLevelStates[plName] != nil {
			// Still desired
		} else if plState.emptyHandler != nil && plState.emptyHandler.IsEmpty() {
			// undesired, empty, and never going to get another request
			klog.V(3).Infof("Priority level %s removed from implementation", plName)
		} else {
			if plState.emptyHandler == nil {
				klog.V(3).Infof("Priority level %s became undesired", plName)
				newState := *plState
				plState = &newState
				plState.emptyHandler = &emptyRelay{reqMgr: reqMgr}
				newlyQuiescent = append(newlyQuiescent, plState)
			}
			newRMState.priorityLevelStates[plName] = plState
			if !plState.config.Exempt {
				shareSum += float64(plState.config.AssuredConcurrencyShares)
			}
		}
	}
	for plName, plState := range newRMState.priorityLevelStates {
		if plState.config.Exempt {
			klog.V(5).Infof("Using exempt priority level %s: quiescent=%v", plName, plState.emptyHandler != nil)
			continue
		}
		plState.concurrencyLimit = int(math.Ceil(float64(reqMgr.serverConcurrencyLimit) * float64(plState.config.AssuredConcurrencyShares) / shareSum))
		metrics.UpdateSharedConcurrencyLimit(plName, plState.concurrencyLimit)
		if plState.queues == nil {
			klog.V(5).Infof("Introducing priority level %s: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, plState.concurrencyLimit, plState.emptyHandler != nil, plState.config.AssuredConcurrencyShares, shareSum)
			plState.queues = reqMgr.queueSetFactory.NewQueueSet(plName, plState.concurrencyLimit, int(plState.config.Queues), int(plState.config.QueueLengthLimit), reqMgr.requestWaitLimit)
		} else {
			klog.V(5).Infof("Retaining priority level %s: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, plState.concurrencyLimit, plState.emptyHandler != nil, plState.config.AssuredConcurrencyShares, shareSum)
			plState.queues.SetConfiguration(plState.concurrencyLimit, int(plState.config.Queues), int(plState.config.QueueLengthLimit), reqMgr.requestWaitLimit)
		}
	}
	reqMgr.curState.Store(newRMState)
	klog.V(5).Infof("Switched to new RequestManagementState")
	// We do the following only after updating curState to guarantee
	// that if Wait returns `quiescent==true` then a fresh load from
	// curState will yield an requestManagerState that is at least
	// as up-to-date as the data here.
	for _, plState := range newlyQuiescent {
		plState.queues.Quiesce(plState.emptyHandler)
	}
}

func (reqMgr *requestManager) syncFlowSchemaStatus(fs *rmtypesv1a1.FlowSchema, isDangling bool) {
	danglingCondition := rmtypesv1a1.GetFlowSchemaConditionByType(fs, rmtypesv1a1.FlowSchemaConditionDangling)
	if danglingCondition == nil {
		danglingCondition = &rmtypesv1a1.FlowSchemaCondition{
			Type: rmtypesv1a1.FlowSchemaConditionDangling,
		}
	}

	switch {
	case isDangling && danglingCondition.Status != corev1.ConditionTrue:
		danglingCondition.Status = corev1.ConditionTrue
		danglingCondition.LastTransitionTime = metav1.Now()
	case !isDangling && danglingCondition.Status != corev1.ConditionFalse:
		danglingCondition.Status = corev1.ConditionFalse
		danglingCondition.LastTransitionTime = metav1.Now()
	default:
		// the dangling status is already in sync, skip updating
		return
	}

	rmtypesv1a1.SetFlowSchemaCondition(fs, *danglingCondition)

	_, err := reqMgr.flowcontrolClient.FlowSchemas().UpdateStatus(fs)
	if err != nil {
		klog.Warningf("failed updating condition for flow-schema %s", fs.Name)
	}
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
	er.reqMgr.wg.Decrement()
}

func (er *emptyRelay) IsEmpty() bool {
	er.RLock()
	defer func() { er.RUnlock() }()
	return er.empty
}
