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
func (reqMgmt *requestManagementSystem) initializeConfigController(informerFactory kubeinformers.SharedInformerFactory) {
	reqMgmt.configQueue = workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(200*time.Millisecond, 8*time.Hour), "req_mgmt_config_queue")
	fci := informerFactory.Flowcontrol().V1alpha1()
	pli := fci.PriorityLevelConfigurations()
	fsi := fci.FlowSchemas()
	reqMgmt.plLister = pli.Lister()
	reqMgmt.plInformerSynced = pli.Informer().HasSynced
	reqMgmt.fsLister = fsi.Lister()
	reqMgmt.fsInformerSynced = fsi.Informer().HasSynced
	pli.Informer().AddEventHandler(reqMgmt)
	fsi.Informer().AddEventHandler(reqMgmt)
}

func (reqMgmt *requestManagementSystem) triggerReload() {
	klog.V(7).Info("triggered request-management system reloading")
	reqMgmt.configQueue.Add(0)
}

// OnAdd handles notification from an informer of object creation
func (reqMgmt *requestManagementSystem) OnAdd(obj interface{}) {
	reqMgmt.triggerReload()
}

// OnUpdate handles notification from an informer of object update
func (reqMgmt *requestManagementSystem) OnUpdate(oldObj, newObj interface{}) {
	reqMgmt.triggerReload()
}

// OnDelete handles notification from an informer of object deletion
func (reqMgmt *requestManagementSystem) OnDelete(obj interface{}) {
	reqMgmt.triggerReload()
}

func (reqMgmt *requestManagementSystem) Run(stopCh <-chan struct{}) error {
	defer reqMgmt.configQueue.ShutDown()
	klog.Info("Starting reqmgmt config controller")
	if ok := cache.WaitForCacheSync(stopCh, reqMgmt.readyFunc, reqMgmt.plInformerSynced, reqMgmt.fsInformerSynced); !ok {
		return fmt.Errorf("Never achieved initial sync")
	}
	klog.Info("Running reqmgmt config worker")
	wait.Until(reqMgmt.runWorker, time.Second, stopCh)
	klog.Info("Shutting down reqmgmt config worker")
	return nil
}

func (reqMgmt *requestManagementSystem) runWorker() {
	for reqMgmt.processNextWorkItem() {
	}
}

func (reqMgmt *requestManagementSystem) processNextWorkItem() bool {
	obj, shutdown := reqMgmt.configQueue.Get()
	if shutdown {
		return false
	}

	func(obj interface{}) {
		defer reqMgmt.configQueue.Done(obj)
		if !reqMgmt.syncOne() {
			reqMgmt.configQueue.AddRateLimited(obj)
		} else {
			reqMgmt.configQueue.Forget(obj)
		}
	}(obj)

	return true
}

// syncOne attempts to sync all the config for the reqmgmt filter.  It
// either succeeds and returns `true` or logs an error and returns
// `false`.
func (reqMgmt *requestManagementSystem) syncOne() bool {
	all := labels.Everything()
	newPLs, err := reqMgmt.plLister.List(all)
	if err != nil {
		klog.Errorf("Unable to list PriorityLevelConfiguration objects: %s", err.Error())
		return false
	}
	newFSs, err := reqMgmt.fsLister.List(all)
	if err != nil {
		klog.Errorf("Unable to list FlowSchema objects: %s", err.Error())
		return false
	}
	reqMgmt.digestConfigObjects(newPLs, newFSs)
	return true
}

func (reqMgmt *requestManagementSystem) digestConfigObjects(newPLs []*rmtypesv1a1.PriorityLevelConfiguration, newFSs []*rmtypesv1a1.FlowSchema) {
	oldRMState := reqMgmt.curState.Load().(*requestManagementState)
	var shareSum float64
	newRMState := &requestManagementState{
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
			if state.config.Exempt {
				newRMState.priorityLevelStates[pl.Name] = state
				continue
			}

			oState := *state
			state = &oState
			state.config = pl.Spec
			if state.emptyHandler != nil { // it was undesired, but no longer
				klog.V(3).Infof("Priority level %s was undesired and has become desired again", pl.Name)
				state.emptyHandler = nil
				state.queues.Quiesce(nil)
			}
		}
		shareSum += float64(state.config.AssuredConcurrencyShares)
		newRMState.priorityLevelStates[pl.Name] = state
		plByName[pl.Name] = newPLs[i]
	}

	fsSeq := make(rmtypesv1a1.FlowSchemaSequence, 0, len(newFSs))
	for i, fs := range newFSs {
		_, flowSchemaExists := newRMState.priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]
		reqMgmt.syncFlowSchemaStatus(fs, !flowSchemaExists)
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
		} else if !plState.config.Exempt {
			oState := *plState
			plState = &oState
			if plState.emptyHandler == nil {
				klog.V(3).Infof("Priority level %s became undesired", plName)
				plState.emptyHandler = &emptyRelay{reqMgmt: reqMgmt}
				newlyQuiescent = append(newlyQuiescent, plState)
			}
			newRMState.priorityLevelStates[plName] = plState
			shareSum += float64(plState.config.AssuredConcurrencyShares)
		} else {
			// keeps exempt or global-default
			newRMState.priorityLevelStates[plName] = plState
		}
	}
	for plName, plState := range newRMState.priorityLevelStates {
		if plState.config.Exempt {
			klog.V(5).Infof("Using exempt priority level %s: quiescent=%v", plName, plState.emptyHandler != nil)
			continue
		}
		plState.concurrencyLimit = int(math.Ceil(float64(reqMgmt.serverConcurrencyLimit) * float64(plState.config.AssuredConcurrencyShares) / shareSum))
		if plState.queues == nil {
			klog.V(5).Infof("Introducing priority level %s: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, plState.concurrencyLimit, plState.emptyHandler != nil, plState.config.AssuredConcurrencyShares, shareSum)
			plState.queues = reqMgmt.queueSetFactory.NewQueueSet(plState.concurrencyLimit, int(plState.config.Queues), int(plState.config.QueueLengthLimit), reqMgmt.requestWaitLimit)
		} else {
			klog.V(5).Infof("Retaining priority level %s: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, plState.concurrencyLimit, plState.emptyHandler != nil, plState.config.AssuredConcurrencyShares, shareSum)
			plState.queues.SetConfiguration(plState.concurrencyLimit, int(plState.config.Queues), int(plState.config.QueueLengthLimit), reqMgmt.requestWaitLimit)
		}
	}
	reqMgmt.curState.Store(newRMState)
	klog.V(5).Infof("Switched to new RequestManagementState")
	// We do the following only after updating curState to guarantee
	// that if Wait returns `quiescent==true` then a fresh load from
	// curState will yield an requestManagementState that is at least
	// as up-to-date as the data here.
	for _, plState := range newlyQuiescent {
		plState.queues.Quiesce(plState.emptyHandler)
	}
}

func (reqMgmt *requestManagementSystem) syncFlowSchemaStatus(fs *rmtypesv1a1.FlowSchema, isDangling bool) {
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

	_, err := reqMgmt.flowcontrolClient.FlowSchemas().UpdateStatus(fs)
	if err != nil {
		klog.Warningf("failed updating condition for flow-schema %s", fs.Name)
	}
}

type emptyRelay struct {
	sync.RWMutex
	reqMgmt *requestManagementSystem
	empty   bool
}

var _ fq.EmptyHandler = &emptyRelay{}

func (er *emptyRelay) HandleEmpty() {
	er.Lock()
	defer func() { er.Unlock() }()
	er.empty = true
	er.reqMgmt.configQueue.Add(0)
}

func (er *emptyRelay) IsEmpty() bool {
	er.RLock()
	defer func() { er.RUnlock() }()
	return er.empty
}
