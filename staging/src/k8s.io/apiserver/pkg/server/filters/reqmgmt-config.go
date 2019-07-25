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
	"math"
	"sort"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	cache "k8s.io/client-go/tools/cache"
	"k8s.io/klog"

	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
)

// initialSync does the initial setup of the configuration of the
// reqmgmt filter.  It either succeeds and returns `true` or logs an
// error and returns `false.
func (reqMgmt *requestManagement) initialSync() bool {
	rms := &RMState{
		flowSchemas:         make(FlowSchemaSeq, 0),
		priorityLevelStates: make(map[string]*PriorityLevelState),
	}
	reqMgmt.curState.Store(rms)
	reqMgmt.plInformer.AddEventHandler(reqMgmt)
	reqMgmt.fsInformer.AddEventHandler(reqMgmt)
	stopCh := make(chan struct{})
	go reqMgmt.plInformer.Run(stopCh)
	go reqMgmt.fsInformer.Run(stopCh)
	if ok := cache.WaitForCacheSync(stopCh, reqMgmt.plInformer.HasSynced, reqMgmt.fsInformer.HasSynced); !ok {
		return false
	}
	ok := reqMgmt.syncOne()
	if !ok {
		klog.Error("Unable to do initial config sync")
		return false
	}
	go reqMgmt.Run(stopCh)
	return true
}

// OnAdd handles notification from an informer of object creation
func (reqMgmt *requestManagement) OnAdd(obj interface{}) {
	reqMgmt.configQueue.Add(0)
}

// OnUpdate handles notification from an informer of object update
func (reqMgmt *requestManagement) OnUpdate(oldObj, newObj interface{}) {
	reqMgmt.OnAdd(newObj)
}

// OnDelete handles notification from an informer of object deletion
func (reqMgmt *requestManagement) OnDelete(obj interface{}) {
	reqMgmt.OnAdd(obj)
}

func (reqMgmt *requestManagement) Run(stopCh <-chan struct{}) {
	defer reqMgmt.configQueue.ShutDown()
	klog.Info("Starting reqmgmt config controller")
	go wait.Until(reqMgmt.runWorker, time.Second, stopCh)
	klog.Info("Started reqmgmt config worker")
	<-stopCh
	klog.Info("Shutting down reqmgmt config worker")
}

func (reqMgmt *requestManagement) runWorker() {
	for reqMgmt.processNextWorkItem() {
	}
}

func (reqMgmt *requestManagement) processNextWorkItem() bool {
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
func (reqMgmt *requestManagement) syncOne() bool {
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
	oldRMS := reqMgmt.curState.Load().(*RMState)
	var shareSum float64
	newRMS := &RMState{
		priorityLevelStates: make(map[string]*PriorityLevelState, len(newPLs)),
	}
	newlyQuiescent := make([]*PriorityLevelState, 0)
	var numExempt, numDefault int
	for _, pl := range newPLs {
		state := oldRMS.priorityLevelStates[pl.Name]
		if state == nil {
			state = &PriorityLevelState{
				config: pl.Spec,
			}
		} else {
			oState := *state
			state = &oState
			state.config = pl.Spec
			if state.emptyHandler != nil { // it was undesired, but no longer
				klog.V(3).Infof("Priority level %s was undesired and has become desired again", pl.Name)
				state.emptyHandler = nil
				state.fqs.Quiesce(nil)
			}
		}
		if state.config.GlobalDefault {
			numDefault++
		}
		if state.config.Exempt {
			numExempt++
		} else {
			shareSum += float64(state.config.AssuredConcurrencyShares)
		}
		newRMS.priorityLevelStates[pl.Name] = state
	}
	fsSeq := make(FlowSchemaSeq, len(newFSs))
	for _, fs := range newFSs {
		if !warnFlowSchemaSpec(fs.Name, &fs.Spec, newRMS.priorityLevelStates, oldRMS.priorityLevelStates) {
			continue
		}
		fsSeq = append(fsSeq, fs)
	}
	sort.Sort(fsSeq)
	newRMS.flowSchemas = fsSeq
	// TODO: https://github.com/kubernetes/enhancements/blob/735aef1c6158bb30dd17321994d53edb25fbe7c0/keps/sig-api-machinery/20190228-priority-and-fairness.md#default-behavior
	for plName, plState := range oldRMS.priorityLevelStates {
		if newRMS.priorityLevelStates[plName] != nil {
			// Still desired
		} else if plState.emptyHandler != nil && plState.emptyHandler.IsEmpty() {
			// undesired, empty, and never going to get another request
			klog.V(3).Infof("Priority level %s removed from implementation", plName)
		} else {
			if plState.emptyHandler == nil {
				klog.V(3).Infof("Priority level %s became undesired", plName)
				plState.emptyHandler = &emptyRelay{reqMgmt: reqMgmt}
				newlyQuiescent = append(newlyQuiescent, plState)
			}
			newRMS.priorityLevelStates[plName] = plState
			if !plState.config.Exempt {
				shareSum += float64(plState.config.AssuredConcurrencyShares)
			}
		}
	}
	for _, plState := range newRMS.priorityLevelStates {
		if plState.config.Exempt {
			continue
		}
		plState.concurrencyLimit = int(math.Ceil(float64(reqMgmt.serverConcurrencyLimit) * float64(plState.config.AssuredConcurrencyShares) / shareSum))
		if plState.fqs == nil {
			plState.fqs = reqMgmt.fairQueuingFactory.NewFairQueuingSystem(plState.concurrencyLimit, int(plState.config.Queues), int(plState.config.QueueLengthLimit), reqMgmt.requestWaitLimit, reqMgmt.clk)
		} else {
			plState.fqs.SetConfiguration(plState.concurrencyLimit, int(plState.config.Queues), int(plState.config.QueueLengthLimit), reqMgmt.requestWaitLimit)
		}
	}
	reqMgmt.curState.Store(newRMS)
	// We do the following only after updating curState to guarantee
	// that if Wait returns `quiescent==true` then a fresh load from
	// curState will yield an RMState that is at least as up-to-date
	// as the data here.
	for _, plState := range newlyQuiescent {
		plState.fqs.Quiesce(plState.emptyHandler)
	}
	return true
}

func warnFlowSchemaSpec(fsName string, spec *rmtypesv1a1.FlowSchemaSpec, newPriorities, oldPriorities map[string]*PriorityLevelState) bool {
	plName := spec.PriorityLevelConfiguration.Name
	if newPriorities[plName] == nil {
		problem := "non-existent"
		if oldPriorities[plName] != nil {
			problem = "undesired"
		}
		klog.Warningf("FlowSchema %s references %s priority level %s and will thus not match any requests", fsName, problem, plName)
		return false
	}
	return true
}

type emptyRelay struct {
	sync.RWMutex
	reqMgmt *requestManagement
	empty   bool
}

var _ EmptyHandler = &emptyRelay{}

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

var _ sort.Interface = FlowSchemaSeq(nil)

func (a FlowSchemaSeq) Len() int {
	return len(a)
}

func (a FlowSchemaSeq) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func (a FlowSchemaSeq) Less(i, j int) bool {
	return a[i].Spec.MatchingPrecedence < a[j].Spec.MatchingPrecedence
}
