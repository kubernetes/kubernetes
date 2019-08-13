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
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	kubeinformers "k8s.io/client-go/informers"
	cache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
)

// initializeConfigController sets up the controller that processes
// config API objects.
func (reqMgmt *requestManagementSystem) initializeConfigController(informerFactory kubeinformers.SharedInformerFactory) {
	reqMgmt.configQueue = workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(200*time.Millisecond, 8*time.Hour), "req_mgmt_config_queue")
	fci := informerFactory.Flowcontrol().V1alpha1()
	pli := fci.PriorityLevelConfigurations()
	fsi := fci.FlowSchemas()
	reqMgmt.plInformer = pli.Informer()
	reqMgmt.plLister = pli.Lister()
	reqMgmt.fsInformer = fsi.Informer()
	reqMgmt.fsLister = fsi.Lister()
	reqMgmt.plInformer.AddEventHandler(reqMgmt)
	reqMgmt.fsInformer.AddEventHandler(reqMgmt)
}

// OnAdd handles notification from an informer of object creation
func (reqMgmt *requestManagementSystem) OnAdd(obj interface{}) {
	reqMgmt.configQueue.Add(0)
}

// OnUpdate handles notification from an informer of object update
func (reqMgmt *requestManagementSystem) OnUpdate(oldObj, newObj interface{}) {
	reqMgmt.OnAdd(newObj)
}

// OnDelete handles notification from an informer of object deletion
func (reqMgmt *requestManagementSystem) OnDelete(obj interface{}) {
	reqMgmt.OnAdd(obj)
}

func (reqMgmt *requestManagementSystem) Run(stopCh <-chan struct{}) error {
	defer reqMgmt.configQueue.ShutDown()
	klog.Info("Starting reqmgmt config controller")
	if ok := cache.WaitForCacheSync(stopCh, reqMgmt.plInformer.HasSynced, reqMgmt.fsInformer.HasSynced); !ok {
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
		priorityLevelStates: make(map[string]*priorityLevelState, len(newPLs)),
	}
	newlyQuiescent := make([]*priorityLevelState, 0)
	var nameOfExempt, nameOfDefault string
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
				klog.V(3).Infof("Priority level %s was undesired and has become desired again", pl.Name)
				state.emptyHandler = nil
				state.queues.Quiesce(nil)
			}
		}
		if state.config.GlobalDefault {
			nameOfDefault = pl.Name
		}
		if state.config.Exempt {
			nameOfExempt = pl.Name
		} else {
			shareSum += float64(state.config.AssuredConcurrencyShares)
		}
		newRMState.priorityLevelStates[pl.Name] = state
	}
	fsSeq := make(FlowSchemaSequence, 0, len(newFSs))
	for _, fs := range newFSs {
		if !warnFlowSchemaSpec(fs.Name, &fs.Spec, newRMState.priorityLevelStates, oldRMState.priorityLevelStates) {
			continue
		}
		fsSeq = append(fsSeq, fs)
	}
	sort.Sort(fsSeq)
	if nameOfExempt == "" {
		nameOfExempt = newRMState.generateExemptPL()
	}
	if nameOfDefault == "" {
		nameOfDefault = newRMState.generateDefaultPL(&shareSum)
	}
	fsSeq = append(fsSeq, newFSAllObj("backstop to "+nameOfExempt, nameOfExempt, 9999, true, groups(user.SystemPrivilegedGroup)))
	fsSeq = append(fsSeq, newFSAllObj("backstop to "+nameOfDefault, nameOfDefault, 9999, false, groups(user.AllAuthenticated, user.AllUnauthenticated)))
	newRMState.flowSchemas = fsSeq
	for _, fs := range fsSeq {
		klog.V(5).Infof("Using FlowSchema %s: %#+v", fs.Name, fs.Spec)
	}
	for plName, plState := range oldRMState.priorityLevelStates {
		if newRMState.priorityLevelStates[plName] != nil {
			// Still desired
		} else if plState.emptyHandler != nil && plState.emptyHandler.IsEmpty() {
			// undesired, empty, and never going to get another request
			klog.V(3).Infof("Priority level %s removed from implementation", plName)
		} else {
			oState := *plState
			plState = &oState
			if plState.emptyHandler == nil {
				klog.V(3).Infof("Priority level %s became undesired", plName)
				plState.emptyHandler = &emptyRelay{reqMgmt: reqMgmt}
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

func warnFlowSchemaSpec(fsName string, spec *rmtypesv1a1.FlowSchemaSpec, newPriorities, oldPriorities map[string]*priorityLevelState) bool {
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

func (newRMState *requestManagementState) generateExemptPL() string {
	nameOfExempt := newRMState.genName("system-top")
	klog.Warningf("No Exempt PriorityLevelConfiguration found, imagining one named %q", nameOfExempt)
	newRMState.priorityLevelStates[nameOfExempt] = &priorityLevelState{
		config: DefaultPriorityLevelConfigurationObjects()[0].Spec,
	}
	return nameOfExempt
}

func (newRMState *requestManagementState) generateDefaultPL(shareSum *float64) string {
	nameOfDefault := newRMState.genName("workload-low")
	klog.Warningf("No GlobalDefault PriorityLevelConfiguration found, imagining one named %q", nameOfDefault)
	newRMState.priorityLevelStates[nameOfDefault] = &priorityLevelState{
		config: DefaultPriorityLevelConfigurationObjects()[1].Spec,
	}
	*shareSum += float64(newRMState.priorityLevelStates[nameOfDefault].config.AssuredConcurrencyShares)
	return nameOfDefault
}

func (newRMState *requestManagementState) genName(base string) string {
	for i := 1; true; i++ {
		extended := strings.TrimSuffix(fmt.Sprintf("%s-%d", base, i), "-1")
		if newRMState.priorityLevelStates[extended] == nil {
			return extended
		}
	}
	return "from an impossible place"
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

var _ sort.Interface = FlowSchemaSequence(nil)

func (a FlowSchemaSequence) Len() int {
	return len(a)
}

func (a FlowSchemaSequence) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func (a FlowSchemaSequence) Less(i, j int) bool {
	return a[i].Spec.MatchingPrecedence < a[j].Spec.MatchingPrecedence
}
