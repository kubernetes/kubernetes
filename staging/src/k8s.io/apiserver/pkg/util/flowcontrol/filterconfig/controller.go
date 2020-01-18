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

package filterconfig

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pkg/errors"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/util/apihelpers"
	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fqs "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	fctypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	fcclientv1a1 "k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1"
	fclistersv1a1 "k8s.io/client-go/listers/flowcontrol/v1alpha1"
)

// This file contains a simple local (to the apiserver) controller
// that digests API Priority and Fairness config objects (FlowSchema
// and PriorityLevelConfiguration) into the data structure that the
// filter uses.  At this first level of development this controller
// takes the simplest possible approach: whenever notified of any
// change to any config object, all them are read and processed as a
// whole.

// Controller maintains eventual consistency with the API objects that
// configure API Priority and Fairness, and provides a procedural
// interface to the configured behavior.
type Controller interface {
	// Run runs the controller, returning after it is stopped
	Run(stopCh <-chan struct{}) error

	// GetCurrentState returns the current configuration, as a
	// procedural interface value
	GetCurrentState() State
}

// State is the configuration at a point in time.
type State interface {

	// Match classifies the given request to the proper FlowSchema and
	// returns the relevant features of that schema and its associated
	// priority level.
	//
	// For every controller C, for every State X returned from
	// C.GetCurrentState(), and for every QueueSet S returned from
	// X.Match(...): if a call S.Wait(...) returns with
	// tryAnother==true and this happens before a call to
	// Controller::GetCurrentState() returns Y then either Y will not
	// return S from Match or Y reflects a more up-to-date
	// configuration than X.
	Match(RequestDigest) (flowSchemaName string, distinguisherMethod *fctypesv1a1.FlowDistinguisherMethod, priorityLevelName string, plEnablement fctypesv1a1.PriorityLevelEnablement, queues fq.QueueSet)
}

// RequestDigest holds necessary info from request for flow-control
type RequestDigest struct {
	RequestInfo *request.RequestInfo
	User        user.Info
}

// `*configController` implements Controller
type configController struct {
	queueSetFactory fq.QueueSetFactory

	// configQueue holds TypedConfigObjectReference values, identifying
	// config objects that need to be processed
	configQueue workqueue.RateLimitingInterface

	plLister         fclistersv1a1.PriorityLevelConfigurationLister
	plInformerSynced cache.InformerSynced

	fsLister         fclistersv1a1.FlowSchemaLister
	fsInformerSynced cache.InformerSynced

	flowcontrolClient fcclientv1a1.FlowcontrolV1alpha1Interface

	// serverConcurrencyLimit is the limit on the server's total
	// number of non-exempt requests being served at once.  This comes
	// from server configuration.
	serverConcurrencyLimit int

	// requestWaitLimit comes from server configuration.
	requestWaitLimit time.Duration

	// curState holds a pointer to the current configState.
	// That is, `Load()` produces a `*configState`.  When a
	// config work queue worker processes a configuration change, it
	// stores a new pointer here --- it does NOT side-effect the old
	// `configState` value.  The new `configState` has
	// a freshly constructed slice of FlowSchema pointers and a
	// freshly constructed map of priority level states.
	curState atomic.Value
}

// `*configState` implements State.
type configState struct {
	// flowSchemas holds the flow schema objects, sorted by increasing
	// numerical (decreasing logical) matching precedence.  Every
	// FlowSchema in this slice is immutable.
	flowSchemas apihelpers.FlowSchemaSequence

	// priorityLevelStates maps the PriorityLevelConfiguration object
	// name to the state for that level.  Every field of every
	// priorityLevelState in here is immutable after this state is
	// stored in the controller.  Every name referenced from a member
	// of `flowSchemas` has an entry here.
	priorityLevelStates map[string]*priorityLevelState
}

// priorityLevelState holds the state specific to a priority level.
type priorityLevelState struct {
	// config holds the configuration after defaulting logic has been applied.
	// Exempt may be true while there are queues, in the case of a priority
	// level that recently switched from being non-exempt to exempt and whose
	// queues are still draining.
	// If there are queues then their parameters are here.
	config fctypesv1a1.PriorityLevelConfigurationSpec

	// qsConfig holds the QueueSetConfig derived from `config` if
	// config is not exempt, nil otherwise.  Every non-nil value
	// points to a value that has been approved by the CheckConfig
	// method of the controller's QueueSetFactory.
	// qsConfig *fq.QueueSetConfig

	// qsCompleter holds the QueueSetCompleter derived from `config`
	// and `queues` if config is not exempt, nil otherwise.
	qsCompleter fq.QueueSetCompleter

	queues fq.QueueSet

	// Non-nil while waiting for queues to drain.
	// May be non-nil only if queues is non-nil.
	// May be non-nil while exempt.
	emptyHandler *emptyRelay
}

var _ Controller = (*configController)(nil)
var _ State = (*configState)(nil)

// NewController constructs a new Controller
func NewController(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient fcclientv1a1.FlowcontrolV1alpha1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
) Controller {
	grc := counter.NoOp{}
	return NewTestableController(
		informerFactory,
		flowcontrolClient,
		serverConcurrencyLimit,
		requestWaitLimit,
		fqs.NewQueueSetFactory(&clock.RealClock{}, grc),
	)
}

// NewTestableController is extra flexible to facilitate testing
func NewTestableController(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient fcclientv1a1.FlowcontrolV1alpha1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	queueSetFactory fq.QueueSetFactory,
) Controller {
	cfgCtl := &configController{
		queueSetFactory:        queueSetFactory,
		serverConcurrencyLimit: serverConcurrencyLimit,
		requestWaitLimit:       requestWaitLimit,
		flowcontrolClient:      flowcontrolClient,
	}
	klog.V(2).Infof("NewTestableController with serverConcurrencyLimit=%d, requestWaitLimit=%s", serverConcurrencyLimit, requestWaitLimit)
	cfgCtl.initializeConfigController(informerFactory)
	emptyCfgState := &configState{
		priorityLevelStates: make(map[string]*priorityLevelState),
	}
	cfgCtl.curState.Store(emptyCfgState)
	cfgCtl.digestConfigObjects(nil, nil)
	return cfgCtl
}

func (cfgCtl *configController) GetCurrentState() State {
	return cfgCtl.curState.Load().(*configState)
}

// initializeConfigController sets up the controller that processes
// config API objects.
func (cfgCtl *configController) initializeConfigController(informerFactory kubeinformers.SharedInformerFactory) {
	cfgCtl.configQueue = workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(200*time.Millisecond, 8*time.Hour), "priority_and_fairness_config_queue")
	fci := informerFactory.Flowcontrol().V1alpha1()
	pli := fci.PriorityLevelConfigurations()
	fsi := fci.FlowSchemas()
	cfgCtl.plLister = pli.Lister()
	cfgCtl.plInformerSynced = pli.Informer().HasSynced
	cfgCtl.fsLister = fsi.Lister()
	cfgCtl.fsInformerSynced = fsi.Informer().HasSynced
	pli.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pl := obj.(*fctypesv1a1.PriorityLevelConfiguration)
			klog.V(7).Infof("triggered API priority and fairness config reloading due to creation of PLC %s", pl.Name)
			cfgCtl.configQueue.Add(0)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			newPL := newObj.(*fctypesv1a1.PriorityLevelConfiguration)
			oldPL := oldObj.(*fctypesv1a1.PriorityLevelConfiguration)
			if !apiequality.Semantic.DeepEqual(oldPL.Spec, newPL.Spec) {
				klog.V(7).Infof("triggered API priority and fairness config reloading due to spec update of PLC %s", newPL.Name)
				cfgCtl.configQueue.Add(0)
			}
		},
		DeleteFunc: func(obj interface{}) {
			name, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			klog.V(7).Infof("triggered API priority and fairness config reloading due to deletion of PLC %s", name)
			cfgCtl.configQueue.Add(0)

		}})
	fsi.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			fs := obj.(*fctypesv1a1.FlowSchema)
			klog.V(7).Infof("triggered API priority and fairness config reloading due to creation of FS %s", fs.Name)
			cfgCtl.configQueue.Add(0)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			newFS := newObj.(*fctypesv1a1.FlowSchema)
			oldFS := oldObj.(*fctypesv1a1.FlowSchema)
			if !apiequality.Semantic.DeepEqual(oldFS.Spec, newFS.Spec) {
				klog.V(7).Infof("triggered API priority and fairness config reloading due to spec update of FS %s", newFS.Name)
				cfgCtl.configQueue.Add(0)
			}
		},
		DeleteFunc: func(obj interface{}) {
			name, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			klog.V(7).Infof("triggered API priority and fairness config reloading due to deletion of FS %s", name)
			cfgCtl.configQueue.Add(0)

		}})
}

func (cfgCtl *configController) Run(stopCh <-chan struct{}) error {
	defer cfgCtl.configQueue.ShutDown()
	klog.Info("Starting API Priority and Fairness config controller")
	if ok := cache.WaitForCacheSync(stopCh, cfgCtl.plInformerSynced, cfgCtl.fsInformerSynced); !ok {
		return fmt.Errorf("Never achieved initial sync")
	}
	klog.Info("Running API Priority and Fairness config worker")
	wait.Until(cfgCtl.runWorker, time.Second, stopCh)
	klog.Info("Shutting down API Priority and Fairness config worker")
	return nil
}

func (cfgCtl *configController) runWorker() {
	for cfgCtl.processNextWorkItem() {
	}
}

func (cfgCtl *configController) processNextWorkItem() bool {
	obj, shutdown := cfgCtl.configQueue.Get()
	if shutdown {
		return false
	}

	func(obj interface{}) {
		defer cfgCtl.configQueue.Done(obj)
		if !cfgCtl.syncOne() {
			cfgCtl.configQueue.AddRateLimited(obj)
		} else {
			cfgCtl.configQueue.Forget(obj)
		}
	}(obj)

	return true
}

// syncOne attempts to sync all the API Priority and Fairness config
// objects.  It either succeeds and returns `true` or logs an error
// and returns `false`.
func (cfgCtl *configController) syncOne() bool {
	all := labels.Everything()
	newPLs, err := cfgCtl.plLister.List(all)
	if err != nil {
		klog.Errorf("Unable to list PriorityLevelConfiguration objects: %s", err.Error())
		return false
	}
	newFSs, err := cfgCtl.fsLister.List(all)
	if err != nil {
		klog.Errorf("Unable to list FlowSchema objects: %s", err.Error())
		return false
	}
	cfgCtl.digestConfigObjects(newPLs, newFSs)
	return true
}

// cfgMeal is the data involved in the process of digesting the API
// objects that configure API Priority and Fairness.  All the config
// objects are digested together, because this is the simplest way to
// cope with the various dependencies between objects.  The process of
// digestion is done in four passes over config objects --- three
// passes over PriorityLevelConfigurations and one pass over the
// FlowSchemas --- with the work dvided among the passes according to
// those dependencies.
type cfgMeal struct {
	cfgCtl *configController

	// The prior result
	oldCfgState *configState

	// The new result being computed
	newCfgState *configState

	// The sum of the concurrency shares of the priority levels in the
	// new configuration
	shareSum float64

	// This holds the priority levels that _start_ quiescing in this
	// meal.  We delay those starts until after the new state is
	// stored in the controller, so that any reload after those starts
	// sees the stored state or a later one.
	newlyQuiescent []*priorityLevelState

	// These keep track of which mandatory objects have been digested
	haveExemptPL, haveCatchAllPL, haveExemptFS, haveCatchAllFS bool
}

// digestConfigObjects is given all the API objects that configure
// cfgCtl and writes its consequent new configState.
func (cfgCtl *configController) digestConfigObjects(newPLs []*fctypesv1a1.PriorityLevelConfiguration, newFSs []*fctypesv1a1.FlowSchema) {
	meal := cfgMeal{
		cfgCtl:      cfgCtl,
		oldCfgState: cfgCtl.curState.Load().(*configState),
		newCfgState: &configState{
			priorityLevelStates: make(map[string]*priorityLevelState),
		},
	}

	meal.digestNewPLs(newPLs)
	meal.digestFlowSchemas(newFSs)
	meal.processOldPLs()

	// Supply missing mandatory PriorityLevelConfiguration objects
	if !meal.haveExemptPL {
		meal.newCfgState.imaginePL(cfgCtl.queueSetFactory, fcboot.MandatoryPriorityLevelConfigurationExempt, cfgCtl.requestWaitLimit, &meal.shareSum)
	}
	if !meal.haveCatchAllPL {
		meal.newCfgState.imaginePL(cfgCtl.queueSetFactory, fcboot.MandatoryPriorityLevelConfigurationCatchAll, cfgCtl.requestWaitLimit, &meal.shareSum)
	}

	meal.finishQueueSetReconfigs()

	// The new config has been constructed, pass to filter for use.
	cfgCtl.curState.Store(meal.newCfgState)
	klog.V(5).Infof("Switched to new API Priority and Farness configuration")

	// We delay the calls to QueueSet::Quiesce so that the following
	// proof works.
	//
	// 1. For any QueueSet S: if a call S.Wait() returns with
	//    tryAnother==true then a call to S.Quiesce with a non-nil
	//    handler happened before that return.  This is from the
	//    contract of QueueSet.
	//
	// 2. Every S.Quiesce call with a non-nil handler happens after a
	//    call to `cfgCtl.curState.Store(X)` for which S is in
	//    newlyQuiescent and S's priority level is not referenced from
	//    any FlowSchema in X.  This is established by the text of
	//    this function and the immutability of each FlowSchemaSpec
	//    and of each priorityLevelState in X after X is stored.  The
	//    FlowSchema objects are from an informer so their
	//    immutability is normal.  That informer and the
	//    priorityLevelState objects are not exported from this
	//    controller, so checking the promises is a local matter.
	//
	// 3. If a call to S.Wait that returns with tryAnother==true
	//    happens before a call to `curState.Load()` that returns a
	//    value Y then either (3a) Y sends no traffic to S or (3b) Y
	//    is a value stored after X.  This implements the sequencing
	//    promise of State::Match.  Chaining together the "happens
	//    before" relationships of (2), (1), and the if condition in
	//    (3) implies that `curState.Store(X)` happens before
	//    `curState.Load()` returns Y.  The fact that this function
	//    stores a fresh pointer in curState each time, together with
	//    the contract of `atomic.Value`, implies that either Y is X
	//    (which, in turn, implies 3a) or Y is a value stored later
	//    (which is 3b).
	for _, plState := range meal.newlyQuiescent {
		plState.queues.Quiesce(plState.emptyHandler)
	}
}

// Digest the new set of PriorityLevelConfiguration objects.
// Pretend broken ones do not exist.
func (meal *cfgMeal) digestNewPLs(newPLs []*fctypesv1a1.PriorityLevelConfiguration) {
	for _, pl := range newPLs {
		state := meal.oldCfgState.priorityLevelStates[pl.Name]
		var err error
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
		state.qsCompleter, err = qscOfPL(meal.cfgCtl.queueSetFactory, state.queues, pl, meal.cfgCtl.requestWaitLimit)
		if err != nil {
			klog.Warningf(err.Error())
			continue
		}
		if state.config.Limited != nil {
			meal.shareSum += float64(state.config.Limited.AssuredConcurrencyShares)
		}
		meal.haveExemptPL = meal.haveExemptPL || pl.Name == fctypesv1a1.PriorityLevelConfigurationNameExempt
		meal.haveCatchAllPL = meal.haveCatchAllPL || pl.Name == fctypesv1a1.PriorityLevelConfigurationNameCatchAll
		meal.newCfgState.priorityLevelStates[pl.Name] = state
	}
}

// Digest the given FlowSchema objects.  Ones that reference a missing
// or broken priority level are not to be passed on to the filter for
// use.  We do this before holding over old priority levels so that
// requests stop going to those levels and FlowSchemaStatus values
// reflect this.  This function also adds any missing mandatory
// FlowSchema objects.
func (meal *cfgMeal) digestFlowSchemas(newFSs []*fctypesv1a1.FlowSchema) {
	fsSeq := make(apihelpers.FlowSchemaSequence, 0, len(newFSs))
	for i, fs := range newFSs {
		_, goodPriorityRef := meal.newCfgState.priorityLevelStates[fs.Spec.PriorityLevelConfiguration.Name]

		// Ensure the object's status reflects whether its priority
		// level reference is broken.
		//
		// TODO: consider
		// k8s.io/apimachinery/pkg/util/errors.NewAggregate
		// errors from all of these and return it at the end.
		//
		// TODO: consider not even trying if server is not handling
		// requests yet.
		meal.cfgCtl.syncFlowSchemaStatus(fs, !goodPriorityRef)

		if !goodPriorityRef {
			continue
		}
		fsSeq = append(fsSeq, newFSs[i])
		meal.haveExemptFS = meal.haveExemptFS || fs.Name == fctypesv1a1.FlowSchemaNameExempt
		meal.haveCatchAllFS = meal.haveCatchAllFS || fs.Name == fctypesv1a1.FlowSchemaNameCatchAll
	}
	// sort into the order to be used for matching
	sort.Sort(fsSeq)

	// Supply missing mandatory FlowSchemas, in correct position
	if !meal.haveExemptFS {
		fsSeq = append(apihelpers.FlowSchemaSequence{fcboot.MandatoryFlowSchemaExempt}, fsSeq...)
	}
	if !meal.haveCatchAllFS {
		fsSeq = append(fsSeq, fcboot.MandatoryFlowSchemaCatchAll)
	}

	meal.newCfgState.flowSchemas = fsSeq
	if klog.V(5) {
		for _, fs := range fsSeq {
			klog.Infof("Using FlowSchema %s: %#+v", fs.Name, fs.Spec)
		}
	}
}

// Consider all the priority levels in the previous configuration.
// Keep the ones that are in the new config, supply mandatory
// behavior, or still have non-empty queues; for the rest: drop it if
// it has no queues, otherwise start the quiescing process if that has
// not already been started.
func (meal *cfgMeal) processOldPLs() {
	for plName, plState := range meal.oldCfgState.priorityLevelStates {
		if meal.newCfgState.priorityLevelStates[plName] != nil {
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
		if plName == fctypesv1a1.PriorityLevelConfigurationNameExempt && !meal.haveExemptPL || plName == fctypesv1a1.PriorityLevelConfigurationNameCatchAll && !meal.haveCatchAllPL {
			klog.V(3).Infof("Retaining old priority level %q with Type=%v because of lack of replacement", plName, plState.config.Type)
		} else {
			if plState.queues == nil {
				klog.V(3).Infof("Removing undesired priority level %q, Type=%v", plName, plState.config.Type)
				continue
			}
			if plState.emptyHandler == nil {
				klog.V(3).Infof("Priority level %q became undesired", plName)
				plState.emptyHandler = &emptyRelay{cfgCtl: meal.cfgCtl}
				meal.newlyQuiescent = append(meal.newlyQuiescent, plState)
			}
		}
		if plState.config.Limited != nil {
			meal.shareSum += float64(plState.config.Limited.AssuredConcurrencyShares)
		}
		meal.haveExemptPL = meal.haveExemptPL || plName == fctypesv1a1.PriorityLevelConfigurationNameExempt
		meal.haveCatchAllPL = meal.haveCatchAllPL || plName == fctypesv1a1.PriorityLevelConfigurationNameCatchAll
		meal.newCfgState.priorityLevelStates[plName] = plState
	}
}

// For all the priority levels of the new config, divide up the
// server's total concurrency limit among them and create/update their
// QueueSets.
func (meal *cfgMeal) finishQueueSetReconfigs() {
	for plName, plState := range meal.newCfgState.priorityLevelStates {
		if plState.config.Limited == nil {
			klog.V(5).Infof("Using exempt priority level %q: quiescent=%v", plName, plState.emptyHandler != nil)
			continue
		}

		concurrencyLimit := int(math.Ceil(float64(meal.cfgCtl.serverConcurrencyLimit) * float64(plState.config.Limited.AssuredConcurrencyShares) / meal.shareSum))
		metrics.UpdateSharedConcurrencyLimit(plName, concurrencyLimit)

		if plState.queues == nil {
			klog.V(5).Infof("Introducing queues for priority level %q: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, concurrencyLimit, plState.emptyHandler != nil, plState.config.Limited.AssuredConcurrencyShares, meal.shareSum)
		} else {
			klog.V(5).Infof("Retaining queues for priority level %q: config=%#+v, concurrencyLimit=%d, quiescent=%v (shares=%v, shareSum=%v)", plName, plState.config, concurrencyLimit, plState.emptyHandler != nil, plState.config.Limited.AssuredConcurrencyShares, meal.shareSum)
		}
		plState.queues = plState.qsCompleter.GetQueueSet(fq.DispatchingConfig{ConcurrencyLimit: concurrencyLimit})
	}
}

// qscOfPL returns a pointer to an appropriate QueuingConfig or nil
// if no limiting is called for.  Returns nil and an error if the given
// object is malformed in a way that is a problem for this package.
func qscOfPL(qsf fq.QueueSetFactory, queues fq.QueueSet, pl *fctypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration) (fq.QueueSetCompleter, error) {
	if (pl.Spec.Type == fctypesv1a1.PriorityLevelEnablementExempt) != (pl.Spec.Limited == nil) {
		return nil, errors.New("broken union structure at the top")
	}
	if pl.Spec.Limited == nil {
		return nil, nil
	}
	if (pl.Spec.Limited.LimitResponse.Type == fctypesv1a1.LimitResponseTypeReject) != (pl.Spec.Limited.LimitResponse.Queuing == nil) {
		return nil, errors.New("broken union structure for limit response")
	}
	qcAPI := pl.Spec.Limited.LimitResponse.Queuing
	qcQS := fq.QueuingConfig{Name: pl.Name}
	if qcAPI != nil {
		qcQS = fq.QueuingConfig{Name: pl.Name,
			DesiredNumQueues: int(qcAPI.Queues),
			QueueLengthLimit: int(qcAPI.QueueLengthLimit),
			HandSize:         int(qcAPI.HandSize),
			RequestWaitLimit: requestWaitLimit,
		}
	}
	var qsc fq.QueueSetCompleter
	var err error
	if queues != nil {
		qsc, err = queues.QualifyQueuingConfig(qcQS)
	} else {
		qsc, err = qsf.QualifyQueuingConfig(qcQS)
	}
	if err != nil {
		err = errors.Wrap(err, fmt.Sprintf("priority level %q has QueuingConfiguration %#+v, which is invalid", pl.Name, *qcAPI))
	}
	return qsc, err
}

func (cfgCtl *configController) syncFlowSchemaStatus(fs *fctypesv1a1.FlowSchema, isDangling bool) {
	danglingCondition := apihelpers.GetFlowSchemaConditionByType(fs, fctypesv1a1.FlowSchemaConditionDangling)
	if danglingCondition == nil {
		danglingCondition = &fctypesv1a1.FlowSchemaCondition{
			Type: fctypesv1a1.FlowSchemaConditionDangling,
		}
	}

	switch {
	case isDangling && danglingCondition.Status != fctypesv1a1.ConditionTrue:
		danglingCondition.Status = fctypesv1a1.ConditionTrue
		danglingCondition.LastTransitionTime = metav1.Now()
	case !isDangling && danglingCondition.Status != fctypesv1a1.ConditionFalse:
		danglingCondition.Status = fctypesv1a1.ConditionFalse
		danglingCondition.LastTransitionTime = metav1.Now()
	default:
		// the dangling status is already in sync, skip updating
		return
	}

	apihelpers.SetFlowSchemaCondition(fs, *danglingCondition)

	_, err := cfgCtl.flowcontrolClient.FlowSchemas().UpdateStatus(fs)
	if err != nil {
		klog.Warningf("failed updating condition for flow-schema %s", fs.Name)
	}
}

// imaginePL adds a priority level based on one of the mandatory ones
func (cfgState *configState) imaginePL(qsf fq.QueueSetFactory, proto *fctypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration, shareSum *float64) {
	klog.Warningf("No %s PriorityLevelConfiguration found, imagining one", proto.Name)
	qsCompleter, err := qscOfPL(qsf, nil, proto, requestWaitLimit)
	if err != nil {
		// This can not happen because proto is one of the mandatory
		// objects and these are not erroneous
		panic(err)
	}
	cfgState.priorityLevelStates[proto.Name] = &priorityLevelState{
		config:      proto.Spec,
		qsCompleter: qsCompleter,
	}
	if proto.Spec.Limited != nil {
		*shareSum += float64(proto.Spec.Limited.AssuredConcurrencyShares)
	}
	return
}

func (cfgState *configState) Match(rd RequestDigest) (string, *fctypesv1a1.FlowDistinguisherMethod, string, fctypesv1a1.PriorityLevelEnablement, fq.QueueSet) {
	var fs *fctypesv1a1.FlowSchema
	for _, fs = range cfgState.flowSchemas {
		if matchesFlowSchema(rd, fs) {
			plName := fs.Spec.PriorityLevelConfiguration.Name
			plState := cfgState.priorityLevelStates[plName]
			return fs.Name, fs.Spec.DistinguisherMethod, plName, plState.config.Type, plState.queues
		}
	}
	// This can never happen because every configState has a
	// FlowSchema that matches everything
	panic(rd)
}

type emptyRelay struct {
	sync.RWMutex
	cfgCtl *configController
	empty  bool
}

var _ fq.EmptyHandler = &emptyRelay{}

func (er *emptyRelay) HandleEmpty() {
	er.Lock()
	defer er.Unlock()
	er.empty = true
	// TODO: to support testing of the config controller, extend
	// goroutine tracking to the config queue and worker
	er.cfgCtl.configQueue.Add(0)
}

func (er *emptyRelay) IsEmpty() bool {
	er.RLock()
	defer er.RUnlock()
	return er.empty
}
