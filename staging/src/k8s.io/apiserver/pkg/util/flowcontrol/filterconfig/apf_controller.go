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
	"context"
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
	apierrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/util/apihelpers"
	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fqs "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset"
	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
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
// change to any config object, or when any priority level that is
// undesired becomes completely unused, all the config objects are
// read and processed as a whole.

// To support synchronization in testing, the controller keeps track
// of the latest priority level referenced by a FlowSchema of this
// name.
const tracerName = "tracer.kube-system"

// Controller maintains eventual consistency with the API objects that
// configure API Priority and Fairness, and provides a procedural
// interface to the configured behavior.
type Controller interface {
	// Run runs the controller, returning after it is stopped
	Run(stopCh <-chan struct{}) error

	// Match classifies the given request to the proper FlowSchema and
	// returns the relevant features of that schema and its associated
	// priority level.  If `startFn==nil` then the caller should
	// immediately execute the request.  Otherwise the caller must
	// call startFn exactly once and do as it says.
	Match(RequestDigest) (flowSchemaName string, distinguisherMethod *fctypesv1a1.FlowDistinguisherMethod, priorityLevelName string, startFn StartFunction)
}

// StartFunction begins the process of handlig a request.  If the
// request gets queued then this function uses the given hashValue as
// the source of entropy as it shuffle-shards the request into a
// queue.  The descr1 and descr2 values play no role in the logic but
// appear in log messages.  This method does not return until the
// queuing, if any, for this request is done.  If `execute` is false
// then `afterExecution` is irrelevant and the request should be
// rejected.  Otherwise the request should be executed and
// `afterExecution` must be called exactly once.
type StartFunction func(ctx context.Context, hashValue uint64) (execute bool, afterExecution func())

// RequestDigest holds necessary info from request for flow-control
type RequestDigest struct {
	RequestInfo *request.RequestInfo
	User        user.Info
}

// `*configController` implements Controller.  The methods of this
// type and cfgMeal follow the convention that the suffix "Locked"
// means that the caller must hold the configController lock.
type configController struct {
	queueSetFactory fq.QueueSetFactory

	// configQueue holds `(interface{})(0)` when the configuration
	// objects need to be reprocessed.
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

	// This must be locked while accessing flowSchemas or
	// priorityLevelStates.  It is the lock involved in
	// LockingWriteMultiple.
	lock sync.Mutex

	// tracedValue holds the latest MatchingPrecedence from a
	// FlowSchema with the tracer name.
	tracedValue int32

	// this condition is broadcast whenever tracedValue changes
	traceCond sync.Cond

	// flowSchemas holds the flow schema objects, sorted by increasing
	// numerical (decreasing logical) matching precedence.  Every
	// FlowSchema in this slice is immutable.
	flowSchemas apihelpers.FlowSchemaSequence

	// priorityLevelStates maps the PriorityLevelConfiguration object
	// name to the state for that level.  Every name referenced from a
	// member of `flowSchemas` has an entry here.
	priorityLevelStates map[string]*priorityLevelState
}

// priorityLevelState holds the state specific to a priority level.
type priorityLevelState struct {
	// config holds the configuration after defaulting logic has been applied.
	// If there are queues then their parameters are here.
	config fctypesv1a1.PriorityLevelConfigurationSpec

	// qsCompleter holds the QueueSetCompleter derived from `config`
	// and `queues` if config is not exempt, nil otherwise.
	qsCompleter fq.QueueSetCompleter

	// The QueueSet for this priority level.  This is nil if and only
	// if the priority level is exempt.
	queues fq.QueueSet

	// quiescing==true indicates that this priority level should be
	// removed when its queues have all drained.  May be true only if
	// queues is non-nil.
	quiescing bool

	// number of goroutines between Controller::Match and calling the
	// returned StartFunction
	numPending int
}

var _ Controller = (*configController)(nil)

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
		priorityLevelStates:    make(map[string]*priorityLevelState),
	}
	cfgCtl.traceCond = *sync.NewCond(&cfgCtl.lock)
	klog.V(2).Infof("NewTestableController with serverConcurrencyLimit=%d, requestWaitLimit=%s", serverConcurrencyLimit, requestWaitLimit)
	cfgCtl.initializeConfigController(informerFactory)
	// ensure the data structure reflects the mandatory config
	cfgCtl.lockAndDigestConfigObjects(nil, nil)
	return cfgCtl
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

func (cfgCtl *configController) hasWork() bool {
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	return cfgCtl.configQueue.Len() != 0
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
	err = cfgCtl.digestConfigObjects(newPLs, newFSs)
	if err == nil {
		return true
	}
	klog.Error(err)
	return false
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

	newPLStates map[string]*priorityLevelState

	// The sum of the concurrency shares of the priority levels in the
	// new configuration
	shareSum float64

	// These keep track of which mandatory objects have been digested
	haveExemptPL, haveCatchAllPL, haveExemptFS, haveCatchAllFS bool

	// Buffered FlowSchema status updates to do.  Do them when the
	// lock is not held, to avoid a deadlock due to such a request
	// provoking a call into this controller while the lock held
	// waiting on that request to complete.
	fsStatusUpdates []fsStatusUpdate
}

// A buffered set of status updates for a FlowSchema
type fsStatusUpdate struct {
	flowSchema *fctypesv1a1.FlowSchema
	condition  fctypesv1a1.FlowSchemaCondition
	oldStatus  fctypesv1a1.ConditionStatus
}

// digestConfigObjects is given all the API objects that configure
// cfgCtl and writes its consequent new configState.
func (cfgCtl *configController) digestConfigObjects(newPLs []*fctypesv1a1.PriorityLevelConfiguration, newFSs []*fctypesv1a1.FlowSchema) error {
	fsStatusUpdates := cfgCtl.lockAndDigestConfigObjects(newPLs, newFSs)
	var errs []error
	for _, fsu := range fsStatusUpdates {
		fs2 := fsu.flowSchema.DeepCopy()
		klog.V(4).Infof("Writing %#+v to FlowSchema %s because its previous Status was %q", fsu.condition, fs2.Name, fsu.oldStatus)
		apihelpers.SetFlowSchemaCondition(fs2, fsu.condition)
		_, err := cfgCtl.flowcontrolClient.FlowSchemas().UpdateStatus(fs2)
		if err != nil {
			errs = append(errs, errors.Wrap(err, fmt.Sprintf("failed to set a status.condition for FlowSchema %s", fs2.Name)))
		}
	}
	if len(errs) == 0 {
		return nil
	}
	return apierrors.NewAggregate(errs)
}

func (cfgCtl *configController) lockAndDigestConfigObjects(newPLs []*fctypesv1a1.PriorityLevelConfiguration, newFSs []*fctypesv1a1.FlowSchema) []fsStatusUpdate {
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	meal := cfgMeal{
		cfgCtl:      cfgCtl,
		newPLStates: make(map[string]*priorityLevelState),
	}

	meal.digestNewPLsLocked(newPLs)
	meal.digestFlowSchemasLocked(newFSs)
	meal.processOldPLsLocked()

	// Supply missing mandatory PriorityLevelConfiguration objects
	if !meal.haveExemptPL {
		meal.imaginePL(fcboot.MandatoryPriorityLevelConfigurationExempt, cfgCtl.requestWaitLimit)
	}
	if !meal.haveCatchAllPL {
		meal.imaginePL(fcboot.MandatoryPriorityLevelConfigurationCatchAll, cfgCtl.requestWaitLimit)
	}

	meal.finishQueueSetReconfigsLocked()

	// The new config has been constructed
	cfgCtl.priorityLevelStates = meal.newPLStates
	klog.V(5).Infof("Switched to new API Priority and Fairness configuration")
	return meal.fsStatusUpdates
}

// Digest the new set of PriorityLevelConfiguration objects.
// Pretend broken ones do not exist.
func (meal *cfgMeal) digestNewPLsLocked(newPLs []*fctypesv1a1.PriorityLevelConfiguration) {
	for _, pl := range newPLs {
		state := meal.cfgCtl.priorityLevelStates[pl.Name]
		if state == nil {
			state = &priorityLevelState{}
		}
		qsCompleter, err := qscOfPL(meal.cfgCtl.queueSetFactory, state.queues, pl.Name, &pl.Spec, meal.cfgCtl.requestWaitLimit)
		if err != nil {
			klog.Warningf("Ignoring PriorityLevelConfiguration object %s because its spec (%#+v) is broken: %s", pl.Name, fcfmt.Fmt(pl.Spec), err.Error())
			continue
		}
		meal.newPLStates[pl.Name] = state
		state.config = pl.Spec
		state.qsCompleter = qsCompleter
		if state.quiescing { // it was undesired, but no longer
			klog.V(3).Infof("Priority level %q was undesired and has become desired again", pl.Name)
			state.quiescing = false
		}
		if state.config.Limited != nil {
			meal.shareSum += float64(state.config.Limited.AssuredConcurrencyShares)
		}
		meal.haveExemptPL = meal.haveExemptPL || pl.Name == fctypesv1a1.PriorityLevelConfigurationNameExempt
		meal.haveCatchAllPL = meal.haveCatchAllPL || pl.Name == fctypesv1a1.PriorityLevelConfigurationNameCatchAll
	}
}

// Digest the given FlowSchema objects.  Ones that reference a missing
// or broken priority level are not to be passed on to the filter for
// use.  We do this before holding over old priority levels so that
// requests stop going to those levels and FlowSchemaStatus values
// reflect this.  This function also adds any missing mandatory
// FlowSchema objects.  The given objects must all have distinct
// names.
func (meal *cfgMeal) digestFlowSchemasLocked(newFSs []*fctypesv1a1.FlowSchema) {
	fsSeq := make(apihelpers.FlowSchemaSequence, 0, len(newFSs))
	fsMap := make(map[string]*fctypesv1a1.FlowSchema, len(newFSs))
	for i, fs := range newFSs {
		otherFS := fsMap[fs.Name]
		if otherFS != nil {
			// This client is forbidden to do this.
			panic(fmt.Sprintf("Given two FlowSchema objects with the same name: %#+v and %#+v", fcfmt.Fmt(otherFS), fcfmt.Fmt(fs)))
		}
		fsMap[fs.Name] = fs
		if fs.Name == tracerName {
			meal.cfgCtl.setTracedValueLocked(fs.Spec.MatchingPrecedence)
		}
		_, goodPriorityRef := meal.newPLStates[fs.Spec.PriorityLevelConfiguration.Name]

		// Ensure the object's status reflects whether its priority
		// level reference is broken.
		//
		// TODO: consider not even trying if server is not handling
		// requests yet.
		meal.presyncFlowSchemaStatus(fs, !goodPriorityRef)

		if !goodPriorityRef {
			klog.V(6).Infof("Ignoring FlowSchema %s because of bad priority level reference %q", fs.Name, fs.Spec.PriorityLevelConfiguration.Name)
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

	meal.cfgCtl.flowSchemas = fsSeq
	if klog.V(5) {
		for _, fs := range fsSeq {
			klog.Infof("Using FlowSchema %#+v", fcfmt.Fmt(fs))
		}
	}
}

// Consider all the priority levels in the previous configuration.
// Keep the ones that are in the new config, supply mandatory
// behavior, or are still busy; for the rest: drop it if it has no
// queues, otherwise start the quiescing process if that has not
// already been started.
func (meal *cfgMeal) processOldPLsLocked() {
	for plName, plState := range meal.cfgCtl.priorityLevelStates {
		if meal.newPLStates[plName] != nil {
			// Still desired and already updated
			continue
		}
		if plName == fctypesv1a1.PriorityLevelConfigurationNameExempt && !meal.haveExemptPL || plName == fctypesv1a1.PriorityLevelConfigurationNameCatchAll && !meal.haveCatchAllPL {
			klog.V(3).Infof("Retaining mandatory priority level %q despite lack of API object", plName)
		} else {
			if plState.queues == nil || plState.numPending == 0 && plState.queues.IsIdle() {
				// Either there are no queues or they are done
				// draining and no use is coming from another
				// goroutine
				klog.V(3).Infof("Removing undesired priority level %q, Type=%v", plName, plState.config.Type)
				continue
			}
			if !plState.quiescing {
				klog.V(3).Infof("Priority level %q became undesired", plName)
				plState.quiescing = true
			}
		}
		var err error
		plState.qsCompleter, err = qscOfPL(meal.cfgCtl.queueSetFactory, plState.queues, plName, &plState.config, meal.cfgCtl.requestWaitLimit)
		if err != nil {
			// This can not happen because qscOfPL already approved this config
			panic(fmt.Sprintf("%s from name=%q spec=%#+v", err.Error(), plName, fcfmt.Fmt(plState.config)))
		}
		if plState.config.Limited != nil {
			meal.shareSum += float64(plState.config.Limited.AssuredConcurrencyShares)
		}
		meal.haveExemptPL = meal.haveExemptPL || plName == fctypesv1a1.PriorityLevelConfigurationNameExempt
		meal.haveCatchAllPL = meal.haveCatchAllPL || plName == fctypesv1a1.PriorityLevelConfigurationNameCatchAll
		meal.newPLStates[plName] = plState
	}
}

// For all the priority levels of the new config, divide up the
// server's total concurrency limit among them and create/update their
// QueueSets.
func (meal *cfgMeal) finishQueueSetReconfigsLocked() {
	for plName, plState := range meal.newPLStates {
		if plState.config.Limited == nil {
			klog.V(5).Infof("Using exempt priority level %q: quiescing=%v", plName, plState.quiescing)
			continue
		}

		// The use of math.Ceil here means that the results might sum
		// to a little more than serverConcurrencyLimit but the
		// difference will be negligible.
		concurrencyLimit := int(math.Ceil(float64(meal.cfgCtl.serverConcurrencyLimit) * float64(plState.config.Limited.AssuredConcurrencyShares) / meal.shareSum))
		metrics.UpdateSharedConcurrencyLimit(plName, concurrencyLimit)

		if plState.queues == nil {
			klog.V(5).Infof("Introducing queues for priority level %q: config=%#+v, concurrencyLimit=%d, quiescing=%v (shares=%v, shareSum=%v)", plName, fcfmt.Fmt(plState.config), concurrencyLimit, plState.quiescing, plState.config.Limited.AssuredConcurrencyShares, meal.shareSum)
		} else {
			klog.V(5).Infof("Retaining queues for priority level %q: config=%#+v, concurrencyLimit=%d, quiescing=%v, numPending=%d (shares=%v, shareSum=%v)", plName, fcfmt.Fmt(plState.config), concurrencyLimit, plState.quiescing, plState.numPending, plState.config.Limited.AssuredConcurrencyShares, meal.shareSum)
		}
		plState.queues = plState.qsCompleter.Complete(fq.DispatchingConfig{ConcurrencyLimit: concurrencyLimit})
	}
}

// qscOfPL returns a pointer to an appropriate QueuingConfig or nil
// if no limiting is called for.  Returns nil and an error if the given
// object is malformed in a way that is a problem for this package.
func qscOfPL(qsf fq.QueueSetFactory, queues fq.QueueSet, plName string, plSpec *fctypesv1a1.PriorityLevelConfigurationSpec, requestWaitLimit time.Duration) (fq.QueueSetCompleter, error) {
	if (plSpec.Type == fctypesv1a1.PriorityLevelEnablementExempt) != (plSpec.Limited == nil) {
		return nil, errors.New("broken union structure at the top")
	}
	if (plSpec.Type == fctypesv1a1.PriorityLevelEnablementExempt) != (plName == fctypesv1a1.PriorityLevelConfigurationNameExempt) {
		// This package does not attempt to cope with a priority level dynamically switching between exempt and not.
		return nil, errors.New("non-alignment between name and type")
	}
	if plSpec.Limited == nil {
		return nil, nil
	}
	if (plSpec.Limited.LimitResponse.Type == fctypesv1a1.LimitResponseTypeReject) != (plSpec.Limited.LimitResponse.Queuing == nil) {
		return nil, errors.New("broken union structure for limit response")
	}
	qcAPI := plSpec.Limited.LimitResponse.Queuing
	qcQS := fq.QueuingConfig{Name: plName}
	if qcAPI != nil {
		qcQS = fq.QueuingConfig{Name: plName,
			DesiredNumQueues: int(qcAPI.Queues),
			QueueLengthLimit: int(qcAPI.QueueLengthLimit),
			HandSize:         int(qcAPI.HandSize),
			RequestWaitLimit: requestWaitLimit,
		}
	}
	var qsc fq.QueueSetCompleter
	var err error
	if queues != nil {
		qsc, err = queues.BeginConfigChange(qcQS)
	} else {
		qsc, err = qsf.BeginConstruction(qcQS)
	}
	if err != nil {
		err = errors.Wrap(err, fmt.Sprintf("priority level %q has QueuingConfiguration %#+v, which is invalid", plName, *qcAPI))
	}
	return qsc, err
}

func (meal *cfgMeal) presyncFlowSchemaStatus(fs *fctypesv1a1.FlowSchema, isDangling bool) {
	danglingCondition := apihelpers.GetFlowSchemaConditionByType(fs, fctypesv1a1.FlowSchemaConditionDangling)
	if danglingCondition == nil {
		danglingCondition = &fctypesv1a1.FlowSchemaCondition{
			Type: fctypesv1a1.FlowSchemaConditionDangling,
		}
	}
	desiredStatus := fctypesv1a1.ConditionFalse
	if isDangling {
		desiredStatus = fctypesv1a1.ConditionTrue
	}
	if danglingCondition.Status == desiredStatus {
		return
	}
	meal.fsStatusUpdates = append(meal.fsStatusUpdates, fsStatusUpdate{
		flowSchema: fs,
		condition: fctypesv1a1.FlowSchemaCondition{
			Type:               fctypesv1a1.FlowSchemaConditionDangling,
			Status:             desiredStatus,
			LastTransitionTime: metav1.Now(),
		},
		oldStatus: danglingCondition.Status})
}

// imaginePL adds a priority level based on one of the mandatory ones
func (meal *cfgMeal) imaginePL(proto *fctypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration) {
	klog.Warningf("No %s PriorityLevelConfiguration found, imagining one", proto.Name)
	qsCompleter, err := qscOfPL(meal.cfgCtl.queueSetFactory, nil, proto.Name, &proto.Spec, requestWaitLimit)
	if err != nil {
		// This can not happen because proto is one of the mandatory
		// objects and these are not erroneous
		panic(err)
	}
	meal.newPLStates[proto.Name] = &priorityLevelState{
		config:      proto.Spec,
		qsCompleter: qsCompleter,
	}
	if proto.Spec.Limited != nil {
		meal.shareSum += float64(proto.Spec.Limited.AssuredConcurrencyShares)
	}
	return
}

func (cfgCtl *configController) Match(rd RequestDigest) (string, *fctypesv1a1.FlowDistinguisherMethod, string, StartFunction) {
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	var fs *fctypesv1a1.FlowSchema
	for _, fs = range cfgCtl.flowSchemas {
		if matchesFlowSchema(rd, fs) {
			plName := fs.Spec.PriorityLevelConfiguration.Name
			plState := cfgCtl.priorityLevelStates[plName]
			if plState.config.Type == fctypesv1a1.PriorityLevelEnablementExempt {
				klog.V(7).Infof("Match(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, immediate", rd, fs.Name, fs.Spec.DistinguisherMethod, plName)
				return fs.Name, fs.Spec.DistinguisherMethod, plName, nil
			}
			plState.numPending++
			matchID := &plName
			klog.V(7).Infof("Match(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, matchID=%p", rd, fs.Name, fs.Spec.DistinguisherMethod, plName, matchID)
			var startCalls int32
			startFn := func(ctx context.Context, hashValue uint64) (execute bool, afterExecution func()) {
				newStarts := atomic.AddInt32(&startCalls, 1)
				klog.V(7).Infof("For matchID=%p, startFn(ctx, %v) startCalls:=%d", matchID, hashValue, newStarts)
				if newStarts > 1 {
					panic(fmt.Sprintf("Match(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, matchID=%p startCalls:=%d", rd, fs.Name, fs.Spec.DistinguisherMethod, plName, matchID, newStarts))
				}
				req := func() fq.Request {
					cfgCtl.lock.Lock()
					defer cfgCtl.lock.Unlock()
					plState.numPending--
					req, idle1 := plState.queues.StartRequest(ctx, hashValue, rd.RequestInfo, rd.User)
					if idle1 {
						cfgCtl.maybeReapLocked(plName, plState)
					}
					return req
				}()
				if req == nil {
					klog.V(7).Infof("For matchID=%p, startFn(ctx, %v): reject", matchID, hashValue)
					return false, func() {}
				}
				exec, idle2, afterExec := req.Wait()
				execID := &exec
				if idle2 {
					cfgCtl.maybeReap(plName)
				}
				klog.V(7).Infof("For matchID=%p, startFn(..) => exec=%v, execID=%p", matchID, exec, execID)
				var afterCalls int32
				return exec, func() {
					newAfter := atomic.AddInt32(&afterCalls, 1)
					klog.V(7).Infof("For matchID=%p, execID=%p, after() calls:=%d", matchID, execID, newAfter)
					if newAfter > 1 {
						panic(fmt.Sprintf("Match(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, matchID=%p, execID=%p, afterCalls:=%d", rd, fs.Name, fs.Spec.DistinguisherMethod, plName, matchID, execID, newAfter))
					}
					idle3 := afterExec()
					if idle3 {
						cfgCtl.maybeReap(plName)
					}
				}
			}
			return fs.Name, fs.Spec.DistinguisherMethod, plName, startFn
		}
	}
	var lastFS *fctypesv1a1.FlowSchema
	if len(cfgCtl.flowSchemas) > 0 {
		lastFS = cfgCtl.flowSchemas[len(cfgCtl.flowSchemas)-1]
	}
	// This can never happen because every configState has a
	// FlowSchema that matches everything.
	panic(fmt.Sprintf("No match; rd=%#+v, lastFS=%#+v", rd, fcfmt.Fmt(lastFS)))
}

// Call this after getting a clue that the given priority level is undesired and idle
func (cfgCtl *configController) maybeReap(plName string) {
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	plState := cfgCtl.priorityLevelStates[plName]
	if plState != nil && (plState.queues == nil || plState.quiescing && plState.numPending == 0 && plState.queues.IsIdle()) {
		klog.V(3).Infof("Triggered API Priority and Fairness config reload because priority level %s became idle", plName)
		cfgCtl.configQueue.Add(0)
	}
}

// Call this if both (1) plState.queues is non-nil and reported being
// idle, and (2) cfgCtl's lock has not been released since then.
func (cfgCtl *configController) maybeReapLocked(plName string, plState *priorityLevelState) {
	if !(plState.quiescing && plState.numPending == 0) {
		return
	}
	klog.V(3).Infof("Triggered API Priority and Fairness config reload because priority level %s became idle", plName)
	cfgCtl.configQueue.Add(0)
}

func (cfgCtl *configController) setTracedValueLocked(val int32) {
	klog.V(4).Infof("Setting traced value to %v", val)
	cfgCtl.tracedValue = val
	cfgCtl.traceCond.Broadcast()
}

func (cfgCtl *configController) waitForTracedValue(val int32) {
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	for cfgCtl.tracedValue < val {
		klog.V(4).Infof("Waiting for traced reference to reach %v", val)
		cfgCtl.traceCond.Wait()
	}
	klog.V(4).Infof("Traced reference became %v", cfgCtl.tracedValue)
}
