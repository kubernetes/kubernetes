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
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/pkg/errors"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	apitypes "k8s.io/apimachinery/pkg/types"
	apierrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/util/apihelpers"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
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

// `*configController` maintains eventual consistency with the API
// objects that configure API Priority and Fairness, and provides a
// procedural interface to the configured behavior.  The methods of
// this type and cfgMeal follow the convention that the suffix
// "Locked" means that the caller must hold the configController lock.
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
	// the API object or prototype prescribing this level.  Nothing
	// reached through this pointer is mutable.
	pl *fctypesv1a1.PriorityLevelConfiguration

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

// NewTestableController is extra flexible to facilitate testing
func newTestableController(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient fcclientv1a1.FlowcontrolV1alpha1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	queueSetFactory fq.QueueSetFactory,
) *configController {
	cfgCtl := &configController{
		queueSetFactory:        queueSetFactory,
		serverConcurrencyLimit: serverConcurrencyLimit,
		requestWaitLimit:       requestWaitLimit,
		flowcontrolClient:      flowcontrolClient,
		priorityLevelStates:    make(map[string]*priorityLevelState),
	}
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
			klog.V(7).Infof("Triggered API priority and fairness config reloading due to creation of PLC %s", pl.Name)
			cfgCtl.configQueue.Add(0)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			newPL := newObj.(*fctypesv1a1.PriorityLevelConfiguration)
			oldPL := oldObj.(*fctypesv1a1.PriorityLevelConfiguration)
			if !apiequality.Semantic.DeepEqual(oldPL.Spec, newPL.Spec) {
				klog.V(7).Infof("Triggered API priority and fairness config reloading due to spec update of PLC %s", newPL.Name)
				cfgCtl.configQueue.Add(0)
			}
		},
		DeleteFunc: func(obj interface{}) {
			name, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			klog.V(7).Infof("Triggered API priority and fairness config reloading due to deletion of PLC %s", name)
			cfgCtl.configQueue.Add(0)

		}})
	fsi.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			fs := obj.(*fctypesv1a1.FlowSchema)
			klog.V(7).Infof("Triggered API priority and fairness config reloading due to creation of FS %s", fs.Name)
			cfgCtl.configQueue.Add(0)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			newFS := newObj.(*fctypesv1a1.FlowSchema)
			oldFS := oldObj.(*fctypesv1a1.FlowSchema)
			if !apiequality.Semantic.DeepEqual(oldFS.Spec, newFS.Spec) {
				klog.V(7).Infof("Triggered API priority and fairness config reloading due to spec update of FS %s", newFS.Name)
				cfgCtl.configQueue.Add(0)
			}
		},
		DeleteFunc: func(obj interface{}) {
			name, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			klog.V(7).Infof("Triggered API priority and fairness config reloading due to deletion of FS %s", name)
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

	// These keep track of which mandatory priority level config
	// objects have been digested
	haveExemptPL, haveCatchAllPL bool

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
	oldValue   fctypesv1a1.FlowSchemaCondition
}

// digestConfigObjects is given all the API objects that configure
// cfgCtl and writes its consequent new configState.
func (cfgCtl *configController) digestConfigObjects(newPLs []*fctypesv1a1.PriorityLevelConfiguration, newFSs []*fctypesv1a1.FlowSchema) error {
	fsStatusUpdates := cfgCtl.lockAndDigestConfigObjects(newPLs, newFSs)
	var errs []error
	for _, fsu := range fsStatusUpdates {
		enc, err := json.Marshal(fsu.condition)
		if err != nil {
			// should never happen because these conditions are created here and well formed
			panic(fmt.Sprintf("Failed to json.Marshall(%#+v): %s", fsu.condition, err.Error()))
		}
		klog.V(4).Infof("Writing Condition %s to FlowSchema %s because its previous value was %s", string(enc), fsu.flowSchema.Name, fcfmt.Fmt(fsu.oldValue))
		_, err = cfgCtl.flowcontrolClient.FlowSchemas().Patch(context.TODO(), fsu.flowSchema.Name, apitypes.StrategicMergePatchType, []byte(fmt.Sprintf(`{"status": {"conditions": [ %s ] } }`, string(enc))), metav1.PatchOptions{FieldManager: "api-priority-and-fairness-config-consumer-v1"}, "status")
		if err != nil {
			errs = append(errs, errors.Wrap(err, fmt.Sprintf("failed to set a status.condition for FlowSchema %s", fsu.flowSchema.Name)))
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
		qsCompleter, err := qscOfPL(meal.cfgCtl.queueSetFactory, state.queues, pl, meal.cfgCtl.requestWaitLimit)
		if err != nil {
			klog.Warningf("Ignoring PriorityLevelConfiguration object %s because its spec (%s) is broken: %s", pl.Name, fcfmt.Fmt(pl.Spec), err)
			continue
		}
		meal.newPLStates[pl.Name] = state
		state.pl = pl
		state.qsCompleter = qsCompleter
		if state.quiescing { // it was undesired, but no longer
			klog.V(3).Infof("Priority level %q was undesired and has become desired again", pl.Name)
			state.quiescing = false
		}
		if state.pl.Spec.Limited != nil {
			meal.shareSum += float64(state.pl.Spec.Limited.AssuredConcurrencyShares)
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
	var haveExemptFS, haveCatchAllFS bool
	for i, fs := range newFSs {
		otherFS := fsMap[fs.Name]
		if otherFS != nil {
			// This client is forbidden to do this.
			panic(fmt.Sprintf("Given two FlowSchema objects with the same name: %s and %s", fcfmt.Fmt(otherFS), fcfmt.Fmt(fs)))
		}
		fsMap[fs.Name] = fs
		_, goodPriorityRef := meal.newPLStates[fs.Spec.PriorityLevelConfiguration.Name]

		// Ensure the object's status reflects whether its priority
		// level reference is broken.
		//
		// TODO: consider not even trying if server is not handling
		// requests yet.
		meal.presyncFlowSchemaStatus(fs, !goodPriorityRef, fs.Spec.PriorityLevelConfiguration.Name)

		if !goodPriorityRef {
			klog.V(6).Infof("Ignoring FlowSchema %s because of bad priority level reference %q", fs.Name, fs.Spec.PriorityLevelConfiguration.Name)
			continue
		}
		fsSeq = append(fsSeq, newFSs[i])
		haveExemptFS = haveExemptFS || fs.Name == fctypesv1a1.FlowSchemaNameExempt
		haveCatchAllFS = haveCatchAllFS || fs.Name == fctypesv1a1.FlowSchemaNameCatchAll
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

	meal.cfgCtl.flowSchemas = fsSeq
	if klog.V(5) {
		for _, fs := range fsSeq {
			klog.Infof("Using FlowSchema %s", fcfmt.Fmt(fs))
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
			// BTW, we know the Spec has not changed because the
			// mandatory objects have immutable Specs
			klog.V(3).Infof("Retaining mandatory priority level %q despite lack of API object", plName)
		} else {
			if plState.queues == nil || plState.numPending == 0 && plState.queues.IsIdle() {
				// Either there are no queues or they are done
				// draining and no use is coming from another
				// goroutine
				klog.V(3).Infof("Removing undesired priority level %q (nilQueues=%v), Type=%v", plName, plState.queues == nil, plState.pl.Spec.Type)
				continue
			}
			if !plState.quiescing {
				klog.V(3).Infof("Priority level %q became undesired", plName)
				plState.quiescing = true
			}
		}
		var err error
		plState.qsCompleter, err = qscOfPL(meal.cfgCtl.queueSetFactory, plState.queues, plState.pl, meal.cfgCtl.requestWaitLimit)
		if err != nil {
			// This can not happen because qscOfPL already approved this config
			panic(fmt.Sprintf("%s from name=%q spec=%s", err, plName, fcfmt.Fmt(plState.pl.Spec)))
		}
		if plState.pl.Spec.Limited != nil {
			// We deliberately include the lingering priority levels
			// here so that their queues get some concurrency and they
			// continue to drain.  During this interim a lingering
			// priority level continues to get a concurrency
			// allocation determined by all the share values in the
			// regular way.
			meal.shareSum += float64(plState.pl.Spec.Limited.AssuredConcurrencyShares)
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
		if plState.pl.Spec.Limited == nil {
			klog.V(5).Infof("Using exempt priority level %q: quiescing=%v", plName, plState.quiescing)
			continue
		}

		// The use of math.Ceil here means that the results might sum
		// to a little more than serverConcurrencyLimit but the
		// difference will be negligible.
		concurrencyLimit := int(math.Ceil(float64(meal.cfgCtl.serverConcurrencyLimit) * float64(plState.pl.Spec.Limited.AssuredConcurrencyShares) / meal.shareSum))
		metrics.UpdateSharedConcurrencyLimit(plName, concurrencyLimit)

		if plState.queues == nil {
			klog.V(5).Infof("Introducing queues for priority level %q: config=%s, concurrencyLimit=%d, quiescing=%v (shares=%v, shareSum=%v)", plName, fcfmt.Fmt(plState.pl.Spec), concurrencyLimit, plState.quiescing, plState.pl.Spec.Limited.AssuredConcurrencyShares, meal.shareSum)
		} else {
			klog.V(5).Infof("Retaining queues for priority level %q: config=%s, concurrencyLimit=%d, quiescing=%v, numPending=%d (shares=%v, shareSum=%v)", plName, fcfmt.Fmt(plState.pl.Spec), concurrencyLimit, plState.quiescing, plState.numPending, plState.pl.Spec.Limited.AssuredConcurrencyShares, meal.shareSum)
		}
		plState.queues = plState.qsCompleter.Complete(fq.DispatchingConfig{ConcurrencyLimit: concurrencyLimit})
	}
}

// qscOfPL returns a pointer to an appropriate QueuingConfig or nil
// if no limiting is called for.  Returns nil and an error if the given
// object is malformed in a way that is a problem for this package.
func qscOfPL(qsf fq.QueueSetFactory, queues fq.QueueSet, pl *fctypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration) (fq.QueueSetCompleter, error) {
	if (pl.Spec.Type == fctypesv1a1.PriorityLevelEnablementExempt) != (pl.Spec.Limited == nil) {
		return nil, errors.New("broken union structure at the top")
	}
	if (pl.Spec.Type == fctypesv1a1.PriorityLevelEnablementExempt) != (pl.Name == fctypesv1a1.PriorityLevelConfigurationNameExempt) {
		// This package does not attempt to cope with a priority level dynamically switching between exempt and not.
		return nil, errors.New("non-alignment between name and type")
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
		qsc, err = queues.BeginConfigChange(qcQS)
	} else {
		qsc, err = qsf.BeginConstruction(qcQS)
	}
	if err != nil {
		err = errors.Wrap(err, fmt.Sprintf("priority level %q has QueuingConfiguration %#+v, which is invalid", pl.Name, qcAPI))
	}
	return qsc, err
}

func (meal *cfgMeal) presyncFlowSchemaStatus(fs *fctypesv1a1.FlowSchema, isDangling bool, plName string) {
	danglingCondition := apihelpers.GetFlowSchemaConditionByType(fs, fctypesv1a1.FlowSchemaConditionDangling)
	if danglingCondition == nil {
		danglingCondition = &fctypesv1a1.FlowSchemaCondition{
			Type: fctypesv1a1.FlowSchemaConditionDangling,
		}
	}
	desiredStatus := fctypesv1a1.ConditionFalse
	var desiredReason, desiredMessage string
	if isDangling {
		desiredStatus = fctypesv1a1.ConditionTrue
		desiredReason = "NotFound"
		desiredMessage = fmt.Sprintf("This FlowSchema references the PriorityLevelConfiguration object named %q but there is no such object", plName)
	} else {
		desiredReason = "Found"
		desiredMessage = fmt.Sprintf("This FlowSchema references the PriorityLevelConfiguration object named %q and it exists", plName)
	}
	if danglingCondition.Status == desiredStatus && danglingCondition.Reason == desiredReason && danglingCondition.Message == desiredMessage {
		return
	}
	meal.fsStatusUpdates = append(meal.fsStatusUpdates, fsStatusUpdate{
		flowSchema: fs,
		condition: fctypesv1a1.FlowSchemaCondition{
			Type:               fctypesv1a1.FlowSchemaConditionDangling,
			Status:             desiredStatus,
			LastTransitionTime: metav1.Now(),
			Reason:             desiredReason,
			Message:            desiredMessage,
		},
		oldValue: *danglingCondition})
}

// imaginePL adds a priority level based on one of the mandatory ones
func (meal *cfgMeal) imaginePL(proto *fctypesv1a1.PriorityLevelConfiguration, requestWaitLimit time.Duration) {
	klog.V(3).Infof("No %s PriorityLevelConfiguration found, imagining one", proto.Name)
	qsCompleter, err := qscOfPL(meal.cfgCtl.queueSetFactory, nil, proto, requestWaitLimit)
	if err != nil {
		// This can not happen because proto is one of the mandatory
		// objects and these are not erroneous
		panic(err)
	}
	meal.newPLStates[proto.Name] = &priorityLevelState{
		pl:          proto,
		qsCompleter: qsCompleter,
	}
	if proto.Spec.Limited != nil {
		meal.shareSum += float64(proto.Spec.Limited.AssuredConcurrencyShares)
	}
	return
}

type immediateRequest struct{}

func (immediateRequest) Finish(execute func()) bool {
	execute()
	return false
}

// startRequest classifies and, if appropriate, enqueues the request.
// Returns a nil Request if and only if the request is to be rejected.
// The returned bool indicates whether the request is exempt from
// limitation.  The startWaitingTime is when the request started
// waiting in its queue, or `Time{}` if this did not happen.
func (cfgCtl *configController) startRequest(ctx context.Context, rd RequestDigest) (fs *fctypesv1a1.FlowSchema, pl *fctypesv1a1.PriorityLevelConfiguration, isExempt bool, req fq.Request, startWaitingTime time.Time) {
	klog.V(7).Infof("startRequest(%#+v)", rd)
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	for _, fs := range cfgCtl.flowSchemas {
		if matchesFlowSchema(rd, fs) {
			plName := fs.Spec.PriorityLevelConfiguration.Name
			plState := cfgCtl.priorityLevelStates[plName]
			if plState.pl.Spec.Type == fctypesv1a1.PriorityLevelEnablementExempt {
				klog.V(7).Infof("startRequest(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, immediate", rd, fs.Name, fs.Spec.DistinguisherMethod, plName)
				return fs, plState.pl, true, immediateRequest{}, time.Time{}
			}
			var numQueues int32
			if plState.pl.Spec.Limited.LimitResponse.Type == fctypesv1a1.LimitResponseTypeQueue {
				numQueues = plState.pl.Spec.Limited.LimitResponse.Queuing.Queues

			}
			var hashValue uint64
			if numQueues > 1 {
				flowDistinguisher := computeFlowDistinguisher(rd, fs.Spec.DistinguisherMethod)
				hashValue = hashFlowID(fs.Name, flowDistinguisher)
			}
			startWaitingTime = time.Now()
			klog.V(7).Infof("startRequest(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, numQueues=%d", rd, fs.Name, fs.Spec.DistinguisherMethod, plName, numQueues)
			req, idle := plState.queues.StartRequest(ctx, hashValue, fs.Name, rd.RequestInfo, rd.User)
			if idle {
				cfgCtl.maybeReapLocked(plName, plState)
			}
			return fs, plState.pl, false, req, startWaitingTime
		}
	}
	// This can never happen because every configState has a
	// FlowSchema that matches everything.  If somehow control reaches
	// here, panic with some relevant information.
	var catchAll *fctypesv1a1.FlowSchema
	for _, fs := range cfgCtl.flowSchemas {
		if fs.Name == fctypesv1a1.FlowSchemaNameCatchAll {
			catchAll = fs
		}
	}
	panic(fmt.Sprintf("No match; rd=%#+v, catchAll=%s", rd, fcfmt.Fmt(catchAll)))
}

// Call this after getting a clue that the given priority level is undesired and idle
func (cfgCtl *configController) maybeReap(plName string) {
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	plState := cfgCtl.priorityLevelStates[plName]
	if plState == nil {
		klog.V(7).Infof("plName=%s, plState==nil", plName)
		return
	}
	if plState.queues != nil {
		useless := plState.quiescing && plState.numPending == 0 && plState.queues.IsIdle()
		klog.V(7).Infof("plState.quiescing=%v, plState.numPending=%d, useless=%v", plState.quiescing, plState.numPending, useless)
		if !useless {
			return
		}
	}
	klog.V(3).Infof("Triggered API priority and fairness config reloading because priority level %s is undesired and idle", plName)
	cfgCtl.configQueue.Add(0)
}

// Call this if both (1) plState.queues is non-nil and reported being
// idle, and (2) cfgCtl's lock has not been released since then.
func (cfgCtl *configController) maybeReapLocked(plName string, plState *priorityLevelState) {
	if !(plState.quiescing && plState.numPending == 0) {
		return
	}
	klog.V(3).Infof("Triggered API priority and fairness config reloading because priority level %s is undesired and idle", plName)
	cfgCtl.configQueue.Add(0)
}

// computeFlowDistinguisher extracts the flow distinguisher according to the given method
func computeFlowDistinguisher(rd RequestDigest, method *fctypesv1a1.FlowDistinguisherMethod) string {
	if method == nil {
		return ""
	}
	switch method.Type {
	case fctypesv1a1.FlowDistinguisherMethodByUserType:
		return rd.User.GetName()
	case fctypesv1a1.FlowDistinguisherMethodByNamespaceType:
		return rd.RequestInfo.Namespace
	default:
		// this line shall never reach
		panic("invalid flow-distinguisher method")
	}
}

func hashFlowID(fsName, fDistinguisher string) uint64 {
	hash := sha256.New()
	var sep = [1]byte{0}
	hash.Write([]byte(fsName))
	hash.Write(sep[:])
	hash.Write([]byte(fDistinguisher))
	var sum [32]byte
	hash.Sum(sum[:0])
	return binary.LittleEndian.Uint64(sum[:8])
}
