/*
Copyright 2015 The Kubernetes Authors.

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

package cache

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientgofeaturegate "k8s.io/client-go/features"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// This file implements a low-level controller that is used in
// sharedIndexInformer, which is an implementation of
// SharedIndexInformer.  Such informers, in turn, are key components
// in the high level controllers that form the backbone of the
// Kubernetes control plane.  Look at those for examples, or the
// example in
// https://github.com/kubernetes/client-go/tree/master/examples/workqueue
// .

// Config contains all the settings for one of these low-level controllers.
type Config struct {
	// The queue for your objects - has to be a DeltaFIFO due to
	// assumptions in the implementation. Your Process() function
	// should accept the output of this Queue's Pop() method.
	Queue

	// Something that can list and watch your objects.
	ListerWatcher

	// Process can process a popped Deltas.
	Process ProcessFunc

	// ProcessBatch can process a batch of popped Deltas, which should return `TransactionError` if not all items
	// in the batch were successfully processed.
	//
	// For batch processing to be used:
	// * ProcessBatch must be non-nil
	// * Queue must implement QueueWithBatch
	// * The client InOrderInformersBatchProcess feature gate must be enabled
	//
	// If any of those are false, Process is used and no batch processing is done.
	ProcessBatch ProcessBatchFunc

	// ObjectType is an example object of the type this controller is
	// expected to handle.
	ObjectType runtime.Object

	// ObjectDescription is the description to use when logging type-specific information about this controller.
	ObjectDescription string

	// FullResyncPeriod is the period at which ShouldResync is considered.
	FullResyncPeriod time.Duration

	// MinWatchTimeout, if set, will define the minimum timeout for watch requests send
	// to kube-apiserver. However, values lower than 5m will not be honored to avoid
	// negative performance impact on controlplane.
	// Optional - if unset a default value of 5m will be used.
	MinWatchTimeout time.Duration

	// ShouldResync is periodically used by the reflector to determine
	// whether to Resync the Queue. If ShouldResync is `nil` or
	// returns true, it means the reflector should proceed with the
	// resync.
	ShouldResync ShouldResyncFunc

	// Called whenever the ListAndWatch drops the connection with an error.
	//
	// Contextual logging: WatchErrorHandlerWithContext should be used instead of WatchErrorHandler in code which supports contextual logging.
	WatchErrorHandler WatchErrorHandler

	// Called whenever the ListAndWatch drops the connection with an error
	// and WatchErrorHandler is not set.
	WatchErrorHandlerWithContext WatchErrorHandlerWithContext

	// WatchListPageSize is the requested chunk size of initial and relist watch lists.
	WatchListPageSize int64
}

// ShouldResyncFunc is a type of function that indicates if a reflector should perform a
// resync or not. It can be used by a shared informer to support multiple event handlers with custom
// resync periods.
type ShouldResyncFunc func() bool

// ProcessFunc processes a single object.
type ProcessFunc func(obj interface{}, isInInitialList bool) error

// ProcessBatchFunc processes multiple objects in batch.
// The deltas must not contain multiple entries for the same object.
type ProcessBatchFunc func(deltas []Delta, isInInitialList bool) error

// `*controller` implements Controller
type controller struct {
	config         Config
	reflector      *Reflector
	reflectorMutex sync.RWMutex
	clock          clock.Clock
}

// Controller is a low-level controller that is parameterized by a
// Config and used in sharedIndexInformer.
type Controller interface {
	// RunWithContext does two things.  One is to construct and run a Reflector
	// to pump objects/notifications from the Config's ListerWatcher
	// to the Config's Queue and possibly invoke the occasional Resync
	// on that Queue.  The other is to repeatedly Pop from the Queue
	// and process with the Config's ProcessFunc.  Both of these
	// continue until the context is canceled.
	//
	// It's an error to call RunWithContext more than once.
	// RunWithContext blocks; call via go.
	RunWithContext(ctx context.Context)

	// Run does the same as RunWithContext with a stop channel instead of
	// a context.
	//
	// Contextual logging: RunWithcontext should be used instead of Run in code which supports contextual logging.
	Run(stopCh <-chan struct{})

	// HasSynced delegates to the Config's Queue
	HasSynced() bool

	// HasSyncedChecker enables waiting for syncing without polling.
	// The returned DoneChecker can be passed to WaitFor.
	// It delegates to the Config's Queue.
	HasSyncedChecker() DoneChecker

	// LastSyncResourceVersion delegates to the Reflector when there
	// is one, otherwise returns the empty string
	LastSyncResourceVersion() string
}

// New makes a new Controller from the given Config.
func New(c *Config) Controller {
	ctlr := &controller{
		config: *c,
		clock:  &clock.RealClock{},
	}
	return ctlr
}

// Run implements [Controller.Run].
func (c *controller) Run(stopCh <-chan struct{}) {
	c.RunWithContext(wait.ContextForChannel(stopCh))
}

// RunWithContext implements [Controller.RunWithContext].
func (c *controller) RunWithContext(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)
	go func() {
		<-ctx.Done()
		c.config.Queue.Close()
	}()
	logger := klog.FromContext(ctx)
	r := NewReflectorWithOptions(
		c.config.ListerWatcher,
		c.config.ObjectType,
		c.config.Queue,
		ReflectorOptions{
			Logger:          &logger,
			ResyncPeriod:    c.config.FullResyncPeriod,
			MinWatchTimeout: c.config.MinWatchTimeout,
			TypeDescription: c.config.ObjectDescription,
			Clock:           c.clock,
		},
	)
	r.ShouldResync = c.config.ShouldResync
	r.WatchListPageSize = c.config.WatchListPageSize
	if c.config.WatchErrorHandler != nil {
		r.watchErrorHandler = func(_ context.Context, r *Reflector, err error) {
			c.config.WatchErrorHandler(r, err)
		}
	} else if c.config.WatchErrorHandlerWithContext != nil {
		r.watchErrorHandler = c.config.WatchErrorHandlerWithContext
	}

	c.reflectorMutex.Lock()
	c.reflector = r
	c.reflectorMutex.Unlock()

	var wg wait.Group

	wg.StartWithContext(ctx, r.RunWithContext)

	wait.UntilWithContext(ctx, c.processLoop, time.Second)
	wg.Wait()
}

// Returns true once this controller has completed an initial resource listing
func (c *controller) HasSynced() bool {
	return c.config.Queue.HasSynced()
}

// HasSyncedChecker enables waiting for syncing without polling.
// The returned DoneChecker can be passed to [WaitFor].
// It delegates to the Config's Queue.
func (c *controller) HasSyncedChecker() DoneChecker {
	return c.config.Queue.HasSyncedChecker()
}

func (c *controller) LastSyncResourceVersion() string {
	c.reflectorMutex.RLock()
	defer c.reflectorMutex.RUnlock()
	if c.reflector == nil {
		return ""
	}
	return c.reflector.LastSyncResourceVersion()
}

// processLoop drains the work queue.
// TODO: Consider doing the processing in parallel. This will require a little thought
// to make sure that we don't end up processing the same object multiple times
// concurrently.
func (c *controller) processLoop(ctx context.Context) {
	useBatchProcess := false
	batchQueue, ok := c.config.Queue.(QueueWithBatch)
	if ok && c.config.ProcessBatch != nil && clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InOrderInformersBatchProcess) {
		useBatchProcess = true
	}
	for {
		select {
		case <-ctx.Done():
			return
		default:
			var err error
			if useBatchProcess {
				err = batchQueue.PopBatch(c.config.ProcessBatch, PopProcessFunc(c.config.Process))
			} else {
				// otherwise fallback to non-batch process behavior
				_, err = c.config.Pop(PopProcessFunc(c.config.Process))
			}
			if err != nil {
				if errors.Is(err, ErrFIFOClosed) {
					return
				}
			}
		}
	}
}

// ResourceEventHandler can handle notifications for events that
// happen to a resource. The events are informational only, so you
// can't return an error.  The handlers MUST NOT modify the objects
// received; this concerns not only the top level of structure but all
// the data structures reachable from it.
//   - OnAdd is called when an object is added.
//   - OnUpdate is called when an object is modified. Note that oldObj is the
//     last known state of the object-- it is possible that several changes
//     were combined together, so you can't use this to see every single
//     change. OnUpdate is also called when a re-list happens, and it will
//     get called even if nothing changed. This is useful for periodically
//     evaluating or syncing something.
//   - OnDelete will get the final state of the item if it is known, otherwise
//     it will get an object of type DeletedFinalStateUnknown. This can
//     happen if the watch is closed and misses the delete event and we don't
//     notice the deletion until the subsequent re-list.
type ResourceEventHandler interface {
	OnAdd(obj interface{}, isInInitialList bool)
	OnUpdate(oldObj, newObj interface{})
	OnDelete(obj interface{})
}

// ResourceEventHandlerFuncs is an adaptor to let you easily specify as many or
// as few of the notification functions as you want while still implementing
// ResourceEventHandler.  This adapter does not remove the prohibition against
// modifying the objects.
//
// See ResourceEventHandlerDetailedFuncs if your use needs to propagate
// HasSynced.
type ResourceEventHandlerFuncs struct {
	AddFunc    func(obj interface{})
	UpdateFunc func(oldObj, newObj interface{})
	DeleteFunc func(obj interface{})
}

// OnAdd calls AddFunc if it's not nil.
func (r ResourceEventHandlerFuncs) OnAdd(obj interface{}, isInInitialList bool) {
	if r.AddFunc != nil {
		r.AddFunc(obj)
	}
}

// OnUpdate calls UpdateFunc if it's not nil.
func (r ResourceEventHandlerFuncs) OnUpdate(oldObj, newObj interface{}) {
	if r.UpdateFunc != nil {
		r.UpdateFunc(oldObj, newObj)
	}
}

// OnDelete calls DeleteFunc if it's not nil.
func (r ResourceEventHandlerFuncs) OnDelete(obj interface{}) {
	if r.DeleteFunc != nil {
		r.DeleteFunc(obj)
	}
}

// ResourceEventHandlerDetailedFuncs is exactly like ResourceEventHandlerFuncs
// except its AddFunc accepts the isInInitialList parameter, for propagating
// HasSynced.
type ResourceEventHandlerDetailedFuncs struct {
	AddFunc    func(obj interface{}, isInInitialList bool)
	UpdateFunc func(oldObj, newObj interface{})
	DeleteFunc func(obj interface{})
}

// OnAdd calls AddFunc if it's not nil.
func (r ResourceEventHandlerDetailedFuncs) OnAdd(obj interface{}, isInInitialList bool) {
	if r.AddFunc != nil {
		r.AddFunc(obj, isInInitialList)
	}
}

// OnUpdate calls UpdateFunc if it's not nil.
func (r ResourceEventHandlerDetailedFuncs) OnUpdate(oldObj, newObj interface{}) {
	if r.UpdateFunc != nil {
		r.UpdateFunc(oldObj, newObj)
	}
}

// OnDelete calls DeleteFunc if it's not nil.
func (r ResourceEventHandlerDetailedFuncs) OnDelete(obj interface{}) {
	if r.DeleteFunc != nil {
		r.DeleteFunc(obj)
	}
}

// FilteringResourceEventHandler applies the provided filter to all events coming
// in, ensuring the appropriate nested handler method is invoked. An object
// that starts passing the filter after an update is considered an add, and an
// object that stops passing the filter after an update is considered a delete.
// Like the handlers, the filter MUST NOT modify the objects it is given.
type FilteringResourceEventHandler struct {
	FilterFunc func(obj interface{}) bool
	Handler    ResourceEventHandler
}

// OnAdd calls the nested handler only if the filter succeeds
func (r FilteringResourceEventHandler) OnAdd(obj interface{}, isInInitialList bool) {
	if !r.FilterFunc(obj) {
		return
	}
	r.Handler.OnAdd(obj, isInInitialList)
}

// OnUpdate ensures the proper handler is called depending on whether the filter matches
func (r FilteringResourceEventHandler) OnUpdate(oldObj, newObj interface{}) {
	newer := r.FilterFunc(newObj)
	older := r.FilterFunc(oldObj)
	switch {
	case newer && older:
		r.Handler.OnUpdate(oldObj, newObj)
	case newer && !older:
		r.Handler.OnAdd(newObj, false)
	case !newer && older:
		r.Handler.OnDelete(oldObj)
	default:
		// do nothing
	}
}

// OnDelete calls the nested handler only if the filter succeeds
func (r FilteringResourceEventHandler) OnDelete(obj interface{}) {
	if !r.FilterFunc(obj) {
		return
	}
	r.Handler.OnDelete(obj)
}

// DeletionHandlingMetaNamespaceKeyFunc checks for
// DeletedFinalStateUnknown objects before calling
// MetaNamespaceKeyFunc.
func DeletionHandlingMetaNamespaceKeyFunc(obj interface{}) (string, error) {
	if d, ok := obj.(DeletedFinalStateUnknown); ok {
		return d.Key, nil
	}
	return MetaNamespaceKeyFunc(obj)
}

// DeletionHandlingObjectToName checks for
// DeletedFinalStateUnknown objects before calling
// ObjectToName.
func DeletionHandlingObjectToName(obj interface{}) (ObjectName, error) {
	if d, ok := obj.(DeletedFinalStateUnknown); ok {
		return ParseObjectName(d.Key)
	}
	return ObjectToName(obj)
}

// InformerOptions configure a Reflector.
type InformerOptions struct {
	// Logger, if not nil, is used instead of klog.Background() for logging.
	Logger *klog.Logger

	// ListerWatcher implements List and Watch functions for the source of the resource
	// the informer will be informing about.
	ListerWatcher ListerWatcher

	// ObjectType is an object of the type that informer is expected to receive.
	ObjectType runtime.Object

	// Handler defines functions that should called on object mutations.
	Handler ResourceEventHandler

	// ResyncPeriod is the underlying Reflector's resync period. If non-zero, the store
	// is re-synced with that frequency - Modify events are delivered even if objects
	// didn't change.
	// This is useful for synchronizing objects that configure external resources
	// (e.g. configure cloud provider functionalities).
	// Optional - if unset, store resyncing is not happening periodically.
	ResyncPeriod time.Duration

	// MinWatchTimeout, if set, will define the minimum timeout for watch requests send
	// to kube-apiserver. However, values lower than 5m will not be honored to avoid
	// negative performance impact on controlplane.
	// Optional - if unset a default value of 5m will be used.
	MinWatchTimeout time.Duration

	// Indexers, if set, are the indexers for the received objects to optimize
	// certain queries.
	// Optional - if unset no indexes are maintained.
	Indexers Indexers

	// Transform function, if set, will be called on all objects before they will be
	// put into the Store and corresponding Add/Modify/Delete handlers will be invoked
	// for them.
	// Optional - if unset no additional transforming is happening.
	Transform TransformFunc

	// Identifier is used to identify the FIFO for metrics and logging purposes.
	// If not set, metrics will not be published.
	Identifier InformerNameAndResource

	// FIFOMetricsProvider is the metrics provider for the FIFO queue.
	// If not set, metrics will be no-ops.
	FIFOMetricsProvider FIFOMetricsProvider
}

// NewInformerWithOptions returns a Store and a controller for populating the store
// while also providing event notifications. You should only used the returned
// Store for Get/List operations; Add/Modify/Deletes will cause the event
// notifications to be faulty.
func NewInformerWithOptions(options InformerOptions) (Store, Controller) {
	var clientState Store
	if options.Indexers == nil {
		clientState = NewStore(DeletionHandlingMetaNamespaceKeyFunc)
	} else {
		clientState = NewIndexer(DeletionHandlingMetaNamespaceKeyFunc, options.Indexers)
	}
	return clientState, newInformer(clientState, options, DeletionHandlingMetaNamespaceKeyFunc)
}

// NewInformer returns a Store and a controller for populating the store
// while also providing event notifications. You should only used the returned
// Store for Get/List operations; Add/Modify/Deletes will cause the event
// notifications to be faulty.
//
// Parameters:
//   - lw is list and watch functions for the source of the resource you want to
//     be informed of.
//   - objType is an object of the type that you expect to receive.
//   - resyncPeriod: if non-zero, will re-list this often (you will get OnUpdate
//     calls, even if nothing changed). Otherwise, re-list will be delayed as
//     long as possible (until the upstream source closes the watch or times out,
//     or you stop the controller).
//   - h is the object you want notifications sent to.
//
// Deprecated: Use NewInformerWithOptions instead.
func NewInformer(
	lw ListerWatcher,
	objType runtime.Object,
	resyncPeriod time.Duration,
	h ResourceEventHandler,
) (Store, Controller) {
	// This will hold the client state, as we know it.
	clientState := NewStore(DeletionHandlingMetaNamespaceKeyFunc)

	options := InformerOptions{
		ListerWatcher: lw,
		ObjectType:    objType,
		Handler:       h,
		ResyncPeriod:  resyncPeriod,
	}
	return clientState, newInformer(clientState, options, DeletionHandlingMetaNamespaceKeyFunc)
}

// NewIndexerInformer returns an Indexer and a Controller for populating the index
// while also providing event notifications. You should only used the returned
// Index for Get/List operations; Add/Modify/Deletes will cause the event
// notifications to be faulty.
//
// Parameters:
//   - lw is list and watch functions for the source of the resource you want to
//     be informed of.
//   - objType is an object of the type that you expect to receive.
//   - resyncPeriod: if non-zero, will re-list this often (you will get OnUpdate
//     calls, even if nothing changed). Otherwise, re-list will be delayed as
//     long as possible (until the upstream source closes the watch or times out,
//     or you stop the controller).
//   - h is the object you want notifications sent to.
//   - indexers is the indexer for the received object type.
//
// Deprecated: Use NewInformerWithOptions instead.
func NewIndexerInformer(
	lw ListerWatcher,
	objType runtime.Object,
	resyncPeriod time.Duration,
	h ResourceEventHandler,
	indexers Indexers,
) (Indexer, Controller) {
	// This will hold the client state, as we know it.
	clientState := NewIndexer(DeletionHandlingMetaNamespaceKeyFunc, indexers)

	options := InformerOptions{
		ListerWatcher: lw,
		ObjectType:    objType,
		Handler:       h,
		ResyncPeriod:  resyncPeriod,
		Indexers:      indexers,
	}
	return clientState, newInformer(clientState, options, DeletionHandlingMetaNamespaceKeyFunc)
}

// NewTransformingInformer returns a Store and a controller for populating
// the store while also providing event notifications. You should only used
// the returned Store for Get/List operations; Add/Modify/Deletes will cause
// the event notifications to be faulty.
// The given transform function will be called on all objects before they will
// put into the Store and corresponding Add/Modify/Delete handlers will
// be invoked for them.
//
// Deprecated: Use NewInformerWithOptions instead.
func NewTransformingInformer(
	lw ListerWatcher,
	objType runtime.Object,
	resyncPeriod time.Duration,
	h ResourceEventHandler,
	transformer TransformFunc,
) (Store, Controller) {
	// This will hold the client state, as we know it.
	clientState := NewStore(DeletionHandlingMetaNamespaceKeyFunc)

	options := InformerOptions{
		ListerWatcher: lw,
		ObjectType:    objType,
		Handler:       h,
		ResyncPeriod:  resyncPeriod,
		Transform:     transformer,
	}
	return clientState, newInformer(clientState, options, DeletionHandlingMetaNamespaceKeyFunc)
}

// NewTransformingIndexerInformer returns an Indexer and a controller for
// populating the index while also providing event notifications. You should
// only used the returned Index for Get/List operations; Add/Modify/Deletes
// will cause the event notifications to be faulty.
// The given transform function will be called on all objects before they will
// be put into the Index and corresponding Add/Modify/Delete handlers will
// be invoked for them.
//
// Deprecated: Use NewInformerWithOptions instead.
func NewTransformingIndexerInformer(
	lw ListerWatcher,
	objType runtime.Object,
	resyncPeriod time.Duration,
	h ResourceEventHandler,
	indexers Indexers,
	transformer TransformFunc,
) (Indexer, Controller) {
	// This will hold the client state, as we know it.
	clientState := NewIndexer(DeletionHandlingMetaNamespaceKeyFunc, indexers)

	options := InformerOptions{
		ListerWatcher: lw,
		ObjectType:    objType,
		Handler:       h,
		ResyncPeriod:  resyncPeriod,
		Indexers:      indexers,
		Transform:     transformer,
	}
	return clientState, newInformer(clientState, options, DeletionHandlingMetaNamespaceKeyFunc)
}

// Multiplexes updates in the form of a list of Deltas into a Store, and informs
// a given handler of events OnUpdate, OnAdd, OnDelete
func processDeltas(
	logger klog.Logger,
	// Object which receives event notifications from the given deltas
	handler ResourceEventHandler,
	clientState Store,
	deltas Deltas,
	isInInitialList bool,
	keyFunc KeyFunc,
) error {
	// from oldest to newest
	for _, d := range deltas {
		obj := d.Object

		switch d.Type {
		case ReplacedAll:
			info, ok := obj.(ReplacedAllInfo)
			if !ok {
				return fmt.Errorf("ReplacedAll did not contain ReplacedAllInfo: %T", obj)
			}
			if err := processReplacedAllInfo(logger, handler, info, clientState, isInInitialList, keyFunc); err != nil {
				return err
			}
		case SyncAll:
			_, ok := obj.(SyncAllInfo)
			if !ok {
				return fmt.Errorf("SyncAll did not contain SyncAllInfo: %T", obj)
			}
			objs := clientState.List()
			for _, obj := range objs {
				handler.OnUpdate(obj, obj)
			}
			return nil
		case Sync, Replaced, Added, Updated:
			if old, exists, err := clientState.Get(obj); err == nil && exists {
				if err := clientState.Update(obj); err != nil {
					return err
				}
				handler.OnUpdate(old, obj)
			} else {
				if err := clientState.Add(obj); err != nil {
					return err
				}
				handler.OnAdd(obj, isInInitialList)
			}
		case Deleted:
			if err := clientState.Delete(obj); err != nil {
				return err
			}
			handler.OnDelete(obj)
		}
	}
	return nil
}

// processDeltasInBatch applies a batch of Delta objects to the given Store and
// notifies the ResourceEventHandler of add, update, or delete events.
//
// If the Store supports transactions (TransactionStore), all Deltas are applied
// atomically in a single transaction and corresponding handler callbacks are
// executed afterward. Otherwise, each Delta is processed individually.
//
// Returns an error if any Delta or transaction fails. For TransactionError,
// only successful operations trigger callbacks.
func processDeltasInBatch(
	logger klog.Logger,
	handler ResourceEventHandler,
	clientState Store,
	deltas []Delta,
	isInInitialList bool,
	keyFunc KeyFunc,
) error {
	// from oldest to newest
	txns := make([]Transaction, 0)
	callbacks := make([]func(), 0)
	txnStore, txnSupported := clientState.(TransactionStore)
	if !txnSupported {
		var errs []error
		for _, delta := range deltas {
			if err := processDeltas(logger, handler, clientState, Deltas{delta}, isInInitialList, keyFunc); err != nil {
				errs = append(errs, err)
			}
		}
		if len(errs) > 0 {
			return fmt.Errorf("unexpected error when handling deltas: %v", errs)
		}
		return nil
	}
	// deltasList is a list of unique objects
	for _, d := range deltas {
		obj := d.Object
		switch d.Type {
		case Sync, Replaced, Added, Updated:
			// it will only return one old object for each because items are unique
			if old, exists, err := clientState.Get(obj); err == nil && exists {
				txn := Transaction{
					Type:   TransactionTypeUpdate,
					Object: obj,
				}
				txns = append(txns, txn)
				callbacks = append(callbacks, func() {
					handler.OnUpdate(old, obj)
				})
			} else {
				txn := Transaction{
					Type:   TransactionTypeAdd,
					Object: obj,
				}
				txns = append(txns, txn)
				callbacks = append(callbacks, func() {
					handler.OnAdd(obj, isInInitialList)
				})
			}
		case Deleted:
			txn := Transaction{
				Type:   TransactionTypeDelete,
				Object: obj,
			}
			txns = append(txns, txn)
			callbacks = append(callbacks, func() {
				handler.OnDelete(obj)
			})
		default:
			return fmt.Errorf("Delta type %s is not supported in batch processing", d.Type)
		}
	}

	err := txnStore.Transaction(txns...)
	if err != nil {
		// if txn had error, only execute the callbacks for the successful ones
		for _, i := range err.SuccessfulIndices {
			if i < len(callbacks) {
				callbacks[i]()
			}
		}
		// formatting the error so txns doesn't escape and keeps allocated in the stack.
		return fmt.Errorf("not all items in the batch successfully processed: %s", err.Error())
	}
	for _, callback := range callbacks {
		callback()
	}
	return nil
}

func processReplacedAllInfo(logger klog.Logger, handler ResourceEventHandler, info ReplacedAllInfo, clientState Store, isInInitialList bool, keyFunc KeyFunc) error {
	var deletions []DeletedFinalStateUnknown
	type replacement struct {
		oldObj interface{}
		newObj interface{}
	}
	replacements := make([]replacement, 0, len(info.Objects))

	err := reconcileReplacement(logger, nil, clientState, info.Objects, keyFunc,
		func(obj DeletedFinalStateUnknown) error {
			deletions = append(deletions, obj)
			return nil
		},
		func(obj interface{}) error {
			// This behavior matches processDeltas handling of Replace deltas
			if old, exists, err := clientState.Get(obj); err == nil && exists {
				replacements = append(replacements, replacement{newObj: obj, oldObj: old})
			} else {
				replacements = append(replacements, replacement{newObj: obj})
			}
			return nil
		},
	)
	if err != nil {
		return err
	}

	// Replace the client state first so the store reflects the events handlers are given
	if err := clientState.Replace(info.Objects, info.ResourceVersion); err != nil {
		return err
	}
	// Processing all deletions first matches behavior of RealFIFO#Replace
	for _, objToDelete := range deletions {
		handler.OnDelete(objToDelete)
	}
	// Processing adds/updates in order observed by reconcileReplacement matches behavior of RealFIFO#Replace
	for _, r := range replacements {
		if r.oldObj != nil {
			handler.OnUpdate(r.oldObj, r.newObj)
		} else {
			handler.OnAdd(r.newObj, isInInitialList)
		}
	}
	return nil
}

// newInformer returns a controller for populating the store while also
// providing event notifications.
//
// Parameters
//   - clientState is the store you want to populate
//   - options contain the options to configure the controller
func newInformer(clientState Store, options InformerOptions, keyFunc KeyFunc) Controller {
	// This will hold incoming changes. Note how we pass clientState in as a
	// KeyLister, that way resync operations will result in the correct set
	// of update/delete deltas.

	logger := klog.Background()
	if options.Logger != nil {
		logger = *options.Logger
	}
	logger, fifo := newQueueFIFO(logger, options.ObjectType, clientState, options.Transform, options.Identifier, options.FIFOMetricsProvider)

	cfg := &Config{
		Queue:            fifo,
		ListerWatcher:    options.ListerWatcher,
		ObjectType:       options.ObjectType,
		FullResyncPeriod: options.ResyncPeriod,
		MinWatchTimeout:  options.MinWatchTimeout,

		Process: func(obj interface{}, isInInitialList bool) error {
			if deltas, ok := obj.(Deltas); ok {
				// This must be the logger *of the fifo*.
				return processDeltas(logger, options.Handler, clientState, deltas, isInInitialList, keyFunc)
			}
			return errors.New("object given as Process argument is not Deltas")
		},
		ProcessBatch: func(deltaList []Delta, isInInitialList bool) error {
			// Same here.
			return processDeltasInBatch(logger, options.Handler, clientState, deltaList, isInInitialList, keyFunc)
		},
	}
	return New(cfg)
}

// newQueueFIFO constructs a new FIFO, choosing between real and delta FIFO
// depending on the InOrderInformers feature gate.
//
// It returns the FIFO and the logger used by the FIFO.
// That logger includes the name used for the FIFO,
// in contrast to the logger which was passed in.
func newQueueFIFO(logger klog.Logger, objectType any, clientState Store, transform TransformFunc, identifier InformerNameAndResource, metricsProvider FIFOMetricsProvider) (klog.Logger, Queue) {
	if clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InOrderInformers) {
		options := RealFIFOOptions{
			Logger:          &logger,
			Name:            fmt.Sprintf("RealFIFO %T", objectType),
			KeyFunction:     MetaNamespaceKeyFunc,
			Transformer:     transform,
			Identifier:      identifier,
			MetricsProvider: metricsProvider,
		}
		// If atomic events are enabled, unset clientState in the case of atomic events as we cannot pass a
		// store to an atomic fifo.
		if clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.AtomicFIFO) {
			options.AtomicEvents = true
		} else {
			options.KnownObjects = clientState
		}
		f := NewRealFIFOWithOptions(options)
		return f.logger, f
	} else {
		f := NewDeltaFIFOWithOptions(DeltaFIFOOptions{
			Logger:                &logger,
			Name:                  fmt.Sprintf("DeltaFIFO %T", objectType),
			KnownObjects:          clientState,
			EmitDeltaTypeReplaced: true,
			Transformer:           transform,
		})
		return f.logger, f
	}
}
