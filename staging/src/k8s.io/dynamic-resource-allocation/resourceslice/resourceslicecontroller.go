/*
Copyright 2024 The Kubernetes Authors.

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

package resourceslice

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	cgocore "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	draclient "k8s.io/dynamic-resource-allocation/client"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

const (
	// poolNameIndex is the name for the ResourceSlice store's index function,
	// which is to index by ResourceSlice.Spec.Pool.Name
	poolNameIndex = "poolName"

	// Including adds in the mutation cache is not safe: We could add a slice, store it,
	// and then the slice gets deleted without the informer hearing anything about that.
	// Then the obsolete slice remains in the mutation cache.
	//
	// To mitigate this, we use a TTL and check a pool again once added slices expire.
	DefaultMutationCacheTTL = time.Minute

	// DefaultSyncDelay defines how long to wait between receiving the most recent
	// informer event and syncing again. This is long enough that the informer cache
	// should be up-to-date (matters mostly for deletes because an out-dated cache
	// causes redundant delete API calls) and not too long that a human mistake
	// doesn't get fixed while that human is waiting for it.
	DefaultSyncDelay = 30 * time.Second
)

// Controller synchronizes information about resources of one driver with
// ResourceSlice objects. It supports node-local and network-attached
// resources. A DRA driver for node-local resources typically runs this
// controller as part of its kubelet plugin.
type Controller struct {
	cancel         func(cause error)
	driverName     string
	owner          *Owner
	resourceClient *draclient.Client
	coreClient     cgocore.CoreV1Interface
	wg             sync.WaitGroup
	// The queue is keyed with the pool name that needs work.
	queue            workqueue.TypedRateLimitingInterface[string]
	sliceStore       cache.MutationCache
	mutationCacheTTL time.Duration
	syncDelay        time.Duration
	errorHandler     func(ctx context.Context, err error, msg string)

	// Last time that a ResourceSlice of a pool was created.
	// At that time + cache mutation TTL do we have to sync again
	// because the locally cached slice might have stayed in the
	// cache erronously (not removed on delete by someone else)
	// and we have to check again once it has been removed from
	// the cache.
	//
	// It's not sufficient to schedule a delayed sync because
	// another sync scheduled by an event overwrites the older
	// one, so we would sync too soon and then not again.
	//
	// The key is the pool name. This makes each time entry
	// unique for syncPool calls for the pool.
	lastAddByPool map[string]time.Time

	// Must use atomic access...
	numCreates int64
	numUpdates int64
	numDeletes int64

	mutex sync.RWMutex

	// When receiving updates from the driver, the entire pointer replaced,
	// so it is okay to not do a deep copy of it when reading it. Only reading
	// the pointer itself must be protected by a read lock.
	resources *DriverResources
}

// +k8s:deepcopy-gen=true

// DriverResources is a complete description of all resources synchronized by the controller.
type DriverResources struct {
	// Each driver may manage different resource pools.
	Pools map[string]Pool
}

// +k8s:deepcopy-gen=true

// Pool is the collection of devices belonging to the same pool.
type Pool struct {
	// NodeSelector may be different for each pool. Must not get set together
	// with Resources.NodeName. If nil and Resources.NodeName is not set,
	// then devices are available on all nodes.
	NodeSelector *v1.NodeSelector

	// Generation can be left at zero. It gets bumped up automatically
	// by the controller.
	Generation int64

	// Slices is a list of all ResourceSlices that the driver
	// wants to publish for this pool. The driver must ensure
	// that each resulting slice is valid. See the API
	// definition for details, in particular the limit on
	// the number of devices.
	//
	// If slices are not valid, then the controller will
	// log errors produced by the apiserver.
	//
	// Drivers should publish at least one slice for each
	// pool that they normally manage, even if that slice
	// is empty. "Empty pool" is different from "no pool"
	// because it shows that the driver is up-and-running
	// and simply doesn't have any devices.
	Slices []Slice
}

// +k8s:deepcopy-gen=true

// Slice is turned into one ResourceSlice by the controller.
type Slice struct {
	// Devices lists all devices which are part of the slice.
	Devices                []resourceapi.Device
	SharedCounters         []resourceapi.CounterSet
	PerDeviceNodeSelection *bool
}

// +k8s:deepcopy-gen=true

// Owner is the resource which is meant to be listed as owner of the resource slices.
// For a node the UID may be left blank. The controller will look it up automatically.
type Owner struct {
	APIVersion string
	Kind       string
	Name       string
	UID        types.UID
}

// StartController constructs a new controller and starts it.
func StartController(ctx context.Context, options Options) (*Controller, error) {
	logger := klog.FromContext(ctx)
	c, err := newController(ctx, options)
	if err != nil {
		return nil, fmt.Errorf("create controller: %w", err)
	}

	logger.V(3).Info("Starting")
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		defer logger.V(3).Info("Stopping")
		c.run(ctx)
	}()
	return c, nil
}

// Options contains various optional settings for [StartController].
type Options struct {
	// DriverName is the required name of the DRA driver.
	DriverName string

	// KubeClient is used to read Node objects (if necessary) and to access
	// ResourceSlices. It must be specified.
	KubeClient kubernetes.Interface

	// If the owner is a v1.Node, then the NodeName field in the
	// ResourceSlice objects is set and used to identify objects
	// managed by the controller. The UID is not needed in that
	// case, the controller will determine it automatically.
	//
	// The owner must be cluster-scoped. This is not always possible,
	// therefore it is optional. A driver without a owner must take
	// care that remaining slices get deleted manually as part of
	// a driver uninstall because garbage collection won't work.
	Owner *Owner

	// This is the initial desired set of slices. Nil means "no resources".
	Resources *DriverResources

	// Queue can be used to override the default work queue implementation.
	Queue workqueue.TypedRateLimitingInterface[string]

	// MutationCacheTTL can be used to change the default TTL of one minute.
	// See source code for details.
	MutationCacheTTL *time.Duration

	// SyncDelay defines how long to wait between receiving the most recent
	// informer event and syncing again. The default is 30 seconds.
	//
	// This is long enough that the informer cache should be up-to-date
	// (matter mostly for deletes because an out-dated cache causes
	// redundant delete API calls) and not too long that a human mistake
	// doesn't get fixed while that human is waiting for it.
	SyncDelay *time.Duration

	// ErrorHandler will get called whenever the controller encounters
	// a problem while trying to publish ResourceSlices. The controller
	// will retry once the handler returns. What the handler does with
	// that information is up to the handler. It could log the error,
	// replace the slices if they cannot be published (see below),
	// or force the program running the controller to fail by exiting.
	//
	// If some fields were dropped because the cluster does not support
	// the feature they depend on, then the error is or wraps an
	// [DroppedFieldsError] instance. Use [errors.As] to convert to that
	// type:
	//    var droppedFields *resourceslice.DroppedFieldsError
	//    if errors.As(err, &droppedFields) { ... do something with droppedFields ... }
	//
	// The default is [utilruntime.HandleErrorWithContext] which just logs
	// the problem.
	ErrorHandler func(ctx context.Context, err error, msg string)
}

// DroppedFieldsError is reported through the ErrorHandler in [Options] if
// a slice could not be published exactly as desired by the driver.
type DroppedFieldsError struct {
	PoolName                  string
	SliceIndex                int
	DesiredSlice, ActualSlice *resourceapi.ResourceSlice
}

func (err *DroppedFieldsError) Error() string {
	// We cannot depend on go-cmp to include a diff here (not suitable for production code).
	// The diff might be too large, too. But we can make some educated guesses....
	disabled := err.DisabledFeatures()
	if len(disabled) == 0 {
		// If we get here, DisabledFeatures needs to be updated.
		disabled = []string{"unknown"}
	}
	return fmt.Sprintf("pool %q, slice #%d: some fields were dropped by the apiserver, probably because these features are disabled: %s", err.PoolName, err.SliceIndex, strings.Join(disabled, " "))
}

func (err *DroppedFieldsError) DisabledFeatures() []string {
	var disabled []string

	// Both slices should have the same number of devices, but better check it.
	for i := 0; i < len(err.DesiredSlice.Spec.Devices) && i < len(err.ActualSlice.Spec.Devices); i++ {
		if len(err.DesiredSlice.Spec.Devices[i].Taints) > len(err.ActualSlice.Spec.Devices[i].Taints) {
			disabled = append(disabled, "DRADeviceTaints")
			break
		}
	}

	// Dropped fields for partitionable devices can be detected without looking at the devices themselves.
	if ptr.Deref(err.DesiredSlice.Spec.PerDeviceNodeSelection, false) && !ptr.Deref(err.ActualSlice.Spec.PerDeviceNodeSelection, false) ||
		len(err.DesiredSlice.Spec.SharedCounters) > len(err.ActualSlice.Spec.SharedCounters) {
		disabled = append(disabled, "DRAPartitionableDevices")
	}

	// The number of binding conditions for both slices should be the same. If they differ,
	// it indicates that the DRADeviceBindingConditions feature is disabled.
	for i := 0; i < len(err.DesiredSlice.Spec.Devices) && i < len(err.ActualSlice.Spec.Devices); i++ {
		if len(err.DesiredSlice.Spec.Devices[i].BindingConditions) != len(err.ActualSlice.Spec.Devices[i].BindingConditions) ||
			len(err.DesiredSlice.Spec.Devices[i].BindingFailureConditions) != len(err.ActualSlice.Spec.Devices[i].BindingFailureConditions) {
			disabled = append(disabled, "DRADeviceBindingConditions")
			break
		}
	}

	// Dropped fields for consumable capacity can be detected with allowMultipleAllocations flag without looking at individual device capacity.
	for i := 0; i < len(err.DesiredSlice.Spec.Devices) && i < len(err.ActualSlice.Spec.Devices); i++ {
		if err.DesiredSlice.Spec.Devices[i].AllowMultipleAllocations != nil && err.ActualSlice.Spec.Devices[i].AllowMultipleAllocations == nil {
			disabled = append(disabled, "DRAConsumableCapacity")
			break
		}
	}

	return disabled
}

var _ error = &DroppedFieldsError{}

// Stop cancels all background activity and blocks until the controller has stopped.
func (c *Controller) Stop() {
	if c == nil {
		return
	}
	c.cancel(errors.New("ResourceSlice controller was asked to stop"))
	c.wg.Wait()
}

// Update sets the new desired state of the resource information.
//
// The controller is doing a deep copy, so the caller may update
// the instance once Update returns. Nil is valid and the same
// as an empty resources struct.
func (c *Controller) Update(resources *DriverResources) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Sync all old pools..
	if c.resources != nil {
		for poolName := range c.resources.Pools {
			c.queue.Add(poolName)
		}
	}

	if resources == nil {
		c.resources = &DriverResources{}
	} else {
		c.resources = resources.DeepCopy()
		roundTaintTimeAdded(c.resources)
	}

	// ... and the new ones (might be the same).
	for poolName := range c.resources.Pools {
		c.queue.Add(poolName)
	}
}

// roundTaintTimeAdded rounds all timestamps to seconds because that is all
// that we can store. Without this we would get semantic differences between
// desired and actual stored slice.
func roundTaintTimeAdded(resources *DriverResources) {
	for _, pool := range resources.Pools {
		for _, slice := range pool.Slices {
			for _, device := range slice.Devices {
				for _, taint := range device.Taints {
					if taint.TimeAdded != nil {
						taint.TimeAdded.Time = taint.TimeAdded.Time.Round(time.Second)
					}
				}
			}
		}
	}
}

// GetStats provides some insights into operations of the controller.
func (c *Controller) GetStats() Stats {
	s := Stats{
		NumCreates: atomic.LoadInt64(&c.numCreates),
		NumUpdates: atomic.LoadInt64(&c.numUpdates),
		NumDeletes: atomic.LoadInt64(&c.numDeletes),
	}
	return s
}

type Stats struct {
	// NumCreates counts the number of ResourceSlices that got created.
	NumCreates int64
	// NumUpdates counts the number of ResourceSlices that got update.
	NumUpdates int64
	// NumDeletes counts the number of ResourceSlices that got deleted.
	NumDeletes int64
}

// newController creates a new controller.
func newController(ctx context.Context, options Options) (*Controller, error) {
	if options.KubeClient == nil {
		return nil, errors.New("KubeClient is nil")
	}
	if options.DriverName == "" {
		return nil, errors.New("DRA driver name is empty")
	}

	ctx, cancel := context.WithCancelCause(ctx)

	c := &Controller{
		cancel:           cancel,
		resourceClient:   draclient.New(options.KubeClient),
		coreClient:       options.KubeClient.CoreV1(),
		driverName:       options.DriverName,
		owner:            options.Owner.DeepCopy(),
		queue:            options.Queue,
		mutationCacheTTL: ptr.Deref(options.MutationCacheTTL, DefaultMutationCacheTTL),
		syncDelay:        ptr.Deref(options.SyncDelay, DefaultSyncDelay),
		errorHandler:     options.ErrorHandler,
		lastAddByPool:    make(map[string]time.Time),
	}
	if c.queue == nil {
		c.queue = workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "node_resource_slices"},
		)
	}
	if c.errorHandler == nil {
		c.errorHandler = func(ctx context.Context, err error, msg string) {
			utilruntime.HandleErrorWithContext(ctx, err, msg)
		}
	}
	if err := c.initInformer(ctx); err != nil {
		return nil, err
	}

	c.Update(options.Resources)

	return c, nil
}

// initInformer initializes the informer used to watch for changes to the resources slice.
func (c *Controller) initInformer(ctx context.Context) error {
	logger := klog.FromContext(ctx)

	// We always filter by driver name, by node name only for node-local resources.
	selector := fields.Set{
		resourceapi.ResourceSliceSelectorDriver:   c.driverName,
		resourceapi.ResourceSliceSelectorNodeName: "",
	}
	if c.owner != nil && c.owner.APIVersion == "v1" && c.owner.Kind == "Node" {
		selector[resourceapi.ResourceSliceSelectorNodeName] = c.owner.Name
	}
	tweakListOptions := func(options *metav1.ListOptions) {
		options.FieldSelector = selector.String()
	}
	indexers := cache.Indexers{
		poolNameIndex: func(obj interface{}) ([]string, error) {
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return []string{}, nil
			}
			return []string{slice.Spec.Pool.Name}, nil
		},
	}
	informer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
				tweakListOptions(&options)
				slices, err := c.resourceClient.ResourceSlices().List(ctx, options)
				if err == nil {
					logger.V(5).Info("Listed ResourceSlices", "resourceAPI", c.resourceClient.CurrentAPI(), "numSlices", len(slices.Items), "listMeta", slices.ListMeta)
				} else {
					logger.V(5).Info("Listed ResourceSlices", "resourceAPI", c.resourceClient.CurrentAPI(), "err", err)
				}
				return slices, err
			},
			WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
				tweakListOptions(&options)
				w, err := c.resourceClient.ResourceSlices().Watch(ctx, options)
				logger.V(5).Info("Started watching ResourceSlices", "resourceAPI", c.resourceClient.CurrentAPI(), "err", err)
				return w, err
			},
		},
		&resourceapi.ResourceSlice{},
		// No resync because all it would do is periodically trigger syncing pools
		// again by reporting all slices as updated with the object as old/new.
		// Our sync method is deterministic (or should be!), so repeating it
		// won't change the outcome.
		0,
		indexers,
	)
	c.sliceStore = cache.NewIntegerResourceVersionMutationCache(logger, informer.GetStore(), informer.GetIndexer(), c.mutationCacheTTL, true /* includeAdds */)
	handler, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlice add", "slice", klog.KObj(slice))
			c.queue.AddAfter(slice.Spec.Pool.Name, c.syncDelay)
			logger.V(5).Info("Scheduled sync", "poolName", slice.Spec.Pool.Name, "at", time.Now().Add(c.syncDelay))
		},
		UpdateFunc: func(old, new any) {
			oldSlice, ok := old.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			newSlice, ok := new.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			if loggerV := logger.V(6); loggerV.Enabled() {
				loggerV.Info("ResourceSlice update", "slice", klog.KObj(newSlice), "diff", diff.Diff(oldSlice, newSlice))
			} else {
				logger.V(5).Info("ResourceSlice update", "slice", klog.KObj(newSlice))
			}
			c.queue.AddAfter(oldSlice.Spec.Pool.Name, c.syncDelay)
			logger.V(5).Info("Scheduled sync", "pool", oldSlice.Spec.Pool.Name, "at", time.Now().Add(c.syncDelay))
			if oldSlice.Spec.Pool.Name != newSlice.Spec.Pool.Name {
				c.queue.AddAfter(newSlice.Spec.Pool.Name, c.syncDelay)
				logger.V(5).Info("Scheduled sync", "poolName", newSlice.Spec.Pool.Name, "at", time.Now().Add(c.syncDelay))
			}
		},
		DeleteFunc: func(obj any) {
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlice delete", "slice", klog.KObj(slice))
			c.queue.AddAfter(slice.Spec.Pool.Name, c.syncDelay)
			logger.V(5).Info("Scheduled sync", "poolName", slice.Spec.Pool.Name, "at", time.Now().Add(c.syncDelay))
		},
	})
	if err != nil {
		return fmt.Errorf("registering event handler on the ResourceSlice informer: %w", err)
	}
	// Start informer and wait for our cache to be populated.
	logger.V(3).Info("Starting ResourceSlice informer and waiting for it to sync")
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		defer logger.V(3).Info("ResourceSlice informer has stopped")
		defer c.queue.ShutDown() // Once we get here, we must have been asked to stop.
		informer.Run(ctx.Done())
	}()
	for !handler.HasSynced() {
		select {
		case <-time.After(time.Second):
		case <-ctx.Done():
			return fmt.Errorf("sync ResourceSlice informer: %w", context.Cause(ctx))
		}
	}
	logger.V(3).Info("ResourceSlice informer has synced")
	return nil
}

// run is running in the background.
func (c *Controller) run(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	poolName, shutdown := c.queue.Get()
	if shutdown {
		return false
	}
	defer c.queue.Done(poolName)
	logger := klog.FromContext(ctx)

	err := c.syncPool(klog.NewContext(ctx, klog.LoggerWithValues(logger, "poolName", poolName)), poolName)
	if err != nil {
		c.errorHandler(ctx, err, "processing ResourceSlice objects")
		c.queue.AddRateLimited(poolName)

		// Return without removing the work item from the queue.
		// It will be retried.
		return true
	}

	c.queue.Forget(poolName)
	return true
}

// syncPool processes one pool. Only runs inside a single worker, so there
// is no need for locking except when accessing c.resources, which may
// be updated at any time by the user of the controller.
func (c *Controller) syncPool(ctx context.Context, poolName string) error {
	logger := klog.FromContext(ctx)
	start := time.Now()

	// Gather information about the actual and desired state.
	var slices []*resourceapi.ResourceSlice
	objs, err := c.sliceStore.ByIndex(poolNameIndex, poolName)
	if err != nil {
		return fmt.Errorf("retrieve ResourceSlice objects: %w", err)
	}
	for _, obj := range objs {
		if slice, ok := obj.(*resourceapi.ResourceSlice); ok {
			slices = append(slices, slice)
		}
	}
	var resources *DriverResources
	c.mutex.RLock()
	resources = c.resources
	c.mutex.RUnlock()

	pool, ok := resources.Pools[poolName]
	if !ok {
		if len(slices) > 0 {
			// All are obsolete, pool does not exist anymore.
			logger.V(5).Info("Removing resource slices after pool removal")
			if err := c.removeSlices(ctx, slices); err != nil {
				return fmt.Errorf("remove slices: %w", err)
			}
		}
		// Pool does not exist anymore, nothing more to do.
		return nil
	}

	// Retrieve node object to get UID?
	// The result gets cached and is expected to not change while
	// the controller runs.
	var nodeName string
	if c.owner != nil && c.owner.APIVersion == "v1" && c.owner.Kind == "Node" {
		nodeName = c.owner.Name
		if c.owner.UID == "" {
			node, err := c.coreClient.Nodes().Get(ctx, c.owner.Name, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("retrieve node %q: %w", c.owner.Name, err)
			}
			// There is only one worker, so no locking needed.
			c.owner.UID = node.UID
		}
	}

	// Slices that don't match any driver slice need to be deleted.
	obsoleteSlices := make([]*resourceapi.ResourceSlice, 0, len(slices))

	// Determine highest generation.
	var generation int64
	for _, slice := range slices {
		if slice.Spec.Pool.Generation > generation {
			generation = slice.Spec.Pool.Generation
		}
	}

	// Everything older is obsolete.
	currentSlices := make([]*resourceapi.ResourceSlice, 0, len(slices))
	for _, slice := range slices {
		if slice.Spec.Pool.Generation < generation {
			obsoleteSlices = append(obsoleteSlices, slice)
		} else {
			currentSlices = append(currentSlices, slice)
		}
	}
	logger.V(5).Info("Existing slices", "obsolete", klog.KObjSlice(obsoleteSlices), "current", klog.KObjSlice(currentSlices))

	// Match each existing slice against the desired slices.
	// Two slices "match" if they contain exactly the
	// same device IDs, in an arbitrary order. As a
	// special case, slices are also considered
	// "matched" in the scenario where there's a single
	// existing slice and a single desired slice. Such a
	// matched slice gets updated with the desired
	// content if there is a difference.
	//
	// In the case where there is more than one existing
	// or desired slices, adding or removing devices is
	// done by deleting the old slice and creating a new one.
	//
	// This is primarily a simplification of the code:
	// to support adding or removing devices from
	// existing slices, we would have to identify "most
	// similar" slices (= minimal editing distance).
	//
	// In currentSliceForDesiredSlice we keep track of
	// which desired slice has a matched slice.
	//
	// At the end of the loop, each current slice is either
	// a match or obsolete.
	currentSliceForDesiredSlice := make(map[int]*resourceapi.ResourceSlice, len(pool.Slices))
	if len(currentSlices) == 1 && len(pool.Slices) == 1 {
		// If there's just one existing slice and one desired slice, assume
		// they "matched" such that if required, it is the existing slice
		// which gets updated and we avoid an unnecessary deletion and
		// recreation of the slice.
		currentSliceForDesiredSlice[0] = currentSlices[0]
	} else {
		for _, currentSlice := range currentSlices {
			matched := false
			for i := range pool.Slices {
				if _, ok := currentSliceForDesiredSlice[i]; ok {
					// Already has a match.
					continue
				}
				if sameSlice(currentSlice, &pool.Slices[i]) {
					currentSliceForDesiredSlice[i] = currentSlice
					logger.V(5).Info("Matched existing slice", "slice", klog.KObj(currentSlice), "matchIndex", i)
					matched = true
					break
				}
			}
			if !matched {
				obsoleteSlices = append(obsoleteSlices, currentSlice)
				logger.V(5).Info("Unmatched existing slice", "slice", klog.KObj(currentSlice))
			}
		}
	}

	// Desired metadata which must be set in each slice.
	resourceSliceCount := len(pool.Slices)
	numMatchedSlices := len(currentSliceForDesiredSlice)
	numNewSlices := resourceSliceCount - numMatchedSlices
	desiredPool := resourceapi.ResourcePool{
		Name:               poolName,
		Generation:         generation, // May get updated later.
		ResourceSliceCount: int64(resourceSliceCount),
	}
	desiredAllNodes := pool.NodeSelector == nil && nodeName == ""

	// Now for each desired slice, figure out which of them are changed.
	changedDesiredSlices := sets.New[int]()
	for i, currentSlice := range currentSliceForDesiredSlice {
		// Reordering entries is a difference and causes an update even if the
		// entries are the same.
		if !apiequality.Semantic.DeepEqual(&currentSlice.Spec.Pool, &desiredPool) ||
			!apiequality.Semantic.DeepEqual(currentSlice.Spec.NodeSelector, pool.NodeSelector) ||
			ptr.Deref(currentSlice.Spec.AllNodes, false) != desiredAllNodes ||
			!DevicesDeepEqual(currentSlice.Spec.Devices, pool.Slices[i].Devices) ||
			!apiequality.Semantic.DeepEqual(currentSlice.Spec.SharedCounters, pool.Slices[i].SharedCounters) ||
			!apiequality.Semantic.DeepEqual(currentSlice.Spec.PerDeviceNodeSelection, pool.Slices[i].PerDeviceNodeSelection) {
			changedDesiredSlices.Insert(i)
			logger.V(5).Info("Need to update slice", "slice", klog.KObj(currentSlice), "matchIndex", i)
		}
	}
	logger.V(5).Info("Completed comparison",
		"numObsolete", len(obsoleteSlices),
		"numMatchedSlices", len(currentSliceForDesiredSlice),
		"numChangedMatchedSlices", len(changedDesiredSlices),
		"numNewSlices", numNewSlices,
	)

	bumpedGeneration := false
	switch {
	case pool.Generation > generation:
		// Bump up the generation if the driver asked for it, or
		// start with a non-zero generation.
		generation = pool.Generation
		bumpedGeneration = true
		logger.V(5).Info("Bumped generation to driver-provided generation", "generation", generation)
	case numNewSlices == 0 && len(changedDesiredSlices) <= 1:
		logger.V(5).Info("Kept generation because at most one update API call is necessary", "generation", generation)
	default:
		generation++
		bumpedGeneration = true
		logger.V(5).Info("Bumped generation by one", "generation", generation)
	}
	desiredPool.Generation = generation

	// First delete obsolete slices. If the desired slices are faulty, then it's still better to
	// remove devices that the driver no longer has, even if we cannot publish the new ones.
	if err := c.removeSlices(ctx, obsoleteSlices); err != nil {
		return fmt.Errorf("remove slices: %w", err)
	}

	// Update existing slices.
	for i, currentSlice := range currentSliceForDesiredSlice {
		if !changedDesiredSlices.Has(i) && !bumpedGeneration {
			continue
		}
		slice := currentSlice.DeepCopy()
		slice.Spec.Pool = desiredPool
		// No need to set the node name. If it was different, we wouldn't
		// have listed the existing slice.
		//
		// When adding new fields here, then also extend sliceStored.
		slice.Spec.NodeSelector = pool.NodeSelector
		slice.Spec.AllNodes = refIfNotZero(desiredAllNodes)
		slice.Spec.SharedCounters = pool.Slices[i].SharedCounters
		slice.Spec.PerDeviceNodeSelection = pool.Slices[i].PerDeviceNodeSelection
		// Preserve TimeAdded from existing device, if there is a matching device and taint.
		slice.Spec.Devices = copyTaintTimeAdded(slice.Spec.Devices, pool.Slices[i].Devices)

		actualSlice, err := c.resourceClient.ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("update resource slice: %w", err)
		}
		logger.V(5).Info("Updated existing resource slice", "slice", klog.KObj(slice))
		atomic.AddInt64(&c.numUpdates, 1)
		c.sliceStored(ctx, "update ResourceSlice", poolName, pool, i, slice, actualSlice)
	}

	// Create new slices.
	added := false
	for i := 0; i < len(pool.Slices); i++ {
		if _, ok := currentSliceForDesiredSlice[i]; ok {
			// Was handled above through an update.
			continue
		}
		var ownerReferences []metav1.OwnerReference
		if c.owner != nil {
			ownerReferences = append(ownerReferences,
				metav1.OwnerReference{
					APIVersion: c.owner.APIVersion,
					Kind:       c.owner.Kind,
					Name:       c.owner.Name,
					UID:        c.owner.UID,
					Controller: ptr.To(true),
				},
			)
		}
		generateName := c.driverName + "-"
		if c.owner != nil {
			generateName = c.owner.Name + "-" + generateName
		}
		slice := &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{
				OwnerReferences: ownerReferences,
				GenerateName:    generateName,
			},
			Spec: resourceapi.ResourceSliceSpec{
				Driver:                 c.driverName,
				Pool:                   desiredPool,
				NodeName:               refIfNotZero(nodeName),
				NodeSelector:           pool.NodeSelector,
				AllNodes:               refIfNotZero(desiredAllNodes),
				Devices:                pool.Slices[i].Devices,
				SharedCounters:         pool.Slices[i].SharedCounters,
				PerDeviceNodeSelection: pool.Slices[i].PerDeviceNodeSelection,
			},
		}

		// It can happen that we create a missing slice, some
		// other change than the create causes another sync of
		// the pool, and then a second slice for the same set
		// of devices would get created because the controller has
		// no copy of the first slice instance in its informer
		// cache yet.
		//
		// Using a https://pkg.go.dev/k8s.io/client-go/tools/cache#MutationCache
		// avoids that.
		actualSlice, err := c.resourceClient.ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("create resource slice: %w", err)
		}
		logger.V(5).Info("Created new resource slice", "slice", klog.KObj(actualSlice))
		atomic.AddInt64(&c.numCreates, 1)
		added = true
		c.sliceStored(ctx, "create ResourceSlice", poolName, pool, i, slice, actualSlice)
	}

	now := time.Now()
	if added {
		c.lastAddByPool[poolName] = now
		logger.V(5).Info("Added slices")
	} else if lastAdd, ok := c.lastAddByPool[poolName]; ok && start.After(lastAdd.Add(c.mutationCacheTTL)) {
		// This sync started after the last add expired from the cache,
		// so we are done and don't need to check again.
		delete(c.lastAddByPool, poolName)
		logger.V(5).Info("Done with re-syncing")
	}
	if lastAdd, ok := c.lastAddByPool[poolName]; ok {
		// Need to check again.
		//
		// Scheduling the resync races with scheduling them in informer events, but that's okay:
		// what matters is that we sync at all at some point.
		//
		// lastAdd was taked by time.Now() above and thus is slightly higher or equal
		// to the time taken by the mutation cache when the slice was added, so we
		// can be sure that any sync running at this time will not see the added
		// slice because it will be expired.
		when := lastAdd.Add(c.mutationCacheTTL)
		c.queue.AddAfter(poolName, when.Sub(now))
		logger.V(5).Info("Scheduled re-sync", "at", when)
	}

	return nil
}

func (c *Controller) removeSlices(ctx context.Context, slices []*resourceapi.ResourceSlice) error {
	logger := klog.FromContext(ctx)

	for _, slice := range slices {
		options := metav1.DeleteOptions{
			Preconditions: &metav1.Preconditions{
				UID:             &slice.UID,
				ResourceVersion: &slice.ResourceVersion,
			},
		}
		// It can happen that we sync again shortly after deleting a
		// slice and before the slice gets removed from the informer
		// cache. The MutationCache can't help here because it does not
		// track pending deletes.
		//
		// If this happens, we get a "not found error" and nothing
		// changes on the server. The only downside is the extra API
		// call. This isn't as bad as extra creates.
		err := c.resourceClient.ResourceSlices().Delete(ctx, slice.Name, options)
		switch {
		case err == nil:
			logger.V(5).Info("Deleted obsolete resource slice", "slice", klog.KObj(slice), "deleteOptions", options)
			atomic.AddInt64(&c.numDeletes, 1)
		case apierrors.IsNotFound(err):
			logger.V(5).Info("Resource slice was already deleted earlier", "slice", klog.KObj(slice))
		default:
			return fmt.Errorf("delete resource slice: %w", err)
		}
	}

	return nil
}

// sliceStored gets called after creating or updating a slice.
// The slice might have been modified during the roundtrip
// through the apiserver.
func (c *Controller) sliceStored(ctx context.Context, msg string, poolName string, pool Pool, sliceIndex int, desiredSlice, actualSlice *resourceapi.ResourceSlice) {
	c.sliceStore.Mutation(actualSlice)

	// One difference is normal: the apiserver may have added TimeAdded to taints.
	// This mutates desiredSlice for the DeepEqual below.
	if copyServerDefaults(desiredSlice, actualSlice) {
		pool.Slices[sliceIndex].Devices = actualSlice.Spec.Devices
	}

	// Some fields may have been dropped. When we receive
	// the updated slice through the informer, the
	// DeepEqual fails and the controller would try to
	// update again, etc.  To break that cycle, update our
	// desired state of the world so that it matches what
	// we can store.
	if !apiequality.Semantic.DeepEqual(desiredSlice.Spec.PerDeviceNodeSelection, actualSlice.Spec.PerDeviceNodeSelection) ||
		!apiequality.Semantic.DeepEqual(desiredSlice.Spec.SharedCounters, actualSlice.Spec.SharedCounters) ||
		!apiequality.Semantic.DeepEqual(desiredSlice.Spec.Devices, actualSlice.Spec.Devices) {
		pool.Slices[sliceIndex].PerDeviceNodeSelection = actualSlice.Spec.PerDeviceNodeSelection
		pool.Slices[sliceIndex].SharedCounters = actualSlice.Spec.SharedCounters
		pool.Slices[sliceIndex].Devices = actualSlice.Spec.Devices

		err := &DroppedFieldsError{
			PoolName:     poolName,
			SliceIndex:   sliceIndex,
			DesiredSlice: desiredSlice.DeepCopy(),
			ActualSlice:  actualSlice.DeepCopy(),
		}
		c.errorHandler(ctx, err, msg)
	}
}

func copyServerDefaults(desiredSlice, actualSlice *resourceapi.ResourceSlice) bool {
	copied := false

	// Should have the same length and entries in the same order.
	for i := 0; i < len(desiredSlice.Spec.Devices) && i < len(actualSlice.Spec.Devices); i++ {
		for e := 0; e < len(desiredSlice.Spec.Devices[i].Taints) && e < len(actualSlice.Spec.Devices[i].Taints); e++ {
			if desiredSlice.Spec.Devices[i].Taints[e].TimeAdded == nil && actualSlice.Spec.Devices[i].Taints[e].TimeAdded != nil {
				if !copied {
					desiredSlice.Spec = *desiredSlice.Spec.DeepCopy()
					copied = true
				}
				desiredSlice.Spec.Devices[i].Taints[e].TimeAdded = actualSlice.Spec.Devices[i].Taints[e].TimeAdded
			}
		}
	}
	return copied
}

func sameSlice(existingSlice *resourceapi.ResourceSlice, desiredSlice *Slice) bool {
	if len(existingSlice.Spec.Devices) != len(desiredSlice.Devices) {
		return false
	}

	existingDevices := sets.New[string]()
	for _, device := range existingSlice.Spec.Devices {
		existingDevices.Insert(device.Name)
	}
	for _, device := range desiredSlice.Devices {
		if !existingDevices.Has(device.Name) {
			return false
		}
	}

	// Same number of devices, names all present -> equal.
	return true
}

// copyTaintTimeAdded copies existing TimeAdded values from one slice into
// the other if the other one doesn't have it for a taint. Both input
// slices are read-only.
func copyTaintTimeAdded(from, to []resourceapi.Device) []resourceapi.Device {
	to = slices.Clone(to)
	for i, toDevice := range to {
		index := slices.IndexFunc(from, func(fromDevice resourceapi.Device) bool {
			return fromDevice.Name == toDevice.Name
		})
		if index < 0 {
			// No matching device.
			continue
		}
		fromDevice := from[index]
		for j, toTaint := range toDevice.Taints {
			if toTaint.TimeAdded != nil {
				// Already set.
				continue
			}
			// Preserve the old TimeAdded if all other fields are the same.
			index := slices.IndexFunc(fromDevice.Taints, func(fromTaint resourceapi.DeviceTaint) bool {
				return toTaint.Key == fromTaint.Key &&
					toTaint.Value == fromTaint.Value &&
					toTaint.Effect == fromTaint.Effect
			})
			if index < 0 {
				// No matching old taint.
				continue
			}
			// In practice, devices are unlikely to have many
			// taints.  Just clone the entire device before we
			// motify it, it's unlikely that we do this more than once.
			to[i] = *toDevice.DeepCopy()
			to[i].Taints[j].TimeAdded = fromDevice.Taints[index].TimeAdded
		}
	}
	return to
}

// DevicesDeepEqual compares two slices of Devices. It behaves like
// apiequality.Semantic.DeepEqual, with one small difference:
// a nil DeviceTaint.TimeAdded is equal to a non-nil time.
// Also, rounding to full seconds (caused by round-tripping) is
// tolerated.
func DevicesDeepEqual(a, b []resourceapi.Device) bool {
	return devicesSemantic.DeepEqual(a, b)
}

var devicesSemantic = func() conversion.Equalities {
	semantic := apiequality.Semantic.Copy()
	if err := semantic.AddFunc(deviceTaintEqual); err != nil {
		panic(err)
	}
	return semantic
}()

func deviceTaintEqual(a, b resourceapi.DeviceTaint) bool {
	if a.TimeAdded != nil && b.TimeAdded != nil {
		delta := b.TimeAdded.Time.Sub(a.TimeAdded.Time)
		if delta < -time.Second || delta > time.Second {
			return false
		}
	}
	return a.Key == b.Key &&
		a.Value == b.Value &&
		a.Effect == b.Effect
}

func refIfNotZero[T comparable](t T) *T {
	var zero T
	if t == zero {
		return nil
	}
	return &t
}
