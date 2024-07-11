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
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"

	"k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	resourceinformers "k8s.io/client-go/informers/resource/v1alpha3"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

const (
	// resyncPeriod for informer
	// TODO (https://github.com/kubernetes/kubernetes/issues/123688): disable?
	resyncPeriod = time.Duration(10 * time.Minute)
)

// Controller synchronizes information about resources of one driver with
// ResourceSlice objects. It supports node-local and network-attached
// resources. A DRA driver for node-local resources typically runs this
// controller as part of its kubelet plugin.
type Controller struct {
	cancel     func(cause error)
	driver     string
	owner      Owner
	kubeClient kubernetes.Interface
	wg         sync.WaitGroup
	// The queue is keyed with the pool name that needs work.
	queue      workqueue.TypedRateLimitingInterface[string]
	sliceStore cache.Store

	mutex sync.RWMutex

	// When receiving updates from the driver, the entire pointer replaced,
	// so it is okay to not do a deep copy of it when reading it. Only reading
	// the pointer itself must be protected by a read lock.
	resources *DriverResources
}

// DriverResources is a complete description of all resources synchronized by the controller.
type DriverResources struct {
	// Each driver may manage different resource pools.
	Pools map[string]Pool
}

// Pool is the collection of devices belonging to the same pool.
type Pool struct {
	// NodeSelector may be different for each pool. Must not get set together
	// with Resources.NodeName. It nil and Resources.NodeName is not set,
	// then devices are available on all nodes.
	NodeSelector *v1.NodeSelector

	// Generation can be left at zero. It gets bumped up automatically
	// by the controller.
	Generation int64

	// Device names must be unique inside the pool.
	Devices []resourceapi.Device
}

// Owner is the resource which is meant to be listed as owner of the resource slices.
// For a node the UID may be left blank. The controller will look it up automatically.
type Owner struct {
	APIVersion string
	Kind       string
	Name       string
	UID        types.UID
}

// StartController constructs a new controller and starts it.
// If the owner is a v1.Node, then the NodeName field in the
// ResourceSlice objects is set and used to identify objects
// managed by the controller. The UID is not needed in that
// case, the controller will determine it automatically.
//
// If a kubeClient is provided, then it synchronizes ResourceSlices
// with the resource information provided by plugins. Without it,
// the controller is inactive. This can happen when kubelet is run stand-alone
// without an apiserver. In that case we can't and don't need to publish
// ResourceSlices.
func StartController(ctx context.Context, kubeClient kubernetes.Interface, driver string, owner Owner, resources *DriverResources) *Controller {
	if kubeClient == nil {
		return nil
	}

	logger := klog.FromContext(ctx)
	ctx, cancel := context.WithCancelCause(ctx)

	c := &Controller{
		cancel:     cancel,
		kubeClient: kubeClient,
		driver:     driver,
		owner:      owner,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "node_resource_slices"},
		),
		resources: resources,
	}

	logger.V(3).Info("Starting")
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		defer logger.V(3).Info("Stopping")
		c.run(ctx)
	}()

	// Sync each pool once.
	for poolName := range resources.Pools {
		c.queue.Add(poolName)
	}

	return c
}

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
// The controller takes over ownership, so these resources must
// not get modified after this method returns.
func (c *Controller) Update(resources *DriverResources) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Sync all old pools..
	for poolName := range c.resources.Pools {
		c.queue.Add(poolName)
	}

	c.resources = resources

	// ... and the new ones (might be the same).
	for poolName := range c.resources.Pools {
		c.queue.Add(poolName)
	}
}

// run is running in the background. It handles blocking initialization (like
// syncing the informer) and then syncs the actual with the desired state.
func (c *Controller) run(ctx context.Context) {
	logger := klog.FromContext(ctx)

	// We always filter by driver name, by node name only for node-local resources.
	selector := fields.Set{resourceapi.ResourceSliceSelectorDriver: c.driver}
	if c.owner.APIVersion == "v1" && c.owner.Kind == "Node" {
		selector[resourceapi.ResourceSliceSelectorNodeName] = c.owner.Name
	}
	informer := resourceinformers.NewFilteredResourceSliceInformer(c.kubeClient, resyncPeriod, nil, func(options *metav1.ListOptions) {
		options.FieldSelector = selector.String()
	})
	c.sliceStore = informer.GetStore()
	handler, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlice add", "slice", klog.KObj(slice))
			c.queue.Add(slice.Spec.Pool.Name)
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
				loggerV.Info("ResourceSlice update", "slice", klog.KObj(newSlice), "diff", cmp.Diff(oldSlice, newSlice))
			} else {
				logger.V(5).Info("ResourceSlice update", "slice", klog.KObj(newSlice))
			}
			c.queue.Add(oldSlice.Spec.Pool.Name)
			c.queue.Add(newSlice.Spec.Pool.Name)
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
			c.queue.Add(slice.Spec.Pool.Name)
		},
	})
	if err != nil {
		logger.Error(err, "Registering event handler on the ResourceSlice informer failed, disabling resource monitoring")
		return
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
			return
		}
	}
	logger.V(3).Info("ResourceSlice informer has synced")

	// Seed the

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

	// Panics are caught and treated like errors.
	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("internal error: %v", r)
			}
		}()
		err = c.syncPool(klog.NewContext(ctx, klog.LoggerWithValues(logger, "poolName", poolName)), poolName)
	}()

	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "processing ResourceSlice objects")
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

	// Gather information about the actual and desired state.
	// TODO: index by pool name.
	var slices []*resourceapi.ResourceSlice
	for _, obj := range c.sliceStore.List() {
		if slice, ok := obj.(*resourceapi.ResourceSlice); ok && slice.Spec.Pool.Name == poolName {
			slices = append(slices, slice)
		}
	}
	var resources *DriverResources
	c.mutex.RLock()
	resources = c.resources
	c.mutex.RUnlock()

	// Retrieve node object to get UID?
	// The result gets cached and is expected to not change while
	// the controller runs.
	var nodeName string
	if c.owner.APIVersion == "v1" && c.owner.Kind == "Node" {
		nodeName = c.owner.Name
		if c.owner.UID == "" {
			node, err := c.kubeClient.CoreV1().Nodes().Get(ctx, c.owner.Name, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("retrieve node %q: %w", c.owner.Name, err)
			}
			// There is only one worker, so no locking needed.
			c.owner.UID = node.UID
		}
	}

	// Slices that don't match any driver resource can either be updated (if there
	// are new driver resources that need to be stored) or they need to be deleted.
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
	slices = currentSlices

	if pool, ok := resources.Pools[poolName]; ok {
		if pool.Generation > generation {
			generation = pool.Generation
		}

		// Right now all devices get published in a single slice.
		// We simply pick the first one, if there is one, and copy
		// it in preparation for updating it.
		//
		// TODO: support splitting across slices, with unit tests.
		if len(slices) > 0 {
			obsoleteSlices = append(obsoleteSlices, slices[1:]...)
			slices = []*resourceapi.ResourceSlice{slices[0].DeepCopy()}
		} else {
			slices = []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						GenerateName: c.owner.Name + "-" + c.driver + "-",
					},
				},
			}
		}

		slice := slices[0]
		slice.OwnerReferences = []metav1.OwnerReference{{
			APIVersion: c.owner.APIVersion,
			Kind:       c.owner.Kind,
			Name:       c.owner.Name,
			UID:        c.owner.UID,
			Controller: ptr.To(true),
		}}
		slice.Spec.Driver = c.driver
		slice.Spec.Pool.Name = poolName
		slice.Spec.Pool.Generation = generation
		slice.Spec.Pool.ResourceSliceCount = 1
		slice.Spec.NodeName = nodeName
		slice.Spec.NodeSelector = pool.NodeSelector
		slice.Spec.AllNodes = pool.NodeSelector == nil && nodeName == ""
		slice.Spec.Devices = pool.Devices

		if loggerV := logger.V(6); loggerV.Enabled() {
			// Dump entire resource information.
			loggerV.Info("Syncing resource slices", "obsoleteSlices", klog.KObjSlice(obsoleteSlices), "slices", klog.KObjSlice(slices), "pool", pool)
		} else {
			logger.V(5).Info("Syncing resource slices", "obsoleteSlices", klog.KObjSlice(obsoleteSlices), "slices", klog.KObjSlice(slices), "numDevices", len(pool.Devices))
		}
	} else if len(slices) > 0 {
		// All are obsolete, pool does not exist anymore.

		logger.V(5).Info("Removing resource slices after pool removal", "obsoleteSlices", klog.KObjSlice(obsoleteSlices), "slices", klog.KObjSlice(slices), "numDevices", len(pool.Devices))
		obsoleteSlices = append(obsoleteSlices, slices...)
	}

	// Remove stale slices.
	for _, slice := range obsoleteSlices {
		logger.V(5).Info("Deleting obsolete resource slice", "slice", klog.KObj(slice))
		if err := c.kubeClient.ResourceV1alpha3().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete resource slice: %w", err)
		}
	}

	// Create or update slices.
	for _, slice := range slices {
		if slice.UID == "" {
			logger.V(5).Info("Creating new resource slice", "slice", klog.KObj(slice))
			if _, err := c.kubeClient.ResourceV1alpha3().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{}); err != nil {
				return fmt.Errorf("create resource slice: %w", err)
			}
			continue
		}

		// TODO: switch to SSA once unit testing supports it.
		logger.V(5).Info("Updating existing resource slice", "slice", klog.KObj(slice))
		if _, err := c.kubeClient.ResourceV1alpha3().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("delete resource slice: %w", err)
		}
	}

	return nil
}
