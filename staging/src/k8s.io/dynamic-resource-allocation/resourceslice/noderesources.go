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

	resourceapi "k8s.io/api/resource/v1alpha2"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	resourceinformers "k8s.io/client-go/informers/resource/v1alpha2"
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

// Controller synchronizes information about resources of one
// driver with ResourceSlice objects. It currently supports node-local
// resources. A DRA driver for node-local resources typically runs this
// controller as part of its kubelet plugin.
//
// Support for network-attached resources will be added later.
type Controller struct {
	cancel     func(cause error)
	driverName string
	owner      Owner
	kubeClient kubernetes.Interface
	wg         sync.WaitGroup
	queue      workqueue.TypedRateLimitingInterface[string]
	sliceStore cache.Store

	mutex sync.RWMutex

	// When receiving updates from the driver, the entire pointer replaced,
	// so it is okay to not do a deep copy of it when reading it. Only reading
	// the pointer itself must be protected by a read lock.
	resources *Resources
}

// Resources is a complete description of all resources synchronized by the controller.
type Resources struct {
	// NodeResources are resources that are local to one node.
	NodeResources []*resourceapi.ResourceModel
}

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
func StartController(ctx context.Context, kubeClient kubernetes.Interface, driverName string, owner Owner, resources *Resources) *Controller {
	if kubeClient == nil {
		return nil
	}

	logger := klog.FromContext(ctx)
	ctx, cancel := context.WithCancelCause(ctx)

	c := &Controller{
		cancel:     cancel,
		kubeClient: kubeClient,
		driverName: driverName,
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

	// Sync once.
	c.queue.Add("")

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
func (c *Controller) Update(resources *Resources) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.resources = resources
	c.queue.Add("")
}

// run is running in the background. It handles blocking initialization (like
// syncing the informer) and then syncs the actual with the desired state.
func (c *Controller) run(ctx context.Context) {
	logger := klog.FromContext(ctx)

	// We always filter by driver name, by node name only for node-local resources.
	selector := fields.Set{"driverName": c.driverName}
	if c.owner.APIVersion == "v1" && c.owner.Kind == "Node" {
		selector["nodeName"] = c.owner.Name
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
			c.queue.Add("")
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
			c.queue.Add("")
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
			c.queue.Add("")
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

	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := c.queue.Get()
	if shutdown {
		return false
	}
	defer c.queue.Done(key)

	// Panics are caught and treated like errors.
	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("internal error: %v", r)
			}
		}()
		err = c.sync(ctx)
	}()

	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "processing ResourceSlice objects")
		c.queue.AddRateLimited(key)

		// Return without removing the work item from the queue.
		// It will be retried.
		return true
	}

	c.queue.Forget(key)
	return true
}

func (c *Controller) sync(ctx context.Context) error {
	logger := klog.FromContext(ctx)

	// Gather information about the actual and desired state.
	slices := c.sliceStore.List()
	var resources *Resources
	c.mutex.RLock()
	resources = c.resources
	c.mutex.RUnlock()

	// Resources that are not yet stored in any slice need to be published.
	// Here we track the indices of any resources that are already stored.
	storedResourceIndices := sets.New[int]()

	// Slices that don't match any driver resource can either be updated (if there
	// are new driver resources that need to be stored) or they need to be deleted.
	obsoleteSlices := make([]*resourceapi.ResourceSlice, 0, len(slices))

	// Match slices with resource information.
	for _, obj := range slices {
		slice := obj.(*resourceapi.ResourceSlice)

		// TODO: network-attached resources.
		index := indexOfModel(resources.NodeResources, &slice.ResourceModel)
		if index >= 0 {
			storedResourceIndices.Insert(index)
			continue
		}

		obsoleteSlices = append(obsoleteSlices, slice)
	}

	if loggerV := logger.V(6); loggerV.Enabled() {
		// Dump entire resource information.
		loggerV.Info("Syncing existing driver resource slices with driver resources", "slices", klog.KObjSlice(slices), "resources", resources)
	} else {
		logger.V(5).Info("Syncing existing driver resource slices with driver resources", "slices", klog.KObjSlice(slices), "numResources", len(resources.NodeResources))
	}

	// Retrieve node object to get UID?
	// The result gets cached and is expected to not change while
	// the controller runs.
	if c.owner.UID == "" && c.owner.APIVersion == "v1" && c.owner.Kind == "Node" {
		node, err := c.kubeClient.CoreV1().Nodes().Get(ctx, c.owner.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("retrieve node %q: %w", c.owner.Name, err)
		}
		// There is only one worker, so no locking needed.
		c.owner.UID = node.UID
	}

	// Update stale slices before removing what's left.
	//
	// We don't really know which of these slices might have
	// been used for "the" driver resource because they don't
	// have a unique ID. In practice, a driver is most likely
	// to just give us one ResourceModel, in which case
	// this isn't a problem at all. If we have more than one,
	// then at least conceptually it currently doesn't matter
	// where we publish it.
	//
	// The long-term goal is to move the handling of
	// ResourceSlice objects into the driver, with kubelet
	// just acting as a REST proxy. The advantage of that will
	// be that kubelet won't need to support the same
	// resource API version as the driver and the control plane.
	// With that approach, the driver will be able to match
	// up objects more intelligently.
	numObsoleteSlices := len(obsoleteSlices)
	for index, resource := range resources.NodeResources {
		if storedResourceIndices.Has(index) {
			// No need to do anything, it is already stored exactly
			// like this in an existing slice.
			continue
		}

		if numObsoleteSlices > 0 {
			// Update one existing slice.
			slice := obsoleteSlices[numObsoleteSlices-1]
			numObsoleteSlices--
			slice = slice.DeepCopy()
			slice.ResourceModel = *resource
			logger.V(5).Info("Reusing existing resource slice", "slice", klog.KObj(slice))
			if _, err := c.kubeClient.ResourceV1alpha2().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{}); err != nil {
				return fmt.Errorf("update resource slice: %w", err)
			}
			continue
		}

		// Create a new slice.
		slice := &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: c.owner.Name + "-" + c.driverName + "-",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: c.owner.APIVersion,
						Kind:       c.owner.Kind,
						Name:       c.owner.Name,
						UID:        c.owner.UID,
						Controller: ptr.To(true),
					},
				},
			},
			DriverName:    c.driverName,
			ResourceModel: *resource,
		}
		if c.owner.APIVersion == "v1" && c.owner.Kind == "Node" {
			slice.NodeName = c.owner.Name
		}
		logger.V(5).Info("Creating new resource slice", "slice", klog.KObj(slice))
		if _, err := c.kubeClient.ResourceV1alpha2().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{}); err != nil {
			return fmt.Errorf("create resource slice: %w", err)
		}
	}

	// All remaining slices are truly orphaned.
	for i := 0; i < numObsoleteSlices; i++ {
		slice := obsoleteSlices[i]
		logger.V(5).Info("Deleting obsolete resource slice", "slice", klog.KObj(slice))
		if err := c.kubeClient.ResourceV1alpha2().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{}); err != nil {
			return fmt.Errorf("delete resource slice: %w", err)
		}
	}

	return nil
}

func indexOfModel(models []*resourceapi.ResourceModel, model *resourceapi.ResourceModel) int {
	for index, m := range models {
		if apiequality.Semantic.DeepEqual(m, model) {
			return index
		}
	}
	return -1
}
