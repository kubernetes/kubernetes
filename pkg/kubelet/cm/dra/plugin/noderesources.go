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

package plugin

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha2"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	resourceinformers "k8s.io/client-go/informers/resource/v1alpha2"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
	"k8s.io/utils/ptr"
)

const (
	// resyncPeriod for informer
	// TODO (https://github.com/kubernetes/kubernetes/issues/123688): disable?
	resyncPeriod = time.Duration(10 * time.Minute)
)

// nodeResourcesController collects resource information from all registered
// plugins and synchronizes that information with ResourceSlice objects.
type nodeResourcesController struct {
	ctx        context.Context
	kubeClient kubernetes.Interface
	getNode    func() (*v1.Node, error)
	wg         sync.WaitGroup
	queue      workqueue.TypedRateLimitingInterface[string]
	sliceStore cache.Store

	mutex         sync.RWMutex
	activePlugins map[string]*activePlugin
}

// activePlugin holds the resource information about one plugin
// and the gRPC stream that is used to retrieve that. The context
// used by that stream can be canceled separately to stop
// the monitoring.
type activePlugin struct {
	// cancel is the function which cancels the monitorPlugin goroutine
	// for this plugin.
	cancel func(reason error)

	// resources is protected by the nodeResourcesController read/write lock.
	// When receiving updates from the driver, the entire slice gets replaced,
	// so it is okay to not do a deep copy of it. Only retrieving the slice
	// must be protected by a read lock.
	resources []*resourceapi.ResourceModel
}

// startNodeResourcesController constructs a new controller and starts it.
//
// If a kubeClient is provided, then it synchronizes ResourceSlices
// with the resource information provided by plugins. Without it,
// the controller is inactive. This can happen when kubelet is run stand-alone
// without an apiserver. In that case we can't and don't need to publish
// ResourceSlices.
func startNodeResourcesController(ctx context.Context, kubeClient kubernetes.Interface, getNode func() (*v1.Node, error)) *nodeResourcesController {
	if kubeClient == nil {
		return nil
	}

	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithName(logger, "node resources controller")
	ctx = klog.NewContext(ctx, logger)

	c := &nodeResourcesController{
		ctx:        ctx,
		kubeClient: kubeClient,
		getNode:    getNode,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "node_resource_slices"},
		),
		activePlugins: make(map[string]*activePlugin),
	}

	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		c.run(ctx)
	}()

	return c
}

// waitForStop blocks until all background activity spawned by
// the controller has stopped. The context passed to start must
// be canceled for that to happen.
//
// Not needed at the moment, but if it was, this is what it would
// look like...
// func (c *nodeResourcesController) waitForStop() {
// 	if c == nil {
// 		return
// 	}
//
// 	c.wg.Wait()
// }

// addPlugin is called whenever a plugin has been (re-)registered.
func (c *nodeResourcesController) addPlugin(driverName string, pluginInstance *plugin) {
	if c == nil {
		return
	}

	klog.FromContext(c.ctx).V(2).Info("Adding plugin", "driverName", driverName)
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if active := c.activePlugins[driverName]; active != nil {
		active.cancel(errors.New("plugin has re-registered"))
	}
	active := &activePlugin{}
	cancelCtx, cancel := context.WithCancelCause(c.ctx)
	active.cancel = cancel
	c.activePlugins[driverName] = active
	c.queue.Add(driverName)

	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		c.monitorPlugin(cancelCtx, active, driverName, pluginInstance)
	}()
}

// removePlugin is called whenever a plugin has been unregistered.
func (c *nodeResourcesController) removePlugin(driverName string) {
	if c == nil {
		return
	}

	klog.FromContext(c.ctx).V(2).Info("Removing plugin", "driverName", driverName)
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if active, ok := c.activePlugins[driverName]; ok {
		active.cancel(errors.New("plugin has unregistered"))
		delete(c.activePlugins, driverName)
		c.queue.Add(driverName)
	}
}

// monitorPlugin calls the plugin to retrieve resource information and caches
// all responses that it gets for processing in the sync method. It keeps
// retrying until an error or EOF response indicates that no further data is
// going to be sent, then watch resources of the plugin stops until it
// re-registers.
func (c *nodeResourcesController) monitorPlugin(ctx context.Context, active *activePlugin, driverName string, pluginInstance *plugin) {
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "driverName", driverName)
	logger.Info("Starting to monitor node resources of the plugin")
	defer func() {
		r := recover()
		logger.Info("Stopping to monitor node resources of the plugin", "reason", context.Cause(ctx), "err", ctx.Err(), "recover", r)
	}()

	// Keep trying until canceled.
	for ctx.Err() == nil {
		logger.V(5).Info("Calling NodeListAndWatchResources")
		stream, err := pluginInstance.NodeListAndWatchResources(ctx, new(drapb.NodeListAndWatchResourcesRequest))
		if err != nil {
			switch {
			case status.Convert(err).Code() == codes.Unimplemented:
				// The plugin simply doesn't provide node resources.
				active.cancel(errors.New("plugin does not support node resource reporting"))
			default:
				// This is a problem, report it and retry.
				logger.Error(err, "Creating gRPC stream for node resources failed")
				// TODO (https://github.com/kubernetes/kubernetes/issues/123689): expontential backoff?
				select {
				case <-time.After(5 * time.Second):
				case <-ctx.Done():
				}
			}
			continue
		}
		for {
			response, err := stream.Recv()
			if err != nil {
				switch {
				case errors.Is(err, io.EOF):
					// This is okay. Some plugins might never change their
					// resources after reporting them once.
					active.cancel(errors.New("plugin has closed the stream"))
				case status.Convert(err).Code() == codes.Unimplemented:
					// The plugin has the method, does not really implement it.
					active.cancel(errors.New("plugin does not support node resource reporting"))
				case ctx.Err() == nil:
					// This is a problem, report it and retry.
					logger.Error(err, "Reading node resources from gRPC stream failed")
					// TODO (https://github.com/kubernetes/kubernetes/issues/123689): expontential backoff?
					select {
					case <-time.After(5 * time.Second):
					case <-ctx.Done():
					}
				}
				break
			}

			if loggerV := logger.V(6); loggerV.Enabled() {
				loggerV.Info("Driver resources updated", "resources", response.Resources)
			} else {
				logger.V(5).Info("Driver resources updated", "numResources", len(response.Resources))
			}

			c.mutex.Lock()
			active.resources = response.Resources
			c.mutex.Unlock()
			c.queue.Add(driverName)
		}
	}
}

// run is running in the background. It handles blocking initialization (like
// syncing the informer) and then syncs the actual with the desired state.
func (c *nodeResourcesController) run(ctx context.Context) {
	logger := klog.FromContext(ctx)

	// When kubelet starts, we have two choices:
	// - Sync immediately, which in practice will delete all ResourceSlices
	//   because no plugin has registered yet. We could do a DeleteCollection
	//   to speed this up.
	// - Wait a bit, then sync. If all plugins have re-registered in the meantime,
	//   we might not need to change any ResourceSlice.
	//
	// For now syncing starts immediately, with no DeleteCollection. This
	// can be reconsidered later.

	// Wait until we're able to get a Node object.
	// This means that the object is created on the API server,
	// the kubeclient is functional and the node informer cache is populated with the node object.
	// Without this it doesn't make sense to proceed further as we need a node name and
	// a node UID for this controller to work.
	var node *v1.Node
	var err error
	for {
		node, err = c.getNode()
		if err == nil {
			break
		}
		logger.V(5).Info("Getting Node object failed, waiting", "err", err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Second):
		}
	}

	// We could use an indexer on driver name, but that seems overkill.
	informer := resourceinformers.NewFilteredResourceSliceInformer(c.kubeClient, resyncPeriod, nil, func(options *metav1.ListOptions) {
		options.FieldSelector = "nodeName=" + node.Name
	})
	c.sliceStore = informer.GetStore()
	handler, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlice add", "slice", klog.KObj(slice))
			c.queue.Add(slice.DriverName)
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
			c.queue.Add(newSlice.DriverName)
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
			c.queue.Add(slice.DriverName)
		},
	})
	if err != nil {
		logger.Error(err, "Registering event handler on the ResourceSlice informer failed, disabling resource monitoring")
		return
	}

	// Start informer and wait for our cache to be populated.
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		informer.Run(ctx.Done())
	}()
	for !handler.HasSynced() {
		select {
		case <-time.After(time.Second):
		case <-ctx.Done():
			return
		}
	}
	logger.Info("ResourceSlice informer has synced")

	for c.processNextWorkItem(ctx) {
	}
}

func (c *nodeResourcesController) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := c.queue.Get()
	if shutdown {
		return false
	}
	defer c.queue.Done(key)

	driverName := key

	// Panics are caught and treated like errors.
	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("internal error: %v", r)
			}
		}()
		err = c.sync(ctx, driverName)
	}()

	if err != nil {
		// TODO (https://github.com/kubernetes/enhancements/issues/3077): contextual logging in utilruntime
		utilruntime.HandleError(fmt.Errorf("processing driver %v: %v", driverName, err))
		c.queue.AddRateLimited(key)

		// Return without removing the work item from the queue.
		// It will be retried.
		return true
	}

	c.queue.Forget(key)
	return true
}

func (c *nodeResourcesController) sync(ctx context.Context, driverName string) error {
	logger := klog.FromContext(ctx)

	// Gather information about the actual and desired state.
	slices := c.sliceStore.List()
	var driverResources []*resourceapi.ResourceModel
	c.mutex.RLock()
	if active, ok := c.activePlugins[driverName]; ok {
		// No need for a deep copy, the entire slice gets replaced on writes.
		driverResources = active.resources
	}
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
		if slice.DriverName != driverName {
			continue
		}

		index := indexOfModel(driverResources, &slice.ResourceModel)
		if index >= 0 {
			storedResourceIndices.Insert(index)
			continue
		}

		obsoleteSlices = append(obsoleteSlices, slice)
	}

	if loggerV := logger.V(6); loggerV.Enabled() {
		// Dump entire resource information.
		loggerV.Info("Syncing existing driver node resource slices with driver resources", "slices", klog.KObjSlice(slices), "resources", driverResources)
	} else {
		logger.V(5).Info("Syncing existing driver node resource slices with driver resources", "slices", klog.KObjSlice(slices), "numResources", len(driverResources))
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
	for index, resource := range driverResources {
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
			logger.V(5).Info("Reusing existing node resource slice", "slice", klog.KObj(slice))
			if _, err := c.kubeClient.ResourceV1alpha2().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{}); err != nil {
				return fmt.Errorf("update node resource slice: %w", err)
			}
			continue
		}

		// Although node name and UID are unlikely to change
		// we're getting updated node object just to be on the safe side.
		// It's a cheap operation as it gets an object from the node informer cache.
		node, err := c.getNode()
		if err != nil {
			return fmt.Errorf("retrieve node object: %w", err)
		}

		// Create a new slice.
		slice := &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: node.Name + "-" + driverName + "-",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: v1.SchemeGroupVersion.WithKind("Node").Version,
						Kind:       v1.SchemeGroupVersion.WithKind("Node").Kind,
						Name:       node.Name,
						UID:        node.UID,
						Controller: ptr.To(true),
					},
				},
			},
			NodeName:      node.Name,
			DriverName:    driverName,
			ResourceModel: *resource,
		}
		logger.V(5).Info("Creating new node resource slice", "slice", klog.KObj(slice))
		if _, err := c.kubeClient.ResourceV1alpha2().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{}); err != nil {
			return fmt.Errorf("create node resource slice: %w", err)
		}
	}

	// All remaining slices are truly orphaned.
	for i := 0; i < numObsoleteSlices; i++ {
		slice := obsoleteSlices[i]
		logger.V(5).Info("Deleting obsolete node resource slice", "slice", klog.KObj(slice))
		if err := c.kubeClient.ResourceV1alpha2().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{}); err != nil {
			return fmt.Errorf("delete node resource slice: %w", err)
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
