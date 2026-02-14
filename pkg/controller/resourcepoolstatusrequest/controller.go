/*
Copyright The Kubernetes Authors.

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

package resourcepoolstatusrequest

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	resourcev1 "k8s.io/api/resource/v1"
	resourcev1alpha1 "k8s.io/api/resource/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	resourcev1listers "k8s.io/client-go/listers/resource/v1"
	resourcev1alpha1listers "k8s.io/client-go/listers/resource/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/resourcepoolstatusrequest/metrics"
)

const (
	// ControllerName is the name of this controller.
	ControllerName = "resourcepoolstatusrequest-controller"

	// maxRetries is the number of times a request will be retried before it is dropped.
	maxRetries = 5
)

// Controller manages ResourcePoolStatusRequest processing.
type Controller struct {
	client clientset.Interface

	// requestLister can list/get ResourcePoolStatusRequests from the shared informer's store
	requestLister resourcev1alpha1listers.ResourcePoolStatusRequestLister

	// sliceLister can list/get ResourceSlices from the shared informer's store
	sliceLister resourcev1listers.ResourceSliceLister

	// claimLister can list/get ResourceClaims from the shared informer's store
	claimLister resourcev1listers.ResourceClaimLister

	// requestSynced returns true if the ResourcePoolStatusRequest store has been synced
	requestSynced cache.InformerSynced

	// sliceSynced returns true if the ResourceSlice store has been synced
	sliceSynced cache.InformerSynced

	// claimSynced returns true if the ResourceClaim store has been synced
	claimSynced cache.InformerSynced

	// workqueue is a rate limited work queue for processing ResourcePoolStatusRequests
	workqueue workqueue.TypedRateLimitingInterface[string]

	// poolDataMu protects poolData and allocationData
	poolDataMu sync.RWMutex

	// poolData caches aggregated pool data from ResourceSlices
	// key is "driver/poolName"
	poolData map[string]*poolInfo

	// allocationData caches allocated device counts by pool
	// key is "driver/poolName"
	allocationData map[string]int32
}

// poolInfo aggregates data for a single pool across all its ResourceSlices
type poolInfo struct {
	driver       string
	poolName     string
	nodeName     string
	totalDevices int32
	sliceCount   int32
	generation   int64
}

// NewController creates a new ResourcePoolStatusRequest controller.
func NewController(
	ctx context.Context,
	client clientset.Interface,
	informerFactory informers.SharedInformerFactory,
) (*Controller, error) {
	logger := klog.FromContext(ctx)

	requestInformer := informerFactory.Resource().V1alpha1().ResourcePoolStatusRequests()
	sliceInformer := informerFactory.Resource().V1().ResourceSlices()
	claimInformer := informerFactory.Resource().V1().ResourceClaims()

	c := &Controller{
		client:         client,
		requestLister:  requestInformer.Lister(),
		sliceLister:    sliceInformer.Lister(),
		claimLister:    claimInformer.Lister(),
		requestSynced:  requestInformer.Informer().HasSynced,
		sliceSynced:    sliceInformer.Informer().HasSynced,
		claimSynced:    claimInformer.Informer().HasSynced,
		workqueue:      workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
		poolData:       make(map[string]*poolInfo),
		allocationData: make(map[string]int32),
	}

	// Register metrics
	metrics.Register()

	// Set up event handlers for ResourcePoolStatusRequests
	_, err := requestInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueueRequest(logger, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			// Only re-queue if the spec changed or status is not yet set
			oldReq := old.(*resourcev1alpha1.ResourcePoolStatusRequest)
			newReq := new.(*resourcev1alpha1.ResourcePoolStatusRequest)
			if newReq.Status.ObservationTime == nil {
				c.enqueueRequest(logger, new)
			} else if oldReq.ResourceVersion != newReq.ResourceVersion {
				// Status was already set, but resource version changed
				// This could be a retry after conflict, re-queue if still incomplete
				if newReq.Status.ObservationTime == nil {
					c.enqueueRequest(logger, new)
				}
			}
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add request event handler: %w", err)
	}

	// Set up event handlers for ResourceSlices to rebuild cache
	_, err = sliceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.handleSliceChange(logger, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			c.handleSliceChange(logger, new)
		},
		DeleteFunc: func(obj interface{}) {
			c.handleSliceDelete(logger, obj)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add slice event handler: %w", err)
	}

	// Set up event handlers for ResourceClaims to track allocations
	_, err = claimInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.handleClaimChange(logger)
		},
		UpdateFunc: func(old, new interface{}) {
			c.handleClaimChange(logger)
		},
		DeleteFunc: func(obj interface{}) {
			c.handleClaimChange(logger)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add claim event handler: %w", err)
	}

	logger.Info("ResourcePoolStatusRequest controller initialized")
	return c, nil
}

// Run starts the controller workers.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.workqueue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting ResourcePoolStatusRequest controller")

	// Wait for the caches to be synced before starting workers
	logger.Info("Waiting for informer caches to sync")
	if !cache.WaitForCacheSync(ctx.Done(), c.requestSynced, c.sliceSynced, c.claimSynced) {
		logger.Error(nil, "Failed to wait for caches to sync")
		return
	}

	// Build initial cache data
	c.rebuildPoolCache(logger)
	c.rebuildAllocationCache(logger)

	logger.Info("Starting workers", "count", workers)
	for range workers {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
	logger.Info("Shutting down ResourcePoolStatusRequest controller")
}

// runWorker is a long-running function that will continually call the
// processNextWorkItem function in order to read and process a message on the workqueue.
func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem will read a single work item off the workqueue and
// attempt to process it.
func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := c.workqueue.Get()
	if shutdown {
		return false
	}

	defer c.workqueue.Done(key)

	logger := klog.FromContext(ctx)
	err := c.syncRequest(ctx, key)
	if err == nil {
		c.workqueue.Forget(key)
		return true
	}

	if c.workqueue.NumRequeues(key) < maxRetries {
		logger.Error(err, "Error syncing request, requeuing", "request", key)
		c.workqueue.AddRateLimited(key)
		return true
	}

	logger.Error(err, "Dropping request out of the queue after max retries", "request", key, "retries", maxRetries)
	c.workqueue.Forget(key)
	utilruntime.HandleError(err)

	return true
}

// syncRequest processes a single ResourcePoolStatusRequest.
func (c *Controller) syncRequest(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	startTime := time.Now()

	defer func() {
		metrics.RequestProcessingDuration.Observe(time.Since(startTime).Seconds())
	}()

	// Get the request
	request, err := c.requestLister.Get(key)
	if err != nil {
		// Request was deleted, nothing to do
		return nil
	}

	// Skip if already processed (status.observationTime is set)
	if request.Status.ObservationTime != nil {
		logger.V(4).Info("Request already processed, skipping", "request", key)
		return nil
	}

	logger.V(2).Info("Processing ResourcePoolStatusRequest", "request", key, "driver", request.Spec.Driver)

	// Calculate the pool status
	status := c.calculatePoolStatus(ctx, request)

	// Update the request status
	requestCopy := request.DeepCopy()
	requestCopy.Status = status

	_, err = c.client.ResourceV1alpha1().ResourcePoolStatusRequests().UpdateStatus(ctx, requestCopy, metav1.UpdateOptions{})
	if err != nil {
		metrics.RequestProcessingErrors.Inc()
		return fmt.Errorf("failed to update status for request %s: %w", key, err)
	}

	metrics.RequestsProcessed.Inc()
	logger.V(2).Info("Successfully processed ResourcePoolStatusRequest", "request", key, "poolCount", len(status.Pools))
	return nil
}

// calculatePoolStatus computes the pool status based on current state.
func (c *Controller) calculatePoolStatus(ctx context.Context, request *resourcev1alpha1.ResourcePoolStatusRequest) resourcev1alpha1.ResourcePoolStatusRequestStatus {
	logger := klog.FromContext(ctx)

	c.poolDataMu.RLock()
	defer c.poolDataMu.RUnlock()

	var pools []resourcev1alpha1.PoolStatus
	driver := request.Spec.Driver
	poolNameFilter := request.Spec.PoolName

	// Find matching pools
	for key, info := range c.poolData {
		if info.driver != driver {
			continue
		}
		if poolNameFilter != "" && info.poolName != poolNameFilter {
			continue
		}

		allocatedDevices := c.allocationData[key]
		availableDevices := max(0, info.totalDevices-allocatedDevices)

		pools = append(pools, resourcev1alpha1.PoolStatus{
			Driver:           info.driver,
			PoolName:         info.poolName,
			NodeName:         info.nodeName,
			TotalDevices:     info.totalDevices,
			AllocatedDevices: allocatedDevices,
			AvailableDevices: availableDevices,
			SliceCount:       info.sliceCount,
			Generation:       info.generation,
		})
	}

	// Sort pools by driver, then pool name
	sort.Slice(pools, func(i, j int) bool {
		if pools[i].Driver != pools[j].Driver {
			return pools[i].Driver < pools[j].Driver
		}
		return pools[i].PoolName < pools[j].PoolName
	})

	totalMatchingPools := int32(len(pools))
	truncated := false

	// Apply limit if specified
	limit := int32(100) // default
	if request.Spec.Limit != nil {
		limit = *request.Spec.Limit
	}
	if int32(len(pools)) > limit {
		pools = pools[:limit]
		truncated = true
	}

	now := metav1.Now()
	status := resourcev1alpha1.ResourcePoolStatusRequestStatus{
		ObservationTime:    &now,
		Pools:              pools,
		TotalMatchingPools: totalMatchingPools,
		Truncated:          truncated,
		Conditions: []metav1.Condition{
			{
				Type:               resourcev1alpha1.ResourcePoolStatusRequestConditionComplete,
				Status:             metav1.ConditionTrue,
				LastTransitionTime: now,
				Reason:             "Calculated",
				Message:            fmt.Sprintf("Successfully calculated status for %d pools", len(pools)),
			},
		},
	}

	logger.V(4).Info("Calculated pool status",
		"request", request.Name,
		"totalMatching", totalMatchingPools,
		"returned", len(pools),
		"truncated", truncated)

	return status
}

// enqueueRequest adds a request to the workqueue.
func (c *Controller) enqueueRequest(logger klog.Logger, obj interface{}) {
	request, ok := obj.(*resourcev1alpha1.ResourcePoolStatusRequest)
	if !ok {
		logger.Error(nil, "Failed to cast object to ResourcePoolStatusRequest")
		return
	}

	// Skip if already processed
	if request.Status.ObservationTime != nil {
		return
	}

	c.workqueue.Add(request.Name)
}

// handleSliceChange updates the pool cache when a ResourceSlice changes.
func (c *Controller) handleSliceChange(logger klog.Logger, obj interface{}) {
	slice, ok := obj.(*resourcev1.ResourceSlice)
	if !ok {
		logger.Error(nil, "Failed to cast object to ResourceSlice")
		return
	}

	c.updatePoolData(slice)
}

// handleSliceDelete handles ResourceSlice deletion.
func (c *Controller) handleSliceDelete(logger klog.Logger, obj interface{}) {
	// On delete, rebuild the entire cache to be safe
	c.rebuildPoolCache(logger)
}

// handleClaimChange rebuilds the allocation cache when claims change.
func (c *Controller) handleClaimChange(logger klog.Logger) {
	c.rebuildAllocationCache(logger)
}

// updatePoolData updates the pool cache for a single slice.
func (c *Controller) updatePoolData(slice *resourcev1.ResourceSlice) {
	driver := slice.Spec.Driver
	poolName := slice.Spec.Pool.Name
	key := driver + "/" + poolName

	c.poolDataMu.Lock()
	defer c.poolDataMu.Unlock()

	// Count devices in this slice
	deviceCount := int32(len(slice.Spec.Devices))

	info, exists := c.poolData[key]
	if !exists {
		var nodeName string
		if slice.Spec.NodeName != nil {
			nodeName = *slice.Spec.NodeName
		}
		c.poolData[key] = &poolInfo{
			driver:       driver,
			poolName:     poolName,
			nodeName:     nodeName,
			totalDevices: deviceCount,
			sliceCount:   1,
			generation:   slice.Spec.Pool.Generation,
		}
	} else if slice.Spec.Pool.Generation > info.generation {
		// Update existing pool info generation if newer
		info.generation = slice.Spec.Pool.Generation
	}
}

// rebuildPoolCache rebuilds the entire pool cache from scratch.
func (c *Controller) rebuildPoolCache(logger klog.Logger) {
	slices, err := c.sliceLister.List(labels.Everything())
	if err != nil {
		logger.Error(err, "Failed to list ResourceSlices for cache rebuild")
		return
	}

	newPoolData := make(map[string]*poolInfo)

	for _, slice := range slices {
		driver := slice.Spec.Driver
		poolName := slice.Spec.Pool.Name
		key := driver + "/" + poolName

		deviceCount := int32(len(slice.Spec.Devices))

		info, exists := newPoolData[key]
		if !exists {
			var nodeName string
			if slice.Spec.NodeName != nil {
				nodeName = *slice.Spec.NodeName
			}
			newPoolData[key] = &poolInfo{
				driver:       driver,
				poolName:     poolName,
				nodeName:     nodeName,
				totalDevices: deviceCount,
				sliceCount:   1,
				generation:   slice.Spec.Pool.Generation,
			}
		} else {
			info.totalDevices += deviceCount
			info.sliceCount++
			if slice.Spec.Pool.Generation > info.generation {
				info.generation = slice.Spec.Pool.Generation
			}
		}
	}

	c.poolDataMu.Lock()
	c.poolData = newPoolData
	c.poolDataMu.Unlock()

	logger.V(4).Info("Rebuilt pool cache", "poolCount", len(newPoolData))
}

// rebuildAllocationCache rebuilds the allocation counts from ResourceClaims.
func (c *Controller) rebuildAllocationCache(logger klog.Logger) {
	claims, err := c.claimLister.List(labels.Everything())
	if err != nil {
		logger.Error(err, "Failed to list ResourceClaims for allocation cache rebuild")
		return
	}

	newAllocationData := make(map[string]int32)

	for _, claim := range claims {
		if claim.Status.Allocation == nil {
			continue
		}

		for _, result := range claim.Status.Allocation.Devices.Results {
			key := result.Driver + "/" + result.Pool
			newAllocationData[key]++
		}
	}

	c.poolDataMu.Lock()
	c.allocationData = newAllocationData
	c.poolDataMu.Unlock()

	logger.V(4).Info("Rebuilt allocation cache", "allocationCount", len(newAllocationData))
}
