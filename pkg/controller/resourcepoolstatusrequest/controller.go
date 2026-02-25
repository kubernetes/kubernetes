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
	"time"

	resourcev1alpha1 "k8s.io/api/resource/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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

	// cleanupPollingInterval is how often the controller checks for expired requests.
	cleanupPollingInterval = 10 * time.Minute

	// completedRequestTTL is how long a completed request (with status.observationTime set)
	// is retained before being automatically deleted.
	completedRequestTTL = 1 * time.Hour

	// pendingRequestTTL is how long a pending request (without status.observationTime)
	// is retained before being automatically deleted. This handles stuck requests.
	pendingRequestTTL = 24 * time.Hour
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
		client:        client,
		requestLister: requestInformer.Lister(),
		sliceLister:   sliceInformer.Lister(),
		claimLister:   claimInformer.Lister(),
		requestSynced: requestInformer.Informer().HasSynced,
		sliceSynced:   sliceInformer.Informer().HasSynced,
		claimSynced:   claimInformer.Informer().HasSynced,
		workqueue:     workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
	}

	// Register metrics
	metrics.Register()

	// Set up event handlers for ResourcePoolStatusRequests
	_, err := requestInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueueRequest(logger, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			newReq := new.(*resourcev1alpha1.ResourcePoolStatusRequest)
			if newReq.Status.ObservationTime == nil {
				c.enqueueRequest(logger, new)
			}
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add request event handler: %w", err)
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

	logger.Info("Starting workers", "count", workers)
	for range workers {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	// Start the cleanup goroutine for TTL-based garbage collection
	go wait.UntilWithContext(ctx, c.cleanupExpiredRequests, cleanupPollingInterval)

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

	// Calculate the pool status on-demand from listers
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

// calculatePoolStatus computes the pool status on-demand by reading directly
// from the shared informer listers. No caches are maintained between requests.
func (c *Controller) calculatePoolStatus(ctx context.Context, request *resourcev1alpha1.ResourcePoolStatusRequest) resourcev1alpha1.ResourcePoolStatusRequestStatus {
	logger := klog.FromContext(ctx)

	driver := request.Spec.Driver
	var poolNameFilter string
	if request.Spec.PoolName != nil {
		poolNameFilter = *request.Spec.PoolName
	}

	// Step 1: Aggregate pool data from ResourceSlices.
	// Uses two passes: first finds the max generation per pool, then only
	// counts slices at that generation (older-generation slices are ignored
	// per KEP-5677).
	type poolInfo struct {
		driver       string
		poolName     string
		nodeName     string
		totalDevices int32
		sliceCount   int32
		generation   int64
	}

	slices, err := c.sliceLister.List(labels.Everything())
	if err != nil {
		logger.Error(err, "Failed to list ResourceSlices")
		return errorStatus("Failed to list ResourceSlices: " + err.Error())
	}

	// Pass 1: Find max generation per pool
	maxGeneration := make(map[string]int64)
	for _, slice := range slices {
		if slice.Spec.Driver != driver {
			continue
		}
		slicePoolName := slice.Spec.Pool.Name
		if poolNameFilter != "" && slicePoolName != poolNameFilter {
			continue
		}
		key := slice.Spec.Driver + "/" + slicePoolName
		if gen, exists := maxGeneration[key]; !exists || slice.Spec.Pool.Generation > gen {
			maxGeneration[key] = slice.Spec.Pool.Generation
		}
	}

	// Pass 2: Aggregate only slices at max generation
	poolData := make(map[string]*poolInfo)
	for _, slice := range slices {
		if slice.Spec.Driver != driver {
			continue
		}
		slicePoolName := slice.Spec.Pool.Name
		if poolNameFilter != "" && slicePoolName != poolNameFilter {
			continue
		}

		key := slice.Spec.Driver + "/" + slicePoolName

		// Skip slices from older generations
		if slice.Spec.Pool.Generation < maxGeneration[key] {
			continue
		}

		deviceCount := int32(len(slice.Spec.Devices))
		info, exists := poolData[key]
		if !exists {
			var nodeName string
			if slice.Spec.NodeName != nil {
				nodeName = *slice.Spec.NodeName
			}
			poolData[key] = &poolInfo{
				driver:       slice.Spec.Driver,
				poolName:     slicePoolName,
				nodeName:     nodeName,
				totalDevices: deviceCount,
				sliceCount:   1,
				generation:   maxGeneration[key],
			}
		} else {
			info.totalDevices += deviceCount
			info.sliceCount++
		}
	}

	// Step 2: Count allocations from ResourceClaims
	allocationData := make(map[string]int32)

	claims, err := c.claimLister.List(labels.Everything())
	if err != nil {
		logger.Error(err, "Failed to list ResourceClaims")
		return errorStatus("Failed to list ResourceClaims: " + err.Error())
	}

	for _, claim := range claims {
		if claim.Status.Allocation == nil {
			continue
		}
		for _, result := range claim.Status.Allocation.Devices.Results {
			key := result.Driver + "/" + result.Pool
			allocationData[key]++
		}
	}

	// Step 3: Build pool status list
	var pools []resourcev1alpha1.PoolStatus
	for key, info := range poolData {
		allocatedDevices := allocationData[key]
		availableDevices := max(0, info.totalDevices-allocatedDevices)

		totalDevices := info.totalDevices
		allocDevices := allocatedDevices
		availDevices := availableDevices
		pool := resourcev1alpha1.PoolStatus{
			Driver:           info.driver,
			PoolName:         info.poolName,
			TotalDevices:     &totalDevices,
			AllocatedDevices: &allocDevices,
			AvailableDevices: &availDevices,
			SliceCount:       info.sliceCount,
			Generation:       info.generation,
		}
		if info.nodeName != "" {
			nodeName := info.nodeName
			pool.NodeName = &nodeName
		}
		pools = append(pools, pool)
	}

	// Sort pools by driver, then pool name
	sort.Slice(pools, func(i, j int) bool {
		if pools[i].Driver != pools[j].Driver {
			return pools[i].Driver < pools[j].Driver
		}
		return pools[i].PoolName < pools[j].PoolName
	})

	totalMatchingPools := int32(len(pools))
	var truncation *resourcev1alpha1.TruncationStatus

	// Apply limit if specified
	limit := int32(100) // default
	if request.Spec.Limit != nil {
		limit = *request.Spec.Limit
	}
	if int32(len(pools)) > limit {
		pools = pools[:limit]
		truncatedStatus := resourcev1alpha1.TruncationStatusTruncated
		truncation = &truncatedStatus
	}

	now := metav1.Now()
	status := resourcev1alpha1.ResourcePoolStatusRequestStatus{
		ObservationTime:    &now,
		Pools:              pools,
		TotalMatchingPools: &totalMatchingPools,
		Truncation:         truncation,
		Conditions: []metav1.Condition{
			{
				Type:               resourcev1alpha1.ResourcePoolStatusRequestConditionComplete,
				Status:             metav1.ConditionTrue,
				LastTransitionTime: now,
				Reason:             "CalculationComplete",
				Message:            fmt.Sprintf("Successfully calculated status for %d pools", len(pools)),
			},
		},
	}

	logger.V(4).Info("Calculated pool status",
		"request", request.Name,
		"totalMatching", totalMatchingPools,
		"returned", len(pools),
		"truncation", truncation)

	return status
}

// errorStatus returns a status indicating a processing failure.
func errorStatus(message string) resourcev1alpha1.ResourcePoolStatusRequestStatus {
	now := metav1.Now()
	return resourcev1alpha1.ResourcePoolStatusRequestStatus{
		ObservationTime: &now,
		Conditions: []metav1.Condition{
			{
				Type:               resourcev1alpha1.ResourcePoolStatusRequestConditionFailed,
				Status:             metav1.ConditionTrue,
				LastTransitionTime: now,
				Reason:             "CalculationFailed",
				Message:            message,
			},
		},
	}
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

// cleanupExpiredRequests deletes ResourcePoolStatusRequests that have exceeded their TTL.
// Completed requests (with observationTime set) are deleted after completedRequestTTL.
// Pending requests (without observationTime) are deleted after pendingRequestTTL.
func (c *Controller) cleanupExpiredRequests(ctx context.Context) {
	logger := klog.FromContext(ctx)

	requests, err := c.requestLister.List(labels.Everything())
	if err != nil {
		logger.Error(err, "Failed to list ResourcePoolStatusRequests for cleanup")
		return
	}

	var deletedCount int
	for _, request := range requests {
		if c.shouldDeleteRequest(request) {
			if err := c.client.ResourceV1alpha1().ResourcePoolStatusRequests().Delete(ctx, request.Name, metav1.DeleteOptions{}); err != nil {
				// Ignore NotFound errors - the request may have been deleted by another process
				if !apierrors.IsNotFound(err) {
					logger.Error(err, "Failed to delete expired ResourcePoolStatusRequest", "request", request.Name)
				}
				continue
			}
			deletedCount++
			logger.V(2).Info("Deleted expired ResourcePoolStatusRequest", "request", request.Name)
		}
	}

	if deletedCount > 0 {
		logger.Info("Cleanup completed", "deletedCount", deletedCount)
	}
}

// shouldDeleteRequest determines if a request should be deleted based on TTL.
func (c *Controller) shouldDeleteRequest(request *resourcev1alpha1.ResourcePoolStatusRequest) bool {
	if request.Status.ObservationTime != nil {
		// Completed request: check against completedRequestTTL
		return isOlderThan(request.Status.ObservationTime.Time, completedRequestTTL)
	}
	// Pending request: check against pendingRequestTTL based on creation time
	return isOlderThan(request.CreationTimestamp.Time, pendingRequestTTL)
}

// isOlderThan checks if the given time is older than the specified duration.
func isOlderThan(t time.Time, d time.Duration) bool {
	return !t.IsZero() && time.Since(t) > d
}
