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

package endpointslice

import (
	"fmt"
	"time"

	"golang.org/x/time/rate"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	discoveryinformers "k8s.io/client-go/informers/discovery/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	discoverylisters "k8s.io/client-go/listers/discovery/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	mirroringmetrics "k8s.io/kubernetes/pkg/controller/endpointslicemirroring/metrics"
)

const (
	// maxRetries is the number of times a service will be retried before it is
	// dropped out of the queue. Any sync error, such as a failure to create or
	// update an EndpointSlice could trigger a retry. With the current
	// rate-limiter in use (1s*2^(numRetries-1)) the following numbers represent
	// the sequence of delays between successive queuings of a service.
	//
	// 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 512s, 1000s (max)
	maxRetries = 15

	// endpointSliceChangeMinSyncDelay indicates the mininum delay before
	// queuing a syncService call after an EndpointSlice changes. If
	// endpointUpdatesBatchPeriod is greater than this value, it will be used
	// instead. This helps batch processing of changes to multiple
	// EndpointSlices.
	endpointSliceChangeMinSyncDelay = 1 * time.Second

	// defaultSyncBackOff is the default backoff period for syncService calls.
	defaultSyncBackOff = 1 * time.Second
	// maxSyncBackOff is the max backoff period for syncService calls.
	maxSyncBackOff = 100 * time.Second

	// controllerName is a unique value used with LabelManagedBy to indicated
	// the component managing an EndpointSlice.
	controllerName = "endpointslicemirroring-controller.k8s.io"
)

// NewController creates and initializes a new Controller
func NewController(endpointsInformer coreinformers.EndpointsInformer,
	endpointSliceInformer discoveryinformers.EndpointSliceInformer,
	maxEndpointsPerSlice int32,
	client clientset.Interface,
	endpointUpdatesBatchPeriod time.Duration,
) *Controller {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(klog.Infof)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "endpoint-slice-mirroring-controller"})

	if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("endpoint_slice_mirroring_controller", client.DiscoveryV1beta1().RESTClient().GetRateLimiter())
	}

	mirroringmetrics.RegisterMetrics()

	c := &Controller{
		client: client,
		// This is similar to the DefaultControllerRateLimiter, just with a
		// significantly higher default backoff (1s vs 5ms). This controller
		// processes events that can require significant EndpointSlice changes,
		// such as an update to a Service or Deployment. A more significant
		// rate limit back off here helps ensure that the Controller does not
		// overwhelm the API Server.
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.NewMaxOfRateLimiter(
			workqueue.NewItemExponentialFailureRateLimiter(defaultSyncBackOff, maxSyncBackOff),
			// 10 qps, 100 bucket size. This is only for retry speed and its
			// only the overall factor (not per item).
			&workqueue.BucketRateLimiter{Limiter: rate.NewLimiter(rate.Limit(10), 100)},
		), "endpoint_slice_mirroring"),
		workerLoopPeriod: time.Second,
	}

	endpointsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: c.onEndpointsUpdate,
		UpdateFunc: func(old, cur interface{}) {
			c.onEndpointsUpdate(cur)
		},
		DeleteFunc: c.onEndpointsUpdate,
	})
	c.endpointsLister = endpointsInformer.Lister()
	c.endpointsSynced = endpointsInformer.Informer().HasSynced

	endpointSliceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.onEndpointSliceAdd,
		UpdateFunc: c.onEndpointSliceUpdate,
		DeleteFunc: c.onEndpointSliceDelete,
	})

	c.endpointSliceLister = endpointSliceInformer.Lister()
	c.endpointSlicesSynced = endpointSliceInformer.Informer().HasSynced
	c.endpointSliceTracker = newEndpointSliceTracker()

	c.maxEndpointsPerSlice = maxEndpointsPerSlice

	c.reconciler = &reconciler{
		client:               c.client,
		maxEndpointsPerSlice: c.maxEndpointsPerSlice,
		endpointSliceTracker: c.endpointSliceTracker,
		metricsCache:         mirroringmetrics.NewCache(maxEndpointsPerSlice),
	}

	c.eventBroadcaster = broadcaster
	c.eventRecorder = recorder

	c.endpointUpdatesBatchPeriod = endpointUpdatesBatchPeriod

	return c
}

// Controller manages selector-based service endpoint slices
type Controller struct {
	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	// endpointsLister is able to list/get endpoints and is populated by the
	// shared informer passed to NewController
	endpointsLister corelisters.EndpointsLister
	// endpointsSynced returns true if the endpoints shared informer has been
	// synced at least once. Added as a member to the struct to allow injection
	// for testing.
	endpointsSynced cache.InformerSynced

	// endpointSliceLister is able to list/get endpoint slices and is populated
	// by the shared informer passed to NewController
	endpointSliceLister discoverylisters.EndpointSliceLister
	// endpointSlicesSynced returns true if the endpoint slice shared informer
	// has been synced at least once. Added as a member to the struct to allow
	// injection for testing.
	endpointSlicesSynced cache.InformerSynced
	// endpointSliceTracker tracks the list of EndpointSlices and associated
	// resource versions expected for each Service. It can help determine if a
	// cached EndpointSlice is out of date.
	endpointSliceTracker *endpointSliceTracker

	// reconciler is an util used to reconcile EndpointSlice changes.
	reconciler *reconciler

	// Services that need to be updated. A channel is inappropriate here,
	// because it allows services with lots of pods to be serviced much
	// more often than services with few pods; it also would cause a
	// service that's inserted multiple times to be processed more than
	// necessary.
	queue workqueue.RateLimitingInterface

	// maxEndpointsPerSlice references the maximum number of endpoints that
	// should be added to an EndpointSlice
	maxEndpointsPerSlice int32

	// workerLoopPeriod is the time between worker runs. The workers
	// process the queue of service and pod changes
	workerLoopPeriod time.Duration

	// endpointUpdatesBatchPeriod is an artificial delay added to all service
	// syncs triggered by pod changes. This can be used to reduce overall number
	// of all EndpointSlice updates.
	endpointUpdatesBatchPeriod time.Duration
}

// Run will not return until stopCh is closed.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting EndpointSliceMirroring controller")
	defer klog.Infof("Shutting down EndpointSliceMirroring controller")

	if !cache.WaitForNamedCacheSync("endpoint_slice_mirroring", stopCh, c.endpointsSynced, c.endpointSlicesSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.worker, c.workerLoopPeriod, stopCh)
	}

	go func() {
		defer utilruntime.HandleCrash()
	}()

	<-stopCh
}

// worker runs a worker thread that just dequeues items, processes them, and
// marks them done. You may run as many of these in parallel as you wish; the
// workqueue guarantees that they will not end up processing the same service
// at the same time
func (c *Controller) worker() {
	for c.processNextWorkItem() {
	}
}

func (c *Controller) processNextWorkItem() bool {
	cKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(cKey)

	err := c.syncEndpoints(cKey.(string))
	c.handleErr(err, cKey)

	return true
}

func (c *Controller) handleErr(err error, key interface{}) {
	if err == nil {
		c.queue.Forget(key)
		return
	}

	if c.queue.NumRequeues(key) < maxRetries {
		klog.Warningf("Error mirroring EndpointSlices for %q Endpoints, retrying. Error: %v", key, err)
		c.queue.AddRateLimited(key)
		return
	}

	klog.Warningf("Retry budget exceeded, dropping %q Endpoints out of the queue: %v", key, err)
	c.queue.Forget(key)
	utilruntime.HandleError(err)
}

func (c *Controller) syncEndpoints(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing EndpointSlices for %q Endpoints. (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	endpoints, err := c.endpointsLister.Endpoints(namespace).Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			c.reconciler.deleteEndpoints(namespace, name)
			// The Endpoints resource has been deleted, return nil so that it
			// won't be retried.
			return nil
		}
		return err
	}

	klog.V(5).Infof("About to mirror EndpointSlices for %q Endpoints", key)

	esLabelSelector := labels.Set(map[string]string{
		discovery.LabelServiceName: endpoints.Name,
		discovery.LabelManagedBy:   controllerName,
	}).AsSelectorPreValidated()
	endpointSlices, err := c.endpointSliceLister.EndpointSlices(endpoints.Namespace).List(esLabelSelector)

	if err != nil {
		// Since we're getting stuff from a local cache, it is basically
		// impossible to get this error.
		c.eventRecorder.Eventf(endpoints, v1.EventTypeWarning, "FailedToListEndpointSlices",
			"Error listing EndpointSlices for Service %s/%s: %v", endpoints.Namespace, endpoints.Name, err)
		return err
	}

	err = c.reconciler.reconcile(endpoints, endpointSlices)
	if err != nil {
		c.eventRecorder.Eventf(endpoints, v1.EventTypeWarning, "FailedToUpdateEndpointSlices",
			"Error updating EndpointSlices for Service %s/%s: %v", endpoints.Namespace, endpoints.Name, err)
		return err
	}

	return nil
}

// onEndpointsUpdated queues the Endpoints resource for processing.
func (c *Controller) onEndpointsUpdate(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	c.queue.Add(key)
}

// onEndpointSliceAdd queues a sync for the relevant Service for a sync if the
// EndpointSlice resource version does not match the expected version in the
// endpointSliceTracker.
func (c *Controller) onEndpointSliceAdd(obj interface{}) {
	endpointSlice := obj.(*discovery.EndpointSlice)
	if endpointSlice == nil {
		utilruntime.HandleError(fmt.Errorf("Invalid EndpointSlice provided to onEndpointSliceAdd()"))
		return
	}
	if managedByController(endpointSlice) && c.endpointSliceTracker.Stale(endpointSlice) {
		c.queueEndpointsForEndpointSlice(endpointSlice)
	}
}

// onEndpointSliceUpdate queues a sync for the relevant Service for a sync if
// the EndpointSlice resource version does not match the expected version in the
// endpointSliceTracker or the managed-by value of the EndpointSlice has changed
// from or to this controller.
func (c *Controller) onEndpointSliceUpdate(prevObj, obj interface{}) {
	prevEndpointSlice := obj.(*discovery.EndpointSlice)
	endpointSlice := obj.(*discovery.EndpointSlice)
	if endpointSlice == nil || prevEndpointSlice == nil {
		utilruntime.HandleError(fmt.Errorf("Invalid EndpointSlice provided to onEndpointSliceUpdate()"))
		return
	}
	if managedByChanged(prevEndpointSlice, endpointSlice) || (managedByController(endpointSlice) && c.endpointSliceTracker.Stale(endpointSlice)) {
		c.queueEndpointsForEndpointSlice(endpointSlice)
	}
}

// onEndpointSliceDelete queues a sync for the relevant Service for a sync if the
// EndpointSlice resource version does not match the expected version in the
// endpointSliceTracker.
func (c *Controller) onEndpointSliceDelete(obj interface{}) {
	endpointSlice := getEndpointSliceFromDeleteAction(obj)
	if endpointSlice != nil && managedByController(endpointSlice) && c.endpointSliceTracker.Has(endpointSlice) {
		c.queueEndpointsForEndpointSlice(endpointSlice)
	}
}

// queueServiceForEndpointSlice attempts to queue the corresponding Service for
// the provided EndpointSlice.
func (c *Controller) queueEndpointsForEndpointSlice(endpointSlice *discovery.EndpointSlice) {
	key, err := endpointsControllerKey(endpointSlice)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for EndpointSlice %+v: %v", endpointSlice, err))
		return
	}

	// queue after the max of endpointSliceChangeMinSyncDelay and
	// endpointUpdatesBatchPeriod.
	delay := endpointSliceChangeMinSyncDelay
	if c.endpointUpdatesBatchPeriod > delay {
		delay = c.endpointUpdatesBatchPeriod
	}
	c.queue.AddAfter(key, delay)
}
