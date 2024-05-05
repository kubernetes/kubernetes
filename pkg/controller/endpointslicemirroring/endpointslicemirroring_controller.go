/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslicemirroring

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/time/rate"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	discoveryinformers "k8s.io/client-go/informers/discovery/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	discoverylisters "k8s.io/client-go/listers/discovery/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/endpointslicemirroring/metrics"
	endpointslicepkg "k8s.io/kubernetes/pkg/controller/util/endpointslice"
)

const (
	// maxRetries is the number of times an Endpoints resource will be retried
	// before it is dropped out of the queue. Any sync error, such as a failure
	// to create or update an EndpointSlice could trigger a retry. With the
	// current rate-limiter in use (1s*2^(numRetries-1)) up to a max of 100s.
	// The following numbers represent the sequence of delays between successive
	// queuings of an Endpoints resource.
	//
	// 1s, 2s, 4s, 8s, 16s, 32s, 64s, 100s (max)
	maxRetries = 15

	// defaultSyncBackOff is the default backoff period for syncEndpoints calls.
	defaultSyncBackOff = 1 * time.Second
	// maxSyncBackOff is the max backoff period for syncEndpoints calls.
	maxSyncBackOff = 100 * time.Second

	// controllerName is a unique value used with LabelManagedBy to indicated
	// the component managing an EndpointSlice.
	controllerName = "endpointslicemirroring-controller.k8s.io"
)

// NewController creates and initializes a new Controller
func NewController(ctx context.Context, endpointsInformer coreinformers.EndpointsInformer,
	endpointSliceInformer discoveryinformers.EndpointSliceInformer,
	serviceInformer coreinformers.ServiceInformer,
	maxEndpointsPerSubset int32,
	client clientset.Interface,
	endpointUpdatesBatchPeriod time.Duration,
) *Controller {
	logger := klog.FromContext(ctx)
	broadcaster := record.NewBroadcaster(record.WithContext(ctx))
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "endpoint-slice-mirroring-controller"})

	metrics.RegisterMetrics()

	c := &Controller{
		client: client,
		// This is similar to the DefaultControllerRateLimiter, just with a
		// significantly higher default backoff (1s vs 5ms). This controller
		// processes events that can require significant EndpointSlice changes.
		// A more significant rate limit back off here helps ensure that the
		// Controller does not overwhelm the API Server.
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.NewMaxOfRateLimiter(
			workqueue.NewItemExponentialFailureRateLimiter(defaultSyncBackOff, maxSyncBackOff),
			// 10 qps, 100 bucket size. This is only for retry speed and its
			// only the overall factor (not per item).
			&workqueue.BucketRateLimiter{Limiter: rate.NewLimiter(rate.Limit(10), 100)},
		), "endpoint_slice_mirroring"),
		workerLoopPeriod: time.Second,
	}

	endpointsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.onEndpointsAdd(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.onEndpointsUpdate(logger, oldObj, newObj)
		},
		DeleteFunc: func(obj interface{}) {
			c.onEndpointsDelete(logger, obj)
		},
	})
	c.endpointsLister = endpointsInformer.Lister()
	c.endpointsSynced = endpointsInformer.Informer().HasSynced

	endpointSliceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: c.onEndpointSliceAdd,
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.onEndpointSliceUpdate(logger, oldObj, newObj)
		},
		DeleteFunc: c.onEndpointSliceDelete,
	})

	c.endpointSliceLister = endpointSliceInformer.Lister()
	c.endpointSlicesSynced = endpointSliceInformer.Informer().HasSynced
	c.endpointSliceTracker = endpointsliceutil.NewEndpointSliceTracker()

	c.serviceLister = serviceInformer.Lister()
	c.servicesSynced = serviceInformer.Informer().HasSynced
	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.onServiceAdd,
		UpdateFunc: c.onServiceUpdate,
		DeleteFunc: c.onServiceDelete,
	})

	c.maxEndpointsPerSubset = maxEndpointsPerSubset

	c.reconciler = &reconciler{
		client:                c.client,
		maxEndpointsPerSubset: c.maxEndpointsPerSubset,
		endpointSliceTracker:  c.endpointSliceTracker,
		metricsCache:          metrics.NewCache(maxEndpointsPerSubset),
		eventRecorder:         recorder,
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
	// shared informer passed to NewController.
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
	// resource versions expected for each Endpoints resource. It can help
	// determine if a cached EndpointSlice is out of date.
	endpointSliceTracker *endpointsliceutil.EndpointSliceTracker

	// serviceLister is able to list/get services and is populated by the shared
	// informer passed to NewController.
	serviceLister corelisters.ServiceLister
	// servicesSynced returns true if the services shared informer has been
	// synced at least once. Added as a member to the struct to allow injection
	// for testing.
	servicesSynced cache.InformerSynced

	// reconciler is an util used to reconcile EndpointSlice changes.
	reconciler *reconciler

	// Endpoints that need to be updated. A channel is inappropriate here,
	// because it allows Endpoints with lots of addresses to be serviced much
	// more often than Endpoints with few addresses; it also would cause an
	// Endpoints resource that's inserted multiple times to be processed more
	// than necessary.
	queue workqueue.RateLimitingInterface

	// maxEndpointsPerSubset references the maximum number of endpoints that
	// should be added to an EndpointSlice for an EndpointSubset.
	maxEndpointsPerSubset int32

	// workerLoopPeriod is the time between worker runs. The workers process the
	// queue of changes to Endpoints resources.
	workerLoopPeriod time.Duration

	// endpointUpdatesBatchPeriod is an artificial delay added to all Endpoints
	// syncs triggered by EndpointSlice changes. This can be used to reduce
	// overall number of all EndpointSlice updates.
	endpointUpdatesBatchPeriod time.Duration
}

// Run will not return until stopCh is closed.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	// Start events processing pipeline.
	c.eventBroadcaster.StartLogging(klog.Infof)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.client.CoreV1().Events("")})
	defer c.eventBroadcaster.Shutdown()

	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting EndpointSliceMirroring controller")
	defer logger.Info("Shutting down EndpointSliceMirroring controller")

	if !cache.WaitForNamedCacheSync("endpoint_slice_mirroring", ctx.Done(), c.endpointsSynced, c.endpointSlicesSynced, c.servicesSynced) {
		return
	}

	logger.V(2).Info("Starting worker threads", "total", workers)
	for i := 0; i < workers; i++ {
		go wait.Until(func() { c.worker(logger) }, c.workerLoopPeriod, ctx.Done())
	}

	<-ctx.Done()
}

// worker runs a worker thread that just dequeues items, processes them, and
// marks them done. You may run as many of these in parallel as you wish; the
// workqueue guarantees that they will not end up processing the same service
// at the same time
func (c *Controller) worker(logger klog.Logger) {
	for c.processNextWorkItem(logger) {
	}
}

func (c *Controller) processNextWorkItem(logger klog.Logger) bool {
	cKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(cKey)

	err := c.syncEndpoints(logger, cKey.(string))
	c.handleErr(logger, err, cKey)

	return true
}

func (c *Controller) handleErr(logger klog.Logger, err error, key interface{}) {
	if err == nil {
		c.queue.Forget(key)
		return
	}

	if c.queue.NumRequeues(key) < maxRetries {
		logger.Info("Error mirroring EndpointSlices for Endpoints, retrying", "key", key, "err", err)
		c.queue.AddRateLimited(key)
		return
	}

	logger.Info("Retry budget exceeded, dropping Endpoints out of the queue", "key", key, "err", err)
	c.queue.Forget(key)
	utilruntime.HandleError(err)
}

func (c *Controller) syncEndpoints(logger klog.Logger, key string) error {
	startTime := time.Now()
	defer func() {
		syncDuration := float64(time.Since(startTime).Milliseconds()) / 1000
		metrics.EndpointsSyncDuration.WithLabelValues().Observe(syncDuration)
		logger.V(4).Info("Finished syncing EndpointSlices for Endpoints", "key", key, "elapsedTime", time.Since(startTime))
	}()

	logger.V(4).Info("syncEndpoints", "key", key)

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	endpoints, err := c.endpointsLister.Endpoints(namespace).Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(4).Info("Endpoints not found, cleaning up any mirrored EndpointSlices", "endpoints", klog.KRef(namespace, name))
			c.endpointSliceTracker.DeleteService(namespace, name)
			return c.deleteMirroredSlices(namespace, name)
		}
		return err
	}

	if !c.shouldMirror(endpoints) {
		logger.V(4).Info("Endpoints should not be mirrored, cleaning up any mirrored EndpointSlices", "endpoints", klog.KRef(namespace, name))
		c.endpointSliceTracker.DeleteService(namespace, name)
		return c.deleteMirroredSlices(namespace, name)
	}

	svc, err := c.serviceLister.Services(namespace).Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(4).Info("Service not found, cleaning up any mirrored EndpointSlices", "service", klog.KRef(namespace, name))
			c.endpointSliceTracker.DeleteService(namespace, name)
			return c.deleteMirroredSlices(namespace, name)
		}
		return err
	}

	// If a selector is specified, clean up any mirrored slices.
	if svc.Spec.Selector != nil {
		logger.V(4).Info("Service now has selector, cleaning up any mirrored EndpointSlices", "service", klog.KRef(namespace, name))
		c.endpointSliceTracker.DeleteService(namespace, name)
		return c.deleteMirroredSlices(namespace, name)
	}

	endpointSlices, err := endpointSlicesMirroredForService(c.endpointSliceLister, namespace, name)
	if err != nil {
		return err
	}

	if c.endpointSliceTracker.StaleSlices(svc, endpointSlices) {
		return endpointslicepkg.NewStaleInformerCache("EndpointSlice informer cache is out of date")
	}

	err = c.reconciler.reconcile(logger, endpoints, endpointSlices)
	if err != nil {
		return err
	}

	return nil
}

// queueEndpoints queues the Endpoints resource for processing.
func (c *Controller) queueEndpoints(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v (type %T): %v", obj, obj, err))
		return
	}

	c.queue.Add(key)
}

// shouldMirror returns true if an Endpoints resource should be mirrored by this
// controller. This will be false if:
// - the Endpoints resource is nil.
// - the Endpoints resource has a skip-mirror label.
// - the Endpoints resource has a leader election annotation.
// This does not ensure that a corresponding Service exists with a nil selector.
// That check should be performed separately.
func (c *Controller) shouldMirror(endpoints *v1.Endpoints) bool {
	if endpoints == nil || skipMirror(endpoints.Labels) || hasLeaderElection(endpoints.Annotations) {
		return false
	}

	return true
}

// onServiceAdd queues a sync for the relevant Endpoints resource.
func (c *Controller) onServiceAdd(obj interface{}) {
	service := obj.(*v1.Service)
	if service == nil {
		utilruntime.HandleError(fmt.Errorf("onServiceAdd() expected type v1.Service, got %T", obj))
		return
	}
	if service.Spec.Selector == nil {
		c.queueEndpoints(obj)
	}
}

// onServiceUpdate queues a sync for the relevant Endpoints resource.
func (c *Controller) onServiceUpdate(prevObj, obj interface{}) {
	service := obj.(*v1.Service)
	prevService := prevObj.(*v1.Service)
	if service == nil || prevService == nil {
		utilruntime.HandleError(fmt.Errorf("onServiceUpdate() expected type v1.Service, got %T, %T", prevObj, obj))
		return
	}
	if (service.Spec.Selector == nil) != (prevService.Spec.Selector == nil) {
		c.queueEndpoints(obj)
	}
}

// onServiceDelete queues a sync for the relevant Endpoints resource.
func (c *Controller) onServiceDelete(obj interface{}) {
	service := getServiceFromDeleteAction(obj)
	if service == nil {
		utilruntime.HandleError(fmt.Errorf("onServiceDelete() expected type v1.Service, got %T", obj))
		return
	}
	if service.Spec.Selector == nil {
		c.queueEndpoints(obj)
	}
}

// onEndpointsAdd queues a sync for the relevant Endpoints resource.
func (c *Controller) onEndpointsAdd(logger klog.Logger, obj interface{}) {
	endpoints := obj.(*v1.Endpoints)
	if endpoints == nil {
		utilruntime.HandleError(fmt.Errorf("onEndpointsAdd() expected type v1.Endpoints, got %T", obj))
		return
	}
	if !c.shouldMirror(endpoints) {
		logger.V(5).Info("Skipping mirroring", "endpoints", klog.KObj(endpoints))
		return
	}
	c.queueEndpoints(obj)
}

// onEndpointsUpdate queues a sync for the relevant Endpoints resource.
func (c *Controller) onEndpointsUpdate(logger klog.Logger, prevObj, obj interface{}) {
	endpoints := obj.(*v1.Endpoints)
	prevEndpoints := prevObj.(*v1.Endpoints)
	if endpoints == nil || prevEndpoints == nil {
		utilruntime.HandleError(fmt.Errorf("onEndpointsUpdate() expected type v1.Endpoints, got %T, %T", prevObj, obj))
		return
	}
	if !c.shouldMirror(endpoints) && !c.shouldMirror(prevEndpoints) {
		logger.V(5).Info("Skipping mirroring", "endpoints", klog.KObj(endpoints))
		return
	}
	c.queueEndpoints(obj)
}

// onEndpointsDelete queues a sync for the relevant Endpoints resource.
func (c *Controller) onEndpointsDelete(logger klog.Logger, obj interface{}) {
	endpoints := getEndpointsFromDeleteAction(obj)
	if endpoints == nil {
		utilruntime.HandleError(fmt.Errorf("onEndpointsDelete() expected type v1.Endpoints, got %T", obj))
		return
	}
	if !c.shouldMirror(endpoints) {
		logger.V(5).Info("Skipping mirroring", "endpoints", klog.KObj(endpoints))
		return
	}
	c.queueEndpoints(obj)
}

// onEndpointSliceAdd queues a sync for the relevant Endpoints resource for a
// sync if the EndpointSlice resource version does not match the expected
// version in the endpointSliceTracker.
func (c *Controller) onEndpointSliceAdd(obj interface{}) {
	endpointSlice := obj.(*discovery.EndpointSlice)
	if endpointSlice == nil {
		utilruntime.HandleError(fmt.Errorf("onEndpointSliceAdd() expected type discovery.EndpointSlice, got %T", obj))
		return
	}
	if managedByController(endpointSlice) && c.endpointSliceTracker.ShouldSync(endpointSlice) {
		c.queueEndpointsForEndpointSlice(endpointSlice)
	}
}

// onEndpointSliceUpdate queues a sync for the relevant Endpoints resource for a
// sync if the EndpointSlice resource version does not match the expected
// version in the endpointSliceTracker or the managed-by value of the
// EndpointSlice has changed from or to this controller.
func (c *Controller) onEndpointSliceUpdate(logger klog.Logger, prevObj, obj interface{}) {
	endpointSlice := obj.(*discovery.EndpointSlice)
	prevEndpointSlice := prevObj.(*discovery.EndpointSlice)
	if endpointSlice == nil || prevEndpointSlice == nil {
		utilruntime.HandleError(fmt.Errorf("onEndpointSliceUpdated() expected type discovery.EndpointSlice, got %T, %T", prevObj, obj))
		return
	}
	// EndpointSlice generation does not change when labels change. Although the
	// controller will never change LabelServiceName, users might. This check
	// ensures that we handle changes to this label.
	svcName := endpointSlice.Labels[discovery.LabelServiceName]
	prevSvcName := prevEndpointSlice.Labels[discovery.LabelServiceName]
	if svcName != prevSvcName {
		logger.Info("LabelServiceName changed", "labelServiceName", discovery.LabelServiceName, "oldName", prevSvcName, "newName", svcName, "endpointSlice", klog.KObj(endpointSlice))
		c.queueEndpointsForEndpointSlice(endpointSlice)
		c.queueEndpointsForEndpointSlice(prevEndpointSlice)
		return
	}
	if managedByChanged(prevEndpointSlice, endpointSlice) || (managedByController(endpointSlice) && c.endpointSliceTracker.ShouldSync(endpointSlice)) {
		c.queueEndpointsForEndpointSlice(endpointSlice)
	}
}

// onEndpointSliceDelete queues a sync for the relevant Endpoints resource for a
// sync if the EndpointSlice resource version does not match the expected
// version in the endpointSliceTracker.
func (c *Controller) onEndpointSliceDelete(obj interface{}) {
	endpointSlice := getEndpointSliceFromDeleteAction(obj)
	if endpointSlice == nil {
		utilruntime.HandleError(fmt.Errorf("onEndpointSliceDelete() expected type discovery.EndpointSlice, got %T", obj))
		return
	}
	if managedByController(endpointSlice) && c.endpointSliceTracker.Has(endpointSlice) {
		// This returns false if we didn't expect the EndpointSlice to be
		// deleted. If that is the case, we queue the Service for another sync.
		if !c.endpointSliceTracker.HandleDeletion(endpointSlice) {
			c.queueEndpointsForEndpointSlice(endpointSlice)
		}
	}
}

// queueEndpointsForEndpointSlice attempts to queue the corresponding Endpoints
// resource for the provided EndpointSlice.
func (c *Controller) queueEndpointsForEndpointSlice(endpointSlice *discovery.EndpointSlice) {
	key, err := endpointsControllerKey(endpointSlice)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for EndpointSlice %+v (type %T): %v", endpointSlice, endpointSlice, err))
		return
	}

	c.queue.AddAfter(key, c.endpointUpdatesBatchPeriod)
}

// deleteMirroredSlices will delete and EndpointSlices that have been mirrored
// for Endpoints with this namespace and name.
func (c *Controller) deleteMirroredSlices(namespace, name string) error {
	endpointSlices, err := endpointSlicesMirroredForService(c.endpointSliceLister, namespace, name)
	if err != nil {
		return err
	}

	c.endpointSliceTracker.DeleteService(namespace, name)
	return c.reconciler.deleteEndpoints(namespace, name, endpointSlices)
}

// endpointSlicesMirroredForService returns the EndpointSlices that have been
// mirrored for a Service by this controller.
func endpointSlicesMirroredForService(endpointSliceLister discoverylisters.EndpointSliceLister, namespace, name string) ([]*discovery.EndpointSlice, error) {
	esLabelSelector := labels.Set(map[string]string{
		discovery.LabelServiceName: name,
		discovery.LabelManagedBy:   controllerName,
	}).AsSelectorPreValidated()
	return endpointSliceLister.EndpointSlices(namespace).List(esLabelSelector)
}
