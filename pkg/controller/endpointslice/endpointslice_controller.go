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
	discovery "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
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
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	endpointslicemetrics "k8s.io/kubernetes/pkg/controller/endpointslice/metrics"
	"k8s.io/kubernetes/pkg/controller/endpointslice/topologycache"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
	endpointsliceutil "k8s.io/kubernetes/pkg/controller/util/endpointslice"
	"k8s.io/kubernetes/pkg/features"
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
	controllerName = "endpointslice-controller.k8s.io"
)

// NewController creates and initializes a new Controller
func NewController(podInformer coreinformers.PodInformer,
	serviceInformer coreinformers.ServiceInformer,
	nodeInformer coreinformers.NodeInformer,
	endpointSliceInformer discoveryinformers.EndpointSliceInformer,
	maxEndpointsPerSlice int32,
	client clientset.Interface,
	endpointUpdatesBatchPeriod time.Duration,
) *Controller {
	broadcaster := record.NewBroadcaster()
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "endpoint-slice-controller"})

	if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("endpoint_slice_controller", client.DiscoveryV1().RESTClient().GetRateLimiter())
	}

	endpointslicemetrics.RegisterMetrics()

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
		), "endpoint_slice"),
		workerLoopPeriod: time.Second,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: c.onServiceUpdate,
		UpdateFunc: func(old, cur interface{}) {
			c.onServiceUpdate(cur)
		},
		DeleteFunc: c.onServiceDelete,
	})
	c.serviceLister = serviceInformer.Lister()
	c.servicesSynced = serviceInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addPod,
		UpdateFunc: c.updatePod,
		DeleteFunc: c.deletePod,
	})
	c.podLister = podInformer.Lister()
	c.podsSynced = podInformer.Informer().HasSynced

	c.nodeLister = nodeInformer.Lister()
	c.nodesSynced = nodeInformer.Informer().HasSynced

	endpointSliceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.onEndpointSliceAdd,
		UpdateFunc: c.onEndpointSliceUpdate,
		DeleteFunc: c.onEndpointSliceDelete,
	})

	c.endpointSliceLister = endpointSliceInformer.Lister()
	c.endpointSlicesSynced = endpointSliceInformer.Informer().HasSynced
	c.endpointSliceTracker = endpointsliceutil.NewEndpointSliceTracker()

	c.maxEndpointsPerSlice = maxEndpointsPerSlice

	c.triggerTimeTracker = endpointutil.NewTriggerTimeTracker()

	c.eventBroadcaster = broadcaster
	c.eventRecorder = recorder

	c.endpointUpdatesBatchPeriod = endpointUpdatesBatchPeriod
	c.serviceSelectorCache = endpointutil.NewServiceSelectorCache()

	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareHints) {
		nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    c.addNode,
			UpdateFunc: c.updateNode,
			DeleteFunc: c.deleteNode,
		})

		c.topologyCache = topologycache.NewTopologyCache()
	}

	c.reconciler = &reconciler{
		client:               c.client,
		nodeLister:           c.nodeLister,
		maxEndpointsPerSlice: c.maxEndpointsPerSlice,
		endpointSliceTracker: c.endpointSliceTracker,
		metricsCache:         endpointslicemetrics.NewCache(maxEndpointsPerSlice),
		topologyCache:        c.topologyCache,
	}

	return c
}

// Controller manages selector-based service endpoint slices
type Controller struct {
	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	// serviceLister is able to list/get services and is populated by the
	// shared informer passed to NewController
	serviceLister corelisters.ServiceLister
	// servicesSynced returns true if the service shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	servicesSynced cache.InformerSynced

	// podLister is able to list/get pods and is populated by the
	// shared informer passed to NewController
	podLister corelisters.PodLister
	// podsSynced returns true if the pod shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podsSynced cache.InformerSynced

	// endpointSliceLister is able to list/get endpoint slices and is populated by the
	// shared informer passed to NewController
	endpointSliceLister discoverylisters.EndpointSliceLister
	// endpointSlicesSynced returns true if the endpoint slice shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	endpointSlicesSynced cache.InformerSynced
	// endpointSliceTracker tracks the list of EndpointSlices and associated
	// resource versions expected for each Service. It can help determine if a
	// cached EndpointSlice is out of date.
	endpointSliceTracker *endpointsliceutil.EndpointSliceTracker

	// nodeLister is able to list/get nodes and is populated by the
	// shared informer passed to NewController
	nodeLister corelisters.NodeLister
	// nodesSynced returns true if the node shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	nodesSynced cache.InformerSynced

	// reconciler is an util used to reconcile EndpointSlice changes.
	reconciler *reconciler

	// triggerTimeTracker is an util used to compute and export the
	// EndpointsLastChangeTriggerTime annotation.
	triggerTimeTracker *endpointutil.TriggerTimeTracker

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

	// endpointUpdatesBatchPeriod is an artificial delay added to all service syncs triggered by pod changes.
	// This can be used to reduce overall number of all endpoint slice updates.
	endpointUpdatesBatchPeriod time.Duration

	// serviceSelectorCache is a cache of service selectors to avoid high CPU consumption caused by frequent calls
	// to AsSelectorPreValidated (see #73527)
	serviceSelectorCache *endpointutil.ServiceSelectorCache

	// topologyCache tracks the distribution of Nodes and endpoints across zones
	// to enable TopologyAwareHints.
	topologyCache *topologycache.TopologyCache
}

// Run will not return until stopCh is closed.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	// Start events processing pipeline.
	c.eventBroadcaster.StartLogging(klog.Infof)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.client.CoreV1().Events("")})
	defer c.eventBroadcaster.Shutdown()

	defer c.queue.ShutDown()

	klog.Infof("Starting endpoint slice controller")
	defer klog.Infof("Shutting down endpoint slice controller")

	if !cache.WaitForNamedCacheSync("endpoint_slice", stopCh, c.podsSynced, c.servicesSynced, c.endpointSlicesSynced, c.nodesSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.worker, c.workerLoopPeriod, stopCh)
	}

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

	err := c.syncService(cKey.(string))
	c.handleErr(err, cKey)

	return true
}

func (c *Controller) handleErr(err error, key interface{}) {
	trackSync(err)

	if err == nil {
		c.queue.Forget(key)
		return
	}

	if c.queue.NumRequeues(key) < maxRetries {
		klog.Warningf("Error syncing endpoint slices for service %q, retrying. Error: %v", key, err)
		c.queue.AddRateLimited(key)
		return
	}

	klog.Warningf("Retry budget exceeded, dropping service %q out of the queue: %v", key, err)
	c.queue.Forget(key)
	utilruntime.HandleError(err)
}

func (c *Controller) syncService(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing service %q endpoint slices. (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	service, err := c.serviceLister.Services(namespace).Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			c.triggerTimeTracker.DeleteService(namespace, name)
			c.reconciler.deleteService(namespace, name)
			c.endpointSliceTracker.DeleteService(namespace, name)
			// The service has been deleted, return nil so that it won't be retried.
			return nil
		}
		return err
	}

	if service.Spec.Selector == nil {
		// services without a selector receive no endpoint slices from this controller;
		// these services will receive endpoint slices that are created out-of-band via the REST API.
		return nil
	}

	klog.V(5).Infof("About to update endpoint slices for service %q", key)

	podLabelSelector := labels.Set(service.Spec.Selector).AsSelectorPreValidated()
	pods, err := c.podLister.Pods(service.Namespace).List(podLabelSelector)
	if err != nil {
		// Since we're getting stuff from a local cache, it is basically
		// impossible to get this error.
		c.eventRecorder.Eventf(service, v1.EventTypeWarning, "FailedToListPods",
			"Error listing Pods for Service %s/%s: %v", service.Namespace, service.Name, err)
		return err
	}

	esLabelSelector := labels.Set(map[string]string{
		discovery.LabelServiceName: service.Name,
		discovery.LabelManagedBy:   controllerName,
	}).AsSelectorPreValidated()
	endpointSlices, err := c.endpointSliceLister.EndpointSlices(service.Namespace).List(esLabelSelector)

	if err != nil {
		// Since we're getting stuff from a local cache, it is basically
		// impossible to get this error.
		c.eventRecorder.Eventf(service, v1.EventTypeWarning, "FailedToListEndpointSlices",
			"Error listing Endpoint Slices for Service %s/%s: %v", service.Namespace, service.Name, err)
		return err
	}

	// Drop EndpointSlices that have been marked for deletion to prevent the controller from getting stuck.
	endpointSlices = dropEndpointSlicesPendingDeletion(endpointSlices)

	if c.endpointSliceTracker.StaleSlices(service, endpointSlices) {
		return endpointsliceutil.NewStaleInformerCache("EndpointSlice informer cache is out of date")
	}

	// We call ComputeEndpointLastChangeTriggerTime here to make sure that the
	// state of the trigger time tracker gets updated even if the sync turns out
	// to be no-op and we don't update the EndpointSlice objects.
	lastChangeTriggerTime := c.triggerTimeTracker.
		ComputeEndpointLastChangeTriggerTime(namespace, service, pods)

	err = c.reconciler.reconcile(service, pods, endpointSlices, lastChangeTriggerTime)
	if err != nil {
		c.eventRecorder.Eventf(service, v1.EventTypeWarning, "FailedToUpdateEndpointSlices",
			"Error updating Endpoint Slices for Service %s/%s: %v", service.Namespace, service.Name, err)
		return err
	}

	return nil
}

// onServiceUpdate updates the Service Selector in the cache and queues the Service for processing.
func (c *Controller) onServiceUpdate(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	_ = c.serviceSelectorCache.Update(key, obj.(*v1.Service).Spec.Selector)
	c.queue.Add(key)
}

// onServiceDelete removes the Service Selector from the cache and queues the Service for processing.
func (c *Controller) onServiceDelete(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	c.serviceSelectorCache.Delete(key)
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
	if managedByController(endpointSlice) && c.endpointSliceTracker.ShouldSync(endpointSlice) {
		c.queueServiceForEndpointSlice(endpointSlice)
	}
}

// onEndpointSliceUpdate queues a sync for the relevant Service for a sync if
// the EndpointSlice resource version does not match the expected version in the
// endpointSliceTracker or the managed-by value of the EndpointSlice has changed
// from or to this controller.
func (c *Controller) onEndpointSliceUpdate(prevObj, obj interface{}) {
	prevEndpointSlice := prevObj.(*discovery.EndpointSlice)
	endpointSlice := obj.(*discovery.EndpointSlice)
	if endpointSlice == nil || prevEndpointSlice == nil {
		utilruntime.HandleError(fmt.Errorf("Invalid EndpointSlice provided to onEndpointSliceUpdate()"))
		return
	}
	// EndpointSlice generation does not change when labels change. Although the
	// controller will never change LabelServiceName, users might. This check
	// ensures that we handle changes to this label.
	svcName := endpointSlice.Labels[discovery.LabelServiceName]
	prevSvcName := prevEndpointSlice.Labels[discovery.LabelServiceName]
	if svcName != prevSvcName {
		klog.Warningf("%s label changed from %s  to %s for %s", discovery.LabelServiceName, prevSvcName, svcName, endpointSlice.Name)
		c.queueServiceForEndpointSlice(endpointSlice)
		c.queueServiceForEndpointSlice(prevEndpointSlice)
		return
	}
	if managedByChanged(prevEndpointSlice, endpointSlice) || (managedByController(endpointSlice) && c.endpointSliceTracker.ShouldSync(endpointSlice)) {
		c.queueServiceForEndpointSlice(endpointSlice)
	}
}

// onEndpointSliceDelete queues a sync for the relevant Service for a sync if the
// EndpointSlice resource version does not match the expected version in the
// endpointSliceTracker.
func (c *Controller) onEndpointSliceDelete(obj interface{}) {
	endpointSlice := getEndpointSliceFromDeleteAction(obj)
	if endpointSlice != nil && managedByController(endpointSlice) && c.endpointSliceTracker.Has(endpointSlice) {
		// This returns false if we didn't expect the EndpointSlice to be
		// deleted. If that is the case, we queue the Service for another sync.
		if !c.endpointSliceTracker.HandleDeletion(endpointSlice) {
			c.queueServiceForEndpointSlice(endpointSlice)
		}
	}
}

// queueServiceForEndpointSlice attempts to queue the corresponding Service for
// the provided EndpointSlice.
func (c *Controller) queueServiceForEndpointSlice(endpointSlice *discovery.EndpointSlice) {
	key, err := serviceControllerKey(endpointSlice)
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

func (c *Controller) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	services, err := c.serviceSelectorCache.GetPodServiceMemberships(c.serviceLister, pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to get pod %s/%s's service memberships: %v", pod.Namespace, pod.Name, err))
		return
	}
	for key := range services {
		c.queue.AddAfter(key, c.endpointUpdatesBatchPeriod)
	}
}

func (c *Controller) updatePod(old, cur interface{}) {
	services := endpointutil.GetServicesToUpdateOnPodChange(c.serviceLister, c.serviceSelectorCache, old, cur)
	for key := range services {
		c.queue.AddAfter(key, c.endpointUpdatesBatchPeriod)
	}
}

// When a pod is deleted, enqueue the services the pod used to be a member of
// obj could be an *v1.Pod, or a DeletionFinalStateUnknown marker item.
func (c *Controller) deletePod(obj interface{}) {
	pod := endpointutil.GetPodFromDeleteAction(obj)
	if pod != nil {
		c.addPod(pod)
	}
}

func (c *Controller) addNode(obj interface{}) {
	c.checkNodeTopologyDistribution()
}

func (c *Controller) updateNode(old, cur interface{}) {
	oldNode := old.(*v1.Node)
	curNode := cur.(*v1.Node)

	if topologycache.NodeReady(oldNode.Status) != topologycache.NodeReady(curNode.Status) {
		c.checkNodeTopologyDistribution()
	}
}

func (c *Controller) deleteNode(obj interface{}) {
	c.checkNodeTopologyDistribution()
}

// checkNodeTopologyDistribution updates Nodes in the topology cache and then
// queues any Services that are past the threshold.
func (c *Controller) checkNodeTopologyDistribution() {
	if c.topologyCache == nil {
		return
	}
	nodes, err := c.nodeLister.List(labels.Everything())
	if err != nil {
		klog.Errorf("Error listing Nodes: %v", err)
		return
	}
	c.topologyCache.SetNodes(nodes)
	serviceKeys := c.topologyCache.GetOverloadedServices()
	for _, serviceKey := range serviceKeys {
		klog.V(2).Infof("Queuing %s Service after Node change due to overloading", serviceKey)
		c.queue.Add(serviceKey)
	}
}

// trackSync increments the EndpointSliceSyncs metric with the result of a sync.
func trackSync(err error) {
	metricLabel := "success"
	if err != nil {
		if endpointsliceutil.IsStaleInformerCacheErr(err) {
			metricLabel = "stale"
		} else {
			metricLabel = "error"
		}
	}
	endpointslicemetrics.EndpointSliceSyncs.WithLabelValues(metricLabel).Inc()
}

func dropEndpointSlicesPendingDeletion(endpointSlices []*discovery.EndpointSlice) []*discovery.EndpointSlice {
	n := 0
	for _, endpointSlice := range endpointSlices {
		if endpointSlice.DeletionTimestamp == nil {
			endpointSlices[n] = endpointSlice
			n++
		}
	}
	return endpointSlices[:n]
}
