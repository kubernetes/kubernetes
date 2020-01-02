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
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/controller"
	endpointslicemetrics "k8s.io/kubernetes/pkg/controller/endpointslice/metrics"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
)

const (
	// maxRetries is the number of times a service will be retried before it is
	// dropped out of the queue. Any sync error, such as a failure to create or
	// update an EndpointSlice could trigger a retry. With the current
	// rate-limiter in use (5ms*2^(maxRetries-1)) the following numbers
	// represent the sequence of delays between successive queuings of a
	// service.
	//
	// 5ms, 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1.3s, 2.6s, 5.1s,
	// 10.2s, 20.4s, 41s, 82s
	maxRetries = 15
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
) *Controller {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(klog.Infof)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "endpoint-slice-controller"})

	if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("endpoint_slice_controller", client.DiscoveryV1beta1().RESTClient().GetRateLimiter())
	}

	endpointslicemetrics.RegisterMetrics()

	c := &Controller{
		client:           client,
		queue:            workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "endpoint_slice"),
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
	c.endpointSliceTracker = newEndpointSliceTracker()

	c.maxEndpointsPerSlice = maxEndpointsPerSlice

	c.reconciler = &reconciler{
		client:               c.client,
		nodeLister:           c.nodeLister,
		maxEndpointsPerSlice: c.maxEndpointsPerSlice,
		endpointSliceTracker: c.endpointSliceTracker,
		metricsCache:         endpointslicemetrics.NewCache(maxEndpointsPerSlice),
	}
	c.triggerTimeTracker = endpointutil.NewTriggerTimeTracker()

	c.eventBroadcaster = broadcaster
	c.eventRecorder = recorder

	c.serviceSelectorCache = endpointutil.NewServiceSelectorCache()

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
	endpointSliceTracker *endpointSliceTracker

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

	// serviceSelectorCache is a cache of service selectors to avoid high CPU consumption caused by frequent calls
	// to AsSelectorPreValidated (see #73527)
	serviceSelectorCache *endpointutil.ServiceSelectorCache
}

// Run will not return until stopCh is closed.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting endpoint slice controller")
	defer klog.Infof("Shutting down endpoint slice controller")

	if !cache.WaitForNamedCacheSync("endpoint_slice", stopCh, c.podsSynced, c.servicesSynced) {
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

	err := c.syncService(cKey.(string))
	c.handleErr(err, cKey)

	return true
}

func (c *Controller) handleErr(err error, key interface{}) {
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
	if managedByController(endpointSlice) && c.endpointSliceTracker.Stale(endpointSlice) {
		c.queueServiceForEndpointSlice(endpointSlice)
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
		c.queueServiceForEndpointSlice(endpointSlice)
	}
}

// onEndpointSliceDelete queues a sync for the relevant Service for a sync if the
// EndpointSlice resource version does not match the expected version in the
// endpointSliceTracker.
func (c *Controller) onEndpointSliceDelete(obj interface{}) {
	endpointSlice := getEndpointSliceFromDeleteAction(obj)
	if endpointSlice != nil && managedByController(endpointSlice) && c.endpointSliceTracker.Has(endpointSlice) {
		c.queueServiceForEndpointSlice(endpointSlice)
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
	c.queue.Add(key)
}

func (c *Controller) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	services, err := c.serviceSelectorCache.GetPodServiceMemberships(c.serviceLister, pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to get pod %s/%s's service memberships: %v", pod.Namespace, pod.Name, err))
		return
	}
	for key := range services {
		c.queue.Add(key)
	}
}

func (c *Controller) updatePod(old, cur interface{}) {
	services := endpointutil.GetServicesToUpdateOnPodChange(c.serviceLister, c.serviceSelectorCache, old, cur, podEndpointChanged)
	for key := range services {
		c.queue.Add(key)
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
