/*
Copyright 2021 The Kubernetes Authors.

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

package allocator

import (
	"time"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	allocationinformers "k8s.io/client-go/informers/allocation/v1alpha1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	allocationlisters "k8s.io/client-go/listers/allocation/v1alpha1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
)

const (
	// maxRetries is the number of times a service will be retried before it is dropped out of the queue.
	// With the current rate-limiter in use (5ms*2^(maxRetries-1)) the following numbers represent the
	// sequence of delays between successive queuings of a service.
	//
	// 5ms, 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1.3s, 2.6s, 5.1s, 10.2s, 20.4s, 41s, 82s
	maxRetries = 15
)

// NewClusterIPAllocatorController returns a new *Controller.
func NewClusterIPAllocatorController(serviceInformer coreinformers.ServiceInformer,
	ipRangeInformer allocationinformers.IPRangeInformer,
	ipAddressInformer allocationinformers.IPAddressInformer,
	client clientset.Interface) *Controller {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartStructuredLogging(0)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "clusterip-allocator-controller"})

	if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("clusterip_allocator_controller", client.CoreV1().RESTClient().GetRateLimiter())
	}
	e := &Controller{
		client:           client,
		queue:            workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "ipaddress"),
		workerLoopPeriod: time.Second,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{})
	e.serviceLister = serviceInformer.Lister()
	e.servicesSynced = serviceInformer.Informer().HasSynced

	ipRangeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{})
	e.ipRangeLister = ipRangeInformer.Lister()
	e.ipRangesSynced = ipRangeInformer.Informer().HasSynced

	ipAddressInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{})
	e.ipAddressLister = ipAddressInformer.Lister()
	e.ipAddressSynced = ipAddressInformer.Informer().HasSynced

	e.eventBroadcaster = broadcaster
	e.eventRecorder = recorder

	return e
}

// Controller manages selector-based service ipAddress.
type Controller struct {
	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	// serviceLister is able to list/get services and is populated by the shared informer passed to
	// NewEndpointController.
	serviceLister corelisters.ServiceLister
	// servicesSynced returns true if the service shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	servicesSynced cache.InformerSynced

	// ipRangeLister is able to list/get ipRanges and is populated by the shared informer passed to
	// NewEndpointController.
	ipRangeLister allocationlisters.IPRangeLister
	// ipRangesSynced returns true if the ipRange shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	ipRangesSynced cache.InformerSynced

	// ipAddressLister is able to list/get ipAddress and is populated by the shared informer passed to
	// NewEndpointController.
	ipAddressLister allocationlisters.IPAddressLister
	// ipAddressSynced returns true if the ipAddress shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	ipAddressSynced cache.InformerSynced

	// Services that need to be updated. A channel is inappropriate here,
	// because it allows services with lots of ipRanges to be serviced much
	// more often than services with few ipRanges; it also would cause a
	// service that's inserted multiple times to be processed more than
	// necessary.
	queue workqueue.RateLimitingInterface

	// workerLoopPeriod is the time between worker runs. The workers process the queue of service and ipRange changes.
	workerLoopPeriod time.Duration
}

// Run will not return until stopCh is closed. workers determines how many
// ipAddress will be handled in parallel.
func (e *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer e.queue.ShutDown()

	klog.Infof("Starting clusterip allocator controller")
	defer klog.Infof("Shutting down clusterip allocator controller")

	if !cache.WaitForNamedCacheSync("clusterip allocator", stopCh, e.ipRangesSynced, e.servicesSynced, e.ipAddressSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(e.worker, e.workerLoopPeriod, stopCh)
	}

	go func() {
		defer utilruntime.HandleCrash()
	}()

	<-stopCh
}

// worker runs a worker thread that just dequeues items, processes them, and
// marks them done. You may run as many of these in parallel as you wish; the
// workqueue guarantees that they will not end up processing the same service
// at the same time.
func (e *Controller) worker() {
	for e.processNextWorkItem() {
	}
}

func (e *Controller) processNextWorkItem() bool {
	eKey, quit := e.queue.Get()
	if quit {
		return false
	}
	defer e.queue.Done(eKey)

	err := e.syncIPRange(eKey.(string))
	e.handleErr(err, eKey)

	return true
}

func (e *Controller) handleErr(err error, key interface{}) {
	if err == nil {
		e.queue.Forget(key)
		return
	}

	ns, name, keyErr := cache.SplitMetaNamespaceKey(key.(string))
	if keyErr != nil {
		klog.ErrorS(err, "Failed to split meta namespace cache key", "key", key)
	}

	if e.queue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Error syncing ipAddress, retrying", "service", klog.KRef(ns, name), "err", err)
		e.queue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping service %q out of the queue: %v", key, err)
	e.queue.Forget(key)
	utilruntime.HandleError(err)
}

func (e *Controller) syncIPRange(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing IPRange allocator %q ipAddress. (%v)", key, time.Since(startTime))
	}()

	_, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	klog.V(4).Infof("syncing IPRange allocator %q ipAddress. (%v)", name, time.Since(startTime))

	return nil
}
