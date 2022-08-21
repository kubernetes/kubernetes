/*
Copyright 2022 The Kubernetes Authors.

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

package servicecidrs

import (
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	networkinginformers "k8s.io/client-go/informers/networking/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	networkinglisters "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/iptree"
	netutils "k8s.io/utils/net"
)

const (
	// maxRetries is the number of times a service will be retried before it is dropped out of the queue.
	// With the current rate-limiter in use (5ms*2^(maxRetries-1)) the following numbers represent the
	// sequence of delays between successive queuings of a service.
	//
	// 5ms, 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1.3s, 2.6s, 5.1s, 10.2s, 20.4s, 41s, 82s
	maxRetries        = 15
	controllerName    = "service-cidrs-controller.k8s.io"
	clusterIPKeyIndex = "spec.clusterIP"
)

// NewController returns a new *Controller.
func NewController(
	serviceInformer coreinformers.ServiceInformer,
	serviceCIDRInformer networkinginformers.ServiceCIDRInformer,
	ipAddressInformer networkinginformers.IPAddressInformer,
	client clientset.Interface,
) *Controller {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartStructuredLogging(0)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName})

	c := &Controller{
		client:           client,
		ipQueue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "ipaddresses"),
		cidrQueue:        workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "servicecidrs"),
		svcQueue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "services"),
		treeV4:           iptree.New(false),
		treeV6:           iptree.New(true),
		workerLoopPeriod: time.Second,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addService,
		UpdateFunc: c.updateService,
		DeleteFunc: c.deleteService,
	})
	c.serviceLister = serviceInformer.Lister()
	c.servicesSynced = serviceInformer.Informer().HasSynced

	serviceInformer.Informer().GetIndexer().AddIndexers(cache.Indexers{
		clusterIPKeyIndex: func(obj interface{}) ([]string, error) {
			svc, ok := obj.(*v1.Service)
			if !ok {
				return []string{}, nil
			}
			// We are only interested in active pods with nodeName set
			if len(svc.Spec.ClusterIPs) == 0 || svc.Spec.ClusterIP == v1.ClusterIPNone {
				return []string{}, nil
			}
			return svc.Spec.ClusterIPs, nil
		},
	})

	svcIndexer := serviceInformer.Informer().GetIndexer()
	c.getServicesAssignedToIP = func(ip string) ([]*v1.Service, error) {
		objs, err := svcIndexer.ByIndex(clusterIPKeyIndex, ip)
		if err != nil {
			return nil, err
		}
		svcs := make([]*v1.Service, 0, len(objs))
		for _, obj := range objs {
			svc, ok := obj.(*v1.Service)
			if !ok {
				continue
			}
			svcs = append(svcs, svc)
		}
		return svcs, nil
	}

	serviceCIDRInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addServiceCIDR,
		UpdateFunc: c.updateServiceCIDR,
		DeleteFunc: c.deleteServiceCIDR,
	})
	c.serviceCIDRLister = serviceCIDRInformer.Lister()
	c.serviceCIDRsSynced = serviceCIDRInformer.Informer().HasSynced

	ipAddressInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addIPAddress,
		UpdateFunc: c.updateIPAddress,
		DeleteFunc: c.deleteIPAddress,
	})
	c.ipAddressLister = ipAddressInformer.Lister()
	c.ipAddressSynced = ipAddressInformer.Informer().HasSynced

	c.eventBroadcaster = broadcaster
	c.eventRecorder = recorder

	return c
}

// Controller manages selector-based service ipAddress.
type Controller struct {
	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	getServicesAssignedToIP func(ip string) ([]*v1.Service, error)

	serviceLister  corelisters.ServiceLister
	servicesSynced cache.InformerSynced

	serviceCIDRLister  networkinglisters.ServiceCIDRLister
	serviceCIDRsSynced cache.InformerSynced

	ipAddressLister networkinglisters.IPAddressLister
	ipAddressSynced cache.InformerSynced

	ipQueue   workqueue.RateLimitingInterface
	cidrQueue workqueue.RateLimitingInterface
	svcQueue  workqueue.RateLimitingInterface

	// workerLoopPeriod is the time between worker runs. The workers process the queue of service and ipRange changes.
	workerLoopPeriod time.Duration

	muTree sync.Mutex
	treeV4 *iptree.Tree
	treeV6 *iptree.Tree
}

// Run will not return until stopCh is closed.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.ipQueue.ShutDown()
	defer c.cidrQueue.ShutDown()
	defer c.svcQueue.ShutDown()

	klog.Infof("Starting %s", controllerName)
	defer klog.Infof("Shutting down %s", controllerName)

	if !cache.WaitForNamedCacheSync(controllerName, stopCh, c.servicesSynced, c.serviceCIDRsSynced, c.ipAddressSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.ipWorker, c.workerLoopPeriod, stopCh)
		go wait.Until(c.cidrWorker, c.workerLoopPeriod, stopCh)
		go wait.Until(c.svcWorker, c.workerLoopPeriod, stopCh)
	}
	<-stopCh
}

func (c *Controller) enqueue(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}

	switch v := obj.(type) {
	// Enqueue the Service and the corresponding IPAddresses, we can do it
	// because the Name of the IPAddress object matches the normalized ClusterIP format.
	case *v1.Service:
		c.svcQueue.Add(key)
		for _, clusterIP := range v.Spec.ClusterIPs {
			// normalize
			ip := netutils.ParseIPSloppy(clusterIP)
			if ip != nil {
				c.ipQueue.Add(ip.String())
			}
		}
	case *networkingapiv1alpha1.ServiceCIDR:
		c.cidrQueue.Add(key)
	case *networkingapiv1alpha1.IPAddress:
		c.ipQueue.Add(key)
	default:
		utilruntime.HandleError(fmt.Errorf("unrecognized type %v for object %#v: ", v, obj))
	}
}

func sameStringSlice(a []string, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// needToRemoveFinalizer checks if object is candidate to be deleted
func needToRemoveFinalizer(obj metav1.Object, finalizer string) bool {
	if obj.GetDeletionTimestamp() == nil {
		return false
	}
	for _, f := range obj.GetFinalizers() {
		if f == finalizer {
			return true
		}
	}
	return false
}

// needToAddFinalizer checks if need to add finalizer to object
func needToAddFinalizer(obj metav1.Object, finalizer string) bool {
	if obj.GetDeletionTimestamp() != nil {
		return false
	}
	for _, f := range obj.GetFinalizers() {
		if f == finalizer {
			return false
		}
	}
	return true
}
