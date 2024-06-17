/*
Copyright 2023 The Kubernetes Authors.

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

package controller

import (
	"context"
	"fmt"
	"net"
	"net/netip"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	networkinginformers "k8s.io/client-go/informers/networking/v1alpha1"
	"k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	networkinglisters "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/util/iptree"
	"k8s.io/utils/clock"
	netutils "k8s.io/utils/net"
)

const (
	// maxRetries is the number of times a service will be retried before it is dropped out of the queue.
	// With the current rate-limiter in use (5ms*2^(maxRetries-1)) the following numbers represent the
	// sequence of delays between successive queuings of a service.
	//
	// 5ms, 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1.3s, 2.6s, 5.1s, 10.2s, 20.4s, 41s, 82s
	maxRetries = 15
	workers    = 5
)

// Repair is a controller loop that examines all service ClusterIP allocations and logs any errors,
// and then creates the accurate list of IPAddresses objects with all allocated ClusterIPs.
//
// Handles:
// * Duplicate ClusterIP assignments caused by operator action or undetected race conditions
// * Allocations to services that were not actually created due to a crash or powerloss
// * Migrates old versions of Kubernetes services into the new ipallocator automatically
//   creating the corresponding IPAddress objects
// * IPAddress objects with wrong references or labels
//
// Logs about:
// * ClusterIPs that do not match the currently configured range
//
// There is a one-to-one relation between Service ClusterIPs and IPAddresses.
// The bidirectional relation is achieved using the following fields:
// Service.Spec.Cluster == IPAddress.Name AND IPAddress.ParentRef == Service
//
// The controller use two reconcile loops, one for Services and other for IPAddress.
// The Service reconcile loop verifies the bidirectional relation exists and is correct.
// 1. Service_X [ClusterIP_X]  <------>  IPAddress_X [Ref:Service_X]   ok
// 2. Service_Y [ClusterIP_Y]  <------>  IPAddress_Y [Ref:GatewayA]    !ok, wrong reference
// 3. Service_Z [ClusterIP_Z]  <------>  							   !ok, missing IPAddress
// 4. Service_A [ClusterIP_A]  <------>  IPAddress_A [Ref:Service_B]   !ok, duplicate IPAddress
//    Service_B [ClusterIP_A]  <------> 								only one service can verify the relation
// The IPAddress reconcile loop checks there are no orphan IPAddresses, the rest of the
// cases are covered by the Services loop
// 1.                          <------>  IPAddress_Z [Ref:Service_C]   !ok, orphan IPAddress

type RepairIPAddress struct {
	client   kubernetes.Interface
	interval time.Duration

	serviceLister  corelisters.ServiceLister
	servicesSynced cache.InformerSynced

	serviceCIDRLister networkinglisters.ServiceCIDRLister
	serviceCIDRSynced cache.InformerSynced

	ipAddressLister networkinglisters.IPAddressLister
	ipAddressSynced cache.InformerSynced

	cidrQueue        workqueue.TypedRateLimitingInterface[string]
	svcQueue         workqueue.TypedRateLimitingInterface[string]
	ipQueue          workqueue.TypedRateLimitingInterface[string]
	workerLoopPeriod time.Duration

	muTree sync.Mutex
	tree   *iptree.Tree[string]

	broadcaster events.EventBroadcaster
	recorder    events.EventRecorder
	clock       clock.Clock
}

// NewRepair creates a controller that periodically ensures that all clusterIPs are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
func NewRepairIPAddress(interval time.Duration,
	client kubernetes.Interface,
	serviceInformer coreinformers.ServiceInformer,
	serviceCIDRInformer networkinginformers.ServiceCIDRInformer,
	ipAddressInformer networkinginformers.IPAddressInformer) *RepairIPAddress {
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	recorder := eventBroadcaster.NewRecorder(legacyscheme.Scheme, "ipallocator-repair-controller")

	r := &RepairIPAddress{
		interval:          interval,
		client:            client,
		serviceLister:     serviceInformer.Lister(),
		servicesSynced:    serviceInformer.Informer().HasSynced,
		serviceCIDRLister: serviceCIDRInformer.Lister(),
		serviceCIDRSynced: serviceCIDRInformer.Informer().HasSynced,
		ipAddressLister:   ipAddressInformer.Lister(),
		ipAddressSynced:   ipAddressInformer.Informer().HasSynced,
		cidrQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "servicecidrs"},
		),
		svcQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "services"},
		),
		ipQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "ipaddresses"},
		),
		tree:             iptree.New[string](),
		workerLoopPeriod: time.Second,
		broadcaster:      eventBroadcaster,
		recorder:         recorder,
		clock:            clock.RealClock{},
	}

	_, _ = serviceInformer.Informer().AddEventHandlerWithResyncPeriod(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil {
				r.svcQueue.Add(key)
			}
		},
		UpdateFunc: func(old interface{}, new interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err == nil {
				r.svcQueue.Add(key)
			}
		},
		DeleteFunc: func(obj interface{}) {
			// IndexerInformer uses a delta queue, therefore for deletes we have to use this
			// key function.
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil {
				r.svcQueue.Add(key)
			}
		},
	}, interval)

	_, _ = serviceCIDRInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil {
				r.cidrQueue.Add(key)
			}
		},
		UpdateFunc: func(old interface{}, new interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err == nil {
				r.cidrQueue.Add(key)
			}
		},
		DeleteFunc: func(obj interface{}) {
			// IndexerInformer uses a delta queue, therefore for deletes we have to use this
			// key function.
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil {
				r.cidrQueue.Add(key)
			}
		},
	})

	ipAddressInformer.Informer().AddEventHandlerWithResyncPeriod(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil {
				r.ipQueue.Add(key)
			}
		},
		UpdateFunc: func(old interface{}, new interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err == nil {
				r.ipQueue.Add(key)
			}
		},
		DeleteFunc: func(obj interface{}) {
			// IndexerInformer uses a delta queue, therefore for deletes we have to use this
			// key function.
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil {
				r.ipQueue.Add(key)
			}
		},
	}, interval)

	return r
}

// RunUntil starts the controller until the provided ch is closed.
func (r *RepairIPAddress) RunUntil(onFirstSuccess func(), stopCh chan struct{}) {
	defer r.cidrQueue.ShutDown()
	defer r.ipQueue.ShutDown()
	defer r.svcQueue.ShutDown()
	r.broadcaster.StartRecordingToSink(stopCh)
	defer r.broadcaster.Shutdown()

	klog.Info("Starting ipallocator-repair-controller")
	defer klog.Info("Shutting down ipallocator-repair-controller")

	if !cache.WaitForNamedCacheSync("ipallocator-repair-controller", stopCh, r.ipAddressSynced, r.servicesSynced, r.serviceCIDRSynced) {
		return
	}

	// First sync goes through all the Services and IPAddresses in the cache,
	// once synced, it signals the main loop and works using the handlers, since
	// it's less expensive and more optimal.
	if err := r.runOnce(); err != nil {
		runtime.HandleError(err)
		return
	}
	onFirstSuccess()

	// serialize the operations on ServiceCIDRs
	go wait.Until(r.cidrWorker, r.workerLoopPeriod, stopCh)

	for i := 0; i < workers; i++ {
		go wait.Until(r.ipWorker, r.workerLoopPeriod, stopCh)
		go wait.Until(r.svcWorker, r.workerLoopPeriod, stopCh)
	}

	<-stopCh
}

// runOnce verifies the state of the ClusterIP allocations and returns an error if an unrecoverable problem occurs.
func (r *RepairIPAddress) runOnce() error {
	return retry.RetryOnConflict(retry.DefaultBackoff, r.doRunOnce)
}

// doRunOnce verifies the state of the ClusterIP allocations and returns an error if an unrecoverable problem occurs.
func (r *RepairIPAddress) doRunOnce() error {
	services, err := r.serviceLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}

	// Check every Service's ClusterIP, and rebuild the state as we think it should be.
	for _, svc := range services {
		key, err := cache.MetaNamespaceKeyFunc(svc)
		if err != nil {
			return err
		}
		err = r.syncService(key)
		if err != nil {
			return err
		}
	}

	// We have checked that every Service has its corresponding IP.
	// Check that there is no IP created by the allocator without
	// a Service associated.
	ipLabelSelector := labels.Set(map[string]string{
		networkingv1alpha1.LabelManagedBy: ipallocator.ControllerName,
	}).AsSelectorPreValidated()
	ipAddresses, err := r.ipAddressLister.List(ipLabelSelector)
	if err != nil {
		return fmt.Errorf("unable to refresh the IPAddress block: %v", err)
	}
	// Check every IPAddress matches the corresponding Service, and rebuild the state as we think it should be.
	for _, ipAddress := range ipAddresses {
		key, err := cache.MetaNamespaceKeyFunc(ipAddress)
		if err != nil {
			return err
		}
		err = r.syncIPAddress(key)
		if err != nil {
			return err
		}
	}

	return nil
}

func (r *RepairIPAddress) svcWorker() {
	for r.processNextWorkSvc() {
	}
}

func (r *RepairIPAddress) processNextWorkSvc() bool {
	eKey, quit := r.svcQueue.Get()
	if quit {
		return false
	}
	defer r.svcQueue.Done(eKey)

	err := r.syncService(eKey)
	r.handleSvcErr(err, eKey)

	return true
}

func (r *RepairIPAddress) handleSvcErr(err error, key string) {
	if err == nil {
		r.svcQueue.Forget(key)
		return
	}

	if r.svcQueue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Error syncing Service, retrying", "service", key, "err", err)
		r.svcQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping Service %q out of the queue: %v", key, err)
	r.svcQueue.Forget(key)
	runtime.HandleError(err)
}

// syncServices reconcile the Service ClusterIPs to verify that each one has the corresponding IPAddress object associated
func (r *RepairIPAddress) syncService(key string) error {
	var syncError error
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	svc, err := r.serviceLister.Services(namespace).Get(name)
	if err != nil {
		// nothing to do
		return nil
	}
	if !helper.IsServiceIPSet(svc) {
		// didn't need a ClusterIP
		return nil
	}

	for _, clusterIP := range svc.Spec.ClusterIPs {
		ip := netutils.ParseIPSloppy(clusterIP)
		if ip == nil {
			// ClusterIP is corrupt, ClusterIPs are already validated, but double checking here
			// in case there are some inconsistencies with the parsers
			r.recorder.Eventf(svc, nil, v1.EventTypeWarning, "ClusterIPNotValid", "ClusterIPValidation", "Cluster IP %s is not a valid IP; please recreate Service", ip)
			runtime.HandleError(fmt.Errorf("the ClusterIP %s for Service %s/%s is not a valid IP; please recreate Service", ip, svc.Namespace, svc.Name))
			continue
		}
		// TODO(aojea) Refactor to abstract the IPs checks
		family := getFamilyByIP(ip)

		r.muTree.Lock()
		prefixes := r.tree.GetHostIPPrefixMatches(ipToAddr(ip))
		r.muTree.Unlock()
		if len(prefixes) == 0 {
			// ClusterIP is out of range
			r.recorder.Eventf(svc, nil, v1.EventTypeWarning, "ClusterIPOutOfRange", "ClusterIPAllocation", "Cluster IP [%v]: %s is not within any configured Service CIDR; please recreate service", family, ip)
			runtime.HandleError(fmt.Errorf("the ClusterIP [%v]: %s for Service %s/%s is not within any service CIDR; please recreate", family, ip, svc.Namespace, svc.Name))
			continue
		}

		// Get the IPAddress object associated to the ClusterIP
		ipAddress, err := r.ipAddressLister.Get(ip.String())
		if apierrors.IsNotFound(err) {
			// ClusterIP doesn't seem to be allocated, create it.
			r.recorder.Eventf(svc, nil, v1.EventTypeWarning, "ClusterIPNotAllocated", "ClusterIPAllocation", "Cluster IP [%v]: %s is not allocated; repairing", family, ip)
			runtime.HandleError(fmt.Errorf("the ClusterIP [%v]: %s for Service %s/%s is not allocated; repairing", family, ip, svc.Namespace, svc.Name))
			_, err := r.client.NetworkingV1alpha1().IPAddresses().Create(context.Background(), newIPAddress(ip.String(), svc), metav1.CreateOptions{})
			if err != nil {
				return err
			}
			continue
		}
		if err != nil {
			r.recorder.Eventf(svc, nil, v1.EventTypeWarning, "UnknownError", "ClusterIPAllocation", "Unable to allocate ClusterIP [%v]: %s due to an unknown error", family, ip)
			return fmt.Errorf("unable to allocate ClusterIP [%v]: %s for Service %s/%s due to an unknown error, will retry later: %v", family, ip, svc.Namespace, svc.Name, err)
		}

		// IPAddress that belongs to a Service must reference a Service
		if ipAddress.Spec.ParentRef.Group != "" ||
			ipAddress.Spec.ParentRef.Resource != "services" {
			r.recorder.Eventf(svc, nil, v1.EventTypeWarning, "ClusterIPNotAllocated", "ClusterIPAllocation", "the ClusterIP [%v]: %s for Service %s/%s has a wrong reference; repairing", family, ip, svc.Namespace, svc.Name)
			if err := r.recreateIPAddress(ipAddress.Name, svc); err != nil {
				return err
			}
			continue
		}

		// IPAddress that belongs to a Service must reference the current Service
		if ipAddress.Spec.ParentRef.Namespace != svc.Namespace ||
			ipAddress.Spec.ParentRef.Name != svc.Name {
			// verify that there are no two Services with the same IP, otherwise
			// it will keep deleting and recreating the same IPAddress changing the reference
			refService, err := r.serviceLister.Services(ipAddress.Spec.ParentRef.Namespace).Get(ipAddress.Spec.ParentRef.Name)
			if err != nil {
				r.recorder.Eventf(svc, nil, v1.EventTypeWarning, "ClusterIPNotAllocated", "ClusterIPAllocation", "the ClusterIP [%v]: %s for Service %s/%s has a wrong reference; repairing", family, ip, svc.Namespace, svc.Name)
				if err := r.recreateIPAddress(ipAddress.Name, svc); err != nil {
					return err
				}
				continue
			}
			// the IPAddress is duplicate but current Service is not the referenced, it has to be recreated
			for _, clusterIP := range refService.Spec.ClusterIPs {
				if ipAddress.Name == clusterIP {
					r.recorder.Eventf(svc, nil, v1.EventTypeWarning, "ClusterIPAlreadyAllocated", "ClusterIPAllocation", "Cluster IP [%v]:%s was assigned to multiple services; please recreate service", family, ip)
					runtime.HandleError(fmt.Errorf("the cluster IP [%v]:%s for service %s/%s was assigned to other services %s/%s; please recreate", family, ip, svc.Namespace, svc.Name, refService.Namespace, refService.Name))
					break
				}
			}
		}

		// IPAddress must have the corresponding labels assigned by the allocator
		if !verifyIPAddressLabels(ipAddress) {
			if err := r.recreateIPAddress(ipAddress.Name, svc); err != nil {
				return err
			}
			continue
		}

	}
	return syncError
}

func (r *RepairIPAddress) recreateIPAddress(name string, svc *v1.Service) error {
	err := r.client.NetworkingV1alpha1().IPAddresses().Delete(context.Background(), name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	_, err = r.client.NetworkingV1alpha1().IPAddresses().Create(context.Background(), newIPAddress(name, svc), metav1.CreateOptions{})
	if err != nil {
		return err
	}
	return nil
}

func (r *RepairIPAddress) ipWorker() {
	for r.processNextWorkIp() {
	}
}

func (r *RepairIPAddress) processNextWorkIp() bool {
	eKey, quit := r.ipQueue.Get()
	if quit {
		return false
	}
	defer r.ipQueue.Done(eKey)

	err := r.syncIPAddress(eKey)
	r.handleIPErr(err, eKey)

	return true
}

func (r *RepairIPAddress) handleIPErr(err error, key string) {
	if err == nil {
		r.ipQueue.Forget(key)
		return
	}

	if r.ipQueue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Error syncing Service, retrying", "service", key, "err", err)
		r.ipQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping Service %q out of the queue: %v", key, err)
	r.ipQueue.Forget(key)
	runtime.HandleError(err)
}

// syncIPAddress verify that the IPAddress that are owned by the ipallocator controller reference an existing Service
// to avoid leaking IPAddresses. IPAddresses that are owned by other controllers are not processed to avoid hotloops.
// IPAddress that reference Services and are part of the ClusterIP are validated in the syncService loop.
func (r *RepairIPAddress) syncIPAddress(key string) error {
	ipAddress, err := r.ipAddressLister.Get(key)
	if err != nil {
		// nothing to do
		return nil
	}

	// not mananged by this controller
	if !managedByController(ipAddress) {
		return nil
	}

	// does not reference a Service but created by the service allocator, something else have changed it, delete it
	if ipAddress.Spec.ParentRef.Group != "" || ipAddress.Spec.ParentRef.Resource != "services" {
		runtime.HandleError(fmt.Errorf("IPAddress %s appears to have been modified, not referencing a Service %v: cleaning up", ipAddress.Name, ipAddress.Spec.ParentRef))
		r.recorder.Eventf(ipAddress, nil, v1.EventTypeWarning, "IPAddressNotAllocated", "IPAddressAllocation", "IPAddress %s appears to have been modified, not referencing a Service %v: cleaning up", ipAddress.Name, ipAddress.Spec.ParentRef)
		err := r.client.NetworkingV1alpha1().IPAddresses().Delete(context.Background(), ipAddress.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		return nil
	}

	svc, err := r.serviceLister.Services(ipAddress.Spec.ParentRef.Namespace).Get(ipAddress.Spec.ParentRef.Name)
	if apierrors.IsNotFound(err) {
		// cleaning all IPAddress without an owner reference IF the time since it was created is greater than 60 seconds (default timeout value on the kube-apiserver)
		// This is required because during the Service creation there is a time that the IPAddress object exists but the Service is still being created
		// Assume that CreationTimestamp exists.
		ipLifetime := r.clock.Now().Sub(ipAddress.CreationTimestamp.Time)
		gracePeriod := 60 * time.Second
		if ipLifetime > gracePeriod {
			runtime.HandleError(fmt.Errorf("IPAddress %s appears to have leaked: cleaning up", ipAddress.Name))
			r.recorder.Eventf(ipAddress, nil, v1.EventTypeWarning, "IPAddressNotAllocated", "IPAddressAllocation", "IPAddress: %s for Service %s/%s appears to have leaked: cleaning up", ipAddress.Name, ipAddress.Spec.ParentRef.Namespace, ipAddress.Spec.ParentRef.Name)
			err := r.client.NetworkingV1alpha1().IPAddresses().Delete(context.Background(), ipAddress.Name, metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				return err
			}
		}
		// requeue after the grace period
		r.ipQueue.AddAfter(key, gracePeriod-ipLifetime)
		return nil
	}
	if err != nil {
		runtime.HandleError(fmt.Errorf("unable to get parent Service for IPAddress %s due to an unknown error: %v", ipAddress, err))
		r.recorder.Eventf(ipAddress, nil, v1.EventTypeWarning, "UnknownError", "IPAddressAllocation", "Unable to get parent Service for IPAddress %s due to an unknown error", ipAddress)
		return err
	}
	// The service exists, we have checked in previous loop that all Service to IPAddress are correct
	// but we also have to check the reverse, that the IPAddress to Service relation is correct
	for _, clusterIP := range svc.Spec.ClusterIPs {
		if ipAddress.Name == clusterIP {
			return nil
		}
	}
	runtime.HandleError(fmt.Errorf("the IPAddress: %s for Service %s/%s has a wrong reference %#v; cleaning up", ipAddress.Name, svc.Name, svc.Namespace, ipAddress.Spec.ParentRef))
	r.recorder.Eventf(ipAddress, nil, v1.EventTypeWarning, "IPAddressWrongReference", "IPAddressAllocation", "IPAddress: %s for Service %s/%s has a wrong reference; cleaning up", ipAddress.Name, svc.Namespace, svc.Name)
	err = r.client.NetworkingV1alpha1().IPAddresses().Delete(context.Background(), ipAddress.Name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil

}

func (r *RepairIPAddress) cidrWorker() {
	for r.processNextWorkCIDR() {
	}
}

func (r *RepairIPAddress) processNextWorkCIDR() bool {
	eKey, quit := r.cidrQueue.Get()
	if quit {
		return false
	}
	defer r.cidrQueue.Done(eKey)

	err := r.syncCIDRs()
	r.handleCIDRErr(err, eKey)

	return true
}

func (r *RepairIPAddress) handleCIDRErr(err error, key string) {
	if err == nil {
		r.cidrQueue.Forget(key)
		return
	}

	if r.cidrQueue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Error syncing ServiceCIDR, retrying", "serviceCIDR", key, "err", err)
		r.cidrQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping ServiceCIDR %q out of the queue: %v", key, err)
	r.cidrQueue.Forget(key)
	runtime.HandleError(err)
}

// syncCIDRs rebuilds the radix tree based from the informers cache
func (r *RepairIPAddress) syncCIDRs() error {
	serviceCIDRList, err := r.serviceCIDRLister.List(labels.Everything())
	if err != nil {
		return err
	}

	tree := iptree.New[string]()
	for _, serviceCIDR := range serviceCIDRList {
		for _, cidr := range serviceCIDR.Spec.CIDRs {
			if prefix, err := netip.ParsePrefix(cidr); err == nil { // it can not fail since is already validated
				tree.InsertPrefix(prefix, serviceCIDR.Name)
			}
		}
	}
	r.muTree.Lock()
	defer r.muTree.Unlock()
	r.tree = tree
	return nil
}

func newIPAddress(name string, svc *v1.Service) *networkingv1alpha1.IPAddress {
	family := string(v1.IPv4Protocol)
	if netutils.IsIPv6String(name) {
		family = string(v1.IPv6Protocol)
	}
	return &networkingv1alpha1.IPAddress{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				networkingv1alpha1.LabelIPAddressFamily: family,
				networkingv1alpha1.LabelManagedBy:       ipallocator.ControllerName,
			},
		},
		Spec: networkingv1alpha1.IPAddressSpec{
			ParentRef: serviceToRef(svc),
		},
	}
}

func serviceToRef(svc *v1.Service) *networkingv1alpha1.ParentReference {
	if svc == nil {
		return nil
	}

	return &networkingv1alpha1.ParentReference{
		Group:     "",
		Resource:  "services",
		Namespace: svc.Namespace,
		Name:      svc.Name,
	}
}

func getFamilyByIP(ip net.IP) v1.IPFamily {
	if netutils.IsIPv6(ip) {
		return v1.IPv6Protocol
	}
	return v1.IPv4Protocol
}

// managedByController returns true if the controller of the provided
// EndpointSlices is the EndpointSlice controller.
func managedByController(ip *networkingv1alpha1.IPAddress) bool {
	managedBy, ok := ip.Labels[networkingv1alpha1.LabelManagedBy]
	if !ok {
		return false
	}
	return managedBy == ipallocator.ControllerName
}

func verifyIPAddressLabels(ip *networkingv1alpha1.IPAddress) bool {
	labelFamily, ok := ip.Labels[networkingv1alpha1.LabelIPAddressFamily]
	if !ok {
		return false
	}

	family := string(v1.IPv4Protocol)
	if netutils.IsIPv6String(ip.Name) {
		family = string(v1.IPv6Protocol)
	}
	if family != labelFamily {
		return false
	}
	return managedByController(ip)
}

// TODO(aojea) move to utils, already in pkg/registry/core/service/ipallocator/cidrallocator.go
// ipToAddr converts a net.IP to a netip.Addr
// if the net.IP is not valid it returns an empty netip.Addr{}
func ipToAddr(ip net.IP) netip.Addr {
	// https://pkg.go.dev/net/netip#AddrFromSlice can return an IPv4 in IPv6 format
	// so we have to check the IP family to return exactly the format that we want
	// address, _ := netip.AddrFromSlice(net.ParseIPSloppy(192.168.0.1)) returns
	// an address like ::ffff:192.168.0.1/32
	bytes := ip.To4()
	if bytes == nil {
		bytes = ip.To16()
	}
	// AddrFromSlice returns Addr{}, false if the input is invalid.
	address, _ := netip.AddrFromSlice(bytes)
	return address
}
