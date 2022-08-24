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
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	corev1informers "k8s.io/client-go/informers/core/v1"
	networkingv1alpha1informers "k8s.io/client-go/informers/networking/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	networkingv1alpha1listers "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/pkg/util/iptree"
	netutils "k8s.io/utils/net"
)

const (
	// maxRetries is the number of times a service will be retried before it is dropped out of the queue.
	// With the current rate-limiter in use (5ms*2^(maxRetries-1)) the following numbers represent the
	// sequence of delays between successive queuings of a service.
	//
	// 5ms, 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1.3s, 2.6s, 5.1s, 10.2s, 20.4s, 41s, 82s
	maxRetries     = 15
	controllerName = "bootstrap-service-cidrs-controller.k8s.io"
	serviceName    = "kubernetes"
)

// NewController returns a new *Controller.
func NewController(
	primaryRange net.IPNet,
	secondaryRange net.IPNet,
	publicIP net.IP,
	servicePort int,
	publicServicePort int,
	publicServiceNodePort int,
	endpointsReconciler reconcilers.EndpointReconciler,
	client clientset.Interface,
) *Controller {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartStructuredLogging(0)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName})

	c := &Controller{
		primaryRange:          primaryRange,
		secondaryRange:        secondaryRange,
		endpointReconciler:    endpointsReconciler,
		publicIP:              publicIP,
		servicePort:           servicePort,
		publicServicePort:     publicServicePort,
		publicServiceNodePort: publicServiceNodePort,
		client:                client,
		queue:                 workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "bootstrap-service-cidr"),
		workerLoopPeriod:      time.Second,
		treeV4:                iptree.New(false),
		treeV6:                iptree.New(true),
	}
	// we construct our own informer because we need such a small subset of the information available.
	// The kubernetes.default service
	serviceInformer := corev1informers.NewFilteredServiceInformer(client, metav1.NamespaceDefault, 12*time.Hour,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", "kubernetes").String()
		})

	serviceInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addService,
		UpdateFunc: c.updateService,
		DeleteFunc: c.deleteService,
	})
	c.serviceInformer = serviceInformer
	c.serviceLister = corev1listers.NewServiceLister(serviceInformer.GetIndexer())
	c.servicesSynced = serviceInformer.HasSynced

	serviceCIDRInformer := networkingv1alpha1informers.NewServiceCIDRInformer(client, 12*time.Hour,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	serviceCIDRInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addServiceCIDR,
		UpdateFunc: c.updateServiceCIDR,
		DeleteFunc: c.deleteServiceCIDR,
	})

	c.serviceCIDRInformer = serviceCIDRInformer
	c.serviceCIDRLister = networkingv1alpha1listers.NewServiceCIDRLister(serviceCIDRInformer.GetIndexer())
	c.serviceCIDRsSynced = serviceCIDRInformer.HasSynced

	c.eventBroadcaster = broadcaster
	c.eventRecorder = recorder

	return c
}

// Controller manages selector-based service ipAddress.
type Controller struct {
	primaryRange   net.IPNet
	secondaryRange net.IPNet

	endpointReconciler reconcilers.EndpointReconciler
	// service frontend
	serviceIP   net.IP
	servicePort int

	// public means reachable/used for endpoints
	publicIP              net.IP
	publicServicePort     int
	publicServiceNodePort int

	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	serviceInformer cache.SharedIndexInformer
	serviceLister   corev1listers.ServiceLister
	servicesSynced  cache.InformerSynced

	serviceCIDRInformer cache.SharedIndexInformer
	serviceCIDRLister   networkingv1alpha1listers.ServiceCIDRLister
	serviceCIDRsSynced  cache.InformerSynced

	queue workqueue.RateLimitingInterface

	// workerLoopPeriod is the time between worker runs. The workers process the queue of service and ipRange changes.
	workerLoopPeriod time.Duration

	muTree sync.Mutex
	treeV4 *iptree.Tree
	treeV6 *iptree.Tree
}

// Run will not return until stopCh is closed.
func (c *Controller) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting %s", controllerName)
	defer klog.Infof("Shutting down %s", controllerName)
	go c.serviceInformer.Run(stopCh)
	go c.serviceCIDRInformer.Run(stopCh)

	if !cache.WaitForNamedCacheSync(controllerName, stopCh, c.servicesSynced, c.serviceCIDRsSynced) {
		return
	}

	// just one worker only
	go wait.Until(c.worker, c.workerLoopPeriod, stopCh)

	// trigger first sync if caches are empty
	if len(c.serviceCIDRInformer.GetStore().ListKeys()) == 0 {
		klog.Infof("No ServiceCIDR existing, bootstrapping from flags")
		c.queue.Add("bootstrap")
	}
	<-stopCh
	if c.endpointReconciler == nil {
		return
	}
	c.endpointReconciler.StopReconciling()
	if err := c.endpointReconciler.RemoveEndpoints(serviceName, c.publicIP, c.endpointPorts()); err != nil {
		klog.Errorf("Unable to remove endpoints from kubernetes service: %v", err)
	}
	c.endpointReconciler.Destroy()
}

// createEndpointPortSpec creates the endpoint ports
func (c *Controller) endpointPorts() []v1.EndpointPort {
	return []v1.EndpointPort{{
		Protocol: v1.ProtocolTCP,
		Port:     int32(c.publicServicePort),
		Name:     "https",
	}}
}

func (c *Controller) createDefaultServiceCIDRIfNeeded(ipv4, ipv6 string) error {
	klog.V(4).Infof("Create default ServiceCIDR, ipv4: %s ipv6: %s", ipv4, ipv6)
	s := &networkingapiv1alpha1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kubernetes-default-",
			Labels: map[string]string{
				networkingapiv1alpha1.LabelServiceCIDRFromFlags: "true",
			},
			Finalizers: []string{networkingapiv1alpha1.ServiceCIDRProtectionFinalizer},
		},
		Spec: networkingapiv1alpha1.ServiceCIDRSpec{
			IPv4: ipv4,
			IPv6: ipv6,
		},
	}
	_, err := c.client.NetworkingV1alpha1().ServiceCIDRs().Create(context.TODO(), s, metav1.CreateOptions{})
	if errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// CreateOrUpdateMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (c *Controller) createDefaultServiceIfNeeded(ip net.IP) error {
	klog.V(4).Infof("Create kubernetes.default with IP %s", ip.String())
	servicePorts := []v1.ServicePort{{
		Protocol:   v1.ProtocolTCP,
		Port:       int32(c.servicePort),
		Name:       "https",
		TargetPort: intstr.FromInt(c.publicServicePort),
	}}
	serviceType := v1.ServiceTypeClusterIP
	if c.publicServiceNodePort > 0 {
		servicePorts[0].NodePort = int32(c.publicServiceNodePort)
		serviceType = v1.ServiceTypeNodePort
	}

	singleStack := v1.IPFamilyPolicySingleStack
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
		Spec: v1.ServiceSpec{
			Ports: servicePorts,
			// maintained by this code, not by the pod selector
			Selector:        nil,
			ClusterIP:       ip.String(),
			IPFamilyPolicy:  &singleStack,
			SessionAffinity: v1.ServiceAffinityNone,
			Type:            serviceType,
		},
	}

	_, err := c.client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{})
	if errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

func (c *Controller) worker() {
	for c.processNext() {
	}
}

func (c *Controller) processNext() bool {
	eKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(eKey)

	err := c.sync(eKey.(string))
	c.handleErr(err, eKey)

	return true
}

func (c *Controller) handleErr(err error, key interface{}) {
	if err == nil {
		c.queue.Forget(key)
		return
	}

	if c.queue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Bootstrap Services controller error syncing, retrying", "key", key, "err", err)
		c.queue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping key %q out of the queue: %v", key, err)
	c.queue.Forget(key)
	utilruntime.HandleError(err)
}

func (c *Controller) sync(key string) error {
	startTime := time.Now()
	klog.V(4).Infof("Bootstrap Services controller starting syncing key: %s", key)

	defer func() {
		klog.V(4).Infof("Bootstrap Services controller finished syncing key: %s: %v", key, time.Since(startTime))
	}()

	// obtain configuration from flags
	var ipv4, ipv6 *net.IPNet
	if netutils.IsIPv4CIDR(&c.primaryRange) {
		ipv4 = &c.primaryRange
	} else if netutils.IsIPv4CIDR(&c.secondaryRange) {
		ipv4 = &c.secondaryRange
	}
	if netutils.IsIPv6CIDR(&c.primaryRange) {
		ipv6 = &c.primaryRange
	} else if netutils.IsIPv6CIDR(&c.secondaryRange) {
		ipv6 = &c.secondaryRange
	}

	// check if already exists a ServiceCIDR containing the flags
	// don't look for IP families that are not set in the flags
	var foundIPv4, foundIPv6 *networkingapiv1alpha1.ServiceCIDR
	var err error
	if ipv4 != nil {
		c.muTree.Lock()
		_, v, ok := c.treeV4.LongestPrefixMatch(ipv4.String())
		if ok {
			foundIPv4, err = c.serviceCIDRLister.Get(v.(string))
			if err != nil {
				c.muTree.Unlock()
				return err
			}
		}
		c.muTree.Unlock()
	}
	if ipv6 != nil {
		c.muTree.Lock()
		_, v, ok := c.treeV6.LongestPrefixMatch(ipv6.String())
		if ok {
			foundIPv6, err = c.serviceCIDRLister.Get(v.(string))
			if err != nil {
				c.muTree.Unlock()
				return err
			}
		}
		c.muTree.Unlock()
	}
	// create the serviceCIDR from flags if necessary
	var ipV4String, ipV6String string
	if ipv4 != nil && foundIPv4 == nil {
		ipV4String = ipv4.String()
	}
	if ipv6 != nil && foundIPv6 == nil {
		ipV6String = ipv6.String()
	}
	if ipV4String != "" || ipV6String != "" {
		err = c.createDefaultServiceCIDRIfNeeded(ipV4String, ipV6String)
		if err != nil {
			return err
		}
	}

	// kubernetes.default
	return c.syncService()
}

// syncService handles the kubernetes.default Service
func (c *Controller) syncService() error {
	// check kubernetes.default Service exist
	_, err := c.serviceLister.Services(metav1.NamespaceDefault).Get(serviceName)
	if err != nil {
		// create the Service using the first IP from the primary range
		if errors.IsNotFound(err) {
			serviceIP, err := netutils.GetIndexedIP(&c.primaryRange, 1)
			if err != nil {
				return err
			}
			err = c.createDefaultServiceIfNeeded(serviceIP)
			if err != nil {
				return err
			}
		} else {
			return err
		}
	}
	// if the service exist it must match one of the ServiceCIDRs

	// check endpoints (if necessary)
	if c.endpointReconciler == nil {
		return nil
	}
	if err := c.endpointReconciler.ReconcileEndpoints(serviceName, c.publicIP, c.endpointPorts(), false); err != nil {
		return err
	}
	return nil
}

func (c *Controller) addServiceCIDR(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}
	cidr := obj.(*networkingapiv1alpha1.ServiceCIDR)
	c.muTree.Lock()
	c.treeV4.Insert(cidr.Spec.IPv4, cidr.Name)
	c.treeV6.Insert(cidr.Spec.IPv6, cidr.Name)
	c.muTree.Unlock()
	c.queue.Add(key)
}

// service CIDRs are inmutable, check finalizer and labels weren't changed
func (c *Controller) updateServiceCIDR(oldObj, obj interface{}) {
	//old := oldObj.(*networkingapiv1alpha1.ServiceCIDR)
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}
	cidr := obj.(*networkingapiv1alpha1.ServiceCIDR)
	if cidr.GetDeletionTimestamp() != nil {
		c.muTree.Lock()
		c.treeV4.Delete(cidr.Spec.IPv4)
		c.treeV6.Delete(cidr.Spec.IPv6)
		c.muTree.Unlock()
	} else {
		c.muTree.Lock()
		c.treeV4.Insert(cidr.Spec.IPv4, cidr.Name)
		c.treeV6.Insert(cidr.Spec.IPv6, cidr.Name)
		c.muTree.Unlock()
	}
	c.queue.Add(key)
}

func (c *Controller) deleteServiceCIDR(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}
	cidr, ok := obj.(*networkingapiv1alpha1.ServiceCIDR)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
			return
		}
		cidr, ok = tombstone.Obj.(*networkingapiv1alpha1.ServiceCIDR)
		if !ok {
			klog.V(2).Infof("Tombstone contained unexpected object: %#v", obj)
			return
		}
	}
	c.muTree.Lock()
	c.treeV4.Delete(cidr.Spec.IPv4)
	c.treeV6.Delete(cidr.Spec.IPv6)
	c.muTree.Unlock()
	c.queue.Add(key)
}

func (c *Controller) addService(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}
	c.queue.Add(key)

}

// kubernetes.default Service must have the first IP of the default ServiceCIDR range
func (c *Controller) updateService(oldObj, obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}
	c.queue.Add(key)
}

func (c *Controller) deleteService(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}
	c.queue.Add(key)
}

// subnetContains check if subnet a contains subnet b
func subnetContains(a, b *net.IPNet) bool {
	// check if there are some IPs in common
	if a.Contains(b.IP) || b.Contains(a.IP) {
		aBits, aMask := a.Mask.Size()
		bBits, bMask := b.Mask.Size()
		// different family
		if aMask != bMask {
			return false
		}
		// check a is larger than b (bits is lower)
		return aBits <= bBits
	}
	return false
}
