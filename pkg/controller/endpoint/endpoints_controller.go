/*
Copyright 2014 The Kubernetes Authors.

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

package endpoint

import (
	"fmt"
	"reflect"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/v1/endpoints"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	utilpod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/pkg/util/workqueue"

	"github.com/golang/glog"
)

const (
	// We'll attempt to recompute EVERY service's endpoints at least this
	// often. Higher numbers = lower CPU/network load; lower numbers =
	// shorter amount of time before a mistaken endpoint is corrected.
	FullServiceResyncPeriod = 30 * time.Second

	// We must avoid syncing service until the pod store has synced. If it hasn't synced, to
	// avoid a hot loop, we'll wait this long between checks.
	PodStoreSyncedPollPeriod = 100 * time.Millisecond

	// An annotation on the Service denoting if the endpoints controller should
	// go ahead and create endpoints for unready pods. This annotation is
	// currently only used by StatefulSets, where we need the pod to be DNS
	// resolvable during initialization and termination. In this situation we
	// create a headless Service just for the StatefulSet, and clients shouldn't
	// be using this Service for anything so unready endpoints don't matter.
	// Endpoints of these Services retain their DNS records and continue
	// receiving traffic for the Service from the moment the kubelet starts all
	// containers in the pod and marks it "Running", till the kubelet stops all
	// containers and deletes the pod from the apiserver.
	TolerateUnreadyEndpointsAnnotation = "service.alpha.kubernetes.io/tolerate-unready-endpoints"
)

var (
	keyFunc = cache.DeletionHandlingMetaNamespaceKeyFunc
)

// NewEndpointController returns a new *EndpointController.
func NewEndpointController(podInformer cache.SharedIndexInformer, client clientset.Interface) *EndpointController {
	if client != nil && client.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("endpoint_controller", client.Core().RESTClient().GetRateLimiter())
	}
	e := &EndpointController{
		client: client,
		queue:  workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "endpoint"),
	}

	e.serviceStore.Indexer, e.serviceController = cache.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return e.client.Core().Services(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return e.client.Core().Services(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.Service{},
		// TODO: Can we have much longer period here?
		FullServiceResyncPeriod,
		cache.ResourceEventHandlerFuncs{
			AddFunc: e.enqueueService,
			UpdateFunc: func(old, cur interface{}) {
				e.enqueueService(cur)
			},
			DeleteFunc: e.enqueueService,
		},
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    e.addPod,
		UpdateFunc: e.updatePod,
		DeleteFunc: e.deletePod,
	})
	e.podStore.Indexer = podInformer.GetIndexer()
	e.podController = podInformer.GetController()
	e.podStoreSynced = podInformer.HasSynced

	return e
}

// NewEndpointControllerFromClient returns a new *EndpointController that runs its own informer.
func NewEndpointControllerFromClient(client *clientset.Clientset, resyncPeriod controller.ResyncPeriodFunc) *EndpointController {
	podInformer := informers.NewPodInformer(client, resyncPeriod())
	e := NewEndpointController(podInformer, client)
	e.internalPodInformer = podInformer

	return e
}

// EndpointController manages selector-based service endpoints.
type EndpointController struct {
	client clientset.Interface

	serviceStore cache.StoreToServiceLister
	podStore     cache.StoreToPodLister

	// internalPodInformer is used to hold a personal informer.  If we're using
	// a normal shared informer, then the informer will be started for us.  If
	// we have a personal informer, we must start it ourselves.   If you start
	// the controller using NewEndpointController(passing SharedInformer), this
	// will be null
	internalPodInformer cache.SharedIndexInformer

	// Services that need to be updated. A channel is inappropriate here,
	// because it allows services with lots of pods to be serviced much
	// more often than services with few pods; it also would cause a
	// service that's inserted multiple times to be processed more than
	// necessary.
	queue workqueue.RateLimitingInterface

	// Since we join two objects, we'll watch both of them with
	// controllers.
	serviceController cache.Controller
	podController     cache.Controller
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool
}

// Runs e; will not return until stopCh is closed. workers determines how many
// endpoints will be handled in parallel.
func (e *EndpointController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer e.queue.ShutDown()

	go e.serviceController.Run(stopCh)
	go e.podController.Run(stopCh)

	if !cache.WaitForCacheSync(stopCh, e.podStoreSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(e.worker, time.Second, stopCh)
	}
	go func() {
		defer utilruntime.HandleCrash()
		time.Sleep(5 * time.Minute) // give time for our cache to fill
		e.checkLeftoverEndpoints()
	}()

	if e.internalPodInformer != nil {
		go e.internalPodInformer.Run(stopCh)
	}

	<-stopCh
}

func (e *EndpointController) getPodServiceMemberships(pod *v1.Pod) (sets.String, error) {
	set := sets.String{}
	services, err := e.serviceStore.GetPodServices(pod)
	if err != nil {
		// don't log this error because this function makes pointless
		// errors when no services match.
		return set, nil
	}
	for i := range services {
		key, err := keyFunc(services[i])
		if err != nil {
			return nil, err
		}
		set.Insert(key)
	}
	return set, nil
}

// When a pod is added, figure out what services it will be a member of and
// enqueue them. obj must have *v1.Pod type.
func (e *EndpointController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	services, err := e.getPodServiceMemberships(pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to get pod %v/%v's service memberships: %v", pod.Namespace, pod.Name, err))
		return
	}
	for key := range services {
		e.queue.Add(key)
	}
}

// When a pod is updated, figure out what services it used to be a member of
// and what services it will be a member of, and enqueue the union of these.
// old and cur must be *v1.Pod types.
func (e *EndpointController) updatePod(old, cur interface{}) {
	newPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if newPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}
	services, err := e.getPodServiceMemberships(newPod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to get pod %v/%v's service memberships: %v", newPod.Namespace, newPod.Name, err))
		return
	}

	// Only need to get the old services if the labels changed.
	if !reflect.DeepEqual(newPod.Labels, oldPod.Labels) ||
		!hostNameAndDomainAreEqual(newPod, oldPod) {
		oldServices, err := e.getPodServiceMemberships(oldPod)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Unable to get pod %v/%v's service memberships: %v", oldPod.Namespace, oldPod.Name, err))
			return
		}
		services = services.Union(oldServices)
	}
	for key := range services {
		e.queue.Add(key)
	}
}

func hostNameAndDomainAreEqual(pod1, pod2 *v1.Pod) bool {
	return getHostname(pod1) == getHostname(pod2) &&
		getSubdomain(pod1) == getSubdomain(pod2)
}

func getHostname(pod *v1.Pod) string {
	if len(pod.Spec.Hostname) > 0 {
		return pod.Spec.Hostname
	}
	if pod.Annotations != nil {
		return pod.Annotations[utilpod.PodHostnameAnnotation]
	}
	return ""
}

func getSubdomain(pod *v1.Pod) string {
	if len(pod.Spec.Subdomain) > 0 {
		return pod.Spec.Subdomain
	}
	if pod.Annotations != nil {
		return pod.Annotations[utilpod.PodSubdomainAnnotation]
	}
	return ""
}

// When a pod is deleted, enqueue the services the pod used to be a member of.
// obj could be an *v1.Pod, or a DeletionFinalStateUnknown marker item.
func (e *EndpointController) deletePod(obj interface{}) {
	if _, ok := obj.(*v1.Pod); ok {
		// Enqueue all the services that the pod used to be a member
		// of. This happens to be exactly the same thing we do when a
		// pod is added.
		e.addPod(obj)
		return
	}
	podKey, err := keyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", obj, err))
		return
	}
	glog.V(4).Infof("Pod %q was deleted but we don't have a record of its final state, so it will take up to %v before it will be removed from all endpoint records.", podKey, FullServiceResyncPeriod)

	// TODO: keep a map of pods to services to handle this condition.
}

// obj could be an *v1.Service, or a DeletionFinalStateUnknown marker item.
func (e *EndpointController) enqueueService(obj interface{}) {
	key, err := keyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	e.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and
// marks them done. You may run as many of these in parallel as you wish; the
// workqueue guarantees that they will not end up processing the same service
// at the same time.
func (e *EndpointController) worker() {
	for e.processNextWorkItem() {
	}
}

func (e *EndpointController) processNextWorkItem() bool {
	eKey, quit := e.queue.Get()
	if quit {
		return false
	}
	defer e.queue.Done(eKey)

	err := e.syncService(eKey.(string))
	if err == nil {
		e.queue.Forget(eKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Sync %v failed with %v", eKey, err))
	e.queue.AddRateLimited(eKey)

	return true
}

func (e *EndpointController) syncService(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing service %q endpoints. (%v)", key, time.Now().Sub(startTime))
	}()

	obj, exists, err := e.serviceStore.Indexer.GetByKey(key)
	if err != nil || !exists {
		// Delete the corresponding endpoint, as the service has been deleted.
		// TODO: Please note that this will delete an endpoint when a
		// service is deleted. However, if we're down at the time when
		// the service is deleted, we will miss that deletion, so this
		// doesn't completely solve the problem. See #6877.
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Need to delete endpoint with key %q, but couldn't understand the key: %v", key, err))
			// Don't retry, as the key isn't going to magically become understandable.
			return nil
		}
		err = e.client.Core().Endpoints(namespace).Delete(name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		return nil
	}

	service := obj.(*v1.Service)
	if service.Spec.Selector == nil {
		// services without a selector receive no endpoints from this controller;
		// these services will receive the endpoints that are created out-of-band via the REST API.
		return nil
	}

	glog.V(5).Infof("About to update endpoints for service %q", key)
	pods, err := e.podStore.Pods(service.Namespace).List(labels.Set(service.Spec.Selector).AsSelectorPreValidated())
	if err != nil {
		// Since we're getting stuff from a local cache, it is
		// basically impossible to get this error.
		return err
	}

	subsets := []v1.EndpointSubset{}

	var tolerateUnreadyEndpoints bool
	if v, ok := service.Annotations[TolerateUnreadyEndpointsAnnotation]; ok {
		b, err := strconv.ParseBool(v)
		if err == nil {
			tolerateUnreadyEndpoints = b
		} else {
			utilruntime.HandleError(fmt.Errorf("Failed to parse annotation %v: %v", TolerateUnreadyEndpointsAnnotation, err))
		}
	}

	readyEps := 0
	notReadyEps := 0
	for i := range pods {
		// TODO: Do we need to copy here?
		pod := &(*pods[i])

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]

			portName := servicePort.Name
			portProto := servicePort.Protocol
			portNum, err := podutil.FindPort(pod, servicePort)
			if err != nil {
				glog.V(4).Infof("Failed to find port for service %s/%s: %v", service.Namespace, service.Name, err)
				continue
			}
			if len(pod.Status.PodIP) == 0 {
				glog.V(5).Infof("Failed to find an IP for pod %s/%s", pod.Namespace, pod.Name)
				continue
			}
			if !tolerateUnreadyEndpoints && pod.DeletionTimestamp != nil {
				glog.V(5).Infof("Pod is being deleted %s/%s", pod.Namespace, pod.Name)
				continue
			}

			epp := v1.EndpointPort{Name: portName, Port: int32(portNum), Protocol: portProto}
			epa := v1.EndpointAddress{
				IP:       pod.Status.PodIP,
				NodeName: &pod.Spec.NodeName,
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       pod.ObjectMeta.Namespace,
					Name:            pod.ObjectMeta.Name,
					UID:             pod.ObjectMeta.UID,
					ResourceVersion: pod.ObjectMeta.ResourceVersion,
				}}

			hostname := getHostname(pod)
			if len(hostname) > 0 &&
				getSubdomain(pod) == service.Name &&
				service.Namespace == pod.Namespace {
				epa.Hostname = hostname
			}

			if tolerateUnreadyEndpoints || v1.IsPodReady(pod) {
				subsets = append(subsets, v1.EndpointSubset{
					Addresses: []v1.EndpointAddress{epa},
					Ports:     []v1.EndpointPort{epp},
				})
				readyEps++
			} else {
				glog.V(5).Infof("Pod is out of service: %v/%v", pod.Namespace, pod.Name)
				subsets = append(subsets, v1.EndpointSubset{
					NotReadyAddresses: []v1.EndpointAddress{epa},
					Ports:             []v1.EndpointPort{epp},
				})
				notReadyEps++
			}
		}
	}
	subsets = endpoints.RepackSubsets(subsets)

	// See if there's actually an update here.
	currentEndpoints, err := e.client.Core().Endpoints(service.Namespace).Get(service.Name, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			currentEndpoints = &v1.Endpoints{
				ObjectMeta: v1.ObjectMeta{
					Name:   service.Name,
					Labels: service.Labels,
				},
			}
		} else {
			return err
		}
	}

	if reflect.DeepEqual(currentEndpoints.Subsets, subsets) &&
		reflect.DeepEqual(currentEndpoints.Labels, service.Labels) {
		glog.V(5).Infof("endpoints are equal for %s/%s, skipping update", service.Namespace, service.Name)
		return nil
	}
	newEndpoints := currentEndpoints
	newEndpoints.Subsets = subsets
	newEndpoints.Labels = service.Labels
	if newEndpoints.Annotations == nil {
		newEndpoints.Annotations = make(map[string]string)
	}

	glog.V(4).Infof("Update endpoints for %v/%v, ready: %d not ready: %d", service.Namespace, service.Name, readyEps, notReadyEps)
	createEndpoints := len(currentEndpoints.ResourceVersion) == 0
	if createEndpoints {
		// No previous endpoints, create them
		_, err = e.client.Core().Endpoints(service.Namespace).Create(newEndpoints)
	} else {
		// Pre-existing
		_, err = e.client.Core().Endpoints(service.Namespace).Update(newEndpoints)
	}
	if err != nil {
		if createEndpoints && errors.IsForbidden(err) {
			// A request is forbidden primarily for two reasons:
			// 1. namespace is terminating, endpoint creation is not allowed by default.
			// 2. policy is misconfigured, in which case no service would function anywhere.
			// Given the frequency of 1, we log at a lower level.
			glog.V(5).Infof("Forbidden from creating endpoints: %v", err)
		}
		return err
	}
	return nil
}

// checkLeftoverEndpoints lists all currently existing endpoints and adds their
// service to the queue. This will detect endpoints that exist with no
// corresponding service; these endpoints need to be deleted. We only need to
// do this once on startup, because in steady-state these are detected (but
// some stragglers could have been left behind if the endpoint controller
// reboots).
func (e *EndpointController) checkLeftoverEndpoints() {
	list, err := e.client.Core().Endpoints(v1.NamespaceAll).List(v1.ListOptions{})
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to list endpoints (%v); orphaned endpoints will not be cleaned up. (They're pretty harmless, but you can restart this component if you want another attempt made.)", err))
		return
	}
	for i := range list.Items {
		ep := &list.Items[i]
		key, err := keyFunc(ep)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Unable to get key for endpoint %#v", ep))
			continue
		}
		e.queue.Add(key)
	}
}
