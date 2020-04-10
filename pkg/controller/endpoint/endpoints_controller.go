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
	"context"
	"fmt"
	"reflect"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/v1/endpoints"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
	"k8s.io/kubernetes/pkg/features"
	utillabels "k8s.io/kubernetes/pkg/util/labels"
	utilnet "k8s.io/utils/net"
)

const (
	// maxRetries is the number of times a service will be retried before it is dropped out of the queue.
	// With the current rate-limiter in use (5ms*2^(maxRetries-1)) the following numbers represent the
	// sequence of delays between successive queuings of a service.
	//
	// 5ms, 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1.3s, 2.6s, 5.1s, 10.2s, 20.4s, 41s, 82s
	maxRetries = 15

	// TolerateUnreadyEndpointsAnnotation is an annotation on the Service denoting if the endpoints
	// controller should go ahead and create endpoints for unready pods. This annotation is
	// currently only used by StatefulSets, where we need the pod to be DNS
	// resolvable during initialization and termination. In this situation we
	// create a headless Service just for the StatefulSet, and clients shouldn't
	// be using this Service for anything so unready endpoints don't matter.
	// Endpoints of these Services retain their DNS records and continue
	// receiving traffic for the Service from the moment the kubelet starts all
	// containers in the pod and marks it "Running", till the kubelet stops all
	// containers and deletes the pod from the apiserver.
	// This field is deprecated. v1.Service.PublishNotReadyAddresses will replace it
	// subsequent releases.  It will be removed no sooner than 1.13.
	TolerateUnreadyEndpointsAnnotation = "service.alpha.kubernetes.io/tolerate-unready-endpoints"
)

// NewEndpointController returns a new *EndpointController.
func NewEndpointController(podInformer coreinformers.PodInformer, serviceInformer coreinformers.ServiceInformer,
	endpointsInformer coreinformers.EndpointsInformer, client clientset.Interface, endpointUpdatesBatchPeriod time.Duration) *EndpointController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(klog.Infof)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "endpoint-controller"})

	if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("endpoint_controller", client.CoreV1().RESTClient().GetRateLimiter())
	}
	e := &EndpointController{
		client:           client,
		queue:            workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "endpoint"),
		workerLoopPeriod: time.Second,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: e.onServiceUpdate,
		UpdateFunc: func(old, cur interface{}) {
			e.onServiceUpdate(cur)
		},
		DeleteFunc: e.onServiceDelete,
	})
	e.serviceLister = serviceInformer.Lister()
	e.servicesSynced = serviceInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    e.addPod,
		UpdateFunc: e.updatePod,
		DeleteFunc: e.deletePod,
	})
	e.podLister = podInformer.Lister()
	e.podsSynced = podInformer.Informer().HasSynced

	e.endpointsLister = endpointsInformer.Lister()
	e.endpointsSynced = endpointsInformer.Informer().HasSynced

	e.triggerTimeTracker = endpointutil.NewTriggerTimeTracker()
	e.eventBroadcaster = broadcaster
	e.eventRecorder = recorder

	e.endpointUpdatesBatchPeriod = endpointUpdatesBatchPeriod

	e.serviceSelectorCache = endpointutil.NewServiceSelectorCache()

	return e
}

// EndpointController manages selector-based service endpoints.
type EndpointController struct {
	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	// serviceLister is able to list/get services and is populated by the shared informer passed to
	// NewEndpointController.
	serviceLister corelisters.ServiceLister
	// servicesSynced returns true if the service shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	servicesSynced cache.InformerSynced

	// podLister is able to list/get pods and is populated by the shared informer passed to
	// NewEndpointController.
	podLister corelisters.PodLister
	// podsSynced returns true if the pod shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podsSynced cache.InformerSynced

	// endpointsLister is able to list/get endpoints and is populated by the shared informer passed to
	// NewEndpointController.
	endpointsLister corelisters.EndpointsLister
	// endpointsSynced returns true if the endpoints shared informer has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	endpointsSynced cache.InformerSynced

	// Services that need to be updated. A channel is inappropriate here,
	// because it allows services with lots of pods to be serviced much
	// more often than services with few pods; it also would cause a
	// service that's inserted multiple times to be processed more than
	// necessary.
	queue workqueue.RateLimitingInterface

	// workerLoopPeriod is the time between worker runs. The workers process the queue of service and pod changes.
	workerLoopPeriod time.Duration

	// triggerTimeTracker is an util used to compute and export the EndpointsLastChangeTriggerTime
	// annotation.
	triggerTimeTracker *endpointutil.TriggerTimeTracker

	endpointUpdatesBatchPeriod time.Duration

	// serviceSelectorCache is a cache of service selectors to avoid high CPU consumption caused by frequent calls
	// to AsSelectorPreValidated (see #73527)
	serviceSelectorCache *endpointutil.ServiceSelectorCache
}

// Run will not return until stopCh is closed. workers determines how many
// endpoints will be handled in parallel.
func (e *EndpointController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer e.queue.ShutDown()

	klog.Infof("Starting endpoint controller")
	defer klog.Infof("Shutting down endpoint controller")

	if !cache.WaitForNamedCacheSync("endpoint", stopCh, e.podsSynced, e.servicesSynced, e.endpointsSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(e.worker, e.workerLoopPeriod, stopCh)
	}

	go func() {
		defer utilruntime.HandleCrash()
		e.checkLeftoverEndpoints()
	}()

	<-stopCh
}

// When a pod is added, figure out what services it will be a member of and
// enqueue them. obj must have *v1.Pod type.
func (e *EndpointController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	services, err := e.serviceSelectorCache.GetPodServiceMemberships(e.serviceLister, pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to get pod %s/%s's service memberships: %v", pod.Namespace, pod.Name, err))
		return
	}
	for key := range services {
		e.queue.AddAfter(key, e.endpointUpdatesBatchPeriod)
	}
}

func podToEndpointAddressForService(svc *v1.Service, pod *v1.Pod) (*v1.EndpointAddress, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return podToEndpointAddress(pod), nil
	}

	// api-server service controller ensured that the service got the correct IP Family
	// according to user setup, here we only need to match EndPoint IPs' family to service
	// actual IP family. as in, we don't need to check service.IPFamily

	ipv6ClusterIP := utilnet.IsIPv6String(svc.Spec.ClusterIP)
	for _, podIP := range pod.Status.PodIPs {
		ipv6PodIP := utilnet.IsIPv6String(podIP.IP)
		// same family?
		// TODO (khenidak) when we remove the max of 2 PodIP limit from pods
		// we will have to return multiple endpoint addresses
		if ipv6ClusterIP == ipv6PodIP {
			return &v1.EndpointAddress{
				IP:       podIP.IP,
				NodeName: &pod.Spec.NodeName,
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       pod.ObjectMeta.Namespace,
					Name:            pod.ObjectMeta.Name,
					UID:             pod.ObjectMeta.UID,
					ResourceVersion: pod.ObjectMeta.ResourceVersion,
				}}, nil
		}
	}
	return nil, fmt.Errorf("failed to find a matching endpoint for service %v", svc.Name)
}

func podToEndpointAddress(pod *v1.Pod) *v1.EndpointAddress {
	return &v1.EndpointAddress{
		IP:       pod.Status.PodIP,
		NodeName: &pod.Spec.NodeName,
		TargetRef: &v1.ObjectReference{
			Kind:            "Pod",
			Namespace:       pod.ObjectMeta.Namespace,
			Name:            pod.ObjectMeta.Name,
			UID:             pod.ObjectMeta.UID,
			ResourceVersion: pod.ObjectMeta.ResourceVersion,
		}}
}

func endpointChanged(pod1, pod2 *v1.Pod) bool {
	endpointAddress1 := podToEndpointAddress(pod1)
	endpointAddress2 := podToEndpointAddress(pod2)

	endpointAddress1.TargetRef.ResourceVersion = ""
	endpointAddress2.TargetRef.ResourceVersion = ""

	return !reflect.DeepEqual(endpointAddress1, endpointAddress2)
}

// When a pod is updated, figure out what services it used to be a member of
// and what services it will be a member of, and enqueue the union of these.
// old and cur must be *v1.Pod types.
func (e *EndpointController) updatePod(old, cur interface{}) {
	services := endpointutil.GetServicesToUpdateOnPodChange(e.serviceLister, e.serviceSelectorCache, old, cur, endpointChanged)
	for key := range services {
		e.queue.AddAfter(key, e.endpointUpdatesBatchPeriod)
	}
}

// When a pod is deleted, enqueue the services the pod used to be a member of.
// obj could be an *v1.Pod, or a DeletionFinalStateUnknown marker item.
func (e *EndpointController) deletePod(obj interface{}) {
	pod := endpointutil.GetPodFromDeleteAction(obj)
	if pod != nil {
		e.addPod(pod)
	}
}

// onServiceUpdate updates the Service Selector in the cache and queues the Service for processing.
func (e *EndpointController) onServiceUpdate(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	_ = e.serviceSelectorCache.Update(key, obj.(*v1.Service).Spec.Selector)
	e.queue.Add(key)
}

// onServiceDelete removes the Service Selector from the cache and queues the Service for processing.
func (e *EndpointController) onServiceDelete(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	e.serviceSelectorCache.Delete(key)
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
	e.handleErr(err, eKey)

	return true
}

func (e *EndpointController) handleErr(err error, key interface{}) {
	if err == nil {
		e.queue.Forget(key)
		return
	}

	if e.queue.NumRequeues(key) < maxRetries {
		klog.V(2).Infof("Error syncing endpoints for service %q, retrying. Error: %v", key, err)
		e.queue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping service %q out of the queue: %v", key, err)
	e.queue.Forget(key)
	utilruntime.HandleError(err)
}

func (e *EndpointController) syncService(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing service %q endpoints. (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	service, err := e.serviceLister.Services(namespace).Get(name)
	if err != nil {
		if !errors.IsNotFound(err) {
			return err
		}

		// Delete the corresponding endpoint, as the service has been deleted.
		// TODO: Please note that this will delete an endpoint when a
		// service is deleted. However, if we're down at the time when
		// the service is deleted, we will miss that deletion, so this
		// doesn't completely solve the problem. See #6877.
		err = e.client.CoreV1().Endpoints(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{})
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		e.triggerTimeTracker.DeleteService(namespace, name)
		return nil
	}

	if service.Spec.Selector == nil {
		// services without a selector receive no endpoints from this controller;
		// these services will receive the endpoints that are created out-of-band via the REST API.
		return nil
	}

	klog.V(5).Infof("About to update endpoints for service %q", key)
	pods, err := e.podLister.Pods(service.Namespace).List(labels.Set(service.Spec.Selector).AsSelectorPreValidated())
	if err != nil {
		// Since we're getting stuff from a local cache, it is
		// basically impossible to get this error.
		return err
	}

	// If the user specified the older (deprecated) annotation, we have to respect it.
	tolerateUnreadyEndpoints := service.Spec.PublishNotReadyAddresses
	if v, ok := service.Annotations[TolerateUnreadyEndpointsAnnotation]; ok {
		b, err := strconv.ParseBool(v)
		if err == nil {
			tolerateUnreadyEndpoints = b
		} else {
			utilruntime.HandleError(fmt.Errorf("Failed to parse annotation %v: %v", TolerateUnreadyEndpointsAnnotation, err))
		}
	}

	// We call ComputeEndpointLastChangeTriggerTime here to make sure that the
	// state of the trigger time tracker gets updated even if the sync turns out
	// to be no-op and we don't update the endpoints object.
	endpointsLastChangeTriggerTime := e.triggerTimeTracker.
		ComputeEndpointLastChangeTriggerTime(namespace, service, pods)

	subsets := []v1.EndpointSubset{}
	var totalReadyEps int
	var totalNotReadyEps int

	for _, pod := range pods {
		if len(pod.Status.PodIP) == 0 {
			klog.V(5).Infof("Failed to find an IP for pod %s/%s", pod.Namespace, pod.Name)
			continue
		}
		if !tolerateUnreadyEndpoints && pod.DeletionTimestamp != nil {
			klog.V(5).Infof("Pod is being deleted %s/%s", pod.Namespace, pod.Name)
			continue
		}

		ep, err := podToEndpointAddressForService(service, pod)
		if err != nil {
			// this will happen, if the cluster runs with some nodes configured as dual stack and some as not
			// such as the case of an upgrade..
			klog.V(2).Infof("failed to find endpoint for service:%v with ClusterIP:%v on pod:%v with error:%v", service.Name, service.Spec.ClusterIP, pod.Name, err)
			continue
		}

		epa := *ep
		if endpointutil.ShouldSetHostname(pod, service) {
			epa.Hostname = pod.Spec.Hostname
		}

		// Allow headless service not to have ports.
		if len(service.Spec.Ports) == 0 {
			if service.Spec.ClusterIP == api.ClusterIPNone {
				subsets, totalReadyEps, totalNotReadyEps = addEndpointSubset(subsets, pod, epa, nil, tolerateUnreadyEndpoints)
				// No need to repack subsets for headless service without ports.
			}
		} else {
			for i := range service.Spec.Ports {
				servicePort := &service.Spec.Ports[i]
				portNum, err := podutil.FindPort(pod, servicePort)
				if err != nil {
					klog.V(4).Infof("Failed to find port for service %s/%s: %v", service.Namespace, service.Name, err)
					continue
				}
				epp := endpointPortFromServicePort(servicePort, portNum)

				var readyEps, notReadyEps int
				subsets, readyEps, notReadyEps = addEndpointSubset(subsets, pod, epa, epp, tolerateUnreadyEndpoints)
				totalReadyEps = totalReadyEps + readyEps
				totalNotReadyEps = totalNotReadyEps + notReadyEps
			}
		}
	}
	subsets = endpoints.RepackSubsets(subsets)

	// See if there's actually an update here.
	currentEndpoints, err := e.endpointsLister.Endpoints(service.Namespace).Get(service.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			currentEndpoints = &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:   service.Name,
					Labels: service.Labels,
				},
			}
		} else {
			return err
		}
	}

	createEndpoints := len(currentEndpoints.ResourceVersion) == 0

	if !createEndpoints &&
		apiequality.Semantic.DeepEqual(currentEndpoints.Subsets, subsets) &&
		apiequality.Semantic.DeepEqual(currentEndpoints.Labels, service.Labels) {
		klog.V(5).Infof("endpoints are equal for %s/%s, skipping update", service.Namespace, service.Name)
		return nil
	}
	newEndpoints := currentEndpoints.DeepCopy()
	newEndpoints.Subsets = subsets
	newEndpoints.Labels = service.Labels
	if newEndpoints.Annotations == nil {
		newEndpoints.Annotations = make(map[string]string)
	}

	if !endpointsLastChangeTriggerTime.IsZero() {
		newEndpoints.Annotations[v1.EndpointsLastChangeTriggerTime] =
			endpointsLastChangeTriggerTime.Format(time.RFC3339Nano)
	} else { // No new trigger time, clear the annotation.
		delete(newEndpoints.Annotations, v1.EndpointsLastChangeTriggerTime)
	}

	if newEndpoints.Labels == nil {
		newEndpoints.Labels = make(map[string]string)
	}

	if !helper.IsServiceIPSet(service) {
		newEndpoints.Labels = utillabels.CloneAndAddLabel(newEndpoints.Labels, v1.IsHeadlessService, "")
	} else {
		newEndpoints.Labels = utillabels.CloneAndRemoveLabel(newEndpoints.Labels, v1.IsHeadlessService)
	}

	klog.V(4).Infof("Update endpoints for %v/%v, ready: %d not ready: %d", service.Namespace, service.Name, totalReadyEps, totalNotReadyEps)
	if createEndpoints {
		// No previous endpoints, create them
		_, err = e.client.CoreV1().Endpoints(service.Namespace).Create(context.TODO(), newEndpoints, metav1.CreateOptions{})
	} else {
		// Pre-existing
		_, err = e.client.CoreV1().Endpoints(service.Namespace).Update(context.TODO(), newEndpoints, metav1.UpdateOptions{})
	}
	if err != nil {
		if createEndpoints && errors.IsForbidden(err) {
			// A request is forbidden primarily for two reasons:
			// 1. namespace is terminating, endpoint creation is not allowed by default.
			// 2. policy is misconfigured, in which case no service would function anywhere.
			// Given the frequency of 1, we log at a lower level.
			klog.V(5).Infof("Forbidden from creating endpoints: %v", err)

			// If the namespace is terminating, creates will continue to fail. Simply drop the item.
			if errors.HasStatusCause(err, v1.NamespaceTerminatingCause) {
				return nil
			}
		}

		if createEndpoints {
			e.eventRecorder.Eventf(newEndpoints, v1.EventTypeWarning, "FailedToCreateEndpoint", "Failed to create endpoint for service %v/%v: %v", service.Namespace, service.Name, err)
		} else {
			e.eventRecorder.Eventf(newEndpoints, v1.EventTypeWarning, "FailedToUpdateEndpoint", "Failed to update endpoint %v/%v: %v", service.Namespace, service.Name, err)
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
	list, err := e.endpointsLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to list endpoints (%v); orphaned endpoints will not be cleaned up. (They're pretty harmless, but you can restart this component if you want another attempt made.)", err))
		return
	}
	for _, ep := range list {
		if _, ok := ep.Annotations[resourcelock.LeaderElectionRecordAnnotationKey]; ok {
			// when there are multiple controller-manager instances,
			// we observe that it will delete leader-election endpoints after 5min
			// and cause re-election
			// so skip the delete here
			// as leader-election only have endpoints without service
			continue
		}
		key, err := controller.KeyFunc(ep)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Unable to get key for endpoint %#v", ep))
			continue
		}
		e.queue.Add(key)
	}
}

func addEndpointSubset(subsets []v1.EndpointSubset, pod *v1.Pod, epa v1.EndpointAddress,
	epp *v1.EndpointPort, tolerateUnreadyEndpoints bool) ([]v1.EndpointSubset, int, int) {
	var readyEps int
	var notReadyEps int
	ports := []v1.EndpointPort{}
	if epp != nil {
		ports = append(ports, *epp)
	}
	if tolerateUnreadyEndpoints || podutil.IsPodReady(pod) {
		subsets = append(subsets, v1.EndpointSubset{
			Addresses: []v1.EndpointAddress{epa},
			Ports:     ports,
		})
		readyEps++
	} else if shouldPodBeInEndpoints(pod) {
		klog.V(5).Infof("Pod is out of service: %s/%s", pod.Namespace, pod.Name)
		subsets = append(subsets, v1.EndpointSubset{
			NotReadyAddresses: []v1.EndpointAddress{epa},
			Ports:             ports,
		})
		notReadyEps++
	}
	return subsets, readyEps, notReadyEps
}

func shouldPodBeInEndpoints(pod *v1.Pod) bool {
	switch pod.Spec.RestartPolicy {
	case v1.RestartPolicyNever:
		return pod.Status.Phase != v1.PodFailed && pod.Status.Phase != v1.PodSucceeded
	case v1.RestartPolicyOnFailure:
		return pod.Status.Phase != v1.PodSucceeded
	default:
		return true
	}
}

func endpointPortFromServicePort(servicePort *v1.ServicePort, portNum int) *v1.EndpointPort {
	epp := &v1.EndpointPort{
		Name:     servicePort.Name,
		Port:     int32(portNum),
		Protocol: servicePort.Protocol,
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ServiceAppProtocol) {
		epp.AppProtocol = servicePort.AppProtocol
	}
	return epp
}
