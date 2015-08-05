/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/pkg/service/endpoints_controller.go
package endpointcontroller

import (
	"fmt"
	"reflect"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	// We'll attempt to recompute EVERY service's endpoints at least this
	// often. Higher numbers = lower CPU/network load; lower numbers =
	// shorter amount of time before a mistaken endpoint is corrected.
	FullServiceResyncPeriod = 30 * time.Second

	// We'll keep pod watches open up to this long. In the unlikely case
	// that a watch misdelivers info about a pod, it'll take this long for
	// that mistake to be rectified.
	PodRelistPeriod = 5 * time.Minute
)

var (
	keyFunc = framework.DeletionHandlingMetaNamespaceKeyFunc
)

// NewEndpointController returns a new *EndpointController.
func NewEndpointController(client *client.Client) *EndpointController {
	e := &EndpointController{
		client: client,
		queue:  workqueue.New(),
	}

	e.serviceStore.Store, e.serviceController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return e.client.Services(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return e.client.Services(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Service{},
		FullServiceResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: e.enqueueService,
			UpdateFunc: func(old, cur interface{}) {
				e.enqueueService(cur)
			},
			DeleteFunc: e.enqueueService,
		},
	)

	e.podStore.Store, e.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return e.client.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return e.client.Pods(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Pod{},
		PodRelistPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    e.addPod,
			UpdateFunc: e.updatePod,
			DeleteFunc: e.deletePod,
		},
	)

	return e
}

// EndpointController manages selector-based service endpoints.
type EndpointController struct {
	client *client.Client

	serviceStore cache.StoreToServiceLister
	podStore     cache.StoreToPodLister

	// Services that need to be updated. A channel is inappropriate here,
	// because it allows services with lots of pods to be serviced much
	// more often than services with few pods; it also would cause a
	// service that's inserted multiple times to be processed more than
	// necessary.
	queue *workqueue.Type

	// Since we join two objects, we'll watch both of them with
	// controllers.
	serviceController *framework.Controller
	podController     *framework.Controller
}

// Runs e; will not return until stopCh is closed. workers determines how many
// endpoints will be handled in parallel.
func (e *EndpointController) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()
	go e.serviceController.Run(stopCh)
	go e.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(e.worker, time.Second, stopCh)
	}
	go func() {
		defer util.HandleCrash()
		time.Sleep(5 * time.Minute) // give time for our cache to fill
		e.checkLeftoverEndpoints()
	}()
	<-stopCh
	e.queue.ShutDown()
}

func (e *EndpointController) getPodServiceMemberships(pod *api.Pod) (util.StringSet, error) {
	set := util.StringSet{}
	services, err := e.serviceStore.GetPodServices(pod)
	if err != nil {
		// don't log this error because this function makes pointless
		// errors when no services match.
		return set, nil
	}
	for i := range services {
		key, err := keyFunc(&services[i])
		if err != nil {
			return nil, err
		}
		set.Insert(key)
	}
	return set, nil
}

// When a pod is added, figure out what services it will be a member of and
// enqueue them. obj must have *api.Pod type.
func (e *EndpointController) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	services, err := e.getPodServiceMemberships(pod)
	if err != nil {
		glog.Errorf("Unable to get pod %v/%v's service memberships: %v", pod.Namespace, pod.Name, err)
		return
	}
	for key := range services {
		e.queue.Add(key)
	}
}

// When a pod is updated, figure out what services it used to be a member of
// and what services it will be a member of, and enqueue the union of these.
// old and cur must be *api.Pod types.
func (e *EndpointController) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		return
	}
	newPod := old.(*api.Pod)
	services, err := e.getPodServiceMemberships(newPod)
	if err != nil {
		glog.Errorf("Unable to get pod %v/%v's service memberships: %v", newPod.Namespace, newPod.Name, err)
		return
	}

	oldPod := cur.(*api.Pod)
	// Only need to get the old services if the labels changed.
	if !reflect.DeepEqual(newPod.Labels, oldPod.Labels) {
		oldServices, err := e.getPodServiceMemberships(oldPod)
		if err != nil {
			glog.Errorf("Unable to get pod %v/%v's service memberships: %v", oldPod.Namespace, oldPod.Name, err)
			return
		}
		services = services.Union(oldServices)
	}
	for key := range services {
		e.queue.Add(key)
	}
}

// When a pod is deleted, enqueue the services the pod used to be a member of.
// obj could be an *api.Pod, or a DeletionFinalStateUnknown marker item.
func (e *EndpointController) deletePod(obj interface{}) {
	if _, ok := obj.(*api.Pod); ok {
		// Enqueue all the services that the pod used to be a member
		// of. This happens to be exactly the same thing we do when a
		// pod is added.
		e.addPod(obj)
		return
	}
	podKey, err := keyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
	}
	glog.Infof("Pod %q was deleted but we don't have a record of its final state, so it will take up to %v before it will be removed from all endpoint records.", podKey, FullServiceResyncPeriod)

	// TODO: keep a map of pods to services to handle this condition.
}

// obj could be an *api.Service, or a DeletionFinalStateUnknown marker item.
func (e *EndpointController) enqueueService(obj interface{}) {
	key, err := keyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
	}

	e.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and
// marks them done. You may run as many of these in parallel as you wish; the
// workqueue guarantees that they will not end up processing the same service
// at the same time.
func (e *EndpointController) worker() {
	for {
		func() {
			key, quit := e.queue.Get()
			if quit {
				return
			}
			// Use defer: in the unlikely event that there's a
			// panic, we'd still like this to get marked done--
			// otherwise the controller will not be able to sync
			// this service again until it is restarted.
			defer e.queue.Done(key)
			e.syncService(key.(string))
		}()
	}
}

func (e *EndpointController) syncService(key string) {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing service %q endpoints. (%v)", key, time.Now().Sub(startTime))
	}()
	obj, exists, err := e.serviceStore.Store.GetByKey(key)
	if err != nil || !exists {
		// Delete the corresponding endpoint, as the service has been deleted.
		// TODO: Please note that this will delete an endpoint when a
		// service is deleted. However, if we're down at the time when
		// the service is deleted, we will miss that deletion, so this
		// doesn't completely solve the problem. See #6877.
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			glog.Errorf("Need to delete endpoint with key %q, but couldn't understand the key: %v", key, err)
			// Don't retry, as the key isn't going to magically become understandable.
			return
		}
		err = e.client.Endpoints(namespace).Delete(name)
		if err != nil && !errors.IsNotFound(err) {
			glog.Errorf("Error deleting endpoint %q: %v", key, err)
			e.queue.Add(key) // Retry
		}
		return
	}

	service := obj.(*api.Service)
	if service.Spec.Selector == nil {
		// services without a selector receive no endpoints from this controller;
		// these services will receive the endpoints that are created out-of-band via the REST API.
		return
	}

	glog.V(5).Infof("About to update endpoints for service %q", key)
	pods, err := e.podStore.Pods(service.Namespace).List(labels.Set(service.Spec.Selector).AsSelector())
	if err != nil {
		// Since we're getting stuff from a local cache, it is
		// basically impossible to get this error.
		glog.Errorf("Error syncing service %q: %v", key, err)
		e.queue.Add(key) // Retry
		return
	}

	subsets := []api.EndpointSubset{}
	for i := range pods.Items {
		pod := &pods.Items[i]

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]

			portName := servicePort.Name
			portProto := servicePort.Protocol
			portNum, err := findPort(pod, servicePort)
			if err != nil {
				glog.V(4).Infof("Failed to find port for service %s/%s: %v", service.Namespace, service.Name, err)
				continue
			}
			if len(pod.Status.PodIP) == 0 {
				glog.V(4).Infof("Failed to find an IP for pod %s/%s", pod.Namespace, pod.Name)
				continue
			}

			if !api.IsPodReady(pod) {
				glog.V(5).Infof("Pod is out of service: %v/%v", pod.Namespace, pod.Name)
				continue
			}

			epp := api.EndpointPort{Name: portName, Port: portNum, Protocol: portProto}
			epa := api.EndpointAddress{IP: pod.Status.PodIP, TargetRef: &api.ObjectReference{
				Kind:            "Pod",
				Namespace:       pod.ObjectMeta.Namespace,
				Name:            pod.ObjectMeta.Name,
				UID:             pod.ObjectMeta.UID,
				ResourceVersion: pod.ObjectMeta.ResourceVersion,
			}}
			subsets = append(subsets, api.EndpointSubset{Addresses: []api.EndpointAddress{epa}, Ports: []api.EndpointPort{epp}})
		}
	}
	subsets = endpoints.RepackSubsets(subsets)

	// See if there's actually an update here.
	currentEndpoints, err := e.client.Endpoints(service.Namespace).Get(service.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			currentEndpoints = &api.Endpoints{
				ObjectMeta: api.ObjectMeta{
					Name:   service.Name,
					Labels: service.Labels,
				},
			}
		} else {
			glog.Errorf("Error getting endpoints: %v", err)
			e.queue.Add(key) // Retry
			return
		}
	}
	if reflect.DeepEqual(currentEndpoints.Subsets, subsets) && reflect.DeepEqual(currentEndpoints.Labels, service.Labels) {
		glog.V(5).Infof("endpoints are equal for %s/%s, skipping update", service.Namespace, service.Name)
		return
	}
	newEndpoints := currentEndpoints
	newEndpoints.Subsets = subsets
	newEndpoints.Labels = service.Labels

	if len(currentEndpoints.ResourceVersion) == 0 {
		// No previous endpoints, create them
		_, err = e.client.Endpoints(service.Namespace).Create(newEndpoints)
	} else {
		// Pre-existing
		_, err = e.client.Endpoints(service.Namespace).Update(newEndpoints)
	}
	if err != nil {
		glog.Errorf("Error updating endpoints: %v", err)
		e.queue.Add(key) // Retry
	}
}

// checkLeftoverEndpoints lists all currently existing endpoints and adds their
// service to the queue. This will detect endpoints that exist with no
// corresponding service; these endpoints need to be deleted. We only need to
// do this once on startup, because in steady-state these are detected (but
// some stragglers could have been left behind if the endpoint controller
// reboots).
func (e *EndpointController) checkLeftoverEndpoints() {
	list, err := e.client.Endpoints(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Errorf("Unable to list endpoints (%v); orphaned endpoints will not be cleaned up. (They're pretty harmless, but you can restart this component if you want another attempt made.)", err)
		return
	}
	for i := range list.Items {
		ep := &list.Items[i]
		key, err := keyFunc(ep)
		if err != nil {
			glog.Errorf("Unable to get key for endpoint %#v", ep)
			continue
		}
		e.queue.Add(key)
	}
}

// findPort locates the container port for the given pod and portName.  If the
// targetPort is a number, use that.  If the targetPort is a string, look that
// string up in all named ports in all containers in the target pod.  If no
// match is found, fail.
func findPort(pod *api.Pod, svcPort *api.ServicePort) (int, error) {
	portName := svcPort.TargetPort
	switch portName.Kind {
	case util.IntstrString:
		name := portName.StrVal
		for _, container := range pod.Spec.Containers {
			for _, port := range container.Ports {
				if port.Name == name && port.Protocol == svcPort.Protocol {
					return port.ContainerPort, nil
				}
			}
		}
	case util.IntstrInt:
		return portName.IntVal, nil
	}

	return 0, fmt.Errorf("no suitable port for manifest: %s", pod.UID)
}
