/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package garbagecollector

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/meta/metatypes"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const ResourceResyncTime = 30 * time.Second

type Monitor struct {
	Store      cache.Store
	Controller *framework.Controller
}

type SelfReference struct {
	metatypes.OwnerReference
	// This is needed by the dynamic client
	Namespace string
}

func (s SelfReference) String() string {
	return fmt.Sprintf("%s/%s, namespace: %s, name: %s, uid: %s", s.APIVersion, s.Kind, s.Namespace, s.Name, s.UID)
}

type node struct {
	identity   SelfReference
	dependents []*node
	// When processing an Update event, we need to compare the updated
	// ownerReferences with the owners recorded in the graph.
	owners []metatypes.OwnerReference
}

type EventType string

const Add EventType = "Add"
const Update EventType = "Update"
const Delete EventType = "Delete"

type Event struct {
	Type   EventType
	Obj    interface{}
	OldObj interface{}
}

type Propagator struct {
	eventQueue  *workqueue.Type
	graphLookup map[types.UID]*node
	gc          *GarbageCollector
}

func (p *Propagator) addToGraphLookup(n *node) {
	fmt.Println("CHAO: add to graph:", n.identity)
	p.graphLookup[n.identity.UID] = n
}

func (p *Propagator) removeFromGraphLookup(n *node) {
	fmt.Println("CHAO: remove from graph:", n.identity)
	delete(p.graphLookup, n.identity.UID)
}

func (p *Propagator) updateOwners(n *node) (ownerExists bool) {
	fmt.Println("CHAO: in updateOwners, graphLookUp:", p.graphLookup)
	for _, owner := range n.owners {
		fmt.Println("CHAO: in updateOwners, lookup for:", owner.UID)
		ownerNode, ok := p.graphLookup[owner.UID]
		if !ok {
			fmt.Println("CHAO: lookup not found")
			continue
		}
		fmt.Println("CHAO: lookup found")
		ownerExists = true
		ownerNode.dependents = append(ownerNode.dependents, n)
	}
	return ownerExists
}

// TODO: finish this function.
func equalReferences(a []metatypes.OwnerReference, b []metatypes.OwnerReference) bool {
	if len(a) != len(b) {
		return false
	}
	// sort and compare
	return true
}

// Dequeue from eventQueue, updating graph, populating dirty_queue.
func (p *Propagator) processEvent() {
	key, quit := p.eventQueue.Get()
	if quit {
		return
	}
	defer p.eventQueue.Done(key)
	event, ok := key.(Event)
	if !ok {
		glog.Errorf("expect an Event, got %v", key)
		return
	}
	obj := event.Obj
	accessor, err := meta.Accessor(obj)
	typeAccessor, err := meta.TypeAccessor(obj)
	if err != nil {
		glog.Errorf("cannot access obj: %v", err)
		return
	}
	glog.V(6).Infof("Propagator process object: %s/%s, namespace %s, name %s, event type %s", typeAccessor.GetAPIVersion(), typeAccessor.GetKind(), accessor.GetNamespace(), accessor.GetName(), event.Type)
	switch event.Type {
	case Add:
		// Check if the node already exsits
		_, ok := p.graphLookup[accessor.GetUID()]
		if ok {
			// We can optionally check if the node in the graph is in sync with that in the Store.
			return
		}
		newNode := &node{
			identity: SelfReference{
				OwnerReference: metatypes.OwnerReference{
					APIVersion: typeAccessor.GetAPIVersion(),
					Kind:       typeAccessor.GetKind(),
					UID:        accessor.GetUID(),
					Name:       accessor.GetName(),
				},
				Namespace: accessor.GetNamespace(),
			},
			owners: accessor.GetOwnerReferences(),
		}
		p.addToGraphLookup(newNode)
		ownerExists := p.updateOwners(newNode)
		// Push the object to the dirty queue if none of its owners exists in the Graph.
		if !ownerExists && len(newNode.owners) != 0 {
			fmt.Println("CHAO: directly added to dirty queue: ", newNode.identity)
			p.gc.dirtyQueue.Add(newNode)
		}

	case Update:
		node, ok := p.graphLookup[accessor.GetUID()]
		if !ok {
			glog.Errorf("received an update for %v, but cannot find the node in the graph", accessor.GetUID())
		}
		// TODO: finalizer: Check if ObjectMeta.DeletionTimestamp is updated from nil to non-nil
		// We only need to add/remove owner refs for now
		if !equalReferences(node.owners, accessor.GetOwnerReferences()) {
			ownerExists := p.updateOwners(node)
			if !ownerExists && len(node.owners) != 0 {
				p.gc.dirtyQueue.Add(node)
			}
		}
	case Delete:
		node, ok := p.graphLookup[accessor.GetUID()]
		if !ok {
			glog.V(6).Infof("%v doesn't exist in the graph, this shouldn't happen", accessor.GetUID())
		}
		p.removeFromGraphLookup(node)
		for _, dep := range node.dependents {
			p.gc.dirtyQueue.Add(dep)
		}
	}
}

type GarbageCollector struct {
	restMapper meta.RESTMapper
	clientPool dynamic.ClientPool
	dirtyQueue *workqueue.Type
	monitors   []Monitor
	propagator *Propagator
}

func monitorFor(p *Propagator, clientPool dynamic.ClientPool, resource unversioned.GroupVersionResource) (Monitor, error) {
	// TODO: consider store in one storage.
	glog.V(6).Infof("create storage for resource %s", resource)
	var monitor Monitor
	client, err := clientPool.ClientForGroupVersion(resource.GroupVersion())
	if err != nil {
		return monitor, err
	}
	monitor.Store, monitor.Controller = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				// APIResource.Kind is not used by the dynamic client, so
				// leave it empty. We want to list this resource in all
				// namespaces if it's namespace scoped, so leave
				// APIResource.Namespaced as false is all right.
				apiResource := unversioned.APIResource{Name: resource.Resource}
				// TODO: Probably we should process the UnstructuredList, extracting only the ObjectMeta before caching it.
				return client.Resource(&apiResource, api.NamespaceAll).UnversionedList(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				// APIResource.Kind is not used by the dynamic client, so
				// leave it empty. We want to list this resource in all
				// namespaces if it's namespace scoped, so leave
				// APIResource.Namespaced as false is all right.
				apiResource := unversioned.APIResource{Name: resource.Resource}
				return client.Resource(&apiResource, api.NamespaceAll).UnversionedWatch(options)
			},
		},
		nil,
		ResourceResyncTime,
		framework.ResourceEventHandlerFuncs{
			// Add the event to the propagator's event queue.
			AddFunc: func(obj interface{}) {
				fmt.Println("CHAO: AddFunc: obj=", obj)
				event := Event{
					Type: Add,
					Obj:  obj,
				}
				p.eventQueue.Add(event)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				event := Event{Update, newObj, oldObj}
				p.eventQueue.Add(event)
			},
			DeleteFunc: func(obj interface{}) {
				event := Event{
					Type: Delete,
					Obj:  obj,
				}
				p.eventQueue.Add(event)
			},
		},
	)
	return monitor, nil
}

var ignoredResources = map[unversioned.GroupVersionResource]struct{}{
	unversioned.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "replicationcontrollers"}: struct{}{},
	unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "bindings"}:                              struct{}{},
	unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "componentstatuses"}:                     struct{}{},
	unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "events"}:                                struct{}{},
}

func NewGarbageCollector(clientPool dynamic.ClientPool, resources []unversioned.GroupVersionResource) (*GarbageCollector, error) {
	gc := &GarbageCollector{
		clientPool: clientPool,
		dirtyQueue: workqueue.New(),
		// TODO: should use a dynamic RESTMapper built from the discovery results.
		restMapper: registered.RESTMapper(),
	}
	gc.propagator = &Propagator{
		eventQueue:  workqueue.New(),
		graphLookup: make(map[types.UID]*node),
		gc:          gc,
	}
	for _, resource := range resources {
		if _, ok := ignoredResources[resource]; ok {
			glog.V(6).Infof("ignore resource %#v", resource)
			continue
		}
		monitor, err := monitorFor(gc.propagator, gc.clientPool, resource)
		if err != nil {
			return nil, err
		}
		gc.monitors = append(gc.monitors, monitor)
	}
	return gc, nil
}

func (gc *GarbageCollector) worker() {
	key, quit := gc.dirtyQueue.Get()
	if quit {
		return
	}
	defer gc.dirtyQueue.Done(key)
	err := gc.processItem(key.(*node))
	if err != nil {
		glog.Errorf("Error syncing item %v: %v", key, err)
	}
}
func (gc *GarbageCollector) apiResource(apiVersion, kind, namespace string) (*unversioned.APIResource, error) {
	fqKind := unversioned.FromAPIVersionAndKind(apiVersion, kind)
	mapping, err := gc.restMapper.RESTMapping(fqKind.GroupKind(), apiVersion)
	if err != nil {
		return nil, fmt.Errorf("unable to get REST mapping for kind: %s, version: %s", kind, apiVersion)
	}
	glog.V(6).Infof("map kind %s, version %s to resource %s", kind, apiVersion, mapping.Resource)
	resource := unversioned.APIResource{
		Name:       mapping.Resource,
		Namespaced: namespace != "",
		Kind:       kind,
	}
	return &resource, nil
}

func (gc *GarbageCollector) deleteObject(item *node) error {
	fqKind := unversioned.FromAPIVersionAndKind(item.identity.APIVersion, item.identity.Kind)
	client, err := gc.clientPool.ClientForGroupVersion(fqKind.GroupVersion())
	resource, err := gc.apiResource(item.identity.APIVersion, item.identity.Kind, item.identity.Namespace)
	if err != nil {
		return err
	}
	uid := item.identity.UID
	preconditions := v1.Preconditions{UID: &uid}
	deleteOptions := v1.DeleteOptions{Preconditions: &preconditions}
	return client.Resource(resource, item.identity.Namespace).Delete(item.identity.Name, &deleteOptions)
}

func (gc *GarbageCollector) getObject(item *node) (*runtime.Unstructured, error) {
	fqKind := unversioned.FromAPIVersionAndKind(item.identity.APIVersion, item.identity.Kind)
	client, err := gc.clientPool.ClientForGroupVersion(fqKind.GroupVersion())
	resource, err := gc.apiResource(item.identity.APIVersion, item.identity.Kind, item.identity.Namespace)
	if err != nil {
		return nil, err
	}
	return client.Resource(resource, item.identity.Namespace).Get(item.identity.Name)
}

func (gc *GarbageCollector) processItem(item *node) error {
	// Get the latest item from the API server
	latest, err := gc.getObject(item)
	if err != nil {
		if errors.IsNotFound(err) {
			glog.V(6).Infof("item %v not found, ignore it", item.identity)
			return nil
		}
		return err
	}
	if latest.GetUID() != item.identity.UID {
		glog.V(6).Infof("UID doesn't match, item %v not found, ignore it", item.identity)
		return nil
	}
	ownerReferences := latest.GetOwnerReferences()
	if len(ownerReferences) == 0 {
		glog.V(6).Infof("object %s's doesn't have an owner, continue on next item", item.identity)
		return nil
	}
	for _, reference := range ownerReferences {
		// TODO: need to verify the reference resource is supported by the system.
		fqKind := unversioned.FromAPIVersionAndKind(reference.APIVersion, reference.Kind)
		client, err := gc.clientPool.ClientForGroupVersion(fqKind.GroupVersion())
		if err != nil {
			return err
		}
		resource, err := gc.apiResource(reference.APIVersion, reference.Kind, item.identity.Namespace)
		if err != nil {
			return err
		}
		owner, err := client.Resource(resource, item.identity.Namespace).Get(reference.Name)
		// TODO: need to compare the UID.
		if err == nil {
			if owner.GetUID() != reference.UID {
				glog.V(6).Infof("object %s's owner %s/%s, %s is not found, UID mismatch", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
				continue
			}
			glog.V(6).Infof("object %s has at least an existing owner, will not garbage collect", item.identity.UID)
			return nil
		} else if errors.IsNotFound(err) {
			glog.V(6).Infof("object %s's owner %s/%s, %s is not found", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
		} else {
			return err
		}
	}
	glog.V(6).Infof("none of object %s's owners exist any more, will garbage collect it", item.identity.UID)
	return gc.deleteObject(item)
}

func (gc *GarbageCollector) Run(workers int, stopCh <-chan struct{}) {
	for _, monitor := range gc.monitors {
		go monitor.Controller.Run(stopCh)
	}
	// list
	// TODO: remove
	// go wait.Until(gc.scanner, ResourceResyncTime, stopCh)

	// worker
	go wait.Until(gc.propagator.processEvent, 0, stopCh)

	for i := 0; i < workers; i++ {
		go wait.Until(gc.worker, 0, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down garbage collector")
	gc.dirtyQueue.ShutDown()
	gc.propagator.eventQueue.ShutDown()
}
