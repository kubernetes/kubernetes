/*
Copyright 2016 The Kubernetes Authors.

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
	"reflect"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
)

type eventType int

const (
	addEvent eventType = iota
	updateEvent
	deleteEvent
)

type event struct {
	eventType eventType
	obj       interface{}
	// the update event comes with an old object, but it's not used by the garbage collector.
	oldObj interface{}
}

// GraphBuilder: based on the events supplied by the informers, GraphBuilder updates
// uidToNode, a graph that caches the dependencies as we know, and enqueues
// items to the attemptToDelete and attemptToOrphan.
type GraphBuilder struct {
	restMapper meta.RESTMapper
	// each monitor list/watches a resource, the results are funneled to the
	// dependencyGraphBuilder
	monitors []cache.Controller
	// metaOnlyClientPool uses a special codec, which removes fields except for
	// apiVersion, kind, and metadata during decoding.
	metaOnlyClientPool dynamic.ClientPool
	// used to register exactly once the rate limiters of the clients used by
	// the `monitors`.
	registeredRateLimiterForControllers *RegisteredRateLimiter
	// monitors are the producer of the graphChanges queue, graphBuilder alters
	// the in-memory graph according to the changes.
	graphChanges workqueue.RateLimitingInterface
	// uidToNode doesn't require a lock to protect, because only the
	// single-threaded GraphBuilder.processGraphChanges() reads/writes it.
	uidToNode *concurrentUIDToNode
	// GraphBuilder is the producer of attemptToDelete and attemptToOrphan, GC is the consumer.
	attemptToDelete workqueue.RateLimitingInterface
	attemptToOrphan workqueue.RateLimitingInterface
	// GraphBuilder and GC share the absentOwnerCache. Objects that are known to
	// be non-existent are added to the cached.
	absentOwnerCache *UIDCache
}

func listWatcher(client *dynamic.Client, resource schema.GroupVersionResource) *cache.ListWatch {
	return &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			// APIResource.Kind is not used by the dynamic client, so
			// leave it empty. We want to list this resource in all
			// namespaces if it's namespace scoped, so leave
			// APIResource.Namespaced as false is all right.
			apiResource := metav1.APIResource{Name: resource.Resource}
			return client.ParameterCodec(dynamic.VersionedParameterEncoderWithV1Fallback).
				Resource(&apiResource, metav1.NamespaceAll).
				List(options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			// APIResource.Kind is not used by the dynamic client, so
			// leave it empty. We want to list this resource in all
			// namespaces if it's namespace scoped, so leave
			// APIResource.Namespaced as false is all right.
			apiResource := metav1.APIResource{Name: resource.Resource}
			return client.ParameterCodec(dynamic.VersionedParameterEncoderWithV1Fallback).
				Resource(&apiResource, metav1.NamespaceAll).
				Watch(options)
		},
	}
}

func (gb *GraphBuilder) controllerFor(resource schema.GroupVersionResource, kind schema.GroupVersionKind) (cache.Controller, error) {
	// TODO: consider store in one storage.
	glog.V(5).Infof("create storage for resource %s", resource)
	client, err := gb.metaOnlyClientPool.ClientForGroupVersionKind(kind)
	if err != nil {
		return nil, err
	}
	gb.registeredRateLimiterForControllers.registerIfNotPresent(resource.GroupVersion(), client, "garbage_collector_monitoring")
	setObjectTypeMeta := func(obj interface{}) {
		runtimeObject, ok := obj.(runtime.Object)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("expected runtime.Object, got %#v", obj))
		}
		runtimeObject.GetObjectKind().SetGroupVersionKind(kind)
	}
	_, monitor := cache.NewInformer(
		listWatcher(client, resource),
		nil,
		ResourceResyncTime,
		cache.ResourceEventHandlerFuncs{
			// add the event to the dependencyGraphBuilder's graphChanges.
			AddFunc: func(obj interface{}) {
				setObjectTypeMeta(obj)
				event := &event{
					eventType: addEvent,
					obj:       obj,
				}
				gb.graphChanges.Add(event)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				setObjectTypeMeta(newObj)
				// TODO: check if there are differences in the ownerRefs,
				// finalizers, and DeletionTimestamp; if not, ignore the update.
				event := &event{updateEvent, newObj, oldObj}
				gb.graphChanges.Add(event)
			},
			DeleteFunc: func(obj interface{}) {
				// delta fifo may wrap the object in a cache.DeletedFinalStateUnknown, unwrap it
				if deletedFinalStateUnknown, ok := obj.(cache.DeletedFinalStateUnknown); ok {
					obj = deletedFinalStateUnknown.Obj
				}
				setObjectTypeMeta(obj)
				event := &event{
					eventType: deleteEvent,
					obj:       obj,
				}
				gb.graphChanges.Add(event)
			},
		},
	)
	return monitor, nil
}

func (gb *GraphBuilder) monitorsForResources(resources map[schema.GroupVersionResource]struct{}) error {
	for resource := range resources {
		if _, ok := ignoredResources[resource]; ok {
			glog.V(5).Infof("ignore resource %#v", resource)
			continue
		}
		kind, err := gb.restMapper.KindFor(resource)
		if err != nil {
			nonCoreMsg := fmt.Sprintf(nonCoreMessage, resource)
			utilruntime.HandleError(fmt.Errorf("%v. %s", err, nonCoreMsg))
			continue
		}
		monitor, err := gb.controllerFor(resource, kind)
		if err != nil {
			return err
		}
		gb.monitors = append(gb.monitors, monitor)
	}
	return nil
}

func (gb *GraphBuilder) HasSynced() bool {
	for _, monitor := range gb.monitors {
		if !monitor.HasSynced() {
			return false
		}
	}
	return true
}

func (gb *GraphBuilder) Run(stopCh <-chan struct{}) {
	for _, monitor := range gb.monitors {
		go monitor.Run(stopCh)
	}
	go wait.Until(gb.runProcessGraphChanges, 1*time.Second, stopCh)
}

var ignoredResources = map[schema.GroupVersionResource]struct{}{
	schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "replicationcontrollers"}:              {},
	schema.GroupVersionResource{Group: "", Version: "v1", Resource: "bindings"}:                                           {},
	schema.GroupVersionResource{Group: "", Version: "v1", Resource: "componentstatuses"}:                                  {},
	schema.GroupVersionResource{Group: "", Version: "v1", Resource: "events"}:                                             {},
	schema.GroupVersionResource{Group: "authentication.k8s.io", Version: "v1beta1", Resource: "tokenreviews"}:             {},
	schema.GroupVersionResource{Group: "authorization.k8s.io", Version: "v1beta1", Resource: "subjectaccessreviews"}:      {},
	schema.GroupVersionResource{Group: "authorization.k8s.io", Version: "v1beta1", Resource: "selfsubjectaccessreviews"}:  {},
	schema.GroupVersionResource{Group: "authorization.k8s.io", Version: "v1beta1", Resource: "localsubjectaccessreviews"}: {},
	schema.GroupVersionResource{Group: "apiregistration.k8s.io", Version: "v1alpha1", Resource: "apiservices"}:            {},
}

func (gb *GraphBuilder) enqueueChanges(e *event) {
	gb.graphChanges.Add(e)
}

// addDependentToOwners adds n to owners' dependents list. If the owner does not
// exist in the gb.uidToNode yet, a "virtual" node will be created to represent
// the owner. The "virtual" node will be enqueued to the attemptToDelete, so that
// processItem() will verify if the owner exists according to the API server.
func (gb *GraphBuilder) addDependentToOwners(n *node, owners []metav1.OwnerReference) {
	for _, owner := range owners {
		ownerNode, ok := gb.uidToNode.Read(owner.UID)
		if !ok {
			// Create a "virtual" node in the graph for the owner if it doesn't
			// exist in the graph yet. Then enqueue the virtual node into the
			// attemptToDelete. The garbage processor will enqueue a virtual delete
			// event to delete it from the graph if API server confirms this
			// owner doesn't exist.
			ownerNode = &node{
				identity: objectReference{
					OwnerReference: owner,
					Namespace:      n.identity.Namespace,
				},
				dependents: make(map[*node]struct{}),
			}
			glog.V(5).Infof("add virtual node.identity: %s\n\n", ownerNode.identity)
			gb.uidToNode.Write(ownerNode)
			gb.attemptToDelete.Add(ownerNode)
		}
		ownerNode.addDependent(n)
	}
}

// insertNode insert the node to gb.uidToNode; then it finds all owners as listed
// in n.owners, and adds the node to their dependents list.
func (gb *GraphBuilder) insertNode(n *node) {
	gb.uidToNode.Write(n)
	gb.addDependentToOwners(n, n.owners)
}

// removeDependentFromOwners remove n from owners' dependents list.
func (gb *GraphBuilder) removeDependentFromOwners(n *node, owners []metav1.OwnerReference) {
	for _, owner := range owners {
		ownerNode, ok := gb.uidToNode.Read(owner.UID)
		if !ok {
			continue
		}
		ownerNode.deleteDependent(n)
	}
}

// removeNode removes the node from gb.uidToNode, then finds all
// owners as listed in n.owners, and removes n from their dependents list.
func (gb *GraphBuilder) removeNode(n *node) {
	gb.uidToNode.Delete(n.identity.UID)
	gb.removeDependentFromOwners(n, n.owners)
}

type ownerRefPair struct {
	oldRef metav1.OwnerReference
	newRef metav1.OwnerReference
}

// TODO: profile this function to see if a naive N^2 algorithm performs better
// when the number of references is small.
func referencesDiffs(old []metav1.OwnerReference, new []metav1.OwnerReference) (added []metav1.OwnerReference, removed []metav1.OwnerReference, changed []ownerRefPair) {
	oldUIDToRef := make(map[string]metav1.OwnerReference)
	for i := 0; i < len(old); i++ {
		oldUIDToRef[string(old[i].UID)] = old[i]
	}
	oldUIDSet := sets.StringKeySet(oldUIDToRef)
	newUIDToRef := make(map[string]metav1.OwnerReference)
	for i := 0; i < len(new); i++ {
		newUIDToRef[string(new[i].UID)] = new[i]
	}
	newUIDSet := sets.StringKeySet(newUIDToRef)

	addedUID := newUIDSet.Difference(oldUIDSet)
	removedUID := oldUIDSet.Difference(newUIDSet)
	intersection := oldUIDSet.Intersection(newUIDSet)

	for uid := range addedUID {
		added = append(added, newUIDToRef[uid])
	}
	for uid := range removedUID {
		removed = append(removed, oldUIDToRef[uid])
	}
	for uid := range intersection {
		if !reflect.DeepEqual(oldUIDToRef[uid], newUIDToRef[uid]) {
			changed = append(changed, ownerRefPair{oldRef: oldUIDToRef[uid], newRef: newUIDToRef[uid]})
		}
	}
	return added, removed, changed
}

// returns if the object in the event just transitions to "being deleted".
func deletionStarts(oldObj interface{}, newAccessor metav1.Object) bool {
	// The delta_fifo may combine the creation and update of the object into one
	// event, so if there is no oldObj, we just return if the newObj (via
	// newAccessor) is being deleted.
	if oldObj == nil {
		if newAccessor.GetDeletionTimestamp() == nil {
			return false
		}
		return true
	}
	oldAccessor, err := meta.Accessor(oldObj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("cannot access oldObj: %v", err))
		return false
	}
	return beingDeleted(newAccessor) && !beingDeleted(oldAccessor)
}

func beingDeleted(accessor metav1.Object) bool {
	return accessor.GetDeletionTimestamp() != nil
}

func hasDeleteDependentsFinalizer(accessor metav1.Object) bool {
	finalizers := accessor.GetFinalizers()
	for _, finalizer := range finalizers {
		if finalizer == metav1.FinalizerDeleteDependents {
			return true
		}
	}
	return false
}

func hasOrphanFinalizer(accessor metav1.Object) bool {
	finalizers := accessor.GetFinalizers()
	for _, finalizer := range finalizers {
		if finalizer == metav1.FinalizerOrphanDependents {
			return true
		}
	}
	return false
}

// this function takes newAccessor directly because the caller already
// instantiates an accessor for the newObj.
func startsWaitingForDependentsDeleted(oldObj interface{}, newAccessor metav1.Object) bool {
	return deletionStarts(oldObj, newAccessor) && hasDeleteDependentsFinalizer(newAccessor)
}

// this function takes newAccessor directly because the caller already
// instantiates an accessor for the newObj.
func startsWaitingForDependentsOrphaned(oldObj interface{}, newAccessor metav1.Object) bool {
	return deletionStarts(oldObj, newAccessor) && hasOrphanFinalizer(newAccessor)
}

// if an blocking ownerReference points to an object gets removed, or gets set to
// "BlockOwnerDeletion=false", add the object to the attemptToDelete queue.
func (gb *GraphBuilder) addUnblockedOwnersToDeleteQueue(removed []metav1.OwnerReference, changed []ownerRefPair) {
	for _, ref := range removed {
		if ref.BlockOwnerDeletion != nil && *ref.BlockOwnerDeletion {
			node, found := gb.uidToNode.Read(ref.UID)
			if !found {
				glog.V(5).Infof("cannot find %s in uidToNode", ref.UID)
				continue
			}
			gb.attemptToDelete.Add(node)
		}
	}
	for _, c := range changed {
		wasBlocked := c.oldRef.BlockOwnerDeletion != nil && *c.oldRef.BlockOwnerDeletion
		isUnblocked := c.newRef.BlockOwnerDeletion == nil || (c.newRef.BlockOwnerDeletion != nil && !*c.newRef.BlockOwnerDeletion)
		if wasBlocked && isUnblocked {
			node, found := gb.uidToNode.Read(c.newRef.UID)
			if !found {
				glog.V(5).Infof("cannot find %s in uidToNode", c.newRef.UID)
				continue
			}
			gb.attemptToDelete.Add(node)
		}
	}
}

func (gb *GraphBuilder) processTransitions(oldObj interface{}, newAccessor metav1.Object, n *node) {
	if startsWaitingForDependentsOrphaned(oldObj, newAccessor) {
		glog.V(5).Infof("add %s to the attemptToOrphan", n.identity)
		gb.attemptToOrphan.Add(n)
		return
	}
	if startsWaitingForDependentsDeleted(oldObj, newAccessor) {
		glog.V(2).Infof("add %s to the attemptToDelete, because it's waiting for its dependents to be deleted", n.identity)
		// if the n is added as a "virtual" node, its deletingDependents field is not properly set, so always set it here.
		n.markDeletingDependents()
		for dep := range n.dependents {
			gb.attemptToDelete.Add(dep)
		}
		gb.attemptToDelete.Add(n)
	}
}

func (gb *GraphBuilder) runProcessGraphChanges() {
	for gb.processGraphChanges() {
	}
}

// Dequeueing an event from graphChanges, updating graph, populating dirty_queue.
func (gb *GraphBuilder) processGraphChanges() bool {
	item, quit := gb.graphChanges.Get()
	if quit {
		return false
	}
	defer gb.graphChanges.Done(item)
	event, ok := item.(*event)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("expect a *event, got %v", item))
		return true
	}
	obj := event.obj
	accessor, err := meta.Accessor(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("cannot access obj: %v", err))
		return true
	}
	typeAccessor, err := meta.TypeAccessor(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("cannot access obj: %v", err))
		return true
	}
	glog.V(5).Infof("GraphBuilder process object: %s/%s, namespace %s, name %s, event type %s", typeAccessor.GetAPIVersion(), typeAccessor.GetKind(), accessor.GetNamespace(), accessor.GetName(), event.eventType)
	// Check if the node already exsits
	existingNode, found := gb.uidToNode.Read(accessor.GetUID())
	switch {
	case (event.eventType == addEvent || event.eventType == updateEvent) && !found:
		newNode := &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					APIVersion: typeAccessor.GetAPIVersion(),
					Kind:       typeAccessor.GetKind(),
					UID:        accessor.GetUID(),
					Name:       accessor.GetName(),
				},
				Namespace: accessor.GetNamespace(),
			},
			dependents:         make(map[*node]struct{}),
			owners:             accessor.GetOwnerReferences(),
			deletingDependents: beingDeleted(accessor) && hasDeleteDependentsFinalizer(accessor),
			beingDeleted:       beingDeleted(accessor),
		}
		gb.insertNode(newNode)
		// the underlying delta_fifo may combine a creation and a deletion into
		// one event, so we need to further process the event.
		gb.processTransitions(event.oldObj, accessor, newNode)
	case (event.eventType == addEvent || event.eventType == updateEvent) && found:
		// handle changes in ownerReferences
		added, removed, changed := referencesDiffs(existingNode.owners, accessor.GetOwnerReferences())
		if len(added) != 0 || len(removed) != 0 || len(changed) != 0 {
			// check if the changed dependency graph unblock owners that are
			// waiting for the deletion of their dependents.
			gb.addUnblockedOwnersToDeleteQueue(removed, changed)
			// update the node itself
			existingNode.owners = accessor.GetOwnerReferences()
			// Add the node to its new owners' dependent lists.
			gb.addDependentToOwners(existingNode, added)
			// remove the node from the dependent list of node that are no longer in
			// the node's owners list.
			gb.removeDependentFromOwners(existingNode, removed)
		}

		if beingDeleted(accessor) {
			existingNode.markBeingDeleted()
		}
		gb.processTransitions(event.oldObj, accessor, existingNode)
	case event.eventType == deleteEvent:
		if !found {
			glog.V(5).Infof("%v doesn't exist in the graph, this shouldn't happen", accessor.GetUID())
			return true
		}
		// removeNode updates the graph
		gb.removeNode(existingNode)
		existingNode.dependentsLock.RLock()
		defer existingNode.dependentsLock.RUnlock()
		if len(existingNode.dependents) > 0 {
			gb.absentOwnerCache.Add(accessor.GetUID())
		}
		for dep := range existingNode.dependents {
			gb.attemptToDelete.Add(dep)
		}
		for _, owner := range existingNode.owners {
			ownerNode, found := gb.uidToNode.Read(owner.UID)
			if !found || !ownerNode.isDeletingDependents() {
				continue
			}
			// this is to let attempToDeleteItem check if all the owner's
			// dependents are deleted, if so, the owner will be deleted.
			gb.attemptToDelete.Add(ownerNode)
		}
	}
	return true
}
