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

	"github.com/golang/glog"

	"k8s.io/client-go/pkg/util/clock"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/workqueue"
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

// Propagator: based on the events supplied by the informers, Propagator updates
// uidToNode, a graph that caches the dependencies as we know, and enqueues
// items to the dirtyQueue and orphanQueue.
type Propagator struct {
	eventQueue *workqueue.TimedWorkQueue
	// uidToNode doesn't require a lock to protect, because only the
	// single-threaded Propagator.processEvent() reads/writes it.
	uidToNode *concurrentUIDToNode
	// Propagator is the producer of dirtyQueue and orphanQueue, GC is the consumer.
	dirtyQueue  *workqueue.TimedWorkQueue
	orphanQueue *workqueue.TimedWorkQueue
	// Propagator and GC share the absentOwnerCache. Objects that are known to
	// be non-existent are added to the cached.
	absentOwnerCache *UIDCache
	// Propagator and GC share the same clock.
	clock clock.Clock
}

// addDependentToOwners adds n to owners' dependents list. If the owner does not
// exist in the p.uidToNode yet, a "virtual" node will be created to represent
// the owner. The "virtual" node will be enqueued to the dirtyQueue, so that
// processItem() will verify if the owner exists according to the API server.
func (p *Propagator) addDependentToOwners(n *node, owners []metav1.OwnerReference) {
	for _, owner := range owners {
		ownerNode, ok := p.uidToNode.Read(owner.UID)
		if !ok {
			// Create a "virtual" node in the graph for the owner if it doesn't
			// exist in the graph yet. Then enqueue the virtual node into the
			// dirtyQueue. The garbage processor will enqueue a virtual delete
			// event to delete it from the graph if API server confirms this
			// owner doesn't exist.
			ownerNode = &node{
				identity: objectReference{
					OwnerReference: owner,
					Namespace:      n.identity.Namespace,
				},
				dependents: make(map[*node]struct{}),
			}
			glog.V(6).Infof("add virtual node.identity: %s\n\n", ownerNode.identity)
			p.uidToNode.Write(ownerNode)
			p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: ownerNode})
		}
		ownerNode.addDependent(n)
	}
}

// insertNode insert the node to p.uidToNode; then it finds all owners as listed
// in n.owners, and adds the node to their dependents list.
func (p *Propagator) insertNode(n *node) {
	p.uidToNode.Write(n)
	p.addDependentToOwners(n, n.owners)
}

// removeDependentFromOwners remove n from owners' dependents list.
func (p *Propagator) removeDependentFromOwners(n *node, owners []metav1.OwnerReference) {
	for _, owner := range owners {
		ownerNode, ok := p.uidToNode.Read(owner.UID)
		if !ok {
			continue
		}
		ownerNode.deleteDependent(n)
	}
}

// removeNode removes the node from p.uidToNode, then finds all
// owners as listed in n.owners, and removes n from their dependents list.
func (p *Propagator) removeNode(n *node) {
	p.uidToNode.Delete(n.identity.UID)
	p.removeDependentFromOwners(n, n.owners)
}

type ownerRefPair struct {
	old metav1.OwnerReference
	new metav1.OwnerReference
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
			changed = append(changed, ownerRefPair{old: oldUIDToRef[uid], new: newUIDToRef[uid]})
		}
	}
	return added, removed, changed
}

// returns if the object in the event just transitions to "being deleted".
func deletionStarts(e *event, accessor meta.Object) bool {
	// The delta_fifo may combine the creation and update of the object into one
	// event, so if there is no e.oldObj, we just return if the e.obj is being deleted.
	if e.oldObj == nil {
		if accessor.GetDeletionTimestamp() == nil {
			return false
		}
	} else {
		oldAccessor, err := meta.Accessor(e.oldObj)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("cannot access oldObj: %v", err))
			return false
		}
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("cannot access newObj: %v", err))
			return false
		}
		if accessor.GetDeletionTimestamp() == nil || oldAccessor.GetDeletionTimestamp() != nil {
			return false
		}
	}
	return true
}

func beingDeleted(accessor meta.Object) bool {
	return accessor.GetDeletionTimestamp() != nil
}

func hasDeleteDependentsFinalizer(accessor meta.Object) bool {
	finalizers := accessor.GetFinalizers()
	for _, finalizer := range finalizers {
		if finalizer == v1.FinalizerDeleteDependents {
			return true
		}
	}
	return false
}

func hasOrphanFianlizer(accessor meta.Object) bool {
	finalizers := accessor.GetFinalizers()
	for _, finalizer := range finalizers {
		if finalizer == v1.FinalizerOrphanDependents {
			return true
		}
	}
	return false
}

func startsWaitingForDependentsDeletion(e *event, accessor meta.Object) bool {
	if !deletionStarts(e, accessor) {
		// ignore the event if it's not about an object that starts being deleted.
		return false
	}
	return hasDeleteDependentsFinalizer(accessor)
}

func shouldOrphanDependents(e *event, accessor meta.Object) bool {
	if !deletionStarts(e, accessor) {
		// ignore the event if it's not about an object that starts being deleted.
		return false
	}
	return hasOrphanFianlizer(accessor)
}

// if an blocking ownerReference points to an object gets removed, or get set to
// "BlockOwnerDeletion=false", add the object to the dirty queue.
func (p *Propagator) addUnblockedOwnersToDirtyQueue(removed []metav1.OwnerReference, changed []ownerRefPair) {
	for _, ref := range removed {
		if ref.BlockOwnerDeletion != nil && *ref.BlockOwnerDeletion {
			node, found := p.uidToNode.Read(ref.UID)
			if !found {
				glog.V(6).Infof("cannot find %s in uidToNode", ref.UID)
				continue
			}
			p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: node})
		}
	}
	for _, c := range changed {
		if c.old.BlockOwnerDeletion != nil && *c.old.BlockOwnerDeletion &&
			c.new.BlockOwnerDeletion != nil && !*c.new.BlockOwnerDeletion {
			node, found := p.uidToNode.Read(c.new.UID)
			if !found {
				glog.V(6).Infof("cannot find %s in uidToNode", c.new.UID)
				continue
			}
			p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: node})
		}
	}
}

// Dequeueing an event from eventQueue, updating graph, populating dirty_queue.
func (p *Propagator) processEvent() {
	timedItem, quit := p.eventQueue.Get()
	if quit {
		return
	}
	defer p.eventQueue.Done(timedItem)
	event, ok := timedItem.Object.(*event)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("expect a *event, got %v", timedItem.Object))
		return
	}
	obj := event.obj
	accessor, err := meta.Accessor(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("cannot access obj: %v", err))
		return
	}
	typeAccessor, err := meta.TypeAccessor(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("cannot access obj: %v", err))
		return
	}
	glog.V(6).Infof("Propagator process object: %s/%s, namespace %s, name %s, event type %s", typeAccessor.GetAPIVersion(), typeAccessor.GetKind(), accessor.GetNamespace(), accessor.GetName(), event.eventType)
	// Check if the node already exsits
	existingNode, found := p.uidToNode.Read(accessor.GetUID())
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
		p.insertNode(newNode)
		// the underlying delta_fifo may combine a creation and deletion into one event
		if shouldOrphanDependents(event, accessor) {
			glog.V(6).Infof("add %s to the orphanQueue", newNode.identity)
			p.orphanQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: newNode})
		}
		if startsWaitingForDependentsDeletion(event, accessor) {
			glog.V(2).Infof("add %s to the dirtyQueue, because it's waiting for its dependents to be deleted", newNode.identity)
			p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: newNode})
			for dep := range newNode.dependents {
				p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: dep})
			}
		}
	case (event.eventType == addEvent || event.eventType == updateEvent) && found:
		// caveat: if GC observes the creation of the dependents later than the
		// deletion of the owner, then the orphaning finalizer won't be effective.
		if shouldOrphanDependents(event, accessor) {
			glog.V(6).Infof("add %s to the orphanQueue", existingNode.identity)
			p.orphanQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: existingNode})
		}
		if beingDeleted(accessor) {
			existingNode.beingDeleted = true
		}
		if startsWaitingForDependentsDeletion(event, accessor) {
			glog.V(2).Infof("add %s to the dirtyQueue, because it's waiting for its dependents to be deleted", existingNode.identity)
			// if the existingNode is added as a "virtual" node, its deletingDependents field is not properly set, so always set it here.
			existingNode.deletingDependents = true
			p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: existingNode})
			for dep := range existingNode.dependents {
				p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: dep})
			}
		}
		// add/remove owner refs
		added, removed, changed := referencesDiffs(existingNode.owners, accessor.GetOwnerReferences())
		if len(added) == 0 && len(removed) == 0 && len(changed) == 0 {
			glog.V(6).Infof("The updateEvent %#v doesn't change node references, ignore", event)
			return
		}
		if len(changed) != 0 {
			glog.V(6).Infof("references %#v changed for object %s", changed, existingNode.identity)
		}
		p.addUnblockedOwnersToDirtyQueue(removed, changed)
		// update the node itself
		existingNode.owners = accessor.GetOwnerReferences()
		// Add the node to its new owners' dependent lists.
		p.addDependentToOwners(existingNode, added)
		// remove the node from the dependent list of node that are no longer in
		// the node's owners list.
		p.removeDependentFromOwners(existingNode, removed)
	case event.eventType == deleteEvent:
		if !found {
			glog.V(6).Infof("%v doesn't exist in the graph, this shouldn't happen", accessor.GetUID())
			return
		}
		p.removeNode(existingNode)
		existingNode.dependentsLock.RLock()
		defer existingNode.dependentsLock.RUnlock()
		if len(existingNode.dependents) > 0 {
			absentOwnerCache.Add(accessor.GetUID())
		}
		for dep := range existingNode.dependents {
			p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: dep})
		}
		for _, owner := range existingNode.owners {
			ownerNode, found := p.uidToNode.Read(owner.UID)
			if !found || !ownerNode.deletingDependents {
				continue
			}
			// this is to let processItem check if all the owner's dependents are deleted.
			p.dirtyQueue.Add(&workqueue.TimedWorkQueueItem{StartTime: p.clock.Now(), Object: ownerNode})
		}
	}
	EventProcessingLatency.Observe(sinceInMicroseconds(p.clock, timedItem.StartTime))
}
