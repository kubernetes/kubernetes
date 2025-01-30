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
	"sync"

	"github.com/go-logr/logr"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

type objectReference struct {
	metav1.OwnerReference
	// This is needed by the dynamic client
	Namespace string
}

// String is used when logging an objectReference in text format.
func (s objectReference) String() string {
	return fmt.Sprintf("[%s/%s, namespace: %s, name: %s, uid: %s]", s.APIVersion, s.Kind, s.Namespace, s.Name, s.UID)
}

// MarshalLog is used when logging an objectReference in JSON format.
func (s objectReference) MarshalLog() interface{} {
	return struct {
		Name       string    `json:"name"`
		Namespace  string    `json:"namespace"`
		APIVersion string    `json:"apiVersion"`
		UID        types.UID `json:"uid"`
	}{
		Namespace:  s.Namespace,
		Name:       s.Name,
		APIVersion: s.APIVersion,
		UID:        s.UID,
	}
}

var _ fmt.Stringer = objectReference{}
var _ logr.Marshaler = objectReference{}

// The single-threaded GraphBuilder.processGraphChanges() is the sole writer of the
// nodes. The multi-threaded GarbageCollector.attemptToDeleteItem() reads the nodes.
// WARNING: node has different locks on different fields. setters and getters
// use the respective locks, so the return values of the getters can be
// inconsistent.
type node struct {
	identity objectReference
	// dependents will be read by the orphan() routine, we need to protect it with a lock.
	dependentsLock sync.RWMutex
	// dependents are the nodes that have node.identity as a
	// metadata.ownerReference.
	dependents map[*node]struct{}
	// this is set by processGraphChanges() if the object has non-nil DeletionTimestamp
	// and has the FinalizerDeleteDependents.
	deletingDependents     bool
	deletingDependentsLock sync.RWMutex
	// this records if the object's deletionTimestamp is non-nil.
	beingDeleted     bool
	beingDeletedLock sync.RWMutex
	// this records if the object was constructed virtually and never observed via informer event
	virtual     bool
	virtualLock sync.RWMutex
	// when processing an Update event, we need to compare the updated
	// ownerReferences with the owners recorded in the graph.
	owners []metav1.OwnerReference
}

// clone() must only be called from the single-threaded GraphBuilder.processGraphChanges()
func (n *node) clone() *node {
	c := &node{
		identity:           n.identity,
		dependents:         make(map[*node]struct{}, len(n.dependents)),
		deletingDependents: n.deletingDependents,
		beingDeleted:       n.beingDeleted,
		virtual:            n.virtual,
		owners:             make([]metav1.OwnerReference, 0, len(n.owners)),
	}
	for dep := range n.dependents {
		c.dependents[dep] = struct{}{}
	}
	for _, owner := range n.owners {
		c.owners = append(c.owners, owner)
	}
	return c
}

// An object is on a one way trip to its final deletion if it starts being
// deleted, so we only provide a function to set beingDeleted to true.
func (n *node) markBeingDeleted() {
	n.beingDeletedLock.Lock()
	defer n.beingDeletedLock.Unlock()
	n.beingDeleted = true
}

func (n *node) isBeingDeleted() bool {
	n.beingDeletedLock.RLock()
	defer n.beingDeletedLock.RUnlock()
	return n.beingDeleted
}

func (n *node) markObserved() {
	n.virtualLock.Lock()
	defer n.virtualLock.Unlock()
	n.virtual = false
}
func (n *node) isObserved() bool {
	n.virtualLock.RLock()
	defer n.virtualLock.RUnlock()
	return !n.virtual
}

func (n *node) markDeletingDependents() {
	n.deletingDependentsLock.Lock()
	defer n.deletingDependentsLock.Unlock()
	n.deletingDependents = true
}

func (n *node) isDeletingDependents() bool {
	n.deletingDependentsLock.RLock()
	defer n.deletingDependentsLock.RUnlock()
	return n.deletingDependents
}

func (n *node) addDependent(dependent *node) {
	n.dependentsLock.Lock()
	defer n.dependentsLock.Unlock()
	n.dependents[dependent] = struct{}{}
}

func (n *node) deleteDependent(dependent *node) {
	n.dependentsLock.Lock()
	defer n.dependentsLock.Unlock()
	delete(n.dependents, dependent)
}

func (n *node) dependentsLength() int {
	n.dependentsLock.RLock()
	defer n.dependentsLock.RUnlock()
	return len(n.dependents)
}

// Note that this function does not provide any synchronization guarantees;
// items could be added to or removed from ownerNode.dependents the moment this
// function returns.
func (n *node) getDependents() []*node {
	n.dependentsLock.RLock()
	defer n.dependentsLock.RUnlock()
	var ret []*node
	for dep := range n.dependents {
		ret = append(ret, dep)
	}
	return ret
}

// blockingDependents returns the dependents that are blocking the deletion of
// n, i.e., the dependent that has an ownerReference pointing to n, and
// the BlockOwnerDeletion field of that ownerReference is true.
// Note that this function does not provide any synchronization guarantees;
// items could be added to or removed from ownerNode.dependents the moment this
// function returns.
func (n *node) blockingDependents() []*node {
	dependents := n.getDependents()
	var ret []*node
	for _, dep := range dependents {
		for _, owner := range dep.owners {
			if owner.UID == n.identity.UID && owner.BlockOwnerDeletion != nil && *owner.BlockOwnerDeletion {
				ret = append(ret, dep)
			}
		}
	}
	return ret
}

// ownerReferenceCoordinates returns an owner reference containing only the coordinate fields
// from the input reference (uid, name, kind, apiVersion)
func ownerReferenceCoordinates(ref metav1.OwnerReference) metav1.OwnerReference {
	return metav1.OwnerReference{
		UID:        ref.UID,
		Name:       ref.Name,
		Kind:       ref.Kind,
		APIVersion: ref.APIVersion,
	}
}

// ownerReferenceMatchesCoordinates returns true if all of the coordinate fields match
// between the two references (uid, name, kind, apiVersion)
func ownerReferenceMatchesCoordinates(a, b metav1.OwnerReference) bool {
	return a.UID == b.UID && a.Name == b.Name && a.Kind == b.Kind && a.APIVersion == b.APIVersion
}

// String renders node as a string using fmt. Acquires a read lock to ensure the
// reflective dump of dependents doesn't race with any concurrent writes.
func (n *node) String() string {
	n.dependentsLock.RLock()
	defer n.dependentsLock.RUnlock()
	return fmt.Sprintf("%#v", n)
}

type concurrentUIDToNode struct {
	uidToNodeLock sync.RWMutex
	uidToNode     map[types.UID]*node
}

func (m *concurrentUIDToNode) Write(node *node) {
	m.uidToNodeLock.Lock()
	defer m.uidToNodeLock.Unlock()
	m.uidToNode[node.identity.UID] = node
}

func (m *concurrentUIDToNode) Read(uid types.UID) (*node, bool) {
	m.uidToNodeLock.RLock()
	defer m.uidToNodeLock.RUnlock()
	n, ok := m.uidToNode[uid]
	return n, ok
}

func (m *concurrentUIDToNode) Delete(uid types.UID) {
	m.uidToNodeLock.Lock()
	defer m.uidToNodeLock.Unlock()
	delete(m.uidToNode, uid)
}
