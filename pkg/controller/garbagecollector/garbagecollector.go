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
	"sync"
	"time"

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"

	// import known versions
	_ "k8s.io/client-go/kubernetes"
)

const ResourceResyncTime time.Duration = 0

// GarbageCollector runs reflectors to watch for changes of managed API
// objects, funnels the results to a single-threaded dependencyGraphBuilder,
// which builds a graph caching the dependencies among objects. Triggered by the
// graph changes, the dependencyGraphBuilder enqueues objects that can
// potentially be garbage-collected to the `attemptToDelete` queue, and enqueues
// objects whose dependents need to be orphaned to the `attemptToOrphan` queue.
// The GarbageCollector has workers who consume these two queues, send requests
// to the API server to delete/update the objects accordingly.
// Note that having the dependencyGraphBuilder notify the garbage collector
// ensures that the garbage collector operates with a graph that is at least as
// up to date as the notification is sent.
type GarbageCollector struct {
	restMapper     resettableRESTMapper
	metadataClient metadata.Interface
	// garbage collector attempts to delete the items in attemptToDelete queue when the time is ripe.
	attemptToDelete workqueue.RateLimitingInterface
	// garbage collector attempts to orphan the dependents of the items in the attemptToOrphan queue, then deletes the items.
	attemptToOrphan        workqueue.RateLimitingInterface
	dependencyGraphBuilder *GraphBuilder
	// GC caches the owners that do not exist according to the API server.
	absentOwnerCache *UIDCache
	sharedInformers  controller.InformerFactory

	workerLock sync.RWMutex
}

func NewGarbageCollector(
	metadataClient metadata.Interface,
	mapper resettableRESTMapper,
	deletableResources map[schema.GroupVersionResource]struct{},
	ignoredResources map[schema.GroupResource]struct{},
	sharedInformers controller.InformerFactory,
	informersStarted <-chan struct{},
) (*GarbageCollector, error) {
	attemptToDelete := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_attempt_to_delete")
	attemptToOrphan := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_attempt_to_orphan")
	absentOwnerCache := NewUIDCache(500)
	gc := &GarbageCollector{
		metadataClient:   metadataClient,
		restMapper:       mapper,
		attemptToDelete:  attemptToDelete,
		attemptToOrphan:  attemptToOrphan,
		absentOwnerCache: absentOwnerCache,
	}
	gb := &GraphBuilder{
		metadataClient:   metadataClient,
		informersStarted: informersStarted,
		restMapper:       mapper,
		graphChanges:     workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_graph_changes"),
		uidToNode: &concurrentUIDToNode{
			uidToNode: make(map[types.UID]*node),
		},
		attemptToDelete:  attemptToDelete,
		attemptToOrphan:  attemptToOrphan,
		absentOwnerCache: absentOwnerCache,
		sharedInformers:  sharedInformers,
		ignoredResources: ignoredResources,
	}
	if err := gb.syncMonitors(deletableResources); err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to sync all monitors: %v", err))
	}
	gc.dependencyGraphBuilder = gb

	return gc, nil
}

// resyncMonitors starts or stops resource monitors as needed to ensure that all
// (and only) those resources present in the map are monitored.
func (gc *GarbageCollector) resyncMonitors(deletableResources map[schema.GroupVersionResource]struct{}) error {
	if err := gc.dependencyGraphBuilder.syncMonitors(deletableResources); err != nil {
		return err
	}
	gc.dependencyGraphBuilder.startMonitors()
	return nil
}

func (gc *GarbageCollector) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer gc.attemptToDelete.ShutDown()
	defer gc.attemptToOrphan.ShutDown()
	defer gc.dependencyGraphBuilder.graphChanges.ShutDown()

	klog.Infof("Starting garbage collector controller")
	defer klog.Infof("Shutting down garbage collector controller")

	go gc.dependencyGraphBuilder.Run(stopCh)

	if !controller.WaitForCacheSync("garbage collector", stopCh, gc.dependencyGraphBuilder.IsSynced) {
		return
	}

	klog.Infof("Garbage collector: all resource monitors have synced. Proceeding to collect garbage")

	// gc workers
	for i := 0; i < workers; i++ {
		go wait.Until(gc.runAttemptToDeleteWorker, 1*time.Second, stopCh)
		go wait.Until(gc.runAttemptToOrphanWorker, 1*time.Second, stopCh)
	}

	<-stopCh
}

// resettableRESTMapper is a RESTMapper which is capable of resetting itself
// from discovery.
type resettableRESTMapper interface {
	meta.RESTMapper
	Reset()
}

// Sync periodically resyncs the garbage collector when new resources are
// observed from discovery. When new resources are detected, Sync will stop all
// GC workers, reset gc.restMapper, and resync the monitors.
//
// Note that discoveryClient should NOT be shared with gc.restMapper, otherwise
// the mapper's underlying discovery client will be unnecessarily reset during
// the course of detecting new resources.
func (gc *GarbageCollector) Sync(discoveryClient discovery.ServerResourcesInterface, period time.Duration, stopCh <-chan struct{}) {
	oldResources := make(map[schema.GroupVersionResource]struct{})
	wait.Until(func() {
		// Get the current resource list from discovery.
		newResources := GetDeletableResources(discoveryClient)

		// This can occur if there is an internal error in GetDeletableResources.
		if len(newResources) == 0 {
			klog.V(2).Infof("no resources reported by discovery, skipping garbage collector sync")
			return
		}

		// Decide whether discovery has reported a change.
		if reflect.DeepEqual(oldResources, newResources) {
			klog.V(5).Infof("no resource updates from discovery, skipping garbage collector sync")
			return
		}

		// Ensure workers are paused to avoid processing events before informers
		// have resynced.
		gc.workerLock.Lock()
		defer gc.workerLock.Unlock()

		// Once we get here, we should not unpause workers until we've successfully synced
		attempt := 0
		wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
			attempt++

			// On a reattempt, check if available resources have changed
			if attempt > 1 {
				newResources = GetDeletableResources(discoveryClient)
				if len(newResources) == 0 {
					klog.V(2).Infof("no resources reported by discovery (attempt %d)", attempt)
					return false, nil
				}
			}

			klog.V(2).Infof("syncing garbage collector with updated resources from discovery (attempt %d): %s", attempt, printDiff(oldResources, newResources))

			// Resetting the REST mapper will also invalidate the underlying discovery
			// client. This is a leaky abstraction and assumes behavior about the REST
			// mapper, but we'll deal with it for now.
			gc.restMapper.Reset()
			klog.V(4).Infof("reset restmapper")

			// Perform the monitor resync and wait for controllers to report cache sync.
			//
			// NOTE: It's possible that newResources will diverge from the resources
			// discovered by restMapper during the call to Reset, since they are
			// distinct discovery clients invalidated at different times. For example,
			// newResources may contain resources not returned in the restMapper's
			// discovery call if the resources appeared in-between the calls. In that
			// case, the restMapper will fail to map some of newResources until the next
			// attempt.
			if err := gc.resyncMonitors(newResources); err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to sync resource monitors (attempt %d): %v", attempt, err))
				return false, nil
			}
			klog.V(4).Infof("resynced monitors")

			// wait for caches to fill for a while (our sync period) before attempting to rediscover resources and retry syncing.
			// this protects us from deadlocks where available resources changed and one of our informer caches will never fill.
			// informers keep attempting to sync in the background, so retrying doesn't interrupt them.
			// the call to resyncMonitors on the reattempt will no-op for resources that still exist.
			// note that workers stay paused until we successfully resync.
			if !controller.WaitForCacheSync("garbage collector", waitForStopOrTimeout(stopCh, period), gc.dependencyGraphBuilder.IsSynced) {
				utilruntime.HandleError(fmt.Errorf("timed out waiting for dependency graph builder sync during GC sync (attempt %d)", attempt))
				return false, nil
			}

			// success, break out of the loop
			return true, nil
		}, stopCh)

		// Finally, keep track of our new state. Do this after all preceding steps
		// have succeeded to ensure we'll retry on subsequent syncs if an error
		// occurred.
		oldResources = newResources
		klog.V(2).Infof("synced garbage collector")
	}, period, stopCh)
}

// printDiff returns a human-readable summary of what resources were added and removed
func printDiff(oldResources, newResources map[schema.GroupVersionResource]struct{}) string {
	removed := sets.NewString()
	for oldResource := range oldResources {
		if _, ok := newResources[oldResource]; !ok {
			removed.Insert(fmt.Sprintf("%+v", oldResource))
		}
	}
	added := sets.NewString()
	for newResource := range newResources {
		if _, ok := oldResources[newResource]; !ok {
			added.Insert(fmt.Sprintf("%+v", newResource))
		}
	}
	return fmt.Sprintf("added: %v, removed: %v", added.List(), removed.List())
}

// waitForStopOrTimeout returns a stop channel that closes when the provided stop channel closes or when the specified timeout is reached
func waitForStopOrTimeout(stopCh <-chan struct{}, timeout time.Duration) <-chan struct{} {
	stopChWithTimeout := make(chan struct{})
	go func() {
		select {
		case <-stopCh:
		case <-time.After(timeout):
		}
		close(stopChWithTimeout)
	}()
	return stopChWithTimeout
}

func (gc *GarbageCollector) IsSynced() bool {
	return gc.dependencyGraphBuilder.IsSynced()
}

func (gc *GarbageCollector) runAttemptToDeleteWorker() {
	for gc.attemptToDeleteWorker() {
	}
}

func (gc *GarbageCollector) attemptToDeleteWorker() bool {
	item, quit := gc.attemptToDelete.Get()
	gc.workerLock.RLock()
	defer gc.workerLock.RUnlock()
	if quit {
		return false
	}
	defer gc.attemptToDelete.Done(item)
	n, ok := item.(*node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("expect *node, got %#v", item))
		return true
	}
	err := gc.attemptToDeleteItem(n)
	if err != nil {
		if _, ok := err.(*restMappingError); ok {
			// There are at least two ways this can happen:
			// 1. The reference is to an object of a custom type that has not yet been
			//    recognized by gc.restMapper (this is a transient error).
			// 2. The reference is to an invalid group/version. We don't currently
			//    have a way to distinguish this from a valid type we will recognize
			//    after the next discovery sync.
			// For now, record the error and retry.
			klog.V(5).Infof("error syncing item %s: %v", n, err)
		} else {
			utilruntime.HandleError(fmt.Errorf("error syncing item %s: %v", n, err))
		}
		// retry if garbage collection of an object failed.
		gc.attemptToDelete.AddRateLimited(item)
	} else if !n.isObserved() {
		// requeue if item hasn't been observed via an informer event yet.
		// otherwise a virtual node for an item added AND removed during watch reestablishment can get stuck in the graph and never removed.
		// see https://issue.k8s.io/56121
		klog.V(5).Infof("item %s hasn't been observed via informer yet", n.identity)
		gc.attemptToDelete.AddRateLimited(item)
	}
	return true
}

// isDangling check if a reference is pointing to an object that doesn't exist.
// If isDangling looks up the referenced object at the API server, it also
// returns its latest state.
func (gc *GarbageCollector) isDangling(reference metav1.OwnerReference, item *node) (
	dangling bool, owner *metav1.PartialObjectMetadata, err error) {
	if gc.absentOwnerCache.Has(reference.UID) {
		klog.V(5).Infof("according to the absentOwnerCache, object %s's owner %s/%s, %s does not exist", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
		return true, nil, nil
	}
	// TODO: we need to verify the reference resource is supported by the
	// system. If it's not a valid resource, the garbage collector should i)
	// ignore the reference when decide if the object should be deleted, and
	// ii) should update the object to remove such references. This is to
	// prevent objects having references to an old resource from being
	// deleted during a cluster upgrade.
	resource, namespaced, err := gc.apiResource(reference.APIVersion, reference.Kind)
	if err != nil {
		return false, nil, err
	}

	// TODO: It's only necessary to talk to the API server if the owner node
	// is a "virtual" node. The local graph could lag behind the real
	// status, but in practice, the difference is small.
	owner, err = gc.metadataClient.Resource(resource).Namespace(resourceDefaultNamespace(namespaced, item.identity.Namespace)).Get(reference.Name, metav1.GetOptions{})
	switch {
	case errors.IsNotFound(err):
		gc.absentOwnerCache.Add(reference.UID)
		klog.V(5).Infof("object %s's owner %s/%s, %s is not found", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
		return true, nil, nil
	case err != nil:
		return false, nil, err
	}

	if owner.GetUID() != reference.UID {
		klog.V(5).Infof("object %s's owner %s/%s, %s is not found, UID mismatch", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
		gc.absentOwnerCache.Add(reference.UID)
		return true, nil, nil
	}
	return false, owner, nil
}

// classify the latestReferences to three categories:
// solid: the owner exists, and is not "waitingForDependentsDeletion"
// dangling: the owner does not exist
// waitingForDependentsDeletion: the owner exists, its deletionTimestamp is non-nil, and it has
// FinalizerDeletingDependents
// This function communicates with the server.
func (gc *GarbageCollector) classifyReferences(item *node, latestReferences []metav1.OwnerReference) (
	solid, dangling, waitingForDependentsDeletion []metav1.OwnerReference, err error) {
	for _, reference := range latestReferences {
		isDangling, owner, err := gc.isDangling(reference, item)
		if err != nil {
			return nil, nil, nil, err
		}
		if isDangling {
			dangling = append(dangling, reference)
			continue
		}

		ownerAccessor, err := meta.Accessor(owner)
		if err != nil {
			return nil, nil, nil, err
		}
		if ownerAccessor.GetDeletionTimestamp() != nil && hasDeleteDependentsFinalizer(ownerAccessor) {
			waitingForDependentsDeletion = append(waitingForDependentsDeletion, reference)
		} else {
			solid = append(solid, reference)
		}
	}
	return solid, dangling, waitingForDependentsDeletion, nil
}

func ownerRefsToUIDs(refs []metav1.OwnerReference) []types.UID {
	var ret []types.UID
	for _, ref := range refs {
		ret = append(ret, ref.UID)
	}
	return ret
}

func (gc *GarbageCollector) attemptToDeleteItem(item *node) error {
	klog.V(2).Infof("processing item %s", item.identity)
	// "being deleted" is an one-way trip to the final deletion. We'll just wait for the final deletion, and then process the object's dependents.
	if item.isBeingDeleted() && !item.isDeletingDependents() {
		klog.V(5).Infof("processing item %s returned at once, because its DeletionTimestamp is non-nil", item.identity)
		return nil
	}
	// TODO: It's only necessary to talk to the API server if this is a
	// "virtual" node. The local graph could lag behind the real status, but in
	// practice, the difference is small.
	latest, err := gc.getObject(item.identity)
	switch {
	case errors.IsNotFound(err):
		// the GraphBuilder can add "virtual" node for an owner that doesn't
		// exist yet, so we need to enqueue a virtual Delete event to remove
		// the virtual node from GraphBuilder.uidToNode.
		klog.V(5).Infof("item %v not found, generating a virtual delete event", item.identity)
		gc.dependencyGraphBuilder.enqueueVirtualDeleteEvent(item.identity)
		// since we're manually inserting a delete event to remove this node,
		// we don't need to keep tracking it as a virtual node and requeueing in attemptToDelete
		item.markObserved()
		return nil
	case err != nil:
		return err
	}

	if latest.GetUID() != item.identity.UID {
		klog.V(5).Infof("UID doesn't match, item %v not found, generating a virtual delete event", item.identity)
		gc.dependencyGraphBuilder.enqueueVirtualDeleteEvent(item.identity)
		// since we're manually inserting a delete event to remove this node,
		// we don't need to keep tracking it as a virtual node and requeueing in attemptToDelete
		item.markObserved()
		return nil
	}

	// TODO: attemptToOrphanWorker() routine is similar. Consider merging
	// attemptToOrphanWorker() into attemptToDeleteItem() as well.
	if item.isDeletingDependents() {
		return gc.processDeletingDependentsItem(item)
	}

	// compute if we should delete the item
	ownerReferences := latest.GetOwnerReferences()
	if len(ownerReferences) == 0 {
		klog.V(2).Infof("object %s's doesn't have an owner, continue on next item", item.identity)
		return nil
	}

	solid, dangling, waitingForDependentsDeletion, err := gc.classifyReferences(item, ownerReferences)
	if err != nil {
		return err
	}
	klog.V(5).Infof("classify references of %s.\nsolid: %#v\ndangling: %#v\nwaitingForDependentsDeletion: %#v\n", item.identity, solid, dangling, waitingForDependentsDeletion)

	switch {
	case len(solid) != 0:
		klog.V(2).Infof("object %#v has at least one existing owner: %#v, will not garbage collect", item.identity, solid)
		if len(dangling) == 0 && len(waitingForDependentsDeletion) == 0 {
			return nil
		}
		klog.V(2).Infof("remove dangling references %#v and waiting references %#v for object %s", dangling, waitingForDependentsDeletion, item.identity)
		// waitingForDependentsDeletion needs to be deleted from the
		// ownerReferences, otherwise the referenced objects will be stuck with
		// the FinalizerDeletingDependents and never get deleted.
		ownerUIDs := append(ownerRefsToUIDs(dangling), ownerRefsToUIDs(waitingForDependentsDeletion)...)
		patch := deleteOwnerRefStrategicMergePatch(item.identity.UID, ownerUIDs...)
		_, err = gc.patch(item, patch, func(n *node) ([]byte, error) {
			return gc.deleteOwnerRefJSONMergePatch(n, ownerUIDs...)
		})
		return err
	case len(waitingForDependentsDeletion) != 0 && item.dependentsLength() != 0:
		deps := item.getDependents()
		for _, dep := range deps {
			if dep.isDeletingDependents() {
				// this circle detection has false positives, we need to
				// apply a more rigorous detection if this turns out to be a
				// problem.
				// there are multiple workers run attemptToDeleteItem in
				// parallel, the circle detection can fail in a race condition.
				klog.V(2).Infof("processing object %s, some of its owners and its dependent [%s] have FinalizerDeletingDependents, to prevent potential cycle, its ownerReferences are going to be modified to be non-blocking, then the object is going to be deleted with Foreground", item.identity, dep.identity)
				patch, err := item.unblockOwnerReferencesStrategicMergePatch()
				if err != nil {
					return err
				}
				if _, err := gc.patch(item, patch, gc.unblockOwnerReferencesJSONMergePatch); err != nil {
					return err
				}
				break
			}
		}
		klog.V(2).Infof("at least one owner of object %s has FinalizerDeletingDependents, and the object itself has dependents, so it is going to be deleted in Foreground", item.identity)
		// the deletion event will be observed by the graphBuilder, so the item
		// will be processed again in processDeletingDependentsItem. If it
		// doesn't have dependents, the function will remove the
		// FinalizerDeletingDependents from the item, resulting in the final
		// deletion of the item.
		policy := metav1.DeletePropagationForeground
		return gc.deleteObject(item.identity, &policy)
	default:
		// item doesn't have any solid owner, so it needs to be garbage
		// collected. Also, none of item's owners is waiting for the deletion of
		// the dependents, so set propagationPolicy based on existing finalizers.
		var policy metav1.DeletionPropagation
		switch {
		case hasOrphanFinalizer(latest):
			// if an existing orphan finalizer is already on the object, honor it.
			policy = metav1.DeletePropagationOrphan
		case hasDeleteDependentsFinalizer(latest):
			// if an existing foreground finalizer is already on the object, honor it.
			policy = metav1.DeletePropagationForeground
		default:
			// otherwise, default to background.
			policy = metav1.DeletePropagationBackground
		}
		klog.V(2).Infof("delete object %s with propagation policy %s", item.identity, policy)
		return gc.deleteObject(item.identity, &policy)
	}
}

// process item that's waiting for its dependents to be deleted
func (gc *GarbageCollector) processDeletingDependentsItem(item *node) error {
	blockingDependents := item.blockingDependents()
	if len(blockingDependents) == 0 {
		klog.V(2).Infof("remove DeleteDependents finalizer for item %s", item.identity)
		return gc.removeFinalizer(item, metav1.FinalizerDeleteDependents)
	}
	for _, dep := range blockingDependents {
		if !dep.isDeletingDependents() {
			klog.V(2).Infof("adding %s to attemptToDelete, because its owner %s is deletingDependents", dep.identity, item.identity)
			gc.attemptToDelete.Add(dep)
		}
	}
	return nil
}

// dependents are copies of pointers to the owner's dependents, they don't need to be locked.
func (gc *GarbageCollector) orphanDependents(owner objectReference, dependents []*node) error {
	errCh := make(chan error, len(dependents))
	wg := sync.WaitGroup{}
	wg.Add(len(dependents))
	for i := range dependents {
		go func(dependent *node) {
			defer wg.Done()
			// the dependent.identity.UID is used as precondition
			patch := deleteOwnerRefStrategicMergePatch(dependent.identity.UID, owner.UID)
			_, err := gc.patch(dependent, patch, func(n *node) ([]byte, error) {
				return gc.deleteOwnerRefJSONMergePatch(n, owner.UID)
			})
			// note that if the target ownerReference doesn't exist in the
			// dependent, strategic merge patch will NOT return an error.
			if err != nil && !errors.IsNotFound(err) {
				errCh <- fmt.Errorf("orphaning %s failed, %v", dependent.identity, err)
			}
		}(dependents[i])
	}
	wg.Wait()
	close(errCh)

	var errorsSlice []error
	for e := range errCh {
		errorsSlice = append(errorsSlice, e)
	}

	if len(errorsSlice) != 0 {
		return fmt.Errorf("failed to orphan dependents of owner %s, got errors: %s", owner, utilerrors.NewAggregate(errorsSlice).Error())
	}
	klog.V(5).Infof("successfully updated all dependents of owner %s", owner)
	return nil
}

func (gc *GarbageCollector) runAttemptToOrphanWorker() {
	for gc.attemptToOrphanWorker() {
	}
}

// attemptToOrphanWorker dequeues a node from the attemptToOrphan, then finds its
// dependents based on the graph maintained by the GC, then removes it from the
// OwnerReferences of its dependents, and finally updates the owner to remove
// the "Orphan" finalizer. The node is added back into the attemptToOrphan if any of
// these steps fail.
func (gc *GarbageCollector) attemptToOrphanWorker() bool {
	item, quit := gc.attemptToOrphan.Get()
	gc.workerLock.RLock()
	defer gc.workerLock.RUnlock()
	if quit {
		return false
	}
	defer gc.attemptToOrphan.Done(item)
	owner, ok := item.(*node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("expect *node, got %#v", item))
		return true
	}
	// we don't need to lock each element, because they never get updated
	owner.dependentsLock.RLock()
	dependents := make([]*node, 0, len(owner.dependents))
	for dependent := range owner.dependents {
		dependents = append(dependents, dependent)
	}
	owner.dependentsLock.RUnlock()

	err := gc.orphanDependents(owner.identity, dependents)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("orphanDependents for %s failed with %v", owner.identity, err))
		gc.attemptToOrphan.AddRateLimited(item)
		return true
	}
	// update the owner, remove "orphaningFinalizer" from its finalizers list
	err = gc.removeFinalizer(owner, metav1.FinalizerOrphanDependents)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("removeOrphanFinalizer for %s failed with %v", owner.identity, err))
		gc.attemptToOrphan.AddRateLimited(item)
	}
	return true
}

// *FOR TEST USE ONLY*
// GraphHasUID returns if the GraphBuilder has a particular UID store in its
// uidToNode graph. It's useful for debugging.
// This method is used by integration tests.
func (gc *GarbageCollector) GraphHasUID(u types.UID) bool {
	_, ok := gc.dependencyGraphBuilder.uidToNode.Read(u)
	return ok
}

// GetDeletableResources returns all resources from discoveryClient that the
// garbage collector should recognize and work with. More specifically, all
// preferred resources which support the 'delete', 'list', and 'watch' verbs.
//
// All discovery errors are considered temporary. Upon encountering any error,
// GetDeletableResources will log and return any discovered resources it was
// able to process (which may be none).
func GetDeletableResources(discoveryClient discovery.ServerResourcesInterface) map[schema.GroupVersionResource]struct{} {
	preferredResources, err := discoveryClient.ServerPreferredResources()
	if err != nil {
		if discovery.IsGroupDiscoveryFailedError(err) {
			klog.Warningf("failed to discover some groups: %v", err.(*discovery.ErrGroupDiscoveryFailed).Groups)
		} else {
			klog.Warningf("failed to discover preferred resources: %v", err)
		}
	}
	if preferredResources == nil {
		return map[schema.GroupVersionResource]struct{}{}
	}

	// This is extracted from discovery.GroupVersionResources to allow tolerating
	// failures on a per-resource basis.
	deletableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"delete", "list", "watch"}}, preferredResources)
	deletableGroupVersionResources := map[schema.GroupVersionResource]struct{}{}
	for _, rl := range deletableResources {
		gv, err := schema.ParseGroupVersion(rl.GroupVersion)
		if err != nil {
			klog.Warningf("ignoring invalid discovered resource %q: %v", rl.GroupVersion, err)
			continue
		}
		for i := range rl.APIResources {
			deletableGroupVersionResources[schema.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: rl.APIResources[i].Name}] = struct{}{}
		}
	}

	return deletableGroupVersionResources
}
