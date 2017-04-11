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
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
	// install the prometheus plugin
	_ "k8s.io/kubernetes/pkg/util/workqueue/prometheus"
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
	restMapper meta.RESTMapper
	// clientPool uses the regular dynamicCodec. We need it to update
	// finalizers. It can be removed if we support patching finalizers.
	clientPool dynamic.ClientPool
	// garbage collector attempts to delete the items in attemptToDelete queue when the time is ripe.
	attemptToDelete workqueue.RateLimitingInterface
	// garbage collector attempts to orphan the dependents of the items in the attemptToOrphan queue, then deletes the items.
	attemptToOrphan        workqueue.RateLimitingInterface
	dependencyGraphBuilder *GraphBuilder
	// used to register exactly once the rate limiter of the dynamic client
	// used by the garbage collector controller.
	registeredRateLimiter *RegisteredRateLimiter
	// GC caches the owners that do not exist according to the API server.
	absentOwnerCache *UIDCache
}

func NewGarbageCollector(metaOnlyClientPool dynamic.ClientPool, clientPool dynamic.ClientPool, mapper meta.RESTMapper, deletableResources map[schema.GroupVersionResource]struct{}) (*GarbageCollector, error) {
	attemptToDelete := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_attempt_to_delete")
	attemptToOrphan := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_attempt_to_orphan")
	absentOwnerCache := NewUIDCache(500)
	gc := &GarbageCollector{
		clientPool:            clientPool,
		restMapper:            mapper,
		attemptToDelete:       attemptToDelete,
		attemptToOrphan:       attemptToOrphan,
		registeredRateLimiter: NewRegisteredRateLimiter(deletableResources),
		absentOwnerCache:      absentOwnerCache,
	}
	gb := &GraphBuilder{
		metaOnlyClientPool:                  metaOnlyClientPool,
		registeredRateLimiterForControllers: NewRegisteredRateLimiter(deletableResources),
		restMapper:                          mapper,
		graphChanges:                        workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_graph_changes"),
		uidToNode: &concurrentUIDToNode{
			uidToNode: make(map[types.UID]*node),
		},
		attemptToDelete:  attemptToDelete,
		attemptToOrphan:  attemptToOrphan,
		absentOwnerCache: absentOwnerCache,
	}
	if err := gb.monitorsForResources(deletableResources); err != nil {
		return nil, err
	}
	gc.dependencyGraphBuilder = gb

	return gc, nil
}

func (gc *GarbageCollector) Run(workers int, stopCh <-chan struct{}) {
	defer gc.attemptToDelete.ShutDown()
	defer gc.attemptToOrphan.ShutDown()
	defer gc.dependencyGraphBuilder.graphChanges.ShutDown()

	glog.Infof("Garbage Collector: Initializing")
	gc.dependencyGraphBuilder.Run(stopCh)
	if !cache.WaitForCacheSync(stopCh, gc.dependencyGraphBuilder.HasSynced) {
		return
	}
	glog.Infof("Garbage Collector: All resource monitors have synced. Proceeding to collect garbage")

	// gc workers
	for i := 0; i < workers; i++ {
		go wait.Until(gc.runAttemptToDeleteWorker, 1*time.Second, stopCh)
		go wait.Until(gc.runAttemptToOrphanWorker, 1*time.Second, stopCh)
	}
	Register()
	<-stopCh
	glog.Infof("Garbage Collector: Shutting down")
}

func (gc *GarbageCollector) HasSynced() bool {
	return gc.dependencyGraphBuilder.HasSynced()
}

func (gc *GarbageCollector) runAttemptToDeleteWorker() {
	for gc.attemptToDeleteWorker() {
	}
}

func (gc *GarbageCollector) attemptToDeleteWorker() bool {
	item, quit := gc.attemptToDelete.Get()
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
		// TODO: remove this block when gc starts using dynamic RESTMapper.
		if restMappingError, ok := err.(*restMappingError); ok {
			utilruntime.HandleError(fmt.Errorf("Ignore syncing item %#v: %s", n, restMappingError.Message()))
			// The RESTMapper is static, so no need to retry, otherwise we'll get the same error.
			return true
		}
		utilruntime.HandleError(fmt.Errorf("Error syncing item %#v: %v", n, err))
		// retry if garbage collection of an object failed.
		gc.attemptToDelete.AddRateLimited(item)
	}
	return true
}

func objectReferenceToMetadataOnlyObject(ref objectReference) *metaonly.MetadataOnlyObject {
	return &metaonly.MetadataOnlyObject{
		TypeMeta: metav1.TypeMeta{
			APIVersion: ref.APIVersion,
			Kind:       ref.Kind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ref.Namespace,
			UID:       ref.UID,
			Name:      ref.Name,
		},
	}
}

// isDangling check if a reference is pointing to an object that doesn't exist.
// If isDangling looks up the referenced object at the API server, it also
// returns its latest state.
func (gc *GarbageCollector) isDangling(reference metav1.OwnerReference, item *node) (
	dangling bool, owner *unstructured.Unstructured, err error) {
	if gc.absentOwnerCache.Has(reference.UID) {
		glog.V(5).Infof("according to the absentOwnerCache, object %s's owner %s/%s, %s does not exist", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
		return true, nil, nil
	}
	// TODO: we need to verify the reference resource is supported by the
	// system. If it's not a valid resource, the garbage collector should i)
	// ignore the reference when decide if the object should be deleted, and
	// ii) should update the object to remove such references. This is to
	// prevent objects having references to an old resource from being
	// deleted during a cluster upgrade.
	fqKind := schema.FromAPIVersionAndKind(reference.APIVersion, reference.Kind)
	client, err := gc.clientPool.ClientForGroupVersionKind(fqKind)
	if err != nil {
		return false, nil, err
	}
	resource, err := gc.apiResource(reference.APIVersion, reference.Kind, len(item.identity.Namespace) != 0)
	if err != nil {
		return false, nil, err
	}
	// TODO: It's only necessary to talk to the API server if the owner node
	// is a "virtual" node. The local graph could lag behind the real
	// status, but in practice, the difference is small.
	owner, err = client.Resource(resource, item.identity.Namespace).Get(reference.Name)
	switch {
	case errors.IsNotFound(err):
		gc.absentOwnerCache.Add(reference.UID)
		glog.V(5).Infof("object %s's owner %s/%s, %s is not found", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
		return true, nil, nil
	case err != nil:
		return false, nil, err
	}

	if owner.GetUID() != reference.UID {
		glog.V(5).Infof("object %s's owner %s/%s, %s is not found, UID mismatch", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
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

func (gc *GarbageCollector) generateVirtualDeleteEvent(identity objectReference) {
	event := &event{
		eventType: deleteEvent,
		obj:       objectReferenceToMetadataOnlyObject(identity),
	}
	glog.V(5).Infof("generating virtual delete event for %s\n\n", event.obj)
	gc.dependencyGraphBuilder.enqueueChanges(event)
}

func ownerRefsToUIDs(refs []metav1.OwnerReference) []types.UID {
	var ret []types.UID
	for _, ref := range refs {
		ret = append(ret, ref.UID)
	}
	return ret
}

func (gc *GarbageCollector) attemptToDeleteItem(item *node) error {
	glog.V(2).Infof("processing item %s", item.identity)
	// "being deleted" is an one-way trip to the final deletion. We'll just wait for the final deletion, and then process the object's dependents.
	if item.isBeingDeleted() && !item.isDeletingDependents() {
		glog.V(5).Infof("processing item %s returned at once, because its DeletionTimestamp is non-nil", item.identity)
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
		glog.V(5).Infof("item %v not found, generating a virtual delete event", item.identity)
		gc.generateVirtualDeleteEvent(item.identity)
		return nil
	case err != nil:
		return err
	}

	if latest.GetUID() != item.identity.UID {
		glog.V(5).Infof("UID doesn't match, item %v not found, generating a virtual delete event", item.identity)
		gc.generateVirtualDeleteEvent(item.identity)
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
		glog.V(2).Infof("object %s's doesn't have an owner, continue on next item", item.identity)
		return nil
	}

	solid, dangling, waitingForDependentsDeletion, err := gc.classifyReferences(item, ownerReferences)
	if err != nil {
		return err
	}
	glog.V(5).Infof("classify references of %s.\nsolid: %#v\ndangling: %#v\nwaitingForDependentsDeletion: %#v\n", item.identity, solid, dangling, waitingForDependentsDeletion)

	switch {
	case len(solid) != 0:
		glog.V(2).Infof("object %s has at least one existing owner: %#v, will not garbage collect", solid, item.identity)
		if len(dangling) != 0 || len(waitingForDependentsDeletion) != 0 {
			glog.V(2).Infof("remove dangling references %#v and waiting references %#v for object %s", dangling, waitingForDependentsDeletion, item.identity)
		}
		// waitingForDependentsDeletion needs to be deleted from the
		// ownerReferences, otherwise the referenced objects will be stuck with
		// the FinalizerDeletingDependents and never get deleted.
		patch := deleteOwnerRefPatch(item.identity.UID, append(ownerRefsToUIDs(dangling), ownerRefsToUIDs(waitingForDependentsDeletion)...)...)
		_, err = gc.patchObject(item.identity, patch)
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
				glog.V(2).Infof("processing object %s, some of its owners and its dependent [%s] have FinalizerDeletingDependents, to prevent potential cycle, its ownerReferences are going to be modified to be non-blocking, then the object is going to be deleted with Foreground", item.identity, dep.identity)
				patch, err := item.patchToUnblockOwnerReferences()
				if err != nil {
					return err
				}
				if _, err := gc.patchObject(item.identity, patch); err != nil {
					return err
				}
				break
			}
		}
		glog.V(2).Infof("at least one owner of object %s has FinalizerDeletingDependents, and the object itself has dependents, so it is going to be deleted with Foreground", item.identity)
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
		// the dependents, so GC deletes item with Default.
		glog.V(2).Infof("delete object %s with Default", item.identity)
		return gc.deleteObject(item.identity, nil)
	}
}

// process item that's waiting for its dependents to be deleted
func (gc *GarbageCollector) processDeletingDependentsItem(item *node) error {
	blockingDependents := item.blockingDependents()
	if len(blockingDependents) == 0 {
		glog.V(2).Infof("remove DeleteDependents finalizer for item %s", item.identity)
		return gc.removeFinalizer(item, metav1.FinalizerDeleteDependents)
	}
	for _, dep := range blockingDependents {
		if !dep.isDeletingDependents() {
			glog.V(2).Infof("adding %s to attemptToDelete, because its owner %s is deletingDependents", dep.identity, item.identity)
			gc.attemptToDelete.Add(dep)
		}
	}
	return nil
}

// dependents are copies of pointers to the owner's dependents, they don't need to be locked.
func (gc *GarbageCollector) orphanDependents(owner objectReference, dependents []*node) error {
	var failedDependents []objectReference
	var errorsSlice []error
	for _, dependent := range dependents {
		// the dependent.identity.UID is used as precondition
		patch := deleteOwnerRefPatch(dependent.identity.UID, owner.UID)
		_, err := gc.patchObject(dependent.identity, patch)
		// note that if the target ownerReference doesn't exist in the
		// dependent, strategic merge patch will NOT return an error.
		if err != nil && !errors.IsNotFound(err) {
			errorsSlice = append(errorsSlice, fmt.Errorf("orphaning %s failed with %v", dependent.identity, err))
		}
	}
	if len(failedDependents) != 0 {
		return fmt.Errorf("failed to orphan dependents of owner %s, got errors: %s", owner, utilerrors.NewAggregate(errorsSlice).Error())
	}
	glog.V(5).Infof("successfully updated all dependents of owner %s", owner)
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
		glog.V(5).Infof("orphanDependents for %s failed with %v", owner.identity, err)
		gc.attemptToOrphan.AddRateLimited(item)
		return true
	}
	// update the owner, remove "orphaningFinalizer" from its finalizers list
	err = gc.removeFinalizer(owner, metav1.FinalizerOrphanDependents)
	if err != nil {
		glog.V(5).Infof("removeOrphanFinalizer for %s failed with %v", owner.identity, err)
		gc.attemptToOrphan.AddRateLimited(item)
	}
	return true
}

// *FOR TEST USE ONLY*
// GraphHasUID returns if the GraphBuilder has a particular UID store in its
// uidToNode graph. It's useful for debugging.
// This method is used by integration tests.
func (gc *GarbageCollector) GraphHasUID(UIDs []types.UID) bool {
	for _, u := range UIDs {
		if _, ok := gc.dependencyGraphBuilder.uidToNode.Read(u); ok {
			return true
		}
	}
	return false
}
