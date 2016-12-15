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
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/types"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	// install the prometheus plugin
	_ "k8s.io/kubernetes/pkg/util/workqueue/prometheus"
	"k8s.io/kubernetes/pkg/watch"
)

const ResourceResyncTime time.Duration = 0

// GarbageCollector is responsible for carrying out cascading deletion, and
// removing ownerReferences from the dependents if the owner is deleted with
// DeleteOptions.OrphanDependents=true.
type GarbageCollector struct {
	restMapper meta.RESTMapper
	// metaOnlyClientPool uses a special codec, which removes fields except for
	// apiVersion, kind, and metadata during decoding.
	metaOnlyClientPool dynamic.ClientPool
	// clientPool uses the regular dynamicCodec. We need it to update
	// finalizers. It can be removed if we support patching finalizers.
	clientPool                          dynamic.ClientPool
	dirtyQueue                          workqueue.RateLimitingInterface
	orphanQueue                         workqueue.RateLimitingInterface
	controllers                         []*cache.Controller
	propagator                          *Propagator
	registeredRateLimiter               *RegisteredRateLimiter
	registeredRateLimiterForControllers *RegisteredRateLimiter
	// GC caches the owners that do not exist according to the API server.
	absentOwnerCache *UIDCache
}

func gcListWatcher(client *dynamic.Client, resource schema.GroupVersionResource) *cache.ListWatch {
	return &cache.ListWatch{
		ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
			// APIResource.Kind is not used by the dynamic client, so
			// leave it empty. We want to list this resource in all
			// namespaces if it's namespace scoped, so leave
			// APIResource.Namespaced as false is all right.
			apiResource := metav1.APIResource{Name: resource.Resource}
			return client.ParameterCodec(dynamic.VersionedParameterEncoderWithV1Fallback).
				Resource(&apiResource, v1.NamespaceAll).
				List(&options)
		},
		WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
			// APIResource.Kind is not used by the dynamic client, so
			// leave it empty. We want to list this resource in all
			// namespaces if it's namespace scoped, so leave
			// APIResource.Namespaced as false is all right.
			apiResource := metav1.APIResource{Name: resource.Resource}
			return client.ParameterCodec(dynamic.VersionedParameterEncoderWithV1Fallback).
				Resource(&apiResource, v1.NamespaceAll).
				Watch(&options)
		},
	}
}

func (gc *GarbageCollector) controllerFor(resource schema.GroupVersionResource, kind schema.GroupVersionKind) (*cache.Controller, error) {
	// TODO: consider store in one storage.
	glog.V(6).Infof("create storage for resource %s", resource)
	client, err := gc.metaOnlyClientPool.ClientForGroupVersionKind(kind)
	if err != nil {
		return nil, err
	}
	gc.registeredRateLimiterForControllers.registerIfNotPresent(resource.GroupVersion(), client, "garbage_collector_monitoring")
	setObjectTypeMeta := func(obj interface{}) {
		runtimeObject, ok := obj.(runtime.Object)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("expected runtime.Object, got %#v", obj))
		}
		runtimeObject.GetObjectKind().SetGroupVersionKind(kind)
	}
	_, controller := cache.NewInformer(
		gcListWatcher(client, resource),
		nil,
		ResourceResyncTime,
		cache.ResourceEventHandlerFuncs{
			// add the event to the propagator's eventQueue.
			AddFunc: func(obj interface{}) {
				setObjectTypeMeta(obj)
				event := &event{
					eventType: addEvent,
					obj:       obj,
				}
				gc.propagator.eventQueue.Add(event)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				setObjectTypeMeta(newObj)
				event := &event{updateEvent, newObj, oldObj}
				gc.propagator.eventQueue.Add(event)
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
				gc.propagator.eventQueue.Add(event)
			},
		},
	)
	return controller, nil
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
}

func NewGarbageCollector(metaOnlyClientPool dynamic.ClientPool, clientPool dynamic.ClientPool, mapper meta.RESTMapper, deletableResources map[schema.GroupVersionResource]struct{}) (*GarbageCollector, error) {
	dirtyQueue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_dirty")
	orphanQueue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_orphan")
	absentOwnerCache := NewUIDCache(500)
	gc := &GarbageCollector{
		metaOnlyClientPool:                  metaOnlyClientPool,
		clientPool:                          clientPool,
		restMapper:                          mapper,
		dirtyQueue:                          dirtyQueue,
		orphanQueue:                         orphanQueue,
		registeredRateLimiter:               NewRegisteredRateLimiter(deletableResources),
		registeredRateLimiterForControllers: NewRegisteredRateLimiter(deletableResources),
		absentOwnerCache:                    absentOwnerCache,
	}
	gc.propagator = &Propagator{
		eventQueue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "garbage_collector_event"),
		uidToNode: &concurrentUIDToNode{
			RWMutex:   &sync.RWMutex{},
			uidToNode: make(map[types.UID]*node),
		},
		dirtyQueue:       dirtyQueue,
		orphanQueue:      orphanQueue,
		absentOwnerCache: absentOwnerCache,
	}
	for resource := range deletableResources {
		if _, ok := ignoredResources[resource]; ok {
			glog.V(6).Infof("ignore resource %#v", resource)
			continue
		}
		kind, err := gc.restMapper.KindFor(resource)
		if err != nil {
			return nil, err
		}
		controller, err := gc.controllerFor(resource, kind)
		if err != nil {
			return nil, err
		}
		gc.controllers = append(gc.controllers, controller)
	}
	return gc, nil
}

func (gc *GarbageCollector) Run(workers int, stopCh <-chan struct{}) {
	defer gc.dirtyQueue.ShutDown()
	defer gc.orphanQueue.ShutDown()
	defer gc.propagator.eventQueue.ShutDown()

	glog.Infof("Garbage Collector: Initializing")
	for _, controller := range gc.controllers {
		go controller.Run(stopCh)
	}

	var syncs []cache.InformerSynced
	for _, controller := range gc.controllers {
		syncs = syncs.append(controller.HasSynced())
	}
	if !cache.WaitForCacheSync(stopCh, syncs...) {
		return
	}
	glog.Infof("Garbage Collector: All monitored resources synced. Proceeding to collect garbage")

	// worker
	go wait.Until(gc.propagator.processEvent, 0, stopCh)

	for i := 0; i < workers; i++ {
		go wait.Until(gc.worker, 0, stopCh)
		go wait.Until(gc.orphanFinalizer, 0, stopCh)
	}
	Register()
	<-stopCh
	glog.Infof("Garbage Collector: Shutting down")
}

func (gc *GarbageCollector) worker() {
	item, quit := gc.dirtyQueue.Get()
	if quit {
		return
	}
	defer gc.dirtyQueue.Done(item)
	n, ok := item.(*node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("expect *node, got %#v", item))
	}
	err := gc.processItem(n)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Error syncing item %#v: %v", n, err))
		// retry if garbage collection of an object failed.
		gc.dirtyQueue.AddRateLimited(item)
		return
	}
}

func objectReferenceToMetadataOnlyObject(ref objectReference) *metaonly.MetadataOnlyObject {
	return &metaonly.MetadataOnlyObject{
		TypeMeta: metav1.TypeMeta{
			APIVersion: ref.APIVersion,
			Kind:       ref.Kind,
		},
		ObjectMeta: v1.ObjectMeta{
			Namespace: ref.Namespace,
			UID:       ref.UID,
			Name:      ref.Name,
		},
	}
}

// classify the latestReferences to three categories:
// solid: the owner exists, and is not "waiting"
// dangling: the owner does not exist
// waiting: the owner exists, its deletionTimestamp is non-nil, and it has
// FinalizerDeletingDependents
// This function communicates with the server.
func (gc *GarbageCollector) classifyReferences(item *node, latestReferences []metav1.OwnerReference) (
	solid, dangling, waiting []metav1.OwnerReference, err error) {
	for _, reference := range latestReferences {
		if gc.absentOwnerCache.Has(reference.UID) {
			glog.V(6).Infof("according to the absentOwnerCache, object %s's owner %s/%s, %s does not exist", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
			dangling = append(dangling, reference)
			continue
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
			return solid, dangling, waiting, err
		}
		resource, err := gc.apiResource(reference.APIVersion, reference.Kind, len(item.identity.Namespace) != 0)
		if err != nil {
			return solid, dangling, waiting, err
		}
		owner, err := client.Resource(resource, item.identity.Namespace).Get(reference.Name)
		if err != nil {
			if !errors.IsNotFound(err) {
				return solid, dangling, waiting, err
			}
			gc.absentOwnerCache.Add(reference.UID)
			glog.V(6).Infof("object %s's owner %s/%s, %s is not found", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
			dangling = append(dangling, reference)
		}

		if owner.GetUID() != reference.UID {
			glog.V(6).Infof("object %s's owner %s/%s, %s is not found, UID mismatch", item.identity.UID, reference.APIVersion, reference.Kind, reference.Name)
			gc.absentOwnerCache.Add(reference.UID)
			dangling = append(dangling, reference)
			continue
		}

		ownerAccessor, err := meta.Accessor(owner)
		if err != nil {
			return solid, dangling, waiting, err
		}
		if ownerAccessor.GetDeletionTimestamp() != nil && hasDeleteDependentsFinalizer(ownerAccessor) {
			waiting = append(waiting, reference)
		} else {
			solid = append(solid, reference)
		}
	}
	return solid, dangling, waiting, nil
}

func (gc *GarbageCollector) generateVirtualDeleteEvent(identity objectReference) {
	event := &event{
		eventType: deleteEvent,
		obj:       objectReferenceToMetadataOnlyObject(identity),
	}
	glog.V(6).Infof("generating virtual delete event for %s\n\n", event.obj)
	gc.propagator.eventQueue.Add(event)
}

func ownerRefsToUIDs(refs []metav1.OwnerReference) []types.UID {
	var ret []types.UID
	for _, ref := range refs {
		ret = append(ret, ref.UID)
	}
	return ret
}

func (gc *GarbageCollector) processItem(item *node) error {
	glog.V(2).Infof("processing item %s", item.identity)
	// "being deleted" is an one-way trip to the final deletion. We'll just wait for the final deletion, and then process the object's dependents.
	if item.beingDeleted && !item.deletingDependents {
		glog.V(6).Infof("processing item %s returned at once", item.identity)
		return nil
	}
	// Get the latest item from the API server
	latest, err := gc.getObject(item.identity)
	if err != nil {
		if errors.IsNotFound(err) {
			// the Propagator can add "virtual" node for an owner that doesn't
			// exist yet, so we need to enqueue a virtual Delete event to remove
			// the virtual node from Propagator.uidToNode.
			glog.V(6).Infof("item %v not found, generating a virtual delete event", item.identity)
			gc.generateVirtualDeleteEvent(item.identity)
			return nil
		}
		return err
	}
	if latest.GetUID() != item.identity.UID {
		glog.V(6).Infof("UID doesn't match, item %v not found, generating a virtual delete event", item.identity)
		gc.generateVirtualDeleteEvent(item.identity)
		return nil
	}

	// TODO: orphanFinalizer() routine is similar. Consider merging orphanFinalizer() into processItem() as well.
	if item.deletingDependents {
		return gc.processDeletingDependentsItem(item)
	}

	// compute if we should delete the item
	ownerReferences := latest.GetOwnerReferences()
	if len(ownerReferences) == 0 {
		glog.V(2).Infof("object %s's doesn't have an owner, continue on next item", item.identity)
		return nil
	}

	solid, dangling, waiting, err := gc.classifyReferences(item, ownerReferences)
	if err != nil {
		return err
	}
	glog.V(6).Infof("classify references of %s.\nsolid: %#v\ndangling: %#v\nwaiting: %#v\n", item.identity, solid, dangling, waiting)

	switch {
	case len(solid) != 0:
		glog.V(2).Infof("object %s has at least one existing owner, will not garbage collect", item.identity)
		if len(dangling) != 0 || len(waiting) != 0 {
			glog.V(2).Infof("remove dangling references %#v and waiting references %#v for object %s", dangling, waiting, item.identity)
		}
		patch := deleteOwnerRefPatch(item.identity.UID, append(ownerRefsToUIDs(dangling), ownerRefsToUIDs(waiting)...)...)
		_, err = gc.patchObject(item.identity, patch)
		return err
	case len(waiting) != 0 && len(item.dependents) != 0:
		for dep := range item.dependents {
			if dep.deletingDependents {
				// this circle detection has false positives, we need to
				// apply a more rigorous detection if this turns out to be a
				// problem.
				glog.V(2).Infof("processing object %s, some of its owners and its dependent [%s] have FianlizerDeletingDependents, to prevent potential cycle, its ownerReferences are going to be modified to be non-blocking, then the object is going to be deleted with DeletePropagationForeground", item.identity, dep.identity)
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
		glog.V(2).Infof("at least one owner of object %s has FianlizerDeletingDependents, and the object itself has dependents, so it is going to be deleted with DeletePropagationForeground", item.identity)
		return gc.deleteObject(item.identity, v1.DeletePropagationForeground)
	default:
		glog.V(2).Infof("delete object %s with DeletePropagationDefault", item.identity)
		return gc.deleteObject(item.identity, v1.DeletePropagationDefault)
	}
}

// process item that's waiting for its dependents to be deleted
func (gc *GarbageCollector) processDeletingDependentsItem(item *node) error {
	blockingDependents := item.blockingDependents()
	if len(blockingDependents) == 0 {
		glog.V(2).Infof("remove DeleteDependents finalizer for item %s", item.identity)
		return gc.removeFinalizer(item, v1.FinalizerDeleteDependents)
	} else {
		for _, dep := range blockingDependents {
			if !dep.deletingDependents {
				glog.V(2).Infof("adding %s to dirtyQueue, because its owner %s is deletingDependents", dep.identity, item.identity)
				gc.dirtyQueue.Add(dep)
			}
		}
		return nil
	}
}

// dependents are copies of pointers to the owner's dependents, they don't need to be locked.
func (gc *GarbageCollector) orhpanDependents(owner objectReference, dependents []*node) error {
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
	glog.V(6).Infof("successfully updated all dependents")
	return nil
}

// orphanFinalizer dequeues a node from the orphanQueue, then finds its dependents
// based on the graph maintained by the GC, then removes it from the
// OwnerReferences of its dependents, and finally updates the owner to remove
// the "Orphan" finalizer. The node is added back into the orphanQueue if any of
// these steps fail.
func (gc *GarbageCollector) orphanFinalizer() {
	item, quit := gc.orphanQueue.Get()
	if quit {
		return
	}
	defer gc.orphanQueue.Done(item)
	owner, ok := item.(*node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("expect *node, got %#v", item))
	}
	// we don't need to lock each element, because they never get updated
	owner.dependentsLock.RLock()
	dependents := make([]*node, 0, len(owner.dependents))
	for dependent := range owner.dependents {
		dependents = append(dependents, dependent)
	}
	owner.dependentsLock.RUnlock()

	err := gc.orhpanDependents(owner.identity, dependents)
	if err != nil {
		glog.V(6).Infof("orphanDependents for %s failed with %v", owner.identity, err)
		gc.orphanQueue.AddRateLimited(item)
		return
	}
	// update the owner, remove "orphaningFinalizer" from its finalizers list
	err = gc.removeFinalizer(owner, v1.FinalizerOrphanDependents)
	if err != nil {
		glog.V(6).Infof("removeOrphanFinalizer for %s failed with %v", owner.identity, err)
		gc.orphanQueue.AddRateLimited(item)
	}
}

// *FOR TEST USE ONLY* It's not safe to call this function when the GC is still
// busy.
// GraphHasUID returns if the Propagator has a particular UID store in its
// uidToNode graph. It's useful for debugging.
func (gc *GarbageCollector) GraphHasUID(UIDs []types.UID) bool {
	for _, u := range UIDs {
		if _, ok := gc.propagator.uidToNode.Read(u); ok {
			return true
		}
	}
	return false
}
