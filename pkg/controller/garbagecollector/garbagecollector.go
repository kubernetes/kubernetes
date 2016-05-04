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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const ResourceResyncTime = 30 * time.Second

type Monitor struct {
	Store      cache.Store
	Controller *framework.Controller
}

type GarbageCollector struct {
	restMapper  meta.RESTMapper
	clientPool  dynamic.ClientPool
	dirty_queue *workqueue.Type
	monitors    []Monitor
}

func monitorFor(clientPool dynamic.ClientPool, resource unversioned.GroupVersionResource) (Monitor, error) {
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
		// TODO: It's impossible to understand every Kind in the cluster.
		nil,
		ResourceResyncTime,
		framework.ResourceEventHandlerFuncs{},
	)
	return monitor, nil
}

func NewGarbageCollector(clientPool dynamic.ClientPool, resources []unversioned.GroupVersionResource) (*GarbageCollector, error) {
	gc := &GarbageCollector{
		clientPool:  clientPool,
		dirty_queue: workqueue.New(),
		// TODO: should use a dynamic RESTMapper built from the discovery results.
		restMapper: registered.RESTMapper(),
	}
	for _, resource := range resources {
		monitor, err := monitorFor(gc.clientPool, resource)
		if err != nil {
			return nil, err
		}
		gc.monitors = append(gc.monitors, monitor)
	}
	return gc, nil
}

type itemRef struct {
	Store cache.Store
	Key   string
}

func (gc *GarbageCollector) scanner() {
	for _, monitor := range gc.monitors {
		keys := monitor.Store.ListKeys()
		// TODO: limit the size of the dirty queue
		for _, key := range keys {
			glog.V(6).Infof("add key %s to dirty queue", key)
			gc.dirty_queue.Add(itemRef{monitor.Store, key})
		}
	}
}

func (gc *GarbageCollector) worker() {
	key, quit := gc.dirty_queue.Get()
	if quit {
		return
	}
	defer gc.dirty_queue.Done(key)
	err := gc.processItem(key.(itemRef))
	if err != nil {
		glog.Errorf("Error syncing deployment %v: %v", key, err)
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

func (gc *GarbageCollector) deleteObject(typeAccessor meta.Type, accessor meta.Object) error {
	fqKind := unversioned.FromAPIVersionAndKind(typeAccessor.GetAPIVersion(), typeAccessor.GetKind())
	client, err := gc.clientPool.ClientForGroupVersion(fqKind.GroupVersion())
	resource, err := gc.apiResource(typeAccessor.GetAPIVersion(), typeAccessor.GetKind(), accessor.GetNamespace())
	if err != nil {
		return err
	}
	uid := accessor.GetUID()
	preconditions := v1.Preconditions{UID: &uid}
	deleteOptions := v1.DeleteOptions{Preconditions: &preconditions}
	return client.Resource(resource, accessor.GetNamespace()).Delete(accessor.GetName(), &deleteOptions)
}

func (gc *GarbageCollector) processItem(storeKey itemRef) error {
	obj, exists, err := storeKey.Store.GetByKey(storeKey.Key)
	if err != nil {
		return fmt.Errorf("unable to retrieve object %s from store: %v", storeKey.Key, err)
	}
	if !exists {
		return fmt.Errorf("object has been deleted %s", storeKey.Key)
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return fmt.Errorf("unable to access ObjectMeta of object %s: %v", storeKey.Key, err)
	}
	typeAccessor, err := meta.TypeAccessor(obj)
	if err != nil {
		return fmt.Errorf("unable to access TypeMeta of object %s: %v", storeKey.Key, err)
	}
	references := accessor.GetOwnerReferences()
	if len(references) == 0 {
		glog.V(6).Infof("object %s's doesn't have an owner, continue on next object", storeKey.Key)
		return nil
	}
	for i := 0; i < len(references); i++ {
		// TODO: need to verify the reference resource is supported by the system.
		fqKind := unversioned.FromAPIVersionAndKind(references[i].APIVersion, references[i].Kind)
		client, err := gc.clientPool.ClientForGroupVersion(fqKind.GroupVersion())
		if err != nil {
			return err
		}
		resource, err := gc.apiResource(references[i].APIVersion, references[i].Kind, accessor.GetNamespace())
		if err != nil {
			return err
		}
		_, err = client.Resource(resource, accessor.GetNamespace()).Get(references[i].Name)
		// TODO: need to compare the UID.
		if err == nil {
			glog.V(6).Infof("object %s has at least an existing owner, will not garbage collect", storeKey.Key)
			return nil
		} else if errors.IsNotFound(err) {
			glog.V(6).Infof("object %s's owner %s/%s, %s is not found", storeKey.Key, references[i].APIVersion, references[i].Kind, references[i].Name)
		} else {
			return err
		}
	}
	glog.V(6).Infof("none of object %s's owners exist any more, will garbage collect it", storeKey.Key)
	return gc.deleteObject(typeAccessor, accessor)
}

func (gc *GarbageCollector) Run(workers int, stopCh <-chan struct{}) {
	for _, monitor := range gc.monitors {
		go monitor.Controller.Run(stopCh)
	}
	// list
	go wait.Until(gc.scanner, ResourceResyncTime, stopCh)

	// worker
	for i := 0; i < workers; i++ {
		go wait.Until(gc.worker, 0, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down garbage collector")
	gc.dirty_queue.ShutDown()
}
