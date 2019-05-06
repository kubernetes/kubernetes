/*
Copyright 2018 The Kubernetes Authors.

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

package dynamicinformer

import (
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamiclister"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
)

// NewDynamicSharedInformerFactory constructs a new instance of dynamicSharedInformerFactory for all namespaces.
func NewDynamicSharedInformerFactory(client dynamic.Interface, defaultResync time.Duration) DynamicSharedInformerFactory {
	return NewFilteredDynamicSharedInformerFactory(client, defaultResync, metav1.NamespaceAll, nil)
}

// NewFilteredDynamicSharedInformerFactory constructs a new instance of dynamicSharedInformerFactory.
// Listers obtained via this factory will be subject to the same filters as specified here.
func NewFilteredDynamicSharedInformerFactory(client dynamic.Interface, defaultResync time.Duration, namespace string, tweakListOptions TweakListOptionsFunc) DynamicSharedInformerFactory {
	return &dynamicSharedInformerFactory{
		client:           client,
		defaultResync:    defaultResync,
		namespace:        metav1.NamespaceAll,
		informers:        map[schema.GroupVersionResource]informers.GenericInformer{},
		startedInformers: make(map[schema.GroupVersionResource]bool),
		tweakListOptions: tweakListOptions,
	}
}

type dynamicSharedInformerFactory struct {
	client        dynamic.Interface
	defaultResync time.Duration
	namespace     string

	lock      sync.Mutex
	informers map[schema.GroupVersionResource]informers.GenericInformer
	// startedInformers is used for tracking which informers have been started.
	// This allows Start() to be called multiple times safely.
	startedInformers map[schema.GroupVersionResource]bool
	tweakListOptions TweakListOptionsFunc
}

var _ DynamicSharedInformerFactory = &dynamicSharedInformerFactory{}

func (f *dynamicSharedInformerFactory) ForResource(gvr schema.GroupVersionResource) informers.GenericInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	key := gvr
	informer, exists := f.informers[key]
	if exists {
		return informer
	}

	informer = NewFilteredDynamicInformer(f, f.client, gvr, f.namespace, f.defaultResync, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, f.tweakListOptions)
	f.informers[key] = informer
	f.startedInformers[key] = false

	return informer
}

func (f *dynamicSharedInformerFactory) removeResource(gvr schema.GroupVersionResource, informer cache.SharedIndexInformer) {
	f.lock.Lock()
	defer f.lock.Unlock()

	key := gvr
	registeredInformer, exists := f.informers[key]
	if exists && registeredInformer.Informer() == informer {
		delete(f.informers, key)
		f.startedInformers[key] = false
	}
}

// Start initializes all requested informers.
func (f *dynamicSharedInformerFactory) Start(stopCh <-chan struct{}) {
	f.lock.Lock()
	defer f.lock.Unlock()

	for informerType, informer := range f.informers {
		if !f.startedInformers[informerType] {
			go informer.Informer().Run(stopCh)
			f.startedInformers[informerType] = true
		}
	}
}

// WaitForCacheSync waits for all started informers' cache were synced.
func (f *dynamicSharedInformerFactory) WaitForCacheSync(stopCh <-chan struct{}) map[schema.GroupVersionResource]bool {
	informers := func() map[schema.GroupVersionResource]cache.SharedIndexInformer {
		f.lock.Lock()
		defer f.lock.Unlock()

		informers := map[schema.GroupVersionResource]cache.SharedIndexInformer{}
		for informerType, informer := range f.informers {
			if f.startedInformers[informerType] {
				informers[informerType] = informer.Informer()
			}
		}
		return informers
	}()

	res := map[schema.GroupVersionResource]bool{}
	for informType, informer := range informers {
		res[informType] = cache.WaitForCacheSync(stopCh, informer.HasSynced)
	}
	return res
}

// NewFilteredDynamicInformer constructs a new informer for a dynamic type.
func NewFilteredDynamicInformer(factory *dynamicSharedInformerFactory, client dynamic.Interface, gvr schema.GroupVersionResource, namespace string, resyncPeriod time.Duration, indexers cache.Indexers, tweakListOptions TweakListOptionsFunc) informers.GenericInformer {
	// make local stopCh for dynamic shared informer
	stopCh := make(chan struct{})
	return &dynamicInformer{
		SharedIndexInformer: cache.NewSharedIndexInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					if tweakListOptions != nil {
						tweakListOptions(&options)
					}
					return client.Resource(gvr).Namespace(namespace).List(options)
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					if tweakListOptions != nil {
						tweakListOptions(&options)
					}
					return client.Resource(gvr).Namespace(namespace).Watch(options)
				},
			},
			&unstructured.Unstructured{},
			resyncPeriod,
			indexers,
		),
		gvr:     gvr,
		stopCh:  stopCh,
		factory: factory,
	}
}

type dynamicInformer struct {
	cache.SharedIndexInformer
	gvr    schema.GroupVersionResource
	stopCh chan struct{}
	// once is for safely close stopCh
	once    sync.Once
	factory *dynamicSharedInformerFactory
}

var _ informers.GenericInformer = &dynamicInformer{}

func (d *dynamicInformer) Informer() cache.SharedIndexInformer {
	return d
}

func (d *dynamicInformer) Lister() cache.GenericLister {
	return dynamiclister.NewRuntimeObjectShim(dynamiclister.New(d.GetIndexer(), d.gvr))
}

var _ cache.SharedIndexInformer = &dynamicInformer{}

func (d *dynamicInformer) GetController() cache.Controller {
	return &dummyController{informer: d}
}

// Use local stopCh, the passed in stopCh is ignored.
// The informer is stopped via its controller
func (d *dynamicInformer) Run(stopCh <-chan struct{}) {
	d.SharedIndexInformer.Run(d.stopCh)
}

func (d *dynamicInformer) stop() {
	d.once.Do(func() {
		close(d.stopCh)
	})
}

type dummyController struct {
	informer *dynamicInformer
}

func (v *dummyController) Run(stopCh <-chan struct{}) {
	<-stopCh
	// stop informer
	v.informer.stop()
	// remove it from factory
	v.informer.factory.removeResource(v.informer.gvr, v.informer)
}

func (v *dummyController) HasSynced() bool {
	return v.informer.HasSynced()
}

func (v *dummyController) LastSyncResourceVersion() string {
	return ""
}
