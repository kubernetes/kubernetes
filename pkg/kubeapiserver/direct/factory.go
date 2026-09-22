/*
Copyright The Kubernetes Authors.

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

package direct

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core"
	corev1informers "k8s.io/client-go/informers/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

var podResource = corev1.Resource("pods")

// NewSharedInformerFactory wraps a SharedInformerFactory so that calling Lister()
// returns a Lister that reads directly from the watch cache (RV=0) without
// registering or starting a client-go informer.
func NewSharedInformerFactory(original informers.SharedInformerFactory) informers.SharedInformerFactory {
	return &directSharedInformerFactory{
		SharedInformerFactory: original,
		storages:              map[schema.GroupResource]Storage{},
	}
}

// SetStorage binds the REST storage that listers for gr read from. Listers are handed out
// before the storage layer is built, so they resolve it lazily on first read.
func SetStorage(f informers.SharedInformerFactory, gr schema.GroupResource, storage Storage) {
	if d, ok := f.(*directSharedInformerFactory); ok {
		d.storages[gr] = storage
	}
}

type directSharedInformerFactory struct {
	informers.SharedInformerFactory

	storages map[schema.GroupResource]Storage
}

func (f *directSharedInformerFactory) storageFor(gr schema.GroupResource) Storage {
	return f.storages[gr]
}

func (f *directSharedInformerFactory) clientFor(gvr schema.GroupVersionResource) Client {
	return &storageClient{
		factory: f,
		gvr:     gvr,
	}
}

func (f *directSharedInformerFactory) Core() coreinformers.Interface {
	return &directCoreInformers{
		Interface: f.SharedInformerFactory.Core(),
		factory:   f,
	}
}

func (f *directSharedInformerFactory) ForResource(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
	switch resource.GroupResource() {
	case podResource:
		return &directInformer{
			original: f.SharedInformerFactory,
			gvr:      resource,
			lister:   NewLister[runtime.Object](f.clientFor(resource)),
		}, nil
	default:
		return f.SharedInformerFactory.ForResource(resource)
	}
}

type directInformer struct {
	original informers.SharedInformerFactory
	gvr      schema.GroupVersionResource
	lister   cache.GenericLister
}

func (g *directInformer) Informer() cache.SharedIndexInformer {
	inf, err := g.original.ForResource(g.gvr)
	if err != nil {
		return nil
	}
	return inf.Informer()
}

func (g *directInformer) Lister() cache.GenericLister {
	return g.lister
}

type directCoreInformers struct {
	coreinformers.Interface
	factory *directSharedInformerFactory
}

func (c *directCoreInformers) V1() corev1informers.Interface {
	return &directCoreV1Informers{
		Interface: c.Interface.V1(),
		factory:   c.factory,
	}
}

type directCoreV1Informers struct {
	corev1informers.Interface
	factory *directSharedInformerFactory
}

func (v *directCoreV1Informers) Pods() corev1informers.TypedPodInformer {
	return &directPodInformer{
		TypedPodInformer: v.Interface.Pods(),
		lister:           NewPodLister(v.factory.clientFor(corev1.SchemeGroupVersion.WithResource("pods"))),
	}
}

type directPodInformer struct {
	corev1informers.TypedPodInformer
	lister corev1listers.PodLister
}

var _ corev1informers.TypedPodInformer = &directPodInformer{}

func (d *directPodInformer) Lister() corev1listers.PodLister {
	return d.lister
}
