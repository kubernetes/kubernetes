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

package robustness

import (
	appsv1informers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	appsv1listers "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

type wrappedSharedIndexInformer struct {
	cache.SharedIndexInformer
	indexer cache.Indexer
}

func (w *wrappedSharedIndexInformer) GetIndexer() cache.Indexer { return w.indexer }
func (w *wrappedSharedIndexInformer) GetStore() cache.Store     { return w.indexer }

// WrapSharedIndexInformer decorates a cache.SharedIndexInformer to route its indexer lookups through the fault registry.
func (f *RobustnessTestFixture) WrapSharedIndexInformer(realInformer cache.SharedIndexInformer, name string) cache.SharedIndexInformer {
	return &wrappedSharedIndexInformer{
		SharedIndexInformer: realInformer,
		indexer:             f.WrapIndexer(realInformer.GetIndexer(), name),
	}
}

// wrappedInformer implements any typed client-go Informer interface (Informer() + Lister()).
type wrappedInformer[L any] struct {
	inf    cache.SharedIndexInformer
	lister L
}

func (w *wrappedInformer[L]) Informer() cache.SharedIndexInformer { return w.inf }
func (w *wrappedInformer[L]) Lister() L                           { return w.lister }

// WrapInformer wraps any typed client-go informer so both its Informer() and Lister()
// read through a fault-injecting indexer named by resource (e.g. "pods", "replicasets").
func WrapInformer[L any](f *RobustnessTestFixture, realInformer cache.SharedIndexInformer, resource string, newLister func(cache.Indexer) L) *wrappedInformer[L] {
	inf := f.WrapSharedIndexInformer(realInformer, resource)
	return &wrappedInformer[L]{
		inf:    inf,
		lister: newLister(inf.GetIndexer()),
	}
}

// WrapPodInformer decorates a PodInformer with a fault-injecting indexer named "pods".
func (f *RobustnessTestFixture) WrapPodInformer(realInformer coreinformers.PodInformer) coreinformers.PodInformer {
	return WrapInformer(f, realInformer.Informer(), "pods", corelisters.NewPodLister)
}

// WrapNodeInformer decorates a NodeInformer with a fault-injecting indexer named "nodes".
func (f *RobustnessTestFixture) WrapNodeInformer(realInformer coreinformers.NodeInformer) coreinformers.NodeInformer {
	return WrapInformer(f, realInformer.Informer(), "nodes", corelisters.NewNodeLister)
}

// WrapDaemonSetInformer decorates a DaemonSetInformer with a fault-injecting indexer named "daemonsets".
func (f *RobustnessTestFixture) WrapDaemonSetInformer(realInformer appsv1informers.DaemonSetInformer) appsv1informers.DaemonSetInformer {
	return WrapInformer(f, realInformer.Informer(), "daemonsets", appsv1listers.NewDaemonSetLister)
}
