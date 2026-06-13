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
	"sync"

	appsv1informers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	appsv1listers "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

// wrappedSharedIndexInformer overrides GetIndexer() and GetStore() to return our wrapped indexer.
type wrappedSharedIndexInformer struct {
	cache.SharedIndexInformer
	fixture *RobustnessTestFixture
	name    string

	mu             sync.Mutex
	wrappedIndexer cache.Indexer
}

func (w *wrappedSharedIndexInformer) GetIndexer() cache.Indexer {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.wrappedIndexer == nil {
		w.wrappedIndexer = w.fixture.WrapIndexer(w.SharedIndexInformer.GetIndexer(), w.name)
	}
	return w.wrappedIndexer
}

func (w *wrappedSharedIndexInformer) GetStore() cache.Store {
	return w.GetIndexer()
}

// WrapPodInformer decorates a standard PodInformer to automatically use our fault-injecting indexer cache.
func (f *RobustnessTestFixture) WrapPodInformer(realInformer coreinformers.PodInformer) coreinformers.PodInformer {
	return &wrappedPodInformer{
		PodInformer: realInformer,
		inf: &wrappedSharedIndexInformer{
			SharedIndexInformer: realInformer.Informer(),
			fixture:             f,
			name:                "pod-cache",
		},
		fixture: f,
	}
}

type wrappedPodInformer struct {
	coreinformers.PodInformer
	inf     cache.SharedIndexInformer
	fixture *RobustnessTestFixture
}

func (w *wrappedPodInformer) Informer() cache.SharedIndexInformer {
	return w.inf
}

func (w *wrappedPodInformer) Lister() corelisters.PodLister {
	return corelisters.NewPodLister(w.inf.GetIndexer())
}

// WrapNodeInformer decorates a standard NodeInformer to automatically use our fault-injecting indexer cache.
func (f *RobustnessTestFixture) WrapNodeInformer(realInformer coreinformers.NodeInformer) coreinformers.NodeInformer {
	return &wrappedNodeInformer{
		NodeInformer: realInformer,
		inf: &wrappedSharedIndexInformer{
			SharedIndexInformer: realInformer.Informer(),
			fixture:             f,
			name:                "node-cache",
		},
		fixture: f,
	}
}

type wrappedNodeInformer struct {
	coreinformers.NodeInformer
	inf     cache.SharedIndexInformer
	fixture *RobustnessTestFixture
}

func (w *wrappedNodeInformer) Informer() cache.SharedIndexInformer {
	return w.inf
}

func (w *wrappedNodeInformer) Lister() corelisters.NodeLister {
	return corelisters.NewNodeLister(w.inf.GetIndexer())
}

// WrapDaemonSetInformer decorates a standard DaemonSetInformer to automatically use our fault-injecting indexer cache.
func (f *RobustnessTestFixture) WrapDaemonSetInformer(realInformer appsv1informers.DaemonSetInformer) appsv1informers.DaemonSetInformer {
	return &wrappedDSInformer{
		DaemonSetInformer: realInformer,
		inf: &wrappedSharedIndexInformer{
			SharedIndexInformer: realInformer.Informer(),
			fixture:             f,
			name:                "daemonset-cache",
		},
		fixture: f,
	}
}

type wrappedDSInformer struct {
	appsv1informers.DaemonSetInformer
	inf     cache.SharedIndexInformer
	fixture *RobustnessTestFixture
}

func (w *wrappedDSInformer) Informer() cache.SharedIndexInformer {
	return w.inf
}

func (w *wrappedDSInformer) Lister() appsv1listers.DaemonSetLister {
	return appsv1listers.NewDaemonSetLister(w.inf.GetIndexer())
}
