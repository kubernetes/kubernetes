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

package informers

import (
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
)

// PodInformer is type of SharedIndexInformer which watches and lists all pods.
// Interface provides constructor for informer and lister for pods
type PodInformer interface {
	Informer() framework.SharedIndexInformer
	Lister() *cache.StoreToPodLister
}

type podInformer struct {
	*sharedInformerFactory
}

// Informer checks whether podInformer exists in sharedInformerFactory and if not, it creates new informer of type
// podInformer and connects it to sharedInformerFactory
func (f *podInformer) Informer() framework.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&api.Pod{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewPodInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for podInformer
func (f *podInformer) Lister() *cache.StoreToPodLister {
	informer := f.Informer()
	return &cache.StoreToPodLister{Indexer: informer.GetIndexer()}
}

//*****************************************************************************

// NamespaceInformer is type of SharedIndexInformer which watches and lists all namespaces.
// Interface provides constructor for informer and lister for namsespaces
type NamespaceInformer interface {
	Informer() framework.SharedIndexInformer
	Lister() *cache.IndexerToNamespaceLister
}

type namespaceInformer struct {
	*sharedInformerFactory
}

// Informer checks whether namespaceInformer exists in sharedInformerFactory and if not, it creates new informer of type
// namespaceInformer and connects it to sharedInformerFactory
func (f *namespaceInformer) Informer() framework.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&api.Namespace{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewNamespaceInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for namespaceInformer
func (f *namespaceInformer) Lister() *cache.IndexerToNamespaceLister {
	informer := f.Informer()
	return &cache.IndexerToNamespaceLister{Indexer: informer.GetIndexer()}
}

//*****************************************************************************

// NodeInformer is type of SharedIndexInformer which watches and lists all nodes.
// Interface provides constructor for informer and lister for nodes
type NodeInformer interface {
	Informer() framework.SharedIndexInformer
	Lister() *cache.StoreToNodeLister
}

type nodeInformer struct {
	*sharedInformerFactory
}

// Informer checks whether nodeInformer exists in sharedInformerFactory and if not, it creates new informer of type
// nodeInformer and connects it to sharedInformerFactory
func (f *nodeInformer) Informer() framework.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&api.Node{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewNodeInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for nodeInformer
func (f *nodeInformer) Lister() *cache.StoreToNodeLister {
	informer := f.Informer()
	return &cache.StoreToNodeLister{Store: informer.GetStore()}
}

//*****************************************************************************

// PVCInformer is type of SharedIndexInformer which watches and lists all persistent volume claims.
// Interface provides constructor for informer and lister for persistent volume claims
type PVCInformer interface {
	Informer() framework.SharedIndexInformer
	Lister() *cache.StoreToPVCFetcher
}

type pvcInformer struct {
	*sharedInformerFactory
}

// Informer checks whether pvcInformer exists in sharedInformerFactory and if not, it creates new informer of type
// pvcInformer and connects it to sharedInformerFactory
func (f *pvcInformer) Informer() framework.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&api.PersistentVolumeClaim{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewPVCInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for pvcInformer
func (f *pvcInformer) Lister() *cache.StoreToPVCFetcher {
	informer := f.Informer()
	return &cache.StoreToPVCFetcher{Store: informer.GetStore()}
}

//*****************************************************************************

// PVInformer is type of SharedIndexInformer which watches and lists all persistent volumes.
// Interface provides constructor for informer and lister for persistent volumes
type PVInformer interface {
	Informer() framework.SharedIndexInformer
	Lister() *cache.StoreToPVFetcher
}

type pvInformer struct {
	*sharedInformerFactory
}

// Informer checks whether pvInformer exists in sharedInformerFactory and if not, it creates new informer of type
// pvInformer and connects it to sharedInformerFactory
func (f *pvInformer) Informer() framework.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&api.PersistentVolume{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewPVInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for pvInformer
func (f *pvInformer) Lister() *cache.StoreToPVFetcher {
	informer := f.Informer()
	return &cache.StoreToPVFetcher{Store: informer.GetStore()}
}
