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
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	coreinternallisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
)

// PodInformer is type of SharedIndexInformer which watches and lists all pods.
// Interface provides constructor for informer and lister for pods
type PodInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() *cache.StoreToPodLister
}

type podInformer struct {
	*sharedInformerFactory
}

// Informer checks whether podInformer exists in sharedInformerFactory and if not, it creates new informer of type
// podInformer and connects it to sharedInformerFactory
func (f *podInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.Pod{})
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
	Informer() cache.SharedIndexInformer
	Lister() *cache.IndexerToNamespaceLister
}

type namespaceInformer struct {
	*sharedInformerFactory
}

// Informer checks whether namespaceInformer exists in sharedInformerFactory and if not, it creates new informer of type
// namespaceInformer and connects it to sharedInformerFactory
func (f *namespaceInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.Namespace{})
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

// InternalNamespaceInformer is type of SharedIndexInformer which watches and lists all namespaces.
// Interface provides constructor for informer and lister for namsespaces
type InternalNamespaceInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() coreinternallisters.NamespaceLister
}

type internalNamespaceInformer struct {
	*sharedInformerFactory
}

// Informer checks whether internalNamespaceInformer exists in sharedInformerFactory and if not, it creates new informer of type
// internalNamespaceInformer and connects it to sharedInformerFactory
func (f *internalNamespaceInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&api.Namespace{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewInternalNamespaceInformer(f.internalclient, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for internalNamespaceInformer
func (f *internalNamespaceInformer) Lister() coreinternallisters.NamespaceLister {
	informer := f.Informer()
	return coreinternallisters.NewNamespaceLister(informer.GetIndexer())
}

//*****************************************************************************

// NodeInformer is type of SharedIndexInformer which watches and lists all nodes.
// Interface provides constructor for informer and lister for nodes
type NodeInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() *cache.StoreToNodeLister
}

type nodeInformer struct {
	*sharedInformerFactory
}

// Informer checks whether nodeInformer exists in sharedInformerFactory and if not, it creates new informer of type
// nodeInformer and connects it to sharedInformerFactory
func (f *nodeInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.Node{})
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
	Informer() cache.SharedIndexInformer
	Lister() *cache.StoreToPersistentVolumeClaimLister
}

type pvcInformer struct {
	*sharedInformerFactory
}

// Informer checks whether pvcInformer exists in sharedInformerFactory and if not, it creates new informer of type
// pvcInformer and connects it to sharedInformerFactory
func (f *pvcInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.PersistentVolumeClaim{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewPVCInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for pvcInformer
func (f *pvcInformer) Lister() *cache.StoreToPersistentVolumeClaimLister {
	informer := f.Informer()
	return &cache.StoreToPersistentVolumeClaimLister{Indexer: informer.GetIndexer()}
}

//*****************************************************************************

// PVInformer is type of SharedIndexInformer which watches and lists all persistent volumes.
// Interface provides constructor for informer and lister for persistent volumes
type PVInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() *cache.StoreToPVFetcher
}

type pvInformer struct {
	*sharedInformerFactory
}

// Informer checks whether pvInformer exists in sharedInformerFactory and if not, it creates new informer of type
// pvInformer and connects it to sharedInformerFactory
func (f *pvInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.PersistentVolume{})
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

//*****************************************************************************

// LimitRangeInformer is type of SharedIndexInformer which watches and lists all limit ranges.
// Interface provides constructor for informer and lister for limit ranges.
type LimitRangeInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() *cache.StoreToLimitRangeLister
}

type limitRangeInformer struct {
	*sharedInformerFactory
}

// Informer checks whether pvcInformer exists in sharedInformerFactory and if not, it creates new informer of type
// limitRangeInformer and connects it to sharedInformerFactory
func (f *limitRangeInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.LimitRange{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewLimitRangeInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for limitRangeInformer
func (f *limitRangeInformer) Lister() *cache.StoreToLimitRangeLister {
	informer := f.Informer()
	return &cache.StoreToLimitRangeLister{Indexer: informer.GetIndexer()}
}

//*****************************************************************************

// InternalLimitRangeInformer is type of SharedIndexInformer which watches and lists all limit ranges.
// Interface provides constructor for informer and lister for limit ranges.
type InternalLimitRangeInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() coreinternallisters.LimitRangeLister
}

type internalLimitRangeInformer struct {
	*sharedInformerFactory
}

// Informer checks whether pvcInformer exists in sharedInformerFactory and if not, it creates new informer of type
// internalLimitRangeInformer and connects it to sharedInformerFactory
func (f *internalLimitRangeInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&api.LimitRange{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewInternalLimitRangeInformer(f.internalclient, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for internalLimitRangeInformer
func (f *internalLimitRangeInformer) Lister() coreinternallisters.LimitRangeLister {
	informer := f.Informer()
	return coreinternallisters.NewLimitRangeLister(informer.GetIndexer())
}

//*****************************************************************************

// ReplicationControllerInformer is type of SharedIndexInformer which watches and lists all replication controllers.
// Interface provides constructor for informer and lister for replication controllers.
type ReplicationControllerInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() *cache.StoreToReplicationControllerLister
}

type replicationControllerInformer struct {
	*sharedInformerFactory
}

// Informer checks whether replicationControllerInformer exists in sharedInformerFactory and if not, it creates new informer of type
// replicationControllerInformer and connects it to sharedInformerFactory
func (f *replicationControllerInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.ReplicationController{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewReplicationControllerInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for replicationControllerInformer
func (f *replicationControllerInformer) Lister() *cache.StoreToReplicationControllerLister {
	informer := f.Informer()
	return &cache.StoreToReplicationControllerLister{Indexer: informer.GetIndexer()}
}

//*****************************************************************************

// NewPodInformer returns a SharedIndexInformer that lists and watches all pods
func NewPodInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().Pods(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().Pods(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.Pod{},
		resyncPeriod,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	return sharedIndexInformer
}

// NewNodeInformer returns a SharedIndexInformer that lists and watches all nodes
func NewNodeInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().Nodes().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().Nodes().Watch(options)
			},
		},
		&v1.Node{},
		resyncPeriod,
		cache.Indexers{})

	return sharedIndexInformer
}

// NewPVCInformer returns a SharedIndexInformer that lists and watches all PVCs
func NewPVCInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().PersistentVolumeClaims(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().PersistentVolumeClaims(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.PersistentVolumeClaim{},
		resyncPeriod,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	return sharedIndexInformer
}

// NewPVInformer returns a SharedIndexInformer that lists and watches all PVs
func NewPVInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().PersistentVolumes().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().PersistentVolumes().Watch(options)
			},
		},
		&v1.PersistentVolume{},
		resyncPeriod,
		cache.Indexers{})

	return sharedIndexInformer
}

// NewNamespaceInformer returns a SharedIndexInformer that lists and watches namespaces
func NewNamespaceInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().Namespaces().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().Namespaces().Watch(options)
			},
		},
		&v1.Namespace{},
		resyncPeriod,
		cache.Indexers{})

	return sharedIndexInformer
}

// NewInternalNamespaceInformer returns a SharedIndexInformer that lists and watches namespaces
func NewInternalNamespaceInformer(client internalclientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				internalOptions := api.ListOptions{}
				v1.Convert_v1_ListOptions_To_api_ListOptions(&options, &internalOptions, nil)
				return client.Core().Namespaces().List(internalOptions)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				internalOptions := api.ListOptions{}
				v1.Convert_v1_ListOptions_To_api_ListOptions(&options, &internalOptions, nil)
				return client.Core().Namespaces().Watch(internalOptions)
			},
		},
		&api.Namespace{},
		resyncPeriod,
		cache.Indexers{})

	return sharedIndexInformer
}

// NewLimitRangeInformer returns a SharedIndexInformer that lists and watches all LimitRanges
func NewLimitRangeInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().LimitRanges(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().LimitRanges(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.LimitRange{},
		resyncPeriod,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})

	return sharedIndexInformer
}

// NewInternalLimitRangeInformer returns a SharedIndexInformer that lists and watches all LimitRanges
func NewInternalLimitRangeInformer(internalclient internalclientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				internalOptions := api.ListOptions{}
				v1.Convert_v1_ListOptions_To_api_ListOptions(&options, &internalOptions, nil)
				return internalclient.Core().LimitRanges(v1.NamespaceAll).List(internalOptions)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				internalOptions := api.ListOptions{}
				v1.Convert_v1_ListOptions_To_api_ListOptions(&options, &internalOptions, nil)
				return internalclient.Core().LimitRanges(v1.NamespaceAll).Watch(internalOptions)
			},
		},
		&api.LimitRange{},
		resyncPeriod,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})

	return sharedIndexInformer
}

// NewReplicationControllerInformer returns a SharedIndexInformer that lists and watches all replication controllers.
func NewReplicationControllerInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().ReplicationControllers(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().ReplicationControllers(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.ReplicationController{},
		resyncPeriod,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	return sharedIndexInformer
}

/*****************************************************************************/

// ServiceAccountInformer is type of SharedIndexInformer which watches and lists all ServiceAccounts.
// Interface provides constructor for informer and lister for ServiceAccounts
type ServiceAccountInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() *cache.StoreToServiceAccountLister
}

type serviceAccountInformer struct {
	*sharedInformerFactory
}

// Informer checks whether ServiceAccountInformer exists in sharedInformerFactory and if not, it creates new informer of type
// ServiceAccountInformer and connects it to sharedInformerFactory
func (f *serviceAccountInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&v1.ServiceAccount{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewServiceAccountInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// Lister returns lister for ServiceAccountInformer
func (f *serviceAccountInformer) Lister() *cache.StoreToServiceAccountLister {
	informer := f.Informer()
	return &cache.StoreToServiceAccountLister{Indexer: informer.GetIndexer()}
}

// NewServiceAccountInformer returns a SharedIndexInformer that lists and watches all ServiceAccounts
func NewServiceAccountInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return client.Core().ServiceAccounts(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().ServiceAccounts(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.ServiceAccount{},
		resyncPeriod,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})

	return sharedIndexInformer
}
