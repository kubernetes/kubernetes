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

	"k8s.io/kubernetes/pkg/api/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// StorageClassInformer is type of SharedIndexInformer which watches and lists all storage classes.
// Interface provides constructor for informer and lister for storage classes
type StorageClassInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() cache.StorageClassLister
}

type storageClassInformer struct {
	*sharedInformerFactory
}

func (f *storageClassInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&storage.StorageClass{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return f.client.Storage().StorageClasses().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return f.client.Storage().StorageClasses().Watch(options)
			},
		},
		&storage.StorageClass{},
		f.defaultResync,
		cache.Indexers{},
	)
	f.informers[informerType] = informer

	return informer
}

func (f *storageClassInformer) Lister() cache.StorageClassLister {
	informer := f.Informer()
	return cache.NewStorageClassLister(informer.GetIndexer())
}
