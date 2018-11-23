/*
Copyright 2017 The Kubernetes Authors.

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

package graceful

import (
	"context"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/waitgroup"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
)

var _ storage.Interface = gracefulShutdownStorage{}

func StorageWithGracefulShutdown(delegatedDecorator generic.StorageDecorator) generic.StorageDecorator {
	return func(config *storagebackend.Config,
		objectType runtime.Object,
		resourcePrefix string,
		keyFunc func(obj runtime.Object) (string, error),
		newListFunc func() runtime.Object,
		getAttrsFunc storage.AttrFunc,
		trigger storage.TriggerPublisherFunc) (storage.Interface, factory.DestroyFunc) {

		storage, destroyFunc := delegatedDecorator(config, objectType, resourcePrefix, keyFunc, newListFunc, getAttrsFunc, trigger)

		gracefulStorage := gracefulShutdownStorage{
			wg:                  &waitgroup.SafeWaitGroup{},
			longRunnerCancelMap: make(map[context.Context]context.CancelFunc),
			cancelMapLock:       &sync.Mutex{},
			delegateStorage:     storage,
		}
		return gracefulStorage, func() {
			gracefulStorage.cancelMapLock.Lock()
			defer gracefulStorage.cancelMapLock.Unlock()
			for reqCtx, childCtxCancel := range gracefulStorage.longRunnerCancelMap {
				childCtxCancel()
				delete(gracefulStorage.longRunnerCancelMap, reqCtx)
			}
			destroyFunc()
		}
	}
}

type gracefulShutdownStorage struct {
	wg                  *waitgroup.SafeWaitGroup
	longRunnerCancelMap map[context.Context]context.CancelFunc
	cancelMapLock       *sync.Mutex
	delegateStorage     storage.Interface
}

func (s gracefulShutdownStorage) saveCancelFunc(ctx context.Context, cancel context.CancelFunc) error {
	s.cancelMapLock.Lock()
	defer s.cancelMapLock.Unlock()
	s.longRunnerCancelMap[ctx] = cancel
	return nil
}

func (s gracefulShutdownStorage) popAndCancel(ctx context.Context) {
	s.cancelMapLock.Lock()
	defer s.cancelMapLock.Unlock()
	defer delete(s.longRunnerCancelMap, ctx)
	cancelFunc := s.longRunnerCancelMap[ctx]
	if cancelFunc != nil {
		cancelFunc()
	}
}

func (s gracefulShutdownStorage) Versioner() storage.Versioner {
	return s.delegateStorage.Versioner()
}

func (s gracefulShutdownStorage) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	s.wg.Add(1)
	defer s.wg.Done()
	return s.delegateStorage.Create(ctx, key, obj, out, ttl)
}

func (s gracefulShutdownStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions) error {
	s.wg.Add(1)
	defer s.wg.Done()
	return s.delegateStorage.Delete(ctx, key, out, preconditions)
}

func (s gracefulShutdownStorage) Watch(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	childCtx, cancel := context.WithCancel(ctx)
	s.saveCancelFunc(ctx, cancel)
	defer s.popAndCancel(ctx)
	return s.delegateStorage.Watch(childCtx, key, resourceVersion, pred)
}

// WatchList implements storage.Interface.
func (s gracefulShutdownStorage) WatchList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	childCtx, cancel := context.WithCancel(ctx)
	s.saveCancelFunc(ctx, cancel)
	defer s.popAndCancel(ctx)
	return s.WatchList(childCtx, key, resourceVersion, pred)
}

// Get implements storage.Interface.
func (s gracefulShutdownStorage) Get(ctx context.Context, key string, resourceVersion string, objPtr runtime.Object, ignoreNotFound bool) error {
	s.wg.Add(1)
	defer s.wg.Done()
	return s.delegateStorage.Get(ctx, key, resourceVersion, objPtr, ignoreNotFound)
}

func (s gracefulShutdownStorage) GetToList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	s.wg.Add(1)
	defer s.wg.Done()
	return s.delegateStorage.GetToList(ctx, key, resourceVersion, pred, listObj)
}

func (s gracefulShutdownStorage) List(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	s.wg.Add(1)
	defer s.wg.Done()
	return s.delegateStorage.List(ctx, key, resourceVersion, pred, listObj)
}

// GuaranteedUpdate implements storage.Interface.
func (s gracefulShutdownStorage) GuaranteedUpdate(
	ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, suggestions ...runtime.Object) error {
	s.wg.Add(1)
	defer s.wg.Done()
	return s.delegateStorage.GuaranteedUpdate(ctx, key, ptrToType, ignoreNotFound, preconditions, tryUpdate, suggestions...)
}

// Count implements storage.Interface.
func (s gracefulShutdownStorage) Count(pathPrefix string) (int64, error) {
	s.wg.Add(1)
	defer s.wg.Done()
	return s.delegateStorage.Count(pathPrefix)
}
