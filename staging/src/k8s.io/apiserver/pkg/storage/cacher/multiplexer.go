/*
Copyright 2024 The Kubernetes Authors.

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

package cacher

import (
	"context"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
)

func NewCacheMultiplexer(store storage.Interface, cache *Cacher) CacheMux {
	return CacheMux{
		store: store,
		cache: cache,
	}
}

type CacheMux struct {
	store storage.Interface
	cache *Cacher
}

var _ storage.Interface = (*CacheMux)(nil)

func (m CacheMux) Versioner() storage.Versioner {
	return m.store.Versioner()
}

func (m CacheMux) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	return m.store.Create(ctx, key, obj, out, ttl)
}

func (m CacheMux) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object) error {
	// Ignore the suggestion and try to pass down the current version of the object
	// read from cache.
	if elem, exists, err := m.cache.watchCache.GetByKey(key); err != nil {
		klog.Errorf("GetByKey returned error: %v", err)
	} else if exists {
		// DeepCopy the object since we modify resource version when serializing the
		// current object.
		currObj := elem.(*storeElement).Object.DeepCopyObject()
		return m.store.Delete(ctx, key, out, preconditions, validateDeletion, currObj)
	}
	// If we couldn't get the object, fallback to no-suggestion.
	return m.store.Delete(ctx, key, out, preconditions, validateDeletion, nil)
}

func (m CacheMux) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	// if the watch-list feature wasn't set and the resourceVersion is unset
	// ensure that the rv from which the watch is being served, is the latest
	// one. "latest" is ensured by serving the watch from
	// the underlying storage.
	//
	// it should never happen due to our validation but let's just be super-safe here
	// and disable sendingInitialEvents when the feature wasn't enabled
	if !utilfeature.DefaultFeatureGate.Enabled(features.WatchList) && opts.SendInitialEvents != nil {
		opts.SendInitialEvents = nil
	}
	// TODO: we should eventually get rid of this legacy case
	if utilfeature.DefaultFeatureGate.Enabled(features.WatchFromStorageWithoutResourceVersion) && opts.SendInitialEvents == nil && opts.ResourceVersion == "" {
		return m.store.Watch(ctx, key, opts)
	}
	return m.cache.Watch(ctx, key, opts)
}

func (m CacheMux) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	if opts.ResourceVersion == "" {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility).
		return m.store.Get(ctx, key, opts, objPtr)
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if !m.cache.ready.check() {
			// If Cache is not initialized, delegate Get requests to storage
			// as described in https://kep.k8s.io/4568
			return m.store.Get(ctx, key, opts, objPtr)
		}
	}
	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.
	if !utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if (opts.ResourceVersion == "" || opts.ResourceVersion == "0") && !m.cache.ready.check() {
			// If Cacher is not yet initialized and we don't require any specific
			// minimal resource version, simply forward the request to storage.
			return m.store.Get(ctx, key, opts, objPtr)
		}
		if err := m.cache.ready.wait(ctx); err != nil {
			return errors.NewServiceUnavailable(err.Error())
		}
	}
	return m.cache.Get(ctx, key, opts, objPtr)
}

func (m CacheMux) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	if shouldDelegateList(opts) {
		return m.store.GetList(ctx, key, opts, listObj)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if !m.cache.ready.check() && shouldDelegateListOnNotReadyCache(opts) {
			// If Cacher is not initialized, delegate List requests to storage
			// as described in https://kep.k8s.io/4568
			return m.store.GetList(ctx, key, opts, listObj)
		}
	} else {
		if (opts.ResourceVersion == "" || opts.ResourceVersion == "0") && !m.cache.ready.check() {
			// If Cacher is not yet initialized and we don't require any specific
			// minimal resource version, simply forward the request to storage.
			return m.store.GetList(ctx, key, opts, listObj)
		}
	}
	return m.cache.GetList(ctx, key, opts, listObj)
}

func (m CacheMux) GuaranteedUpdate(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	// Ignore the suggestion and try to pass down the current version of the object
	// read from cache.
	if elem, exists, err := m.cache.watchCache.GetByKey(key); err != nil {
		klog.Errorf("GetByKey returned error: %v", err)
	} else if exists {
		// DeepCopy the object since we modify resource version when serializing the
		// current object.
		currObj := elem.(*storeElement).Object.DeepCopyObject()
		return m.store.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, currObj)
	}
	// If we couldn't get the object, fallback to no-suggestion.
	return m.store.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, nil)
}

func (m CacheMux) Count(key string) (int64, error) {
	return m.store.Count(key)
}

func (m CacheMux) ReadinessCheck() error {
	return m.cache.ReadinessCheck()
}

func (m CacheMux) WaitReady(ctx context.Context) error {
	return m.cache.ready.wait(ctx)
}

func (m CacheMux) RequestWatchProgress(ctx context.Context) error {
	return m.store.RequestWatchProgress(ctx)
}
