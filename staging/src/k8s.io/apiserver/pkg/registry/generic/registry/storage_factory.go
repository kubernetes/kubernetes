/*
Copyright 2015 The Kubernetes Authors.

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

package registry

import (
	"fmt"
	"sync"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	cacherstorage "k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/client-go/tools/cache"
)

// Creates a cacher based given storageConfig.
func StorageWithCacher() generic.StorageDecorator {
	return func(
		storageConfig *storagebackend.ConfigForResource,
		resourcePrefix string,
		keyFunc func(obj runtime.Object) (string, error),
		newFunc func() runtime.Object,
		newListFunc func() runtime.Object,
		getAttrsFunc storage.AttrFunc,
		triggerFuncs storage.IndexerFuncs,
		indexers *cache.Indexers) (storage.Interface, factory.DestroyFunc, error) {

		s, d, err := generic.NewRawStorage(storageConfig, newFunc)
		if err != nil {
			return s, d, err
		}
		if klogV := klog.V(5); klogV.Enabled() {
			//nolint:logcheck // It complains about the key/value pairs because it cannot check them.
			klogV.InfoS("Storage caching is enabled", objectTypeToArgs(newFunc())...)
		}

		cacherConfig := cacherstorage.Config{
			Storage:        s,
			Versioner:      etcd3.APIObjectVersioner{},
			ResourcePrefix: resourcePrefix,
			KeyFunc:        keyFunc,
			NewFunc:        newFunc,
			NewListFunc:    newListFunc,
			GetAttrsFunc:   getAttrsFunc,
			IndexerFuncs:   triggerFuncs,
			Indexers:       indexers,
			Codec:          storageConfig.Codec,
		}
		cacher, err := cacherstorage.NewCacherFromConfig(cacherConfig)
		if err != nil {
			return nil, func() {}, err
		}
		destroyFunc := func() {
			cacher.Stop()
			d()
		}

		// TODO : Remove RegisterStorageCleanup below when PR
		// https://github.com/kubernetes/kubernetes/pull/50690
		// merges as that shuts down storage properly
		RegisterStorageCleanup(destroyFunc)

		return cacher, destroyFunc, nil
	}
}

func objectTypeToArgs(obj runtime.Object) []interface{} {
	// special-case unstructured objects that tell us their apiVersion/kind
	if u, isUnstructured := obj.(*unstructured.Unstructured); isUnstructured {
		if apiVersion, kind := u.GetAPIVersion(), u.GetKind(); len(apiVersion) > 0 && len(kind) > 0 {
			return []interface{}{"apiVersion", apiVersion, "kind", kind}
		}
	}

	// otherwise just return the type
	return []interface{}{"type", fmt.Sprintf("%T", obj)}
}

// TODO : Remove all the code below when PR
// https://github.com/kubernetes/kubernetes/pull/50690
// merges as that shuts down storage properly
// HACK ALERT : Track the destroy methods to call them
// from the test harness. TrackStorageCleanup will be called
// only from the test harness, so Register/Cleanup will be
// no-op at runtime.

var cleanupLock sync.Mutex
var cleanup []func() = nil

func TrackStorageCleanup() {
	cleanupLock.Lock()
	defer cleanupLock.Unlock()

	if cleanup != nil {
		panic("Conflicting storage tracking")
	}
	cleanup = make([]func(), 0)
}

func RegisterStorageCleanup(fn func()) {
	cleanupLock.Lock()
	defer cleanupLock.Unlock()

	if cleanup == nil {
		return
	}
	cleanup = append(cleanup, fn)
}

func CleanupStorage() {
	cleanupLock.Lock()
	old := cleanup
	cleanup = nil
	cleanupLock.Unlock()

	for _, d := range old {
		d()
	}
}
