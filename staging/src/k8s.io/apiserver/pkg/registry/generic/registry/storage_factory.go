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

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	cacherstorage "k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/cacher/watchgroup"
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

		s, d, etcdClient, err := factory.CreateWithEtcdClient(*storageConfig, newFunc, newListFunc, resourcePrefix)
		if err != nil {
			return s, d, err
		}
		if klogV := klog.V(5); klogV.Enabled() {
			//nolint:logcheck // It complains about the key/value pairs because it cannot check them.
			klogV.InfoS("Storage caching is enabled", objectTypeToArgs(newFunc())...)
		}

		// Create a per-resource WatchGroupManager. Each resource type gets its own
		// Manager with its own EtcdGroupStore, ensuring membership isolation.
		var wgManager *watchgroup.Manager
		if etcdClient != nil {
			if rawClient, ok := etcdClient.Client().(*clientv3.Client); ok {
				etcdStore := watchgroup.NewEtcdGroupStore(rawClient, "/watchgroups/", storageConfig.GroupResource)
				wgManager = watchgroup.NewManager(etcdStore, nil)
			}
		}

		cacherConfig := cacherstorage.Config{
			Storage:             s,
			Versioner:           storage.APIObjectVersioner{},
			GroupResource:       storageConfig.GroupResource,
			EventsHistoryWindow: storageConfig.EventsHistoryWindow,
			ResourcePrefix:      resourcePrefix,
			KeyFunc:             keyFunc,
			NewFunc:             newFunc,
			NewListFunc:         newListFunc,
			GetAttrsFunc:        getAttrsFunc,
			IndexerFuncs:        triggerFuncs,
			Indexers:            indexers,
			Codec:               storageConfig.Codec,
			WatchGroupManager:   wgManager,
		}
		cacher, err := cacherstorage.NewCacherFromConfig(cacherConfig)
		if err != nil {
			return nil, func() {}, err
		}

		// Wire up the rebalance callback now that we have the cacher
		if wgManager != nil {
			wgManager.SetOnRebalance(cacher.HandleRebalance)
		}

		delegator := cacherstorage.NewCacheDelegator(cacher, s)
		var once sync.Once
		destroyFunc := func() {
			once.Do(func() {
				delegator.Stop()
				cacher.Stop()
				d()
			})
		}

		return delegator, destroyFunc, nil
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
