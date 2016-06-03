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
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
)

// Creates a cacher on top of the given 'storageInterface'.
func StorageWithCacher(
	storageInterface storage.Interface,
	capacity int,
	objectType runtime.Object,
	resourcePrefix string,
	scopeStrategy rest.NamespaceScopedStrategy,
	newListFunc func() runtime.Object) storage.Interface {

	config := storage.CacherConfig{
		CacheCapacity:  capacity,
		Storage:        storageInterface,
		Versioner:      etcdstorage.APIObjectVersioner{},
		Type:           objectType,
		ResourcePrefix: resourcePrefix,
		NewListFunc:    newListFunc,
	}
	if scopeStrategy.NamespaceScoped() {
		config.KeyFunc = func(obj runtime.Object) (string, error) {
			return storage.NamespaceKeyFunc(resourcePrefix, obj)
		}
	} else {
		config.KeyFunc = func(obj runtime.Object) (string, error) {
			return storage.NoNamespaceKeyFunc(resourcePrefix, obj)
		}
	}

	return storage.NewCacherFromConfig(config)
}
