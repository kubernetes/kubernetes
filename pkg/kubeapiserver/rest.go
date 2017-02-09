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

package kubeapiserver

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	genericapiserver "k8s.io/apiserver/pkg/server"
)

// RESTOptionsFactory is a RESTOptionsGetter for kube apiservers since they do complicated stuff
type RESTOptionsFactory struct {
	DeleteCollectionWorkers int
	EnableGarbageCollection bool
	EnableWatchCache        bool
	StorageFactory          genericapiserver.StorageFactory
}

func (f *RESTOptionsFactory) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	storageConfig, err := f.StorageFactory.NewConfig(resource)
	if err != nil {
		return generic.RESTOptions{}, fmt.Errorf("Unable to find storage destination for %v, due to %v", resource, err.Error())
	}

	ret := generic.RESTOptions{
		StorageConfig:           storageConfig,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: f.DeleteCollectionWorkers,
		EnableGarbageCollection: f.EnableGarbageCollection,
		ResourcePrefix:          f.StorageFactory.ResourcePrefix(resource),
	}
	if f.EnableWatchCache {
		ret.Decorator = genericregistry.StorageWithCacher
	}

	return ret, nil
}
