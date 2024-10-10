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

package generic

import (
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	flowcontrolrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
	"k8s.io/client-go/tools/cache"
)

// RESTOptions is set of resource-specific configuration options to generic registries.
type RESTOptions struct {
	StorageConfig *storagebackend.ConfigForResource
	Decorator     StorageDecorator

	EnableGarbageCollection   bool
	DeleteCollectionWorkers   int
	ResourcePrefix            string
	CountMetricPollPeriod     time.Duration
	StorageObjectCountTracker flowcontrolrequest.StorageObjectCountTracker

	// AllowUnsafeCorruptObjectDeletion, if enabled, will make an attempt to
	// perform the normal deletion flow, but if either of the below occurs:
	// a) the data (associated with the resource being deleted) retrieved
	// from the storage failed to transform properly (eg. decryption failure)
	// b) the data (associated with the resource being deleted) failed to
	// decode properly (eg. corrupt data)
	// it will disregard these errors, bypass the finalzer constraints,
	// deletion hook(s) and go ahead with the deletion flow.
	//
	// WARNING: This will break the cluster if the resource has dependencies
	// Use only if you REALLY know what you are doing.
	// WARNING: Vendor(s) will most likely consider using this option to be
	// in violation of the support of their product.
	//
	// The default value is false, and the user must opt in to enable it.
	AllowUnsafeCorruptObjectDeletion bool
}

// Implement RESTOptionsGetter so that RESTOptions can directly be used when available (i.e. tests)
func (opts RESTOptions) GetRESTOptions(schema.GroupResource, runtime.Object) (RESTOptions, error) {
	return opts, nil
}

type RESTOptionsGetter interface {
	// GetRESTOptions returns the RESTOptions for the given resource and example object.
	// The example object is used to determine the storage version for the resource.
	// If the example object is nil, the storage version will be determined by the resource's default storage version.
	GetRESTOptions(resource schema.GroupResource, example runtime.Object) (RESTOptions, error)
}

// StoreOptions is set of configuration options used to complete generic registries.
type StoreOptions struct {
	RESTOptions RESTOptionsGetter
	TriggerFunc storage.IndexerFuncs
	AttrFunc    storage.AttrFunc
	Indexers    *cache.Indexers
}
