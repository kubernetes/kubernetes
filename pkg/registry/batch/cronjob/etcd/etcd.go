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

package etcd

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/batch"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/registry/batch/cronjob"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// REST implements a RESTStorage for scheduled jobs against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against CronJobs.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST) {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &batch.CronJobList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.CronJobs),
		&batch.CronJob{},
		prefix,
		cronjob.Strategy,
		newListFunc,
		cronjob.GetAttrs,
		storage.NoTriggerPublisher,
	)

	store := &genericregistry.Store{
		NewFunc: func() runtime.Object { return &batch.CronJob{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return genericregistry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return genericregistry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a scheduled job
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*batch.CronJob).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc:           cronjob.MatchCronJob,
		QualifiedResource:       batch.Resource("cronjobs"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate scheduled job creation
		CreateStrategy: cronjob.Strategy,

		// Used to validate scheduled job updates
		UpdateStrategy: cronjob.Strategy,
		DeleteStrategy: cronjob.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}

	statusStore := *store
	statusStore.UpdateStrategy = cronjob.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a resourcequota.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &batch.CronJob{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
