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

package storage

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/registry/batch/job"
	"k8s.io/kubernetes/pkg/registry/cachesize"
)

// JobStorage includes dummy storage for Job.
type JobStorage struct {
	Job    *REST
	Status *StatusREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) JobStorage {
	jobRest, jobStatusRest := NewREST(optsGetter)

	return JobStorage{
		Job:    jobRest,
		Status: jobStatusRest,
	}
}

// REST implements a RESTStorage for jobs against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against Jobs.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &genericregistry.Store{
		Copier:                   api.Scheme,
		NewFunc:                  func() runtime.Object { return &batch.Job{} },
		NewListFunc:              func() runtime.Object { return &batch.JobList{} },
		PredicateFunc:            job.MatchJob,
		DefaultQualifiedResource: batch.Resource("jobs"),
		WatchCacheSize:           cachesize.GetWatchCacheSizeByResource("jobs"),

		CreateStrategy: job.Strategy,
		UpdateStrategy: job.Strategy,
		DeleteStrategy: job.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: job.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = job.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of a resourcequota.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &batch.Job{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
