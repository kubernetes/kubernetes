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
	"strings"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
)

// errFrozen is a transient error to indicate that clients should retry with backoff.
var errFrozen = errors.NewServiceUnavailable("TPR data is temporarily frozen")

// REST implements a RESTStorage for ThirdPartyResourceData.
type REST struct {
	*genericregistry.Store
	kind   string
	frozen atomic.Value
}

// Freeze causes all future calls to Create/Update/Delete/DeleteCollection to return a transient error.
// This is irreversible and meant for use when the TPR data is being deleted or migrated/abandoned.
func (r *REST) Freeze() {
	r.frozen.Store(true)
}

func (r *REST) isFrozen() bool {
	return r.frozen.Load() != nil
}

// Create is a wrapper to support Freeze.
func (r *REST) Create(ctx genericapirequest.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	if r.isFrozen() {
		return nil, errFrozen
	}
	return r.Store.Create(ctx, obj, includeUninitialized)
}

// Update is a wrapper to support Freeze.
func (r *REST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	if r.isFrozen() {
		return nil, false, errFrozen
	}
	return r.Store.Update(ctx, name, objInfo)
}

// Delete is a wrapper to support Freeze.
func (r *REST) Delete(ctx genericapirequest.Context, name string, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	if r.isFrozen() {
		return nil, false, errFrozen
	}
	return r.Store.Delete(ctx, name, options)
}

// DeleteCollection is a wrapper to support Freeze.
func (r *REST) DeleteCollection(ctx genericapirequest.Context, options *metav1.DeleteOptions, listOptions *metainternalversion.ListOptions) (runtime.Object, error) {
	if r.isFrozen() {
		return nil, errFrozen
	}
	return r.Store.DeleteCollection(ctx, options, listOptions)
}

// NewREST returns a registry which will store ThirdPartyResourceData in the given helper
func NewREST(optsGetter generic.RESTOptionsGetter, group, kind string) *REST {
	resource := extensions.Resource("thirdpartyresourcedatas")
	opts, err := optsGetter.GetRESTOptions(resource)
	if err != nil {
		panic(err) // TODO: Propagate error up
	}

	// We explicitly do NOT do any decoration here yet.
	opts.Decorator = generic.UndecoratedStorage // TODO use watchCacheSize=-1 to signal UndecoratedStorage
	opts.ResourcePrefix = "/ThirdPartyResourceData/" + group + "/" + strings.ToLower(kind) + "s"

	store := &genericregistry.Store{
		Copier:            api.Scheme,
		NewFunc:           func() runtime.Object { return &extensions.ThirdPartyResourceData{} },
		NewListFunc:       func() runtime.Object { return &extensions.ThirdPartyResourceDataList{} },
		PredicateFunc:     thirdpartyresourcedata.Matcher,
		QualifiedResource: resource,
		WatchCacheSize:    cachesize.GetWatchCacheSizeByResource(resource.Resource),

		CreateStrategy: thirdpartyresourcedata.Strategy,
		UpdateStrategy: thirdpartyresourcedata.Strategy,
		DeleteStrategy: thirdpartyresourcedata.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: opts, AttrFunc: thirdpartyresourcedata.GetAttrs} // Pass in opts to use UndecoratedStorage and custom ResourcePrefix
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	return &REST{
		Store: store,
		kind:  kind,
	}
}

// Implements the rest.KindProvider interface
func (r *REST) Kind() string {
	return r.kind
}
