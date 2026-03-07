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
	"context"
	"fmt"
	"strconv"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/core/event"
)

// REST implements a RESTStorage for events.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against events.
func NewREST(optsGetter generic.RESTOptionsGetter, ttl uint64) (*REST, error) {
	store := &genericregistry.Store{
		NewFunc:       func() runtime.Object { return &api.Event{} },
		NewListFunc:   func() runtime.Object { return &api.EventList{} },
		PredicateFunc: event.Matcher,
		TTLFunc: func(runtime.Object, uint64, bool) (uint64, error) {
			return ttl, nil
		},
		DefaultQualifiedResource:  api.Resource("events"),
		SingularQualifiedResource: api.Resource("event"),

		CreateStrategy: event.Strategy,
		UpdateStrategy: event.Strategy,
		DeleteStrategy: event.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: event.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}
	return &REST{store}, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"ev"}
}

// Watch overrides the default Watch implementation to validate resourceVersion
// and return a proper 410 Gone error when the resourceVersion is too old,
// ensuring consistent behavior with resources that use the watch cache.
// This fixes issue #137089 where Event watch requests with stale resourceVersion
// would return HTTP 200 with an empty response body instead of the expected 410 error.
func (r *REST) Watch(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	// If resourceVersion is specified, validate it before proceeding
	if options != nil && options.ResourceVersion != "" {
		requestRV, err := strconv.ParseUint(options.ResourceVersion, 10, 64)
		if err != nil {
			return nil, apierrors.NewBadRequest(fmt.Sprintf("invalid resource version: %s", options.ResourceVersion))
		}

		// Get current resource version from storage
		currentRV, err := r.Store.Storage.Storage.GetCurrentResourceVersion(ctx)
		if err != nil {
			// If we can't determine current RV, proceed with the watch request
			// and let the underlying storage handle any errors
			return r.Store.Watch(ctx, options)
		}

		// If requested RV is significantly older than current, return an error
		// The 1000 threshold is consistent with the watch cache's compacted window check
		// Note: This is a conservative check; the actual watch cache may allow larger windows
		if requestRV > 0 && requestRV < currentRV && (currentRV-requestRV) > 1000 {
			return newErrWatcher(apierrors.NewResourceExpired(fmt.Sprintf(
				"too old resource version: %s (%d)", options.ResourceVersion, currentRV)))
		}
	}

	return r.Store.Watch(ctx, options)
}

// errWatcher implements watch.Interface and returns a single error event
type errWatcher struct {
	result chan watch.Event
}

func newErrWatcher(err error) *errWatcher {
	// Create an error event
	errEvent := watch.Event{Type: watch.Error}
	switch err := err.(type) {
	case runtime.Object:
		errEvent.Object = err
	case *apierrors.StatusError:
		errEvent.Object = &err.ErrStatus
	default:
		errEvent.Object = &metav1.Status{
			Status:  metav1.StatusFailure,
			Message: err.Error(),
			Reason:  metav1.StatusReasonExpired,
			Code:    410,
		}
	}

	// Create a watcher with room for a single event, populate it, and close the channel
	watcher := &errWatcher{result: make(chan watch.Event, 1)}
	watcher.result <- errEvent
	close(watcher.result)

	return watcher
}

func (e *errWatcher) ResultChan() <-chan watch.Event {
	return e.result
}

func (e *errWatcher) Stop() {
}

func (e *errWatcher) RequestWatchProgress() error {
	return nil
}
