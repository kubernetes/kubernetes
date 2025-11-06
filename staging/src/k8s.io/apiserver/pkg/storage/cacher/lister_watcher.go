/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"

	"google.golang.org/grpc/metadata"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/consistencydetector"
	"k8s.io/client-go/util/watchlist"
)

// listerWatcher opaques storage.Interface to expose cache.ListerWatcher.
type listerWatcher struct {
	storage         storage.Interface
	resourcePrefix  string
	newListFunc     func() runtime.Object
	contextMetadata metadata.MD

	unsupportedWatchListSemantics    bool
	watchListConsistencyCheckEnabled bool
}

// NewListerWatcher returns a storage.Interface backed ListerWatcher.
func NewListerWatcher(storage storage.Interface, resourcePrefix string, newListFunc func() runtime.Object, contextMetadata metadata.MD) cache.ListerWatcher {
	return &listerWatcher{
		storage:                          storage,
		resourcePrefix:                   resourcePrefix,
		newListFunc:                      newListFunc,
		contextMetadata:                  contextMetadata,
		unsupportedWatchListSemantics:    watchlist.DoesClientNotSupportWatchListSemantics(storage),
		watchListConsistencyCheckEnabled: consistencydetector.IsDataConsistencyDetectionForWatchListEnabled(),
	}
}

// Implements cache.ListerWatcher interface.
func (lw *listerWatcher) List(options metav1.ListOptions) (runtime.Object, error) {
	list := lw.newListFunc()
	pred := storage.SelectionPredicate{
		Label:    labels.Everything(),
		Field:    fields.Everything(),
		Limit:    options.Limit,
		Continue: options.Continue,
	}

	storageOpts := storage.ListOptions{
		ResourceVersionMatch: options.ResourceVersionMatch,
		Predicate:            pred,
		Recursive:            true,
	}

	// The ConsistencyChecker built into reflectors for the WatchList feature is responsible
	// for verifying that the data received from the server (potentially from the watch cache)
	// is consistent with the data stored in etcd.
	//
	// To perform this verification, the checker uses the ResourceVersion obtained from the initial request
	// and sets the ResourceVersionMatch so that it retrieves exactly the same data directly from etcd.
	// This allows comparing both data sources and confirming their consistency.
	//
	// The code below checks whether the incoming request originates from the ConsistencyChecker.
	// If so, it allows explicitly setting the ResourceVersion.
	//
	// As of Oct 2025, reflector on its own is not setting RVM=Exact.
	// However, even if that changes in the meantime, we would have to propagate that
	// down to storage to ensure the correct semantics of the request.
	watchListEnabled := utilfeature.DefaultFeatureGate.Enabled(features.WatchList)
	supportedRVM := options.ResourceVersionMatch == metav1.ResourceVersionMatchExact
	if watchListEnabled && lw.watchListConsistencyCheckEnabled && supportedRVM {
		storageOpts.ResourceVersion = options.ResourceVersion
	}

	ctx := context.Background()
	if lw.contextMetadata != nil {
		ctx = metadata.NewOutgoingContext(ctx, lw.contextMetadata)
	}
	if err := lw.storage.GetList(ctx, lw.resourcePrefix, storageOpts, list); err != nil {
		return nil, err
	}
	return list, nil
}

// Implements cache.ListerWatcher interface.
func (lw *listerWatcher) Watch(options metav1.ListOptions) (watch.Interface, error) {
	pred := storage.Everything
	pred.AllowWatchBookmarks = options.AllowWatchBookmarks
	opts := storage.ListOptions{
		ResourceVersion:   options.ResourceVersion,
		Predicate:         pred,
		Recursive:         true,
		ProgressNotify:    true,
		SendInitialEvents: options.SendInitialEvents,
	}
	ctx := context.Background()
	if lw.contextMetadata != nil {
		ctx = metadata.NewOutgoingContext(ctx, lw.contextMetadata)
	}

	// we need the below check because the listWatcher bypasses the REST layer,
	// so the options are not validated. Without this, we might end up in a situation
	// where streaming is requested, but the FeatureGate is disabled,
	// and the bookmark will not be sent
	//
	// in such a case, client-go is going to fall back to a standard LIST on any error
	// returned for watch-list requests
	if isListWatchRequest(opts) && !utilfeature.DefaultFeatureGate.Enabled(features.WatchList) {
		return nil, fmt.Errorf("sendInitialEvents is forbidden for watch unless the WatchList feature gate is enabled")
	}

	return lw.storage.Watch(ctx, lw.resourcePrefix, opts)
}

func (lw *listerWatcher) IsWatchListSemanticsUnSupported() bool {
	return lw.unsupportedWatchListSemantics
}
