/*
Copyright 2024 The Kubernetes Authors.

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

package consistencydetector

import (
	"context"
	"os"
	"strconv"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

var dataConsistencyDetectionForListFromCacheEnabled = false

func init() {
	dataConsistencyDetectionForListFromCacheEnabled, _ = strconv.ParseBool(os.Getenv("KUBE_LIST_FROM_CACHE_INCONSISTENCY_DETECTOR"))
}

// CheckListFromCacheDataConsistencyIfRequested performs a data consistency check only when
// the KUBE_LIST_FROM_CACHE_INCONSISTENCY_DETECTOR environment variable was set during a binary startup
// for requests that have a high chance of being served from the watch-cache.
//
// The consistency check is meant to be enforced only in the CI, not in production.
// The check ensures that data retrieved by a list api call from the watch-cache
// is exactly the same as data received by the list api call from etcd.
//
// Note that this function will panic when data inconsistency is detected.
// This is intentional because we want to catch it in the CI.
//
// Note that this function doesn't examine the ListOptions to determine
// if the original request has hit the cache because it would be challenging
// to maintain consistency with the server-side implementation.
// For simplicity, we assume that the first request retrieved data from
// the cache (even though this might not be true for some requests)
// and issue the second call to get data from etcd for comparison.
func CheckListFromCacheDataConsistencyIfRequested[T runtime.Object](ctx context.Context, identity string, listItemsFn ListFunc[T], optionsUsedToReceiveList metav1.ListOptions, receivedList runtime.Object) {
	if !dataConsistencyDetectionForListFromCacheEnabled {
		return
	}
	checkListFromCacheDataConsistencyIfRequestedInternal(ctx, identity, listItemsFn, optionsUsedToReceiveList, receivedList)
}

func checkListFromCacheDataConsistencyIfRequestedInternal[T runtime.Object](ctx context.Context, identity string, listItemsFn ListFunc[T], optionsUsedToReceiveList metav1.ListOptions, receivedList runtime.Object) {
	receivedListMeta, err := meta.ListAccessor(receivedList)
	if err != nil {
		panic(err)
	}
	rawListItems, err := meta.ExtractListWithAlloc(receivedList)
	if err != nil {
		panic(err) // this should never happen
	}
	lastSyncedResourceVersion := receivedListMeta.GetResourceVersion()
	CheckDataConsistency(ctx, identity, lastSyncedResourceVersion, listItemsFn, optionsUsedToReceiveList, func() []runtime.Object { return rawListItems })
}
