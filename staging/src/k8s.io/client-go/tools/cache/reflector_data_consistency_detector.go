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

package cache

import (
	"context"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

var dataConsistencyDetectionEnabled = false

func init() {
	dataConsistencyDetectionEnabled, _ = strconv.ParseBool(os.Getenv("KUBE_WATCHLIST_INCONSISTENCY_DETECTOR"))
}

// checkWatchListConsistencyIfRequested performs a data consistency check only when
// the KUBE_WATCHLIST_INCONSISTENCY_DETECTOR environment variable was set during a binary startup.
//
// The consistency check is meant to be enforced only in the CI, not in production.
// The check ensures that data retrieved by the watch-list api call
// is exactly the same as data received by the standard list api call.
//
// Note that this function will panic when data inconsistency is detected.
// This is intentional because we want to catch it in the CI.
func checkWatchListConsistencyIfRequested(stopCh <-chan struct{}, identity string, lastSyncedResourceVersion string, listerWatcher Lister, store Store) {
	if !dataConsistencyDetectionEnabled {
		return
	}
	checkWatchListConsistency(stopCh, identity, lastSyncedResourceVersion, listerWatcher, store)
}

// checkWatchListConsistency exists solely for testing purposes.
// we cannot use checkWatchListConsistencyIfRequested because
// it is guarded by an environmental variable.
// we cannot manipulate the environmental variable because
// it will affect other tests in this package.
func checkWatchListConsistency(stopCh <-chan struct{}, identity string, lastSyncedResourceVersion string, listerWatcher Lister, store Store) {
	klog.Warningf("%s: data consistency check for the watch-list feature is enabled, this will result in an additional call to the API server.", identity)
	opts := metav1.ListOptions{
		ResourceVersion:      lastSyncedResourceVersion,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
	}
	var list runtime.Object
	err := wait.PollUntilContextCancel(wait.ContextForChannel(stopCh), time.Second, true, func(_ context.Context) (done bool, err error) {
		list, err = listerWatcher.List(opts)
		if err != nil {
			// the consistency check will only be enabled in the CI
			// and LIST calls in general will be retired by the client-go library
			// if we fail simply log and retry
			klog.Errorf("failed to list data from the server, retrying until stopCh is closed, err: %v", err)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		klog.Errorf("failed to list data from the server, the watch-list consistency check won't be performed, stopCh was closed, err: %v", err)
		return
	}

	rawListItems, err := meta.ExtractListWithAlloc(list)
	if err != nil {
		panic(err) // this should never happen
	}

	listItems := toMetaObjectSliceOrDie(rawListItems)
	storeItems := toMetaObjectSliceOrDie(store.List())

	sort.Sort(byUID(listItems))
	sort.Sort(byUID(storeItems))

	if !cmp.Equal(listItems, storeItems) {
		klog.Infof("%s: data received by the new watch-list api call is different than received by the standard list api call, diff: %v", identity, cmp.Diff(listItems, storeItems))
		msg := "data inconsistency detected for the watch-list feature, panicking!"
		panic(msg)
	}
}

type byUID []metav1.Object

func (a byUID) Len() int           { return len(a) }
func (a byUID) Less(i, j int) bool { return a[i].GetUID() < a[j].GetUID() }
func (a byUID) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func toMetaObjectSliceOrDie[T any](s []T) []metav1.Object {
	result := make([]metav1.Object, len(s))
	for i, v := range s {
		m, err := meta.Accessor(v)
		if err != nil {
			panic(err)
		}
		result[i] = m
	}
	return result
}
