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

// Package tests contains cacher tests that run embedded etcd. This is to avoid dependency on "testing" in cacher package.
package tests

import (
	"context"
	"fmt"
	"strconv"
	"testing"
	"time"

	"google.golang.org/grpc/metadata"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestEmptyWatchEventCache(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)

	// add a few objects
	v := storage.APIObjectVersioner{}
	lastRV := uint64(0)
	for i := 0; i < 5; i++ {
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("foo-%d", i), Namespace: "test-ns"}}
		out := &example.Pod{}
		key := computePodKey(pod)
		if err := etcdStorage.Create(context.Background(), key, pod, out, 0); err != nil {
			t.Fatalf("Create failed: %v", err)
		}
		var err error
		if lastRV, err = v.ParseResourceVersion(out.ResourceVersion); err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
	}

	cacher, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Given that cacher is always initialized from the "current" version of etcd,
	// we now have a cacher with an empty cache of watch events and a resourceVersion of rv.
	// It should support establishing watches from rv and higher, but not older.

	expectedResourceExpiredError := apierrors.NewResourceExpired("").ErrStatus
	tests := []struct {
		name            string
		resourceVersion uint64
		expectedEvent   *watch.Event
	}{
		{
			name:            "RV-1",
			resourceVersion: lastRV - 1,
			expectedEvent:   &watch.Event{Type: watch.Error, Object: &expectedResourceExpiredError},
		},
		{
			name:            "RV",
			resourceVersion: lastRV,
		},
		{
			name:            "RV+1",
			resourceVersion: lastRV + 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := storage.ListOptions{
				ResourceVersion: strconv.Itoa(int(tt.resourceVersion)),
				Predicate:       storage.Everything,
			}
			watcher, err := cacher.Watch(context.Background(), "/pods/test-ns", opts)
			if err != nil {
				t.Fatalf("Failed to create watch: %v", err)
			}
			defer watcher.Stop()
			select {
			case event := <-watcher.ResultChan():
				if tt.expectedEvent == nil {
					t.Errorf("Unexpected event: type=%#v, object=%#v", event.Type, event.Object)
					break
				}
				if e, a := tt.expectedEvent.Type, event.Type; e != a {
					t.Errorf("Expected: %s, got: %s", e, a)
				}
				if e, a := tt.expectedEvent.Object, event.Object; !apiequality.Semantic.DeepDerivative(e, a) {
					t.Errorf("Expected: %#v, got: %#v", e, a)
				}
			case <-time.After(3 * time.Second):
				if tt.expectedEvent != nil {
					t.Errorf("Failed to get an event")
				}
				// watch remained established successfully
			}
		})
	}
}

func TestWatchStreamSeparation(t *testing.T) {
	tcs := []struct {
		name                         string
		separateCacheWatchRPC        bool
		useWatchCacheContextMetadata bool
		expectBookmarkOnWatchCache   bool
		expectBookmarkOnEtcd         bool
	}{
		{
			name:                       "common RPC > both get bookmarks",
			separateCacheWatchRPC:      false,
			expectBookmarkOnEtcd:       true,
			expectBookmarkOnWatchCache: true,
		},
		{
			name:                         "common RPC & watch cache context > both get bookmarks",
			separateCacheWatchRPC:        false,
			useWatchCacheContextMetadata: true,
			expectBookmarkOnEtcd:         true,
			expectBookmarkOnWatchCache:   true,
		},
		{
			name:                       "separate RPC > only etcd gets bookmarks",
			separateCacheWatchRPC:      true,
			expectBookmarkOnEtcd:       true,
			expectBookmarkOnWatchCache: false,
		},
		{
			name:                         "separate RPC & watch cache context > only watch cache gets bookmarks",
			separateCacheWatchRPC:        true,
			useWatchCacheContextMetadata: true,
			expectBookmarkOnEtcd:         false,
			expectBookmarkOnWatchCache:   true,
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SeparateCacheWatchRPC, tc.separateCacheWatchRPC)()
			_, cacher, _, terminate := testSetupWithEtcdServer(t)
			t.Cleanup(terminate)
			if err := cacher.WaitReady(context.TODO()); err != nil {
				t.Fatalf("unexpected error waiting for the cache to be ready")
			}

			waitContext, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			waitForEtcdBookmark := cacher.WaitForEtcdBookmark(waitContext)

			var out example.Pod
			err := cacher.Create(context.Background(), "foo", &example.Pod{}, &out, 0)
			if err != nil {
				t.Fatal(err)
			}
			versioner := storage.APIObjectVersioner{}
			var lastResourceVersion uint64
			lastResourceVersion, err = versioner.ObjectResourceVersion(&out)
			if err != nil {
				t.Fatal(err)
			}

			var contextMetadata metadata.MD
			if tc.useWatchCacheContextMetadata {
				contextMetadata = cacher.WatchContextMetadata()
			}
			// Wait before sending watch progress request to avoid https://github.com/etcd-io/etcd/issues/17507
			// TODO(https://github.com/etcd-io/etcd/issues/17507): Remove sleep when etcd is upgraded to version with fix.
			time.Sleep(time.Second)
			err = cacher.RequestWatchProgress(metadata.NewOutgoingContext(context.Background(), contextMetadata))
			if err != nil {
				t.Fatal(err)
			}
			// Give time for bookmark to arrive
			time.Sleep(time.Second)

			etcdWatchResourceVersion, err := waitForEtcdBookmark()
			if err != nil {
				t.Fatal(err)
			}
			gotEtcdWatchBookmark := etcdWatchResourceVersion == lastResourceVersion
			if gotEtcdWatchBookmark != tc.expectBookmarkOnEtcd {
				t.Errorf("Unexpected etcd bookmark check result, rv: %d, got: %v, want: %v", etcdWatchResourceVersion, etcdWatchResourceVersion, tc.expectBookmarkOnEtcd)
			}

			watchCacheResourceVersion := cacher.ResourceVersion()
			cacherGotBookmark := watchCacheResourceVersion == lastResourceVersion
			if cacherGotBookmark != tc.expectBookmarkOnWatchCache {
				t.Errorf("Unexpected watch cache bookmark check result, rv: %d, got: %v, want: %v", watchCacheResourceVersion, cacherGotBookmark, tc.expectBookmarkOnWatchCache)
			}
		})
	}
}
