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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/watchlist"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/ptr"

	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
)

func TestDoesClientSupportWatchListSemanticsForKubeClient(t *testing.T) {
	target1 := &cachertesting.MockStorage{}
	if !watchlist.DoesClientNotSupportWatchListSemantics(target1) {
		t.Fatalf("cachertesting.MockStorage should NOT support WatchList semantics")
	}

	server, target2 := newEtcdTestStorage(t, "/pods/")
	defer server.Terminate(t)

	if watchlist.DoesClientNotSupportWatchListSemantics(target2) {
		t.Fatalf("etcd should support WatchList semantics")
	}
}

func TestCacherListerWatcher(t *testing.T) {
	prefix := "/pods/"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	objects := []*example.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}},
	}
	for _, obj := range objects {
		out := &example.Pod{}
		key := computePodKey(obj)
		if err := store.Create(context.Background(), key, obj, out, 0); err != nil {
			t.Fatalf("Create failed: %v", err)
		}
	}

	lw := NewListerWatcher(store, prefix, fn, nil)

	obj, err := lw.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	pl, ok := obj.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", pl)
	}
	if len(pl.Items) != 3 {
		t.Errorf("Expected PodList of length 3 but got %d", len(pl.Items))
	}
}

func TestCacherListerWatcherPagination(t *testing.T) {
	prefix := "/pods/"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	// We need the list to be sorted by name to later check the alphabetical order of
	// returned results.
	objects := []*example.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}},
	}
	for _, obj := range objects {
		out := &example.Pod{}
		key := computePodKey(obj)
		if err := store.Create(context.Background(), key, obj, out, 0); err != nil {
			t.Fatalf("Create failed: %v", err)
		}
	}

	lw := NewListerWatcher(store, prefix, fn, nil)

	obj1, err := lw.List(metav1.ListOptions{Limit: 2})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	limit1, ok := obj1.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", limit1)
	}
	if len(limit1.Items) != 2 {
		t.Errorf("Expected PodList of length 2 but got %d", len(limit1.Items))
	}
	if limit1.Continue == "" {
		t.Errorf("Expected list to have Continue but got none")
	}
	obj2, err := lw.List(metav1.ListOptions{Limit: 2, Continue: limit1.Continue})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	limit2, ok := obj2.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", limit2)
	}
	if limit2.Continue != "" {
		t.Errorf("Expected list not to have Continue, but got %s", limit1.Continue)
	}

	if limit1.Items[0].Name != objects[0].Name {
		t.Errorf("Expected list1.Items[0] to be %s but got %s", objects[0].Name, limit1.Items[0].Name)
	}
	if limit1.Items[1].Name != objects[1].Name {
		t.Errorf("Expected list1.Items[1] to be %s but got %s", objects[1].Name, limit1.Items[1].Name)
	}
	if limit2.Items[0].Name != objects[2].Name {
		t.Errorf("Expected list2.Items[0] to be %s but got %s", objects[2].Name, limit2.Items[0].Name)
	}

}

func TestCacherListerWatcherListWatch(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)

	prefix := "/pods/"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	makePodFn := func() *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"},
		}
	}
	ctx := context.TODO()
	pod := makePodFn()
	key := computePodKey(pod)
	createdPod := &example.Pod{}
	if err := store.Create(ctx, key, makePodFn(), createdPod, 0); err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	lw := NewListerWatcher(store, prefix, fn, nil)
	target := cache.ToListerWatcherWithContext(lw)
	watchListOptions := metav1.ListOptions{
		Watch:               true,
		AllowWatchBookmarks: true,
		SendInitialEvents:   ptr.To(true),
	}
	w, err := target.WatchWithContext(ctx, watchListOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Stop()

	expectedWatchEvents := []watch.Event{
		{Type: watch.Added, Object: createdPod},
		{
			Type: watch.Bookmark,
			Object: &example.Pod{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: createdPod.ResourceVersion,
					Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
				},
			},
		},
	}

	storagetesting.TestCheckResultsInStrictOrder(t, w, expectedWatchEvents)
	storagetesting.TestCheckNoMoreResultsWithIgnoreFunc(t, w, nil)
}

func TestCacherListerWatcherWhenListWatchDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, false)

	prefix := "/pods/"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	lw := NewListerWatcher(store, prefix, fn, nil)
	target := cache.ToListerWatcherWithContext(lw)
	watchListOptions := metav1.ListOptions{
		Watch:               true,
		AllowWatchBookmarks: true,
		SendInitialEvents:   ptr.To(true),
	}
	_, err := target.WatchWithContext(context.TODO(), watchListOptions)
	if err == nil {
		t.Fatalf("Expected error, but got none")
	}

	expectedErrMsg := "sendInitialEvents is forbidden for watch unless the WatchList feature gate is enabled"
	if err.Error() != expectedErrMsg {
		t.Fatalf("Expected error %q, but got %q", expectedErrMsg, err.Error())
	}
}

func TestListerWatcherListResourceVersionPropagation(t *testing.T) {
	scenarios := []struct {
		name                             string
		options                          metav1.ListOptions
		watchListEnabled                 bool
		watchListConsistencyCheckEnabled bool
		expectedStorageResourceVer       string
	}{
		{
			name: "WatchList FG disabled - RV not propagated",
			options: metav1.ListOptions{
				ResourceVersion:      "123",
				ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			},
			watchListEnabled:                 false,
			watchListConsistencyCheckEnabled: true,
			expectedStorageResourceVer:       "",
		},
		{
			name: "WatchList consistency check disabled - RV not propagated",
			options: metav1.ListOptions{
				ResourceVersion:      "123",
				ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			},
			watchListEnabled:                 true,
			watchListConsistencyCheckEnabled: false,
			expectedStorageResourceVer:       "",
		},
		{
			name: "Unsupported RVM - RV not propagated",
			options: metav1.ListOptions{
				ResourceVersion:      "123",
				ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
			},
			watchListEnabled:                 true,
			watchListConsistencyCheckEnabled: true,
			expectedStorageResourceVer:       "",
		},
		{
			name: "all conditions satisfied - RV propagated",
			options: metav1.ListOptions{
				ResourceVersion:      "123",
				ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			},
			watchListEnabled:                 true,
			watchListConsistencyCheckEnabled: true,
			expectedStorageResourceVer:       "123",
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, scenario.watchListEnabled)

			backingStorage := &cachertesting.MockStorage{}
			var capturedOpts storage.ListOptions
			backingStorage.GetListFn = func(ctx context.Context, resPrefix string, opts storage.ListOptions, listObj runtime.Object) error {
				capturedOpts = opts
				return nil
			}

			targetInterface := NewListerWatcher(backingStorage, "/pods/", func() runtime.Object { return &example.PodList{} }, nil)
			target := targetInterface.(*listerWatcher)
			target.watchListConsistencyCheckEnabled = scenario.watchListConsistencyCheckEnabled

			if _, err := target.List(scenario.options); err != nil {
				t.Fatalf("List returned error: %v", err)
			}

			if capturedOpts.ResourceVersion != scenario.expectedStorageResourceVer {
				t.Fatalf("expected storage ResourceVersion %q, got %q", scenario.expectedStorageResourceVer, capturedOpts.ResourceVersion)
			}
		})
	}
}
