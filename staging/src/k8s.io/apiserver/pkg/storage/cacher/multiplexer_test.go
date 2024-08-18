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

package cacher

import (
	"context"
	"errors"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestCacheMuxGetListBypass(t *testing.T) {
	type testCase struct {
		opts         storage.ListOptions
		expectBypass bool
	}
	commonTestCases := []testCase{
		{opts: storage.ListOptions{ResourceVersion: "0"}, expectBypass: false},
		{opts: storage.ListOptions{ResourceVersion: "1"}, expectBypass: false},

		{opts: storage.ListOptions{ResourceVersion: "", Predicate: storage.SelectionPredicate{Continue: "a"}}, expectBypass: true},
		{opts: storage.ListOptions{ResourceVersion: "0", Predicate: storage.SelectionPredicate{Continue: "a"}}, expectBypass: true},
		{opts: storage.ListOptions{ResourceVersion: "1", Predicate: storage.SelectionPredicate{Continue: "a"}}, expectBypass: true},

		{opts: storage.ListOptions{ResourceVersion: "0", Predicate: storage.SelectionPredicate{Limit: 500}}, expectBypass: false},
		{opts: storage.ListOptions{ResourceVersion: "1", Predicate: storage.SelectionPredicate{Limit: 500}}, expectBypass: true},

		{opts: storage.ListOptions{ResourceVersion: "", ResourceVersionMatch: metav1.ResourceVersionMatchExact}, expectBypass: true},
		{opts: storage.ListOptions{ResourceVersion: "0", ResourceVersionMatch: metav1.ResourceVersionMatchExact}, expectBypass: true},
		{opts: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchExact}, expectBypass: true},
	}

	t.Run("ConsistentListFromStorage", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, false)
		testCases := append(commonTestCases,
			testCase{opts: storage.ListOptions{ResourceVersion: ""}, expectBypass: true},
			testCase{opts: storage.ListOptions{ResourceVersion: "", Predicate: storage.SelectionPredicate{Limit: 500}}, expectBypass: true},
		)
		for _, tc := range testCases {
			testCacheMuxGetListBypass(t, tc.opts, tc.expectBypass)
		}

	})
	t.Run("ConsistentListFromCache", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, true)

		// TODO(p0lyn0mial): the following tests assume that etcdfeature.DefaultFeatureSupportChecker.Supports(storage.RequestWatchProgress)
		// evaluates to true. Otherwise the cache will be bypassed and the test will fail.
		//
		// If you were to run only TestGetListCacheBypass you would see that the test fail.
		// However in CI all test are run and there must be a test(s) that properly
		// initialize the storage layer so that the mentioned method evaluates to true
		forceRequestWatchProgressSupport(t)

		testCases := append(commonTestCases,
			testCase{opts: storage.ListOptions{ResourceVersion: ""}, expectBypass: false},
			testCase{opts: storage.ListOptions{ResourceVersion: "", Predicate: storage.SelectionPredicate{Limit: 500}}, expectBypass: false},
		)
		for _, tc := range testCases {
			testCacheMuxGetListBypass(t, tc.opts, tc.expectBypass)
		}
	})
}

func testCacheMuxGetListBypass(t *testing.T, options storage.ListOptions, expectBypass bool) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	mux := NewCacheMultiplexer(backingStorage, cacher)

	result := &example.PodList{}

	if !utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if err := mux.WaitReady(context.Background()); err != nil {
			t.Fatalf("unexpected error waiting for the cache to be ready")
		}
	}

	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.getListFn = func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
		currentResourceVersion := "42"
		switch {
		// request made by getCurrentResourceVersionFromStorage by checking Limit
		case key == cacher.resourcePrefix:
			podList := listObj.(*example.PodList)
			podList.ResourceVersion = currentResourceVersion
			return nil
		// request made by storage.GetList with revision from original request or
		// returned by getCurrentResourceVersionFromStorage
		case opts.ResourceVersion == options.ResourceVersion || opts.ResourceVersion == currentResourceVersion:
			return errDummy
		default:
			t.Fatalf("Unexpected request %+v", opts)
			return nil
		}
	}
	err = mux.GetList(context.TODO(), "pods/ns", options, result)
	if err != nil && err != errDummy {
		t.Fatalf("Unexpected error for List request with options: %v, err: %v", options, err)
	}
	gotBypass := err == errDummy
	if gotBypass != expectBypass {
		t.Errorf("Unexpected bypass result for List request with options %+v, bypass expected: %v, got: %v", options, expectBypass, gotBypass)
	}
}

func TestCacheMuxNonRecursiveGetListBypass(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, false)
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	mux := NewCacheMultiplexer(backingStorage, cacher)

	pred := storage.SelectionPredicate{
		Limit: 500,
	}
	result := &example.PodList{}

	if !utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if err := mux.WaitReady(context.Background()); err != nil {
			t.Fatalf("unexpected error waiting for the cache to be ready")
		}
	}

	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.injectError(errDummy)
	err = mux.GetList(context.TODO(), "pods/ns", storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred,
	}, result)
	if err != nil {
		t.Errorf("GetList with Limit and RV=0 should be served from cache: %v", err)
	}

	err = mux.GetList(context.TODO(), "pods/ns", storage.ListOptions{
		ResourceVersion: "",
		Predicate:       pred,
	}, result)
	if err != errDummy {
		t.Errorf("GetList with Limit without RV=0 should bypass cacher: %v", err)
	}
}

func TestCacheMuxGetBypass(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	mux := NewCacheMultiplexer(backingStorage, cacher)

	result := &example.Pod{}

	if !utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if err := mux.WaitReady(context.Background()); err != nil {
			t.Fatalf("unexpected error waiting for the cache to be ready")
		}
	}

	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.injectError(errDummy)
	err = mux.Get(context.TODO(), "pods/ns/pod-0", storage.GetOptions{
		IgnoreNotFound:  true,
		ResourceVersion: "0",
	}, result)
	if err != nil {
		t.Errorf("Get with RV=0 should be served from cache: %v", err)
	}

	err = mux.Get(context.TODO(), "pods/ns/pod-0", storage.GetOptions{
		IgnoreNotFound:  true,
		ResourceVersion: "",
	}, result)
	if err != errDummy {
		t.Errorf("Get without RV=0 should bypass cacher: %v", err)
	}
}

func TestCacheMuxWatchBypass(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	mux := NewCacheMultiplexer(backingStorage, cacher)

	if !utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if err := mux.WaitReady(context.Background()); err != nil {
			t.Fatalf("unexpected error waiting for the cache to be ready")
		}
	}

	_, err = mux.Watch(context.TODO(), "pod/ns", storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       storage.Everything,
	})
	if err != nil {
		t.Errorf("Watch with RV=0 should be served from cache: %v", err)
	}

	_, err = mux.Watch(context.TODO(), "pod/ns", storage.ListOptions{
		ResourceVersion: "",
		Predicate:       storage.Everything,
	})
	if err != nil {
		t.Errorf("Watch with RV=0 should be served from cache: %v", err)
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchFromStorageWithoutResourceVersion, false)
	_, err = mux.Watch(context.TODO(), "pod/ns", storage.ListOptions{
		ResourceVersion: "",
		Predicate:       storage.Everything,
	})
	if err != nil {
		t.Errorf("With WatchFromStorageWithoutResourceVersion disabled, watch with unset RV should be served from cache: %v", err)
	}

	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.injectError(errDummy)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchFromStorageWithoutResourceVersion, true)
	_, err = mux.Watch(context.TODO(), "pod/ns", storage.ListOptions{
		ResourceVersion: "",
		Predicate:       storage.Everything,
	})
	if !errors.Is(err, errDummy) {
		t.Errorf("With WatchFromStorageWithoutResourceVersion enabled, watch with unset RV should be served from storage: %v", err)
	}
}
