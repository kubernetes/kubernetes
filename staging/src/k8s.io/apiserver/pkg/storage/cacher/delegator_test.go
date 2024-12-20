/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

func TestConsistencyCheckerDigest(t *testing.T) {
	newListFunc := func() runtime.Object { return &example.PodList{} }
	testCases := []struct {
		desc            string
		resourceVersion string
		cacherReady     bool
		cacherItems     []example.Pod
		etcdItems       []example.Pod

		expectListKey    string
		expectDigest     storageDigest
		expectErr        bool
		expectConsistent bool
	}{
		{
			desc:             "not ready",
			cacherReady:      false,
			resourceVersion:  "1",
			expectErr:        true,
			expectConsistent: true,
		},
		{
			desc:            "empty",
			resourceVersion: "1",
			cacherReady:     true,
			expectDigest: storageDigest{
				ResourceVersion: "1",
				CacheDigest:     "cbf29ce484222325",
				EtcdDigest:      "cbf29ce484222325",
			},
			expectConsistent: true,
		},
		{
			desc:            "with one element equal",
			resourceVersion: "2",
			cacherReady:     true,
			cacherItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod", ResourceVersion: "2"}},
			},
			etcdItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod", ResourceVersion: "2"}},
			},
			expectDigest: storageDigest{
				ResourceVersion: "2",
				CacheDigest:     "86bf3a5e80d1c5cb",
				EtcdDigest:      "86bf3a5e80d1c5cb",
			},
			expectConsistent: true,
		},
		{
			desc:            "namespace changes digest",
			resourceVersion: "2",
			cacherReady:     true,
			cacherItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "kube-system", Name: "pod", ResourceVersion: "2"}},
			},
			etcdItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "kube-public", Name: "pod", ResourceVersion: "2"}},
			},
			expectDigest: storageDigest{
				ResourceVersion: "2",
				CacheDigest:     "4ae4e750bd825b17",
				EtcdDigest:      "f940a60af965b03",
			},
			expectConsistent: false,
		},
		{
			desc:            "name changes digest",
			resourceVersion: "2",
			cacherReady:     true,
			cacherItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod2", ResourceVersion: "2"}},
			},
			etcdItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod3", ResourceVersion: "2"}},
			},
			expectDigest: storageDigest{
				ResourceVersion: "2",
				CacheDigest:     "c9120494e4c1897d",
				EtcdDigest:      "c9156494e4c46274",
			},
			expectConsistent: false,
		},
		{
			desc:            "resourceVersion changes digest",
			resourceVersion: "4",
			cacherReady:     true,
			cacherItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod", ResourceVersion: "3"}},
			},
			etcdItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod", ResourceVersion: "4"}},
			},
			expectDigest: storageDigest{
				ResourceVersion: "4",
				CacheDigest:     "86bf3a5e80d1c5ca",
				EtcdDigest:      "86bf3a5e80d1c5cd",
			},
			expectConsistent: false,
		},
		{
			desc:            "watch missed write event",
			resourceVersion: "3",
			cacherReady:     true,
			cacherItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "Default", Name: "pod", ResourceVersion: "2"}},
			},
			etcdItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "Default", Name: "pod", ResourceVersion: "2"}},
				{ObjectMeta: metav1.ObjectMeta{Namespace: "Default", Name: "pod", ResourceVersion: "3"}},
			},
			expectDigest: storageDigest{
				ResourceVersion: "3",
				CacheDigest:     "1859bac707c2cb2b",
				EtcdDigest:      "11d147fc800df0e0",
			},
			expectConsistent: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			etcd := &dummyStorage{
				getListFn: func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
					if key != tc.expectListKey {
						t.Fatalf("Expect GetList key %q, got %q", tc.expectListKey, key)
					}
					if opts.ResourceVersion != tc.resourceVersion {
						t.Fatalf("Expect GetList resourceVersion %q, got %q", tc.resourceVersion, opts.ResourceVersion)
					}
					if opts.ResourceVersionMatch != metav1.ResourceVersionMatchExact {
						t.Fatalf("Expect GetList match exact, got %q", opts.ResourceVersionMatch)
					}
					podList := listObj.(*example.PodList)
					podList.Items = tc.etcdItems
					podList.ResourceVersion = tc.resourceVersion
					return nil
				},
			}
			cacher := &dummyCacher{
				ready:      tc.cacherReady,
				consistent: true,
				dummyStorage: dummyStorage{
					getListFn: func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
						if key != tc.expectListKey {
							t.Fatalf("Expect GetList key %q, got %q", tc.expectListKey, key)
						}
						if opts.ResourceVersion != "0" {
							t.Fatalf("Expect GetList resourceVersion 0, got %q", opts.ResourceVersion)
						}
						if opts.ResourceVersionMatch != metav1.ResourceVersionMatchNotOlderThan {
							t.Fatalf("Expect GetList match not older than, got %q", opts.ResourceVersionMatch)
						}
						podList := listObj.(*example.PodList)
						podList.Items = tc.cacherItems
						podList.ResourceVersion = tc.resourceVersion
						return nil
					},
				},
			}
			checker := newConsistencyChecker("", schema.GroupResource{}, newListFunc, cacher, etcd)
			digest, err := checker.calculateDigests(context.Background())
			if (err != nil) != tc.expectErr {
				t.Fatalf("Expect error: %v, got: %v", tc.expectErr, err)
			}
			if err != nil {
				return
			}
			if *digest != tc.expectDigest {
				t.Errorf("Expect: %+v Got: %+v", &tc.expectDigest, *digest)
			}

			checker.check(context.Background())
			if cacher.consistent != tc.expectConsistent {
				t.Errorf("Expect: %+v Got: %+v", tc.expectConsistent, cacher.consistent)
			}
		})
	}
}

func TestConsistencyCheckerListOpts(t *testing.T) {
	newListFunc := func() runtime.Object { return &example.PodList{} }

	resourceVersion := "50"
	etcdOpts := []storage.ListOptions{}
	etcd := &dummyStorage{
		getListFn: func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
			etcdOpts = append(etcdOpts, opts)
			podList := listObj.(*example.PodList)
			podList.ResourceVersion = resourceVersion
			return nil
		},
	}
	cacherOpts := []storage.ListOptions{}
	cacher := &dummyCacher{
		ready: true,
		dummyStorage: dummyStorage{
			getListFn: func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
				cacherOpts = append(cacherOpts, opts)
				podList := listObj.(*example.PodList)
				podList.ResourceVersion = resourceVersion
				return nil
			},
		},
	}
	checker := newConsistencyChecker("", schema.GroupResource{}, newListFunc, cacher, etcd)
	_, err := checker.calculateDigests(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	wantCacherOpts := []storage.ListOptions{
		{
			ResourceVersion:      "0",
			ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
			Predicate:            storage.Everything,
			Recursive:            true,
		},
	}
	if diff := cmp.Diff(cacherOpts, wantCacherOpts); diff != "" {
		t.Errorf("unexpected list opts (-want +got):\n%s", diff)
	}
	wantEtcdOpts := []storage.ListOptions{
		{
			ResourceVersion:      resourceVersion,
			ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			Predicate: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: storageWatchListPageSize,
			},
			Recursive: true,
		},
	}
	if diff := cmp.Diff(etcdOpts, wantEtcdOpts); diff != "" {
		t.Errorf("unexpected list opts (-want +got):\n%s", diff)
	}
}

func TestConsistencyCheckerDigestMatches(t *testing.T) {
	ctx, store, terminate := testSetup(t)
	t.Cleanup(terminate)

	var out example.Pod
	resourceVersion := ""
	t.Logf("Create %d pods to ensure pagination", storageWatchListPageSize+1)
	for i := 0; i < int(storageWatchListPageSize)+1; i++ {
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: fmt.Sprintf("%d", i)}}
		err := store.Create(ctx, computePodKey(pod), pod, &out, 0)
		if err != nil {
			t.Fatal(err)
		}
		resourceVersion = out.ResourceVersion
	}

	t.Log("Execute list to ensure cache is up to date")
	outList := &example.PodList{}
	err := store.cacher.GetList(ctx, "/pods", storage.ListOptions{ResourceVersion: resourceVersion, Recursive: true, Predicate: storage.Everything, ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan}, outList)
	if err != nil {
		t.Fatal(err)
	}
	if len(outList.Items) != int(storageWatchListPageSize)+1 {
		t.Errorf("Expect to get %d pods, got %d", storageWatchListPageSize+1, len(outList.Items))
	}

	checker := newConsistencyChecker("/pods", schema.GroupResource{}, store.cacher.newListFunc, store.cacher, store.storage)
	digest, err := checker.calculateDigests(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if digest.CacheDigest != digest.EtcdDigest {
		t.Errorf("Expect digests to match, cache: %s etcd: %q", digest.CacheDigest, digest.EtcdDigest)
	}
	if digest.ResourceVersion != resourceVersion {
		t.Errorf("Expect resourceVersion to equal: %q, got %q", resourceVersion, digest.ResourceVersion)
	}
}
