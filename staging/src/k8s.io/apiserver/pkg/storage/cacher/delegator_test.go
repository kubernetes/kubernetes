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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

func TestCalculateDigest(t *testing.T) {
	newListFunc := func() runtime.Object { return &example.PodList{} }
	testCases := []struct {
		desc            string
		resourceVersion string
		cacherReady     bool
		cacherItems     []example.Pod
		etcdItems       []example.Pod
		resourcePrefix  string

		expectListKey string
		expectDigest  storageDigest
		expectErr     bool
	}{
		{
			desc:            "not ready",
			cacherReady:     false,
			resourceVersion: "1",
			expectErr:       true,
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
		},
		{
			desc:            "resourceVersion changes digest",
			resourceVersion: "2",
			cacherReady:     true,
			cacherItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod", ResourceVersion: "3"}},
			},
			etcdItems: []example.Pod{
				{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod", ResourceVersion: "4"}},
			},
			expectDigest: storageDigest{
				ResourceVersion: "2",
				CacheDigest:     "86bf3a5e80d1c5ca",
				EtcdDigest:      "86bf3a5e80d1c5cd",
			},
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
				ready: tc.cacherReady,
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
			checker := newConsistencyChecker(tc.resourcePrefix, newListFunc, cacher, etcd)
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
		})
	}
}
