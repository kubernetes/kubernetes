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
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/consistency"
)

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
	err := store.cacher.GetList(ctx, "/pods/", storage.ListOptions{ResourceVersion: resourceVersion, Recursive: true, Predicate: storage.Everything, ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan}, outList)
	if err != nil {
		t.Fatal(err)
	}
	if len(outList.Items) != int(storageWatchListPageSize)+1 {
		t.Errorf("Expect to get %d pods, got %d", storageWatchListPageSize+1, len(outList.Items))
	}

	checker := consistency.NewChecker("/pods/", schema.GroupResource{}, store.cacher.newListFunc, store.cacher, store.storage)
	digest, err := checker.CalculateDigests(ctx)
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
