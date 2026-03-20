/*
Copyright 2026 The Kubernetes Authors.

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

package kubelet

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

func TestNewNodeHasSyncedFunc(t *testing.T) {
	const nodeName = types.NodeName("test-node")

	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	nodeHasSynced := newNodeHasSyncedFunc(corelisters.NewNodeLister(indexer), nodeName)

	// An empty cache (e.g. the initial List completed before the node was
	// registered) must not be reported as synced.
	if nodeHasSynced() {
		t.Fatal("expected nodeHasSynced to be false while the node is absent from the cache")
	}

	// Once the node appears in the cache, it is synced.
	node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: string(nodeName)}}
	if err := indexer.Add(node); err != nil {
		t.Fatalf("failed to add node to indexer: %v", err)
	}
	if !nodeHasSynced() {
		t.Fatal("expected nodeHasSynced to be true once the node is present in the cache")
	}

	// The signal latches: a later removal from the cache must not flip it back.
	if err := indexer.Delete(node); err != nil {
		t.Fatalf("failed to delete node from indexer: %v", err)
	}
	if !nodeHasSynced() {
		t.Fatal("expected nodeHasSynced to stay true after the node was observed once")
	}
}
