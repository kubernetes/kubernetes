/*
Copyright The Kubernetes Authors.

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

package apiserver

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/sharding"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// calculateShardRange divides the 64-bit hash space evenly across shards and
// returns the hex-encoded start (inclusive) and end (exclusive) for the given index.
func calculateShardRange(index, total int) (start, end string) {
	maxVal := new(big.Int).Lsh(big.NewInt(1), 64) // 2^64
	span := new(big.Int).Div(maxVal, big.NewInt(int64(total)))

	startVal := new(big.Int).Mul(span, big.NewInt(int64(index)))
	endVal := new(big.Int).Mul(span, big.NewInt(int64(index+1)))
	if index == total-1 {
		endVal = maxVal // last shard covers remainder
	}

	start = fmt.Sprintf("%016x", startVal)
	end = fmt.Sprintf("%016x", endVal)
	return start, end
}

func shardSelectorString(index, total int) string {
	start, end := calculateShardRange(index, total)
	return fmt.Sprintf("shardRange(object.metadata.uid,0x%s,0x%s)", start, end)
}

// hexLess compares two lowercase hex strings numerically, matching the
// server-side sharding comparison logic.
func hexLess(a, b string) bool {
	if len(a) != len(b) {
		return len(a) < len(b)
	}
	return a < b
}

// objectInShard returns true if the object's UID hash falls within the given shard range.
func objectInShard(uid string, index, total int) bool {
	start, end := calculateShardRange(index, total)
	hash := sharding.HashField(uid)
	if hexLess(hash, start) {
		return false
	}
	if !hexLess(hash, end) {
		return false
	}
	return true
}

func TestShardedList(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListandWatch, true)

	ctx, client, _, tearDownFn := setup(t)
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "shard-list", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Create a batch of configmaps.
	const numObjects = 20
	created := make([]*v1.ConfigMap, 0, numObjects)
	for range numObjects {
		cm, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "shard-test-",
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create configmap: %v", err)
		}
		created = append(created, cm)
	}

	const numShards = 3
	allFound := make(map[string]bool)

	for shard := range numShards {
		selector := shardSelectorString(shard, numShards)
		list, err := client.CoreV1().ConfigMaps(ns.Name).List(ctx, metav1.ListOptions{
			Selector: selector,
		})
		if err != nil {
			t.Fatalf("shard %d: failed to list: %v", shard, err)
		}

		// The response must include shard info with the selector echoed back.
		if list.ShardInfo == nil {
			t.Errorf("shard %d: expected shardInfo to be set", shard)
		} else if list.ShardInfo.Selector != selector {
			t.Errorf("shard %d: expected shardInfo.selector=%q, got %q", shard, selector, list.ShardInfo.Selector)
		}

		for _, cm := range list.Items {
			uid := string(cm.UID)
			if !objectInShard(uid, shard, numShards) {
				t.Errorf("shard %d: object %s (UID %s) should not be in this shard", shard, cm.Name, uid)
			}
			if allFound[uid] {
				t.Errorf("shard %d: object %s (UID %s) appeared in multiple shards", shard, cm.Name, uid)
			}
			allFound[uid] = true
		}
	}

	// Every created object must appear in exactly one shard.
	for _, cm := range created {
		if !allFound[string(cm.UID)] {
			t.Errorf("object %s (UID %s) was not returned by any shard", cm.Name, cm.UID)
		}
	}
}

func TestShardedWatch(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListandWatch, true)

	ctx, client, _, tearDownFn := setup(t)
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "shard-watch", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Get initial resource version.
	list, err := client.CoreV1().ConfigMaps(ns.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("initial list failed: %v", err)
	}
	rv := list.ResourceVersion

	const numShards = 2

	// Start a watch per shard.
	watchers := make([]watch.Interface, numShards)
	for shard := range numShards {
		w, err := client.CoreV1().ConfigMaps(ns.Name).Watch(ctx, metav1.ListOptions{
			ResourceVersion: rv,
			Selector:        shardSelectorString(shard, numShards),
		})
		if err != nil {
			t.Fatalf("shard %d: failed to start watch: %v", shard, err)
		}
		defer w.Stop()
		watchers[shard] = w
	}

	// Create objects and track which shard should see each one.
	const numObjects = 10
	expectedShard := make(map[string]int) // UID -> shard index

	for range numObjects {
		cm, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "shard-watch-",
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create configmap: %v", err)
		}
		for shard := range numShards {
			if objectInShard(string(cm.UID), shard, numShards) {
				expectedShard[string(cm.UID)] = shard
				break
			}
		}
	}

	// Collect events from each watcher.
	received := make([]map[string]bool, numShards)
	for i := range received {
		received[i] = make(map[string]bool)
	}

	for shard := range numShards {
		collectEvents(t, watchers[shard], received[shard], expectedShard)
	}

	// Verify every object was seen by exactly the right shard.
	for uid, expectedIdx := range expectedShard {
		if !received[expectedIdx][uid] {
			t.Errorf("UID %s: expected in shard %d but not received", uid, expectedIdx)
		}
		for other := range numShards {
			if other != expectedIdx && received[other][uid] {
				t.Errorf("UID %s: received in shard %d but expected only in shard %d", uid, other, expectedIdx)
			}
		}
	}
}

func collectEvents(t *testing.T, w watch.Interface, seen map[string]bool, expected map[string]int) {
	t.Helper()
	timeout := time.After(30 * time.Second)
	remaining := 0
	for range expected {
		remaining++
	}
	// We only need events for UIDs in our expected set that belong to this watcher.
	// But we don't know which watcher this is, so just collect all ADDED events until
	// we've seen enough or timed out.
	for {
		select {
		case evt, ok := <-w.ResultChan():
			if !ok {
				return
			}
			if evt.Type == watch.Added {
				cm, ok := evt.Object.(*v1.ConfigMap)
				if !ok {
					continue
				}
				seen[string(cm.UID)] = true
			}
		case <-timeout:
			return
		}
	}
}

func TestShardedListFeatureGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListandWatch, false)

	ctx, client, _, tearDownFn := setup(t)
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "shard-disabled", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Create a configmap so the namespace is non-empty.
	_, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create configmap: %v", err)
	}

	// List with a shard selector — when the gate is off, the selector should be
	// ignored and all objects returned (not sharded).
	list, err := client.CoreV1().ConfigMaps(ns.Name).List(ctx, metav1.ListOptions{
		Selector: shardSelectorString(0, 2),
	})
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}

	if list.ShardInfo != nil {
		t.Errorf("expected shardInfo=nil when feature gate is disabled")
	}
	if len(list.Items) != 1 {
		t.Errorf("expected 1 item (selector ignored), got %d", len(list.Items))
	}
}

func TestShardedListComplete(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListandWatch, true)

	ctx, client, _, tearDownFn := setup(t)
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "shard-complete", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// An empty/everything selector should return all objects without sharded=true.
	const numObjects = 5
	for range numObjects {
		_, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "complete-",
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create configmap: %v", err)
		}
	}

	// A single shard covering the full range (shard 0 of 1) has empty start and end,
	// which means no selector string — just use a normal list.
	list, err := client.CoreV1().ConfigMaps(ns.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if list.ShardInfo != nil {
		t.Errorf("expected shardInfo=nil for unsharded list")
	}
	if len(list.Items) != numObjects {
		t.Errorf("expected %d items, got %d", numObjects, len(list.Items))
	}
}
