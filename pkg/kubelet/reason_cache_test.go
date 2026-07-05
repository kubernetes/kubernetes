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

package kubelet

import (
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestReasonCache(t *testing.T) {
	// Create test sync result
	syncResult := kubecontainer.PodSyncResult{}
	results := []*kubecontainer.SyncResult{
		// reason cache should be set for SyncResult with StartContainer action and error
		kubecontainer.NewSyncResult(kubecontainer.StartContainer, "container_1"),
		// reason cache should not be set for SyncResult with StartContainer action but without error
		kubecontainer.NewSyncResult(kubecontainer.StartContainer, "container_2"),
		// reason cache should not be set for SyncResult with other actions
		kubecontainer.NewSyncResult(kubecontainer.KillContainer, "container_3"),
	}
	results[0].Fail(kubecontainer.ErrRunContainer, "message_1")
	results[2].Fail(kubecontainer.ErrKillContainer, "message_3")
	syncResult.AddSyncResult(results...)
	uid := types.UID("pod_1")

	reasonCache := NewReasonCache()
	reasonCache.Update(uid, syncResult)
	assertReasonInfo(t, reasonCache, uid, results[0], true)
	assertReasonInfo(t, reasonCache, uid, results[1], false)
	assertReasonInfo(t, reasonCache, uid, results[2], false)

	reasonCache.Remove(uid, results[0].Target.(string))
	assertReasonInfo(t, reasonCache, uid, results[0], false)
}

func TestReasonCacheCleanupOrphanedPods(t *testing.T) {
	reasonCache := NewReasonCache()
	uid1 := types.UID("pod_1")
	uid2 := types.UID("pod_2")
	uid3 := types.UID("pod_3")

	reasonCache.add(uid1, "container_1", kubecontainer.ErrRunContainer, "message_1")
	reasonCache.add(uid2, "container_1", kubecontainer.ErrRunContainer, "message_2")
	reasonCache.add(uid3, "container_1", kubecontainer.ErrRunContainer, "message_3")

	// Clean up all pods except pod_1 and pod_2
	activePods := sets.New[types.UID](uid1, uid2)
	reasonCache.CleanupOrphanedPods(activePods)

	// pod_1 and pod_2 should still be in cache
	if _, ok := reasonCache.Get(uid1, "container_1"); !ok {
		t.Errorf("expected pod_1 to be in cache")
	}
	if _, ok := reasonCache.Get(uid2, "container_1"); !ok {
		t.Errorf("expected pod_2 to be in cache")
	}

	// pod_3 should be removed
	if _, ok := reasonCache.Get(uid3, "container_1"); ok {
		t.Errorf("expected pod_3 to be removed from cache")
	}

	// Test RemovePod
	reasonCache.RemovePod(uid1)
	if _, ok := reasonCache.Get(uid1, "container_1"); ok {
		t.Errorf("expected pod_1 to be removed from cache after RemovePod")
	}
}

func assertReasonInfo(t *testing.T, cache *ReasonCache, uid types.UID, result *kubecontainer.SyncResult, found bool) {
	name := result.Target.(string)
	actualReason, ok := cache.Get(uid, name)
	if ok && !found {
		t.Fatalf("unexpected cache hit: %v, %q", actualReason.Err, actualReason.Message)
	}
	if !ok && found {
		t.Fatalf("corresponding reason info not found")
	}
	if !found {
		return
	}
	reason := result.Error
	message := result.Message
	if actualReason.Err != reason || actualReason.Message != message {
		t.Errorf("expected %v %q, got %v %q", reason, message, actualReason.Err, actualReason.Message)
	}
}
