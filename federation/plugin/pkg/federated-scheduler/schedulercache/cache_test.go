/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package schedulercache

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
)

// TestAssumeReplicaSetScheduled tests that after a replicaSet is assumed, its information is aggregated
// on cluster level.
func TestAssumeReplicaSetScheduled(t *testing.T) {
	clusterName := "cluster"
	testReplicaSets := []*v1beta1.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test"),
		makeBaseReplicaSet(clusterName, "test-1"),
		makeBaseReplicaSet(clusterName, "test-2"),
		makeBaseReplicaSet(clusterName, "test-nonzero"),
	}

	tests := []struct {
		replicaSets []*v1beta1.ReplicaSet
		clusterInfo *ClusterInfo
	}{{
		replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[0]},
		clusterInfo: &ClusterInfo{
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[0]},
		},
	}, {
		replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[1], testReplicaSets[2]},
		clusterInfo: &ClusterInfo{
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[1], testReplicaSets[2]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		for _, rs := range tt.replicaSets {
			if err := cache.AssumeReplicaSet(rs); err != nil {
				t.Fatalf("AssumeReplicaSetScheduled failed: %v", err)
			}
		}
		c := cache.clusters[clusterName]
		if !reflect.DeepEqual(c, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, c, tt.clusterInfo)
		}
	}
}

type testExpireReplicaSetStruct struct {
	rs          *v1beta1.ReplicaSet
	assumedTime time.Time
}

// TestExpireReplicaSet tests that assumed replicaSets will be removed if expired.
// The removal will be reflected in cluster info.
func TestExpireReplicaSet(t *testing.T) {
	clusterName := "cluster"
	testReplicaSets := []*v1beta1.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test-1"),
		makeBaseReplicaSet(clusterName, "test-2"),
	}
	now := time.Now()
	ttl := 10 * time.Second
	tests := []struct {
		replicaSets      []*testExpireReplicaSetStruct
		cleanupTime time.Time

		clusterInfo *ClusterInfo
	}{{ // assumed replicaSet would expires
		replicaSets: []*testExpireReplicaSetStruct{
			{rs: testReplicaSets[0], assumedTime: now},
		},
		cleanupTime: now.Add(2 * ttl),
		clusterInfo:   nil,
	}, { // first one would expire, second one would not.
		replicaSets: []*testExpireReplicaSetStruct{
			{rs: testReplicaSets[0], assumedTime: now},
			{rs: testReplicaSets[1], assumedTime: now.Add(3 * ttl / 2)},
		},
		cleanupTime: now.Add(2 * ttl),
		clusterInfo: &ClusterInfo{
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[1]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)

		for _, replicaSet := range tt.replicaSets {
			if err := cache.assumeReplicaSet(replicaSet.rs, replicaSet.assumedTime); err != nil {
				t.Fatalf("assumeReplicaSet failed: %v", err)
			}
		}
		// replicaSets that have assumedTime + ttl < cleanupTime will get expired and removed
		cache.cleanupAssumedReplicaSet(tt.cleanupTime)
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}
	}
}

// TestAddReplicaSetWillConfirm tests that a replicaSet being Add()ed will be confirmed if assumed.
// The replicaSet info should still exist after manually expiring unconfirmed replicaSets.
func TestAddReplicaSetWillConfirm(t *testing.T) {
	clusterName := "cluster"
	now := time.Now()
	ttl := 10 * time.Second

	testReplicaSets := []*v1beta1.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test-1"),
		makeBaseReplicaSet(clusterName, "test-2"),
	}
	tests := []struct {
		replicaSetsToAssume []*v1beta1.ReplicaSet
		replicaSetsToAdd    []*v1beta1.ReplicaSet

		clusterInfo         *ClusterInfo
	}{{ // two replicaSet were assumed at same time. But first one is called Add() and gets confirmed.
		replicaSetsToAssume: []*v1beta1.ReplicaSet{testReplicaSets[0], testReplicaSets[1]},
		replicaSetsToAdd:    []*v1beta1.ReplicaSet{testReplicaSets[0]},
		clusterInfo: &ClusterInfo{
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[0]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, replicaSetToAssume := range tt.replicaSetsToAssume {
			if err := cache.assumeReplicaSet(replicaSetToAssume, now); err != nil {
				t.Fatalf("assumeReplicaSet failed: %v", err)
			}
		}
		for _, replicaSetToAdd := range tt.replicaSetsToAdd {
			if err := cache.AddReplicaSet(replicaSetToAdd); err != nil {
				t.Fatalf("AddReplicaSet failed: %v", err)
			}
		}
		cache.cleanupAssumedReplicaSet(now.Add(2 * ttl))
		// check after expiration. confirmed replicaSets shouldn't be expired.
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}
	}
}

// TestAddReplicaSetAfterExpiration tests that a replicaSet being Add()ed will be added back if expired.
func TestAddReplicaSetAfterExpiration(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	baseReplicaSet := makeBaseReplicaSet(clusterName, "test")
	tests := []struct {
		replicaSet       *v1beta1.ReplicaSet

		clusterInfo *ClusterInfo
	}{{
		replicaSet: baseReplicaSet,
		clusterInfo: &ClusterInfo{
			replicaSets: []*v1beta1.ReplicaSet{baseReplicaSet},
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		if err := cache.assumeReplicaSet(tt.replicaSet, now); err != nil {
			t.Fatalf("assumeReplicaSet failed: %v", err)
		}
		cache.cleanupAssumedReplicaSet(now.Add(2 * ttl))
		// It should be expired and removed.
		n := cache.clusters[clusterName]
		if n != nil {
			t.Errorf("#%d: expecting nil cluster info, but get=%v", i, n)
		}
		if err := cache.AddReplicaSet(tt.replicaSet); err != nil {
			t.Fatalf("AddReplicaSet failed: %v", err)
		}
		// check after expiration. confirmed replicaSets shouldn't be expired.
		n = cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}
	}
}

// TestUpdateReplicaSet tests that a replicaSet will be updated if added before.
func TestUpdateReplicaSet(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	testReplicaSets := []*v1beta1.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test"),
		makeBaseReplicaSet(clusterName, "test"),
	}
	tests := []struct {
		replicaSetsToAssume []*v1beta1.ReplicaSet
		replicaSetsToAdd    []*v1beta1.ReplicaSet
		replicaSetsToUpdate []*v1beta1.ReplicaSet

		clusterInfo    []*ClusterInfo
	}{{ // add a replicaSet and then update it twice
		replicaSetsToAdd:    []*v1beta1.ReplicaSet{testReplicaSets[0]},
		replicaSetsToUpdate: []*v1beta1.ReplicaSet{testReplicaSets[0], testReplicaSets[1], testReplicaSets[0]},
		clusterInfo: []*ClusterInfo{{
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[1]},
		}, {
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[0]},
		}},
	}}

	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, replicaSetToAdd := range tt.replicaSetsToAdd {
			if err := cache.AddReplicaSet(replicaSetToAdd); err != nil {
				t.Fatalf("AddReplicaSet failed: %v", err)
			}
		}

		for i := range tt.replicaSetsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdateReplicaSet(tt.replicaSetsToUpdate[i-1], tt.replicaSetsToUpdate[i]); err != nil {
				t.Fatalf("UpdateReplicaSet failed: %v", err)
			}
			// check after expiration. confirmed replicaSets shouldn't be expired.
			n := cache.clusters[clusterName]
			if !reflect.DeepEqual(n, tt.clusterInfo[i-1]) {
				t.Errorf("#%d: cluster info get=%s, want=%s", i-1, n, tt.clusterInfo)
			}
		}
	}
}

// TestExpireAddUpdateReplicaSet test the sequence that a replicaSet is expired, added, then updated
func TestExpireAddUpdateReplicaSet(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	testReplicaSets := []*v1beta1.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test"),
		makeBaseReplicaSet(clusterName, "test"),
	}
	tests := []struct {
		replicaSetsToAssume []*v1beta1.ReplicaSet
		replicaSetsToAdd    []*v1beta1.ReplicaSet
		replicaSetsToUpdate []*v1beta1.ReplicaSet

		clusterInfo    []*ClusterInfo
	}{{ // ReplicaSet is assumed, expired, and added. Then it would be updated twice.
		replicaSetsToAssume: []*v1beta1.ReplicaSet{testReplicaSets[0]},
		replicaSetsToAdd:    []*v1beta1.ReplicaSet{testReplicaSets[0]},
		replicaSetsToUpdate: []*v1beta1.ReplicaSet{testReplicaSets[0], testReplicaSets[1], testReplicaSets[0]},
		clusterInfo: []*ClusterInfo{{
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[1]},
		}, {
			replicaSets: []*v1beta1.ReplicaSet{testReplicaSets[0]},
		}},
	}}

	now := time.Now()
	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, replicaSetToAssume := range tt.replicaSetsToAssume {
			if err := cache.assumeReplicaSet(replicaSetToAssume, now); err != nil {
				t.Fatalf("assumeReplicaSet failed: %v", err)
			}
		}
		cache.cleanupAssumedReplicaSet(now.Add(2 * ttl))

		for _, replicaSetToAdd := range tt.replicaSetsToAdd {
			if err := cache.AddReplicaSet(replicaSetToAdd); err != nil {
				t.Fatalf("AddReplicaSet failed: %v", err)
			}
		}

		for i := range tt.replicaSetsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdateReplicaSet(tt.replicaSetsToUpdate[i-1], tt.replicaSetsToUpdate[i]); err != nil {
				t.Fatalf("UpdateReplicaSet failed: %v", err)
			}
			// check after expiration. confirmed replicaSets shouldn't be expired.
			n := cache.clusters[clusterName]
			if !reflect.DeepEqual(n, tt.clusterInfo[i-1]) {
				t.Errorf("#%d: cluster info get=%s, want=%s", i-1, n, tt.clusterInfo)
			}
		}
	}
}

// TestRemoveReplicaSet tests after added replicaSet is removed, its information should also be subtracted.
func TestRemoveReplicaSet(t *testing.T) {
	clusterName := "cluster"
	baseReplicaSet := makeBaseReplicaSet(clusterName, "test")
	tests := []struct {
		replicaSet       *v1beta1.ReplicaSet

		clusterInfo *ClusterInfo
	}{{
		replicaSet: baseReplicaSet,
		clusterInfo: &ClusterInfo{
			replicaSets: []*v1beta1.ReplicaSet{baseReplicaSet},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		if err := cache.AddReplicaSet(tt.replicaSet); err != nil {
			t.Fatalf("AddReplicaSet failed: %v", err)
		}
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}

		if err := cache.RemoveReplicaSet(tt.replicaSet); err != nil {
			t.Fatalf("RemoveReplicaSet failed: %v", err)
		}

		n = cache.clusters[clusterName]
		if n != nil {
			t.Errorf("#%d: expecting replicaSet deleted and nil cluster info, get=%s", i, n)
		}
	}
}

func BenchmarkGetNodeNameToInfoMap1kNodes30kReplicaSets(b *testing.B) {
	cache := setupCacheOf1kNodes30kReplicaSets(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cache.GetClusterNameToInfoMap()
	}
}

func BenchmarkList1kNodes30kReplicaSets(b *testing.B) {
	cache := setupCacheOf1kNodes30kReplicaSets(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cache.List()
	}
}

func BenchmarkExpire100ReplicaSets(b *testing.B) {
	benchmarkExpire(b, 100)
}

func BenchmarkExpire1kReplicaSets(b *testing.B) {
	benchmarkExpire(b, 1000)
}

func BenchmarkExpire10kReplicaSets(b *testing.B) {
	benchmarkExpire(b, 10000)
}

func benchmarkExpire(b *testing.B, replicaSetNum int) {
	now := time.Now()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		cache := setupCacheWithAssumedReplicaSet(b, replicaSetNum, now)
		b.StartTimer()
		cache.cleanupAssumedReplicaSet(now.Add(2 * time.Second))
	}
}

func makeBaseReplicaSet(clusterName, objName string) *v1beta1.ReplicaSet {
	annotations := map[string]string{}
	annotations[unversioned.TargetClusterKey] = clusterName
	return &v1beta1.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Namespace: "cluster_info_cache_test",
			Name:      objName,
			Annotations: annotations,
		},
	}
}

func setupCacheOf1kNodes30kReplicaSets(b *testing.B) Cache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < 1000; i++ {
		clusterName := fmt.Sprintf("cluster-%d", i)
		for j := 0; j < 30; j++ {
			objName := fmt.Sprintf("%s-replicaSet-%d", clusterName, j)
			replicaSet := makeBaseReplicaSet(clusterName, objName)

			if err := cache.AddReplicaSet(replicaSet); err != nil {
				b.Fatalf("AddReplicaSet failed: %v", err)
			}
		}
	}
	return cache
}

func setupCacheWithAssumedReplicaSet(b *testing.B, replicaSetNum int, assumedTime time.Time) *schedulerCache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < replicaSetNum; i++ {
		clusterName := fmt.Sprintf("cluster-%d", i/10)
		objName := fmt.Sprintf("%s-replicaSet-%d", clusterName, i%10)
		replicaSet := makeBaseReplicaSet(clusterName, objName)

		err := cache.assumeReplicaSet(replicaSet, assumedTime)
		if err != nil {
			b.Fatalf("assumeReplicaSetIfBindSucceed failed: %v", err)
		}
	}
	return cache
}

func alwaysTrue() bool {
	return true
}
