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
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
)

// TestAssumeSubReplicaSetScheduled tests that after a subRS is assumed, its information is aggregated
// on cluster level.
func TestAssumeSubReplicaSetScheduled(t *testing.T) {
	clusterName := "cluster"
	testSubReplicaSets := []*federation.SubReplicaSet{
		makeBaseSubReplicaSet(clusterName, "test"),
		makeBaseSubReplicaSet(clusterName, "test-1"),
		makeBaseSubReplicaSet(clusterName, "test-2"),
		makeBaseSubReplicaSet(clusterName, "test-nonzero"),
	}

	tests := []struct {
		subRSs      []*federation.SubReplicaSet
		clusterInfo *ClusterInfo
	}{{
		subRSs: []*federation.SubReplicaSet{testSubReplicaSets[0]},
		clusterInfo: &ClusterInfo{
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[0]},
		},
	}, {
		subRSs: []*federation.SubReplicaSet{testSubReplicaSets[1], testSubReplicaSets[2]},
		clusterInfo: &ClusterInfo{
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[1], testSubReplicaSets[2]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		for _, subRS := range tt.subRSs {
			if err := cache.AssumeSubRSIfBindSucceed(subRS, alwaysTrue); err != nil {
				t.Fatalf("AssumeSubReplicaSetScheduled failed: %v", err)
			}
		}
		c := cache.clusters[clusterName]
		if !reflect.DeepEqual(c, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, c, tt.clusterInfo)
		}
	}
}

type testExpireSubReplicaSetStruct struct {
	subRS         *federation.SubReplicaSet
	assumedTime time.Time
}

// TestExpireSubReplicaSet tests that assumed subRSs will be removed if expired.
// The removal will be reflected in cluster info.
func TestExpireSubReplicaSet(t *testing.T) {
	clusterName := "cluster"
	testSubReplicaSets := []*federation.SubReplicaSet{
		makeBaseSubReplicaSet(clusterName, "test-1"),
		makeBaseSubReplicaSet(clusterName, "test-2"),
	}
	now := time.Now()
	ttl := 10 * time.Second
	tests := []struct {
		subRSs      []*testExpireSubReplicaSetStruct
		cleanupTime time.Time

		clusterInfo *ClusterInfo
	}{{ // assumed subRS would expires
		subRSs: []*testExpireSubReplicaSetStruct{
			{subRS: testSubReplicaSets[0], assumedTime: now},
		},
		cleanupTime: now.Add(2 * ttl),
		clusterInfo:   nil,
	}, { // first one would expire, second one would not.
		subRSs: []*testExpireSubReplicaSetStruct{
			{subRS: testSubReplicaSets[0], assumedTime: now},
			{subRS: testSubReplicaSets[1], assumedTime: now.Add(3 * ttl / 2)},
		},
		cleanupTime: now.Add(2 * ttl),
		clusterInfo: &ClusterInfo{
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[1]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)

		for _, subRS := range tt.subRSs {
			if err := cache.assumeSubRSIfBindSucceed(subRS.subRS, alwaysTrue, subRS.assumedTime); err != nil {
				t.Fatalf("assumeSubReplicaSet failed: %v", err)
			}
		}
		// subRSs that have assumedTime + ttl < cleanupTime will get expired and removed
		cache.cleanupAssumedSubRS(tt.cleanupTime)
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}
	}
}

// TestAddSubReplicaSetWillConfirm tests that a subRS being Add()ed will be confirmed if assumed.
// The subRS info should still exist after manually expiring unconfirmed subRSs.
func TestAddSubReplicaSetWillConfirm(t *testing.T) {
	clusterName := "cluster"
	now := time.Now()
	ttl := 10 * time.Second

	testSubReplicaSets := []*federation.SubReplicaSet{
		makeBaseSubReplicaSet(clusterName, "test-1"),
		makeBaseSubReplicaSet(clusterName, "test-2"),
	}
	tests := []struct {
		subRSsToAssume []*federation.SubReplicaSet
		subRSsToAdd    []*federation.SubReplicaSet

		clusterInfo    *ClusterInfo
	}{{ // two subRS were assumed at same time. But first one is called Add() and gets confirmed.
		subRSsToAssume: []*federation.SubReplicaSet{testSubReplicaSets[0], testSubReplicaSets[1]},
		subRSsToAdd:    []*federation.SubReplicaSet{testSubReplicaSets[0]},
		clusterInfo: &ClusterInfo{
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[0]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, subRSToAssume := range tt.subRSsToAssume {
			if err := cache.assumeSubRSIfBindSucceed(subRSToAssume, alwaysTrue, now); err != nil {
				t.Fatalf("assumeSubReplicaSet failed: %v", err)
			}
		}
		for _, subRSToAdd := range tt.subRSsToAdd {
			if err := cache.AddSubRS(subRSToAdd); err != nil {
				t.Fatalf("AddSubReplicaSet failed: %v", err)
			}
		}
		cache.cleanupAssumedSubRS(now.Add(2 * ttl))
		// check after expiration. confirmed subRSs shouldn't be expired.
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}
	}
}

// TestAddSubReplicaSetAfterExpiration tests that a subRS being Add()ed will be added back if expired.
func TestAddSubReplicaSetAfterExpiration(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	baseSubReplicaSet := makeBaseSubReplicaSet(clusterName, "test")
	tests := []struct {
		subRS       *federation.SubReplicaSet

		clusterInfo *ClusterInfo
	}{{
		subRS: baseSubReplicaSet,
		clusterInfo: &ClusterInfo{
			replicaSets: []*federation.SubReplicaSet{baseSubReplicaSet},
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		if err := cache.assumeSubRSIfBindSucceed(tt.subRS, alwaysTrue, now); err != nil {
			t.Fatalf("assumeSubReplicaSet failed: %v", err)
		}
		cache.cleanupAssumedSubRS(now.Add(2 * ttl))
		// It should be expired and removed.
		n := cache.clusters[clusterName]
		if n != nil {
			t.Errorf("#%d: expecting nil cluster info, but get=%v", i, n)
		}
		if err := cache.AddSubRS(tt.subRS); err != nil {
			t.Fatalf("AddSubReplicaSet failed: %v", err)
		}
		// check after expiration. confirmed subRSs shouldn't be expired.
		n = cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}
	}
}

// TestUpdateSubReplicaSet tests that a subRS will be updated if added before.
func TestUpdateSubReplicaSet(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	testSubReplicaSets := []*federation.SubReplicaSet{
		makeBaseSubReplicaSet(clusterName, "test"),
		makeBaseSubReplicaSet(clusterName, "test"),
	}
	tests := []struct {
		subRSsToAssume []*federation.SubReplicaSet
		subRSsToAdd    []*federation.SubReplicaSet
		subRSsToUpdate []*federation.SubReplicaSet

		clusterInfo    []*ClusterInfo
	}{{ // add a subRS and then update it twice
		subRSsToAdd:    []*federation.SubReplicaSet{testSubReplicaSets[0]},
		subRSsToUpdate: []*federation.SubReplicaSet{testSubReplicaSets[0], testSubReplicaSets[1], testSubReplicaSets[0]},
		clusterInfo: []*ClusterInfo{{
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[1]},
		}, {
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[0]},
		}},
	}}

	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, subRSToAdd := range tt.subRSsToAdd {
			if err := cache.AddSubRS(subRSToAdd); err != nil {
				t.Fatalf("AddSubReplicaSet failed: %v", err)
			}
		}

		for i := range tt.subRSsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdateSubRS(tt.subRSsToUpdate[i-1], tt.subRSsToUpdate[i]); err != nil {
				t.Fatalf("UpdateSubReplicaSet failed: %v", err)
			}
			// check after expiration. confirmed subRSs shouldn't be expired.
			n := cache.clusters[clusterName]
			if !reflect.DeepEqual(n, tt.clusterInfo[i-1]) {
				t.Errorf("#%d: cluster info get=%s, want=%s", i-1, n, tt.clusterInfo)
			}
		}
	}
}

// TestExpireAddUpdateSubReplicaSet test the sequence that a subRS is expired, added, then updated
func TestExpireAddUpdateSubReplicaSet(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	testSubReplicaSets := []*federation.SubReplicaSet{
		makeBaseSubReplicaSet(clusterName, "test"),
		makeBaseSubReplicaSet(clusterName, "test"),
	}
	tests := []struct {
		subRSsToAssume []*federation.SubReplicaSet
		subRSsToAdd    []*federation.SubReplicaSet
		subRSsToUpdate []*federation.SubReplicaSet

		clusterInfo    []*ClusterInfo
	}{{ // SubReplicaSet is assumed, expired, and added. Then it would be updated twice.
		subRSsToAssume: []*federation.SubReplicaSet{testSubReplicaSets[0]},
		subRSsToAdd:    []*federation.SubReplicaSet{testSubReplicaSets[0]},
		subRSsToUpdate: []*federation.SubReplicaSet{testSubReplicaSets[0], testSubReplicaSets[1], testSubReplicaSets[0]},
		clusterInfo: []*ClusterInfo{{
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[1]},
		}, {
			replicaSets: []*federation.SubReplicaSet{testSubReplicaSets[0]},
		}},
	}}

	now := time.Now()
	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, subRSToAssume := range tt.subRSsToAssume {
			if err := cache.assumeSubRSIfBindSucceed(subRSToAssume, alwaysTrue, now); err != nil {
				t.Fatalf("assumeSubReplicaSet failed: %v", err)
			}
		}
		cache.cleanupAssumedSubRS(now.Add(2 * ttl))

		for _, subRSToAdd := range tt.subRSsToAdd {
			if err := cache.AddSubRS(subRSToAdd); err != nil {
				t.Fatalf("AddSubReplicaSet failed: %v", err)
			}
		}

		for i := range tt.subRSsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdateSubRS(tt.subRSsToUpdate[i-1], tt.subRSsToUpdate[i]); err != nil {
				t.Fatalf("UpdateSubReplicaSet failed: %v", err)
			}
			// check after expiration. confirmed subRSs shouldn't be expired.
			n := cache.clusters[clusterName]
			if !reflect.DeepEqual(n, tt.clusterInfo[i-1]) {
				t.Errorf("#%d: cluster info get=%s, want=%s", i-1, n, tt.clusterInfo)
			}
		}
	}
}

// TestRemoveSubReplicaSet tests after added subRS is removed, its information should also be subtracted.
func TestRemoveSubReplicaSet(t *testing.T) {
	clusterName := "cluster"
	baseSubReplicaSet := makeBaseSubReplicaSet(clusterName, "test")
	tests := []struct {
		subRS       *federation.SubReplicaSet

		clusterInfo *ClusterInfo
	}{{
		subRS: baseSubReplicaSet,
		clusterInfo: &ClusterInfo{
			replicaSets: []*federation.SubReplicaSet{baseSubReplicaSet},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		if err := cache.AddSubRS(tt.subRS); err != nil {
			t.Fatalf("AddSubReplicaSet failed: %v", err)
		}
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.clusterInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.clusterInfo)
		}

		if err := cache.RemoveSubRS(tt.subRS); err != nil {
			t.Fatalf("RemoveSubReplicaSet failed: %v", err)
		}

		n = cache.clusters[clusterName]
		if n != nil {
			t.Errorf("#%d: expecting subRS deleted and nil cluster info, get=%s", i, n)
		}
	}
}

func BenchmarkGetNodeNameToInfoMap1kNodes30kSubReplicaSets(b *testing.B) {
	cache := setupCacheOf1kNodes30kSubReplicaSets(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cache.GetClusterNameToInfoMap()
	}
}

func BenchmarkList1kNodes30kSubReplicaSets(b *testing.B) {
	cache := setupCacheOf1kNodes30kSubReplicaSets(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cache.List()
	}
}

func BenchmarkExpire100SubReplicaSets(b *testing.B) {
	benchmarkExpire(b, 100)
}

func BenchmarkExpire1kSubReplicaSets(b *testing.B) {
	benchmarkExpire(b, 1000)
}

func BenchmarkExpire10kSubReplicaSets(b *testing.B) {
	benchmarkExpire(b, 10000)
}

func benchmarkExpire(b *testing.B, subRSNum int) {
	now := time.Now()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		cache := setupCacheWithAssumedSubRS(b, subRSNum, now)
		b.StartTimer()
		cache.cleanupAssumedSubRS(now.Add(2 * time.Second))
	}
}

func makeBaseSubReplicaSet(clusterName, objName string) *federation.SubReplicaSet {
	annotations := map[string]string{}
	annotations[unversioned.TargetClusterKey] = clusterName
	return &federation.SubReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Namespace: "cluster_info_cache_test",
			Name:      objName,
			Annotations: annotations,
		},
	}
}

func setupCacheOf1kNodes30kSubReplicaSets(b *testing.B) Cache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < 1000; i++ {
		clusterName := fmt.Sprintf("cluster-%d", i)
		for j := 0; j < 30; j++ {
			objName := fmt.Sprintf("%s-subRS-%d", clusterName, j)
			subRS := makeBaseSubReplicaSet(clusterName, objName)

			if err := cache.AddSubRS(subRS); err != nil {
				b.Fatalf("AddSubReplicaSet failed: %v", err)
			}
		}
	}
	return cache
}

func setupCacheWithAssumedSubRS(b *testing.B, subRSNum int, assumedTime time.Time) *schedulerCache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < subRSNum; i++ {
		clusterName := fmt.Sprintf("cluster-%d", i/10)
		objName := fmt.Sprintf("%s-subRS-%d", clusterName, i%10)
		subRS := makeBaseSubReplicaSet(clusterName, objName)

		err := cache.assumeSubRSIfBindSucceed(subRS, alwaysTrue, assumedTime)
		if err != nil {
			b.Fatalf("assumeSubReplicaSetIfBindSucceed failed: %v", err)
		}
	}
	return cache
}

func alwaysTrue() bool {
	return true
}
