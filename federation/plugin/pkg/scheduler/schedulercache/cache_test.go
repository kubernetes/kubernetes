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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/labels"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
)

// TestAssumeReplicaSetScheduled tests that after a pod is assumed, its information is aggregated
// on cluster level.
func TestAssumeReplicaSetScheduled(t *testing.T) {
	clusterName := "cluster"
	testReplicaSets := []*extensions.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBaseReplicaSet(clusterName, "test-1", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBaseReplicaSet(clusterName, "test-2", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
		makeBaseReplicaSet(clusterName, "test-nonzero", "", "", []api.ContainerPort{{HostPort: 80}}),
	}

	tests := []struct {
		pods []*extensions.ReplicaSet

		wNodeInfo *ClusterInfo
	}{{
		pods: []*extensions.ReplicaSet{testReplicaSets[0]},
		wNodeInfo: &ClusterInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[0]},
		},
	}, {
		pods: []*extensions.ReplicaSet{testReplicaSets[1], testReplicaSets[2]},
		wNodeInfo: &ClusterInfo{
			requestedResource: &Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[1], testReplicaSets[2]},
		},
	}, { // test non-zero request
		pods: []*extensions.ReplicaSet{testReplicaSets[3]},
		wNodeInfo: &ClusterInfo{
			requestedResource: &Resource{
				MilliCPU: 0,
				Memory:   0,
			},
			nonzeroRequest: &Resource{
				MilliCPU: priorityutil.DefaultMilliCpuRequest,
				Memory:   priorityutil.DefaultMemoryRequest,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[3]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		for _, pod := range tt.pods {
			if err := cache.AssumeReplicaSetIfBindSucceed(pod, alwaysTrue); err != nil {
				t.Fatalf("AssumeReplicaSetScheduled failed: %v", err)
			}
		}
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

type testExpireReplicaSetStruct struct {
	pod         *extensions.ReplicaSet
	assumedTime time.Time
}

// TestExpireReplicaSet tests that assumed pods will be removed if expired.
// The removal will be reflected in cluster info.
func TestExpireReplicaSet(t *testing.T) {
	clusterName := "cluster"
	testReplicaSets := []*extensions.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test-1", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBaseReplicaSet(clusterName, "test-2", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
	}
	now := time.Now()
	ttl := 10 * time.Second
	tests := []struct {
		pods        []*testExpireReplicaSetStruct
		cleanupTime time.Time

		wNodeInfo *ClusterInfo
	}{{ // assumed pod would expires
		pods: []*testExpireReplicaSetStruct{
			{pod: testReplicaSets[0], assumedTime: now},
		},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo:   nil,
	}, { // first one would expire, second one would not.
		pods: []*testExpireReplicaSetStruct{
			{pod: testReplicaSets[0], assumedTime: now},
			{pod: testReplicaSets[1], assumedTime: now.Add(3 * ttl / 2)},
		},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo: &ClusterInfo{
			requestedResource: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[1]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)

		for _, pod := range tt.pods {
			if err := cache.assumeReplicaSetIfBindSucceed(pod.pod, alwaysTrue, pod.assumedTime); err != nil {
				t.Fatalf("assumeReplicaSet failed: %v", err)
			}
		}
		// pods that have assumedTime + ttl < cleanupTime will get expired and removed
		cache.cleanupAssumedReplicaSets(tt.cleanupTime)
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestAddReplicaSetWillConfirm tests that a pod being Add()ed will be confirmed if assumed.
// The pod info should still exist after manually expiring unconfirmed pods.
func TestAddReplicaSetWillConfirm(t *testing.T) {
	clusterName := "cluster"
	now := time.Now()
	ttl := 10 * time.Second

	testReplicaSets := []*extensions.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test-1", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBaseReplicaSet(clusterName, "test-2", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
	}
	tests := []struct {
		podsToAssume []*extensions.ReplicaSet
		podsToAdd    []*extensions.ReplicaSet

		wNodeInfo *ClusterInfo
	}{{ // two pod were assumed at same time. But first one is called Add() and gets confirmed.
		podsToAssume: []*extensions.ReplicaSet{testReplicaSets[0], testReplicaSets[1]},
		podsToAdd:    []*extensions.ReplicaSet{testReplicaSets[0]},
		wNodeInfo: &ClusterInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[0]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAssume := range tt.podsToAssume {
			if err := cache.assumeReplicaSetIfBindSucceed(podToAssume, alwaysTrue, now); err != nil {
				t.Fatalf("assumeReplicaSet failed: %v", err)
			}
		}
		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddReplicaSet(podToAdd); err != nil {
				t.Fatalf("AddReplicaSet failed: %v", err)
			}
		}
		cache.cleanupAssumedReplicaSets(now.Add(2 * ttl))
		// check after expiration. confirmed pods shouldn't be expired.
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestAddReplicaSetAfterExpiration tests that a pod being Add()ed will be added back if expired.
func TestAddReplicaSetAfterExpiration(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	baseReplicaSet := makeBaseReplicaSet(clusterName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		pod *extensions.ReplicaSet

		wNodeInfo *ClusterInfo
	}{{
		pod: baseReplicaSet,
		wNodeInfo: &ClusterInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			replicaSet: []*extensions.ReplicaSet{baseReplicaSet},
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		if err := cache.assumeReplicaSetIfBindSucceed(tt.pod, alwaysTrue, now); err != nil {
			t.Fatalf("assumeReplicaSet failed: %v", err)
		}
		cache.cleanupAssumedReplicaSets(now.Add(2 * ttl))
		// It should be expired and removed.
		n := cache.clusters[clusterName]
		if n != nil {
			t.Errorf("#%d: expecting nil cluster info, but get=%v", i, n)
		}
		if err := cache.AddReplicaSet(tt.pod); err != nil {
			t.Fatalf("AddReplicaSet failed: %v", err)
		}
		// check after expiration. confirmed pods shouldn't be expired.
		n = cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestUpdateReplicaSet tests that a pod will be updated if added before.
func TestUpdateReplicaSet(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	testReplicaSets := []*extensions.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBaseReplicaSet(clusterName, "test", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
	}
	tests := []struct {
		podsToAssume []*extensions.ReplicaSet
		podsToAdd    []*extensions.ReplicaSet
		podsToUpdate []*extensions.ReplicaSet

		wNodeInfo []*ClusterInfo
	}{{ // add a pod and then update it twice
		podsToAdd:    []*extensions.ReplicaSet{testReplicaSets[0]},
		podsToUpdate: []*extensions.ReplicaSet{testReplicaSets[0], testReplicaSets[1], testReplicaSets[0]},
		wNodeInfo: []*ClusterInfo{{
			requestedResource: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[1]},
		}, {
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[0]},
		}},
	}}

	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddReplicaSet(podToAdd); err != nil {
				t.Fatalf("AddReplicaSet failed: %v", err)
			}
		}

		for i := range tt.podsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdateReplicaSet(tt.podsToUpdate[i-1], tt.podsToUpdate[i]); err != nil {
				t.Fatalf("UpdateReplicaSet failed: %v", err)
			}
			// check after expiration. confirmed pods shouldn't be expired.
			n := cache.clusters[clusterName]
			if !reflect.DeepEqual(n, tt.wNodeInfo[i-1]) {
				t.Errorf("#%d: cluster info get=%s, want=%s", i-1, n, tt.wNodeInfo)
			}
		}
	}
}

// TestExpireAddUpdateReplicaSet test the sequence that a pod is expired, added, then updated
func TestExpireAddUpdateReplicaSet(t *testing.T) {
	clusterName := "cluster"
	ttl := 10 * time.Second
	testReplicaSets := []*extensions.ReplicaSet{
		makeBaseReplicaSet(clusterName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBaseReplicaSet(clusterName, "test", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
	}
	tests := []struct {
		podsToAssume []*extensions.ReplicaSet
		podsToAdd    []*extensions.ReplicaSet
		podsToUpdate []*extensions.ReplicaSet

		wNodeInfo []*ClusterInfo
	}{{ // ReplicaSet is assumed, expired, and added. Then it would be updated twice.
		podsToAssume: []*extensions.ReplicaSet{testReplicaSets[0]},
		podsToAdd:    []*extensions.ReplicaSet{testReplicaSets[0]},
		podsToUpdate: []*extensions.ReplicaSet{testReplicaSets[0], testReplicaSets[1], testReplicaSets[0]},
		wNodeInfo: []*ClusterInfo{{
			requestedResource: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[1]},
		}, {
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			replicaSet: []*extensions.ReplicaSet{testReplicaSets[0]},
		}},
	}}

	now := time.Now()
	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAssume := range tt.podsToAssume {
			if err := cache.assumeReplicaSetIfBindSucceed(podToAssume, alwaysTrue, now); err != nil {
				t.Fatalf("assumeReplicaSet failed: %v", err)
			}
		}
		cache.cleanupAssumedReplicaSets(now.Add(2 * ttl))

		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddReplicaSet(podToAdd); err != nil {
				t.Fatalf("AddReplicaSet failed: %v", err)
			}
		}

		for i := range tt.podsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdateReplicaSet(tt.podsToUpdate[i-1], tt.podsToUpdate[i]); err != nil {
				t.Fatalf("UpdateReplicaSet failed: %v", err)
			}
			// check after expiration. confirmed pods shouldn't be expired.
			n := cache.clusters[clusterName]
			if !reflect.DeepEqual(n, tt.wNodeInfo[i-1]) {
				t.Errorf("#%d: cluster info get=%s, want=%s", i-1, n, tt.wNodeInfo)
			}
		}
	}
}

// TestRemoveReplicaSet tests after added pod is removed, its information should also be subtracted.
func TestRemoveReplicaSet(t *testing.T) {
	clusterName := "cluster"
	baseReplicaSet := makeBaseReplicaSet(clusterName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		pod *extensions.ReplicaSet

		wNodeInfo *ClusterInfo
	}{{
		pod: baseReplicaSet,
		wNodeInfo: &ClusterInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			nonzeroRequest: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			replicaSet: []*extensions.ReplicaSet{baseReplicaSet},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		if err := cache.AddReplicaSet(tt.pod); err != nil {
			t.Fatalf("AddReplicaSet failed: %v", err)
		}
		n := cache.clusters[clusterName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: cluster info get=%s, want=%s", i, n, tt.wNodeInfo)
		}

		if err := cache.RemoveReplicaSet(tt.pod); err != nil {
			t.Fatalf("RemoveReplicaSet failed: %v", err)
		}

		n = cache.clusters[clusterName]
		if n != nil {
			t.Errorf("#%d: expecting pod deleted and nil cluster info, get=%s", i, n)
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
		cache.List(labels.Everything())
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

func benchmarkExpire(b *testing.B, podNum int) {
	now := time.Now()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		cache := setupCacheWithAssumedReplicaSets(b, podNum, now)
		b.StartTimer()
		cache.cleanupAssumedReplicaSets(now.Add(2 * time.Second))
	}
}

func makeBaseReplicaSet(clusterName, objName, cpu, mem string, ports []api.ContainerPort) *extensions.ReplicaSet {
	req := api.ResourceList{}
	if cpu != "" {
		req = api.ResourceList{
			api.ResourceCPU:    resource.MustParse(cpu),
			api.ResourceMemory: resource.MustParse(mem),
		}
	}
	return &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Namespace: "cluster_info_cache_test",
			Name:      objName,
		},
		Spec: extensions.ReplicaSet{
			Spec: extensions.ReplicaSetSpec{
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						Containers: []api.Container{{
							Resources: api.ResourceRequirements{
								Requests: req,
							},
							Ports: ports,
						}},
					},
				},
			},
		},
	}
}

func setupCacheOf1kNodes30kReplicaSets(b *testing.B) Cache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < 1000; i++ {
		clusterName := fmt.Sprintf("cluster-%d", i)
		for j := 0; j < 30; j++ {
			objName := fmt.Sprintf("%s-pod-%d", clusterName, j)
			pod := makeBaseReplicaSet(clusterName, objName, "0", "0", nil)

			if err := cache.AddReplicaSet(pod); err != nil {
				b.Fatalf("AddReplicaSet failed: %v", err)
			}
		}
	}
	return cache
}

func setupCacheWithAssumedReplicaSets(b *testing.B, podNum int, assumedTime time.Time) *schedulerCache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < podNum; i++ {
		clusterName := fmt.Sprintf("cluster-%d", i/10)
		objName := fmt.Sprintf("%s-pod-%d", clusterName, i%10)
		pod := makeBaseReplicaSet(clusterName, objName, "0", "0", nil)

		err := cache.assumeReplicaSetIfBindSucceed(pod, alwaysTrue, assumedTime)
		if err != nil {
			b.Fatalf("assumeReplicaSetIfBindSucceed failed: %v", err)
		}
	}
	return cache
}

func alwaysTrue() bool {
	return true
}
