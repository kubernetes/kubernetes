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

package nodeinfocache

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// TestAssumePod tests that after a pod is assumed, its information is aggregated
// on node level.
func TestAssumePod(t *testing.T) {
	nodeName := "node"
	tests := []struct {
		pods []*api.Pod

		wNodeInfo *schedulercache.NodeInfo
	}{{
		pods: []*api.Pod{makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})},
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			PodNum: 1,
		},
	}, {
		pods: []*api.Pod{
			makeBasePod(nodeName, "test-1", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
			makeBasePod(nodeName, "test-2", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}})},
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			PodNum: 2,
		},
	}}

	for i, tt := range tests {
		cache := newNodeInfoCache(time.Second, time.Second, nil)
		for _, pod := range tt.pods {
			err := cache.AssumePod(pod)
			if err != nil {
				t.Fatalf("AssumePod failed: %v", err)
			}
		}
		n := cache.getNodeInfo(nodeName)
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestExpirePod tests that assumed pods will be removed if expired.
// The removal will be reflected in node info.
func TestExpirePod(t *testing.T) {
	nodeName := "node"
	now := time.Now()
	ttl := 10 * time.Second
	tests := []struct {
		pods        []*api.Pod
		assumedTime []time.Time // assume time for individual pods
		cleanupTime time.Time

		wNodeInfo *schedulercache.NodeInfo
	}{{ // assumed pod would expires
		pods:        []*api.Pod{makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})},
		assumedTime: []time.Time{now},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo:   nil,
	}, { // first one would expire, second one would not.
		pods: []*api.Pod{
			makeBasePod(nodeName, "test-1", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
			makeBasePod(nodeName, "test-2", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}})},
		assumedTime: []time.Time{now, now.Add(3 * ttl / 2)},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			PodNum: 1,
		},
	}}

	for i, tt := range tests {
		cache := newNodeInfoCache(ttl, time.Second, nil)

		for j, pod := range tt.pods {
			err := cache.assumePod(pod, tt.assumedTime[j])
			if err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
		}
		// pods that have assumedTime + ttl < cleanupTime will get expired and removed
		cache.cleanupAssumedPods(tt.cleanupTime)
		n := cache.getNodeInfo(nodeName)
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestAddPodWillConfirm tests that a pod being Add()ed will be confirmed if binded.
// The pod info should still exist after manually expiring unconfirmed pods.
func TestAddPodWillConfirm(t *testing.T) {
	nodeName := "node"
	ttl := 10 * time.Second
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		podToAssume *api.Pod
		podToAdd    *api.Pod

		wNodeInfo *schedulercache.NodeInfo
	}{{ // Pod is binded. It should be confirmed.
		podToAssume: basePod,
		podToAdd:    basePod,
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			PodNum: 1,
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newNodeInfoCache(ttl, time.Second, nil)
		err := cache.assumePod(tt.podToAssume, now)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		err = cache.AddPod(tt.podToAdd)
		if err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))
		// check after expiration. confirmed pods shouldn't be expired.
		n := cache.getNodeInfo(nodeName)
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestAddPodAfterExpiration tests that a pod being Add()ed will be added back if expired.
func TestAddPodAfterExpiration(t *testing.T) {
	nodeName := "node"
	ttl := 10 * time.Second
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		podToAssume *api.Pod
		podToAdd    *api.Pod

		wNodeInfo *schedulercache.NodeInfo
	}{{
		podToAssume: basePod,
		podToAdd:    basePod,
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			PodNum: 1,
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newNodeInfoCache(ttl, time.Second, nil)
		err := cache.assumePod(tt.podToAssume, now)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))
		// It should be expired and removed.
		n := cache.getNodeInfo(nodeName)
		if n != nil {
			t.Errorf("#%d: expecting nil node info, but get=%v", i, n)
		}
		err = cache.AddPod(tt.podToAdd)
		if err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		// check after expiration. confirmed pods shouldn't be expired.
		n = cache.getNodeInfo(nodeName)
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestUpdatePod tests that a pod will be updated if added before.
func TestUpdatePod(t *testing.T) {
	nodeName := "node"
	ttl := 10 * time.Second
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		podToAssume *api.Pod
		podToAdd    *api.Pod
		podToUpdate *api.Pod

		wNodeInfo *schedulercache.NodeInfo
	}{{ // Pod is binded. It should be confirmed.
		podToAssume: basePod,
		podToAdd:    basePod,
		podToUpdate: makeBasePod(nodeName, "test", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			PodNum: 1,
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newNodeInfoCache(ttl, time.Second, nil)
		err := cache.assumePod(tt.podToAssume, now)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		err = cache.AddPod(tt.podToAdd)
		if err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		err = cache.UpdatePod(tt.podToAdd, tt.podToUpdate)
		if err != nil {
			t.Fatalf("UpdatePod failed: %v", err)
		}
		// check after expiration. confirmed pods shouldn't be expired.
		n := cache.getNodeInfo(nodeName)
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestRemoveBindedPod tests after binded pod is removed, its information should also be subtracted.
func TestRemoveBindedPod(t *testing.T) {
	nodeName := "node"
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		pod *api.Pod

		wNodeInfo *schedulercache.NodeInfo
	}{{
		pod: basePod,
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			PodNum: 1,
		},
	}}

	for i, tt := range tests {
		cache := newNodeInfoCache(time.Second, time.Second, nil)
		err := cache.AssumePod(tt.pod)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		n := cache.getNodeInfo(nodeName)
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}

		err = cache.RemovePod(tt.pod)
		if err != nil {
			t.Fatalf("RemovePod failed: %v", err)
		}

		n = cache.getNodeInfo(nodeName)
		if n != nil {
			t.Errorf("#%d: expecting pod deleted and nil node info, get=%s", i, n)
		}
	}
}

// TestRemoveBindedPod tests after added pod is removed, its information should also be subtracted.
func TestRemoveAddedPod(t *testing.T) {
	nodeName := "node"
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		pod *api.Pod

		wNodeInfo *schedulercache.NodeInfo
	}{{
		pod: basePod,
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			PodNum: 1,
		},
	}}

	for i, tt := range tests {
		cache := newNodeInfoCache(time.Second, time.Second, nil)
		err := cache.AssumePod(tt.pod)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		err = cache.AddPod(tt.pod)
		if err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		n := cache.getNodeInfo(nodeName)
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}

		err = cache.RemovePod(tt.pod)
		if err != nil {
			t.Fatalf("RemovePod failed: %v", err)
		}

		n = cache.getNodeInfo(nodeName)
		if n != nil {
			t.Errorf("#%d: expecting pod deleted and nil node info, get=%s", i, n)
		}
	}
}

// TestRemoveExpiredPod tests that removing expired pod shouldn't have any error.
func TestRemoveExpiredPod(t *testing.T) {
	nodeName := "node"
	ttl := 10 * time.Second
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		pod       *api.Pod
		wNodeInfo *schedulercache.NodeInfo
	}{{
		pod: basePod,
		wNodeInfo: &schedulercache.NodeInfo{
			RequestedResource: &schedulercache.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			PodNum: 1,
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newNodeInfoCache(ttl, time.Second, nil)
		err := cache.assumePod(tt.pod, now)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))

		n := cache.getNodeInfo(nodeName)
		if n != nil {
			t.Errorf("#%d: expecting pod expired and nil node info, get=%s", i, n)
			continue
		}

		// no error should happen
		err = cache.RemovePod(tt.pod)
		if err != nil {
			t.Fatalf("RemovePod failed: %v", err)
		}
	}
}

func makeBasePod(nodeName, objName, cpu, mem string, ports []api.ContainerPort) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: "node_info_cache_test",
			Name:      objName,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceCPU:    resource.MustParse(cpu),
						api.ResourceMemory: resource.MustParse(mem),
					},
				},
				Ports: ports,
			}},
			NodeName: nodeName,
		},
	}
}
