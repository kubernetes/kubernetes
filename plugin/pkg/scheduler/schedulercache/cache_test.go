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
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

// TestAssumePodScheduled tests that after a pod is assumed, its information is aggregated
// on node level.
func TestAssumePodScheduled(t *testing.T) {
	nodeName := "node"
	testPods := []*api.Pod{
		makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBasePod(nodeName, "test-1", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBasePod(nodeName, "test-2", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
	}

	tests := []struct {
		pods []*api.Pod

		wNodeInfo *NodeInfo
	}{{
		pods: []*api.Pod{testPods[0]},
		wNodeInfo: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			pods: []*api.Pod{testPods[0]},
		},
	}, {
		pods: []*api.Pod{testPods[1], testPods[2]},
		wNodeInfo: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			pods: []*api.Pod{testPods[1], testPods[2]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		for _, pod := range tt.pods {
			err := cache.AssumePodIfBindSucceed(pod, alwaysTrue)
			if err != nil {
				t.Fatalf("AssumePodScheduled failed: %v", err)
			}
		}
		n := cache.nodes[nodeName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestExpirePod tests that assumed pods will be removed if expired.
// The removal will be reflected in node info.
func TestExpirePod(t *testing.T) {
	nodeName := "node"
	testPods := []*api.Pod{
		makeBasePod(nodeName, "test-1", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBasePod(nodeName, "test-2", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
	}
	now := time.Now()
	ttl := 10 * time.Second
	tests := []struct {
		pods        []*api.Pod
		assumedTime []time.Time // assume time for individual pods
		cleanupTime time.Time

		wNodeInfo *NodeInfo
	}{{ // assumed pod would expires
		pods:        []*api.Pod{makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})},
		assumedTime: []time.Time{now},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo:   nil,
	}, { // first one would expire, second one would not.
		pods:        []*api.Pod{testPods[0], testPods[1]},
		assumedTime: []time.Time{now, now.Add(3 * ttl / 2)},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			pods: []*api.Pod{testPods[1]},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)

		for j, pod := range tt.pods {
			err := cache.assumePodIfBindSucceed(pod, alwaysTrue, tt.assumedTime[j])
			if err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
		}
		// pods that have assumedTime + ttl < cleanupTime will get expired and removed
		cache.cleanupAssumedPods(tt.cleanupTime)
		n := cache.nodes[nodeName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestAddPodWillConfirm tests that a pod being Add()ed will be confirmed if assumed.
// The pod info should still exist after manually expiring unconfirmed pods.
func TestAddPodWillConfirm(t *testing.T) {
	nodeName := "node"
	ttl := 10 * time.Second
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		podToAssume *api.Pod
		podToAdd    *api.Pod

		wNodeInfo *NodeInfo
	}{{ // Pod is assumed. It should be confirmed.
		podToAssume: basePod,
		podToAdd:    basePod,
		wNodeInfo: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			pods: []*api.Pod{basePod},
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		err := cache.assumePodIfBindSucceed(tt.podToAssume, alwaysTrue, now)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		err = cache.AddPod(tt.podToAdd)
		if err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))
		// check after expiration. confirmed pods shouldn't be expired.
		n := cache.nodes[nodeName]
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

		wNodeInfo *NodeInfo
	}{{
		podToAssume: basePod,
		podToAdd:    basePod,
		wNodeInfo: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			pods: []*api.Pod{basePod},
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		err := cache.assumePodIfBindSucceed(tt.podToAssume, alwaysTrue, now)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))
		// It should be expired and removed.
		n := cache.nodes[nodeName]
		if n != nil {
			t.Errorf("#%d: expecting nil node info, but get=%v", i, n)
		}
		err = cache.AddPod(tt.podToAdd)
		if err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		// check after expiration. confirmed pods shouldn't be expired.
		n = cache.nodes[nodeName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestUpdatePod tests that a pod will be updated if added before.
func TestUpdatePod(t *testing.T) {
	nodeName := "node"
	ttl := 10 * time.Second
	testPods := []*api.Pod{
		makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}}),
		makeBasePod(nodeName, "test", "200m", "1Ki", []api.ContainerPort{{HostPort: 8080}}),
	}
	tests := []struct {
		podToAssume  *api.Pod
		podToAdd     *api.Pod
		podToUpdate  *api.Pod
		podToUpdate2 *api.Pod

		wNodeInfo  *NodeInfo
		wNodeInfo2 *NodeInfo
	}{{ // Pod is assumed. It should be confirmed. Then it would be updated.
		podToAssume:  testPods[0],
		podToAdd:     testPods[0],
		podToUpdate:  testPods[1],
		podToUpdate2: testPods[0],
		wNodeInfo: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			pods: []*api.Pod{testPods[1]},
		},
		wNodeInfo2: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			pods: []*api.Pod{testPods[0]},
		},
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		err := cache.assumePodIfBindSucceed(tt.podToAssume, alwaysTrue, now)
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
		n := cache.nodes[nodeName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}

		// unlike Add, Update can happen multiple times
		err = cache.UpdatePod(tt.podToUpdate, tt.podToUpdate2)
		if err != nil {
			t.Fatalf("UpdatePod failed: %v", err)
		}
		// check after expiration. confirmed pods shouldn't be expired.
		n = cache.nodes[nodeName]
		if !reflect.DeepEqual(n, tt.wNodeInfo2) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}
	}
}

// TestRemovePod tests after added pod is removed, its information should also be subtracted.
func TestRemovePod(t *testing.T) {
	nodeName := "node"
	basePod := makeBasePod(nodeName, "test", "100m", "500", []api.ContainerPort{{HostPort: 80}})
	tests := []struct {
		pod *api.Pod

		wNodeInfo *NodeInfo
	}{{
		pod: basePod,
		wNodeInfo: &NodeInfo{
			requestedResource: &Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			pods: []*api.Pod{basePod},
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		err := cache.AssumePodIfBindSucceed(tt.pod, alwaysTrue)
		if err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		err = cache.AddPod(tt.pod)
		if err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		n := cache.nodes[nodeName]
		if !reflect.DeepEqual(n, tt.wNodeInfo) {
			t.Errorf("#%d: node info get=%s, want=%s", i, n, tt.wNodeInfo)
		}

		err = cache.RemovePod(tt.pod)
		if err != nil {
			t.Fatalf("RemovePod failed: %v", err)
		}

		n = cache.nodes[nodeName]
		if n != nil {
			t.Errorf("#%d: expecting pod deleted and nil node info, get=%s", i, n)
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

func alwaysTrue() bool {
	return true
}
