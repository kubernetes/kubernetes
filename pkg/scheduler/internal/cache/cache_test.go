/*
Copyright 2015 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func deepEqualWithoutGeneration(t *testing.T, testcase int, actual *nodeInfoListItem, expected *schedulernodeinfo.NodeInfo) {
	if (actual == nil) != (expected == nil) {
		t.Error("One of the actual or expeted is nil and the other is not!")
	}
	// Ignore generation field.
	if actual != nil {
		actual.info.SetGeneration(0)
	}
	if expected != nil {
		expected.SetGeneration(0)
	}
	if actual != nil && !reflect.DeepEqual(actual.info, expected) {
		t.Errorf("#%d: node info get=%s, want=%s", testcase, actual.info, expected)
	}
}

type hostPortInfoParam struct {
	protocol, ip string
	port         int32
}

type hostPortInfoBuilder struct {
	inputs []hostPortInfoParam
}

func newHostPortInfoBuilder() *hostPortInfoBuilder {
	return &hostPortInfoBuilder{}
}

func (b *hostPortInfoBuilder) add(protocol, ip string, port int32) *hostPortInfoBuilder {
	b.inputs = append(b.inputs, hostPortInfoParam{protocol, ip, port})
	return b
}

func (b *hostPortInfoBuilder) build() schedulernodeinfo.HostPortInfo {
	res := make(schedulernodeinfo.HostPortInfo)
	for _, param := range b.inputs {
		res.Add(param.ip, param.protocol, param.port)
	}
	return res
}

func newNodeInfo(requestedResource *schedulernodeinfo.Resource,
	nonzeroRequest *schedulernodeinfo.Resource,
	pods []*v1.Pod,
	usedPorts schedulernodeinfo.HostPortInfo,
	imageStates map[string]*schedulernodeinfo.ImageStateSummary,
) *schedulernodeinfo.NodeInfo {
	nodeInfo := schedulernodeinfo.NewNodeInfo(pods...)
	nodeInfo.SetRequestedResource(requestedResource)
	nodeInfo.SetNonZeroRequest(nonzeroRequest)
	nodeInfo.SetUsedPorts(usedPorts)
	nodeInfo.SetImageStates(imageStates)

	return nodeInfo
}

// TestAssumePodScheduled tests that after a pod is assumed, its information is aggregated
// on node level.
func TestAssumePodScheduled(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-1", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-2", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-nonzero", "", "", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test", "100m", "500", "example.com/foo:3", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-2", "200m", "1Ki", "example.com/foo:5", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test", "100m", "500", "random-invalid-extended-key:100", []v1.ContainerPort{{}}),
	}

	tests := []struct {
		pods []*v1.Pod

		wNodeInfo *schedulernodeinfo.NodeInfo
	}{{
		pods: []*v1.Pod{testPods[0]},
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[1], testPods[2]},
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			[]*v1.Pod{testPods[1], testPods[2]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}, { // test non-zero request
		pods: []*v1.Pod{testPods[3]},
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 0,
				Memory:   0,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: priorityutil.DefaultMilliCPURequest,
				Memory:   priorityutil.DefaultMemoryRequest,
			},
			[]*v1.Pod{testPods[3]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[4]},
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU:        100,
				Memory:          500,
				ScalarResources: map[v1.ResourceName]int64{"example.com/foo": 3},
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[4]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[4], testPods[5]},
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU:        300,
				Memory:          1524,
				ScalarResources: map[v1.ResourceName]int64{"example.com/foo": 8},
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			[]*v1.Pod{testPods[4], testPods[5]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[6]},
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[6]},
			newHostPortInfoBuilder().build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	},
	}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		for _, pod := range tt.pods {
			if err := cache.AssumePod(pod); err != nil {
				t.Fatalf("AssumePod failed: %v", err)
			}
		}
		n := cache.nodes[nodeName]
		deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo)

		for _, pod := range tt.pods {
			if err := cache.ForgetPod(pod); err != nil {
				t.Fatalf("ForgetPod failed: %v", err)
			}
		}
		if cache.nodes[nodeName] != nil {
			t.Errorf("NodeInfo should be cleaned for %s", nodeName)
		}
	}
}

type testExpirePodStruct struct {
	pod         *v1.Pod
	assumedTime time.Time
}

func assumeAndFinishBinding(cache *schedulerCache, pod *v1.Pod, assumedTime time.Time) error {
	if err := cache.AssumePod(pod); err != nil {
		return err
	}
	return cache.finishBinding(pod, assumedTime)
}

// TestExpirePod tests that assumed pods will be removed if expired.
// The removal will be reflected in node info.
func TestExpirePod(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test-1", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-2", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	now := time.Now()
	ttl := 10 * time.Second
	tests := []struct {
		pods        []*testExpirePodStruct
		cleanupTime time.Time

		wNodeInfo *schedulernodeinfo.NodeInfo
	}{{ // assumed pod would expires
		pods: []*testExpirePodStruct{
			{pod: testPods[0], assumedTime: now},
		},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo:   nil,
	}, { // first one would expire, second one would not.
		pods: []*testExpirePodStruct{
			{pod: testPods[0], assumedTime: now},
			{pod: testPods[1], assumedTime: now.Add(3 * ttl / 2)},
		},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			[]*v1.Pod{testPods[1]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)

		for _, pod := range tt.pods {
			if err := assumeAndFinishBinding(cache, pod.pod, pod.assumedTime); err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
		}
		// pods that have assumedTime + ttl < cleanupTime will get expired and removed
		cache.cleanupAssumedPods(tt.cleanupTime)
		n := cache.nodes[nodeName]
		deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo)
	}
}

// TestAddPodWillConfirm tests that a pod being Add()ed will be confirmed if assumed.
// The pod info should still exist after manually expiring unconfirmed pods.
func TestAddPodWillConfirm(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	now := time.Now()
	ttl := 10 * time.Second

	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test-1", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-2", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	tests := []struct {
		podsToAssume []*v1.Pod
		podsToAdd    []*v1.Pod

		wNodeInfo *schedulernodeinfo.NodeInfo
	}{{ // two pod were assumed at same time. But first one is called Add() and gets confirmed.
		podsToAssume: []*v1.Pod{testPods[0], testPods[1]},
		podsToAdd:    []*v1.Pod{testPods[0]},
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAssume := range tt.podsToAssume {
			if err := assumeAndFinishBinding(cache, podToAssume, now); err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
		}
		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddPod(podToAdd); err != nil {
				t.Fatalf("AddPod failed: %v", err)
			}
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))
		// check after expiration. confirmed pods shouldn't be expired.
		n := cache.nodes[nodeName]
		deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo)
	}
}

func TestSnapshot(t *testing.T) {
	nodeName := "node"
	now := time.Now()
	ttl := 10 * time.Second

	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test-1", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-2", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
	}
	tests := []struct {
		podsToAssume []*v1.Pod
		podsToAdd    []*v1.Pod
	}{{ // two pod were assumed at same time. But first one is called Add() and gets confirmed.
		podsToAssume: []*v1.Pod{testPods[0], testPods[1]},
		podsToAdd:    []*v1.Pod{testPods[0]},
	}}

	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAssume := range tt.podsToAssume {
			if err := assumeAndFinishBinding(cache, podToAssume, now); err != nil {
				t.Errorf("assumePod failed: %v", err)
			}
		}
		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddPod(podToAdd); err != nil {
				t.Errorf("AddPod failed: %v", err)
			}
		}

		snapshot := cache.Snapshot()
		if len(snapshot.Nodes) != len(cache.nodes) {
			t.Errorf("Unequal number of nodes in the cache and its snapshot. expeted: %v, got: %v", len(cache.nodes), len(snapshot.Nodes))
		}
		for name, ni := range snapshot.Nodes {
			nItem := cache.nodes[name]
			if !reflect.DeepEqual(ni, nItem.info) {
				t.Errorf("expect \n%+v; got \n%+v", nItem.info, ni)
			}
		}
		if !reflect.DeepEqual(snapshot.AssumedPods, cache.assumedPods) {
			t.Errorf("expect \n%+v; got \n%+v", cache.assumedPods, snapshot.AssumedPods)
		}
	}
}

// TestAddPodWillReplaceAssumed tests that a pod being Add()ed will replace any assumed pod.
func TestAddPodWillReplaceAssumed(t *testing.T) {
	now := time.Now()
	ttl := 10 * time.Second

	assumedPod := makeBasePod(t, "assumed-node-1", "test-1", "100m", "500", "", []v1.ContainerPort{{HostPort: 80}})
	addedPod := makeBasePod(t, "actual-node", "test-1", "100m", "500", "", []v1.ContainerPort{{HostPort: 80}})
	updatedPod := makeBasePod(t, "actual-node", "test-1", "200m", "500", "", []v1.ContainerPort{{HostPort: 90}})

	tests := []struct {
		podsToAssume []*v1.Pod
		podsToAdd    []*v1.Pod
		podsToUpdate [][]*v1.Pod

		wNodeInfo map[string]*schedulernodeinfo.NodeInfo
	}{{
		podsToAssume: []*v1.Pod{assumedPod.DeepCopy()},
		podsToAdd:    []*v1.Pod{addedPod.DeepCopy()},
		podsToUpdate: [][]*v1.Pod{{addedPod.DeepCopy(), updatedPod.DeepCopy()}},
		wNodeInfo: map[string]*schedulernodeinfo.NodeInfo{
			"assumed-node": nil,
			"actual-node": newNodeInfo(
				&schedulernodeinfo.Resource{
					MilliCPU: 200,
					Memory:   500,
				},
				&schedulernodeinfo.Resource{
					MilliCPU: 200,
					Memory:   500,
				},
				[]*v1.Pod{updatedPod.DeepCopy()},
				newHostPortInfoBuilder().add("TCP", "0.0.0.0", 90).build(),
				make(map[string]*schedulernodeinfo.ImageStateSummary),
			),
		},
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAssume := range tt.podsToAssume {
			if err := assumeAndFinishBinding(cache, podToAssume, now); err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
		}
		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddPod(podToAdd); err != nil {
				t.Fatalf("AddPod failed: %v", err)
			}
		}
		for _, podToUpdate := range tt.podsToUpdate {
			if err := cache.UpdatePod(podToUpdate[0], podToUpdate[1]); err != nil {
				t.Fatalf("UpdatePod failed: %v", err)
			}
		}
		for nodeName, expected := range tt.wNodeInfo {
			t.Log(nodeName)
			n := cache.nodes[nodeName]
			deepEqualWithoutGeneration(t, i, n, expected)
		}
	}
}

// TestAddPodAfterExpiration tests that a pod being Add()ed will be added back if expired.
func TestAddPodAfterExpiration(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	ttl := 10 * time.Second
	basePod := makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}})
	tests := []struct {
		pod *v1.Pod

		wNodeInfo *schedulernodeinfo.NodeInfo
	}{{
		pod: basePod,
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{basePod},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}}

	now := time.Now()
	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		if err := assumeAndFinishBinding(cache, tt.pod, now); err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))
		// It should be expired and removed.
		n := cache.nodes[nodeName]
		if n != nil {
			t.Errorf("#%d: expecting nil node info, but get=%v", i, n)
		}
		if err := cache.AddPod(tt.pod); err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		// check after expiration. confirmed pods shouldn't be expired.
		n = cache.nodes[nodeName]
		deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo)
	}
}

// TestUpdatePod tests that a pod will be updated if added before.
func TestUpdatePod(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	ttl := 10 * time.Second
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	tests := []struct {
		podsToAdd    []*v1.Pod
		podsToUpdate []*v1.Pod

		wNodeInfo []*schedulernodeinfo.NodeInfo
	}{{ // add a pod and then update it twice
		podsToAdd:    []*v1.Pod{testPods[0]},
		podsToUpdate: []*v1.Pod{testPods[0], testPods[1], testPods[0]},
		wNodeInfo: []*schedulernodeinfo.NodeInfo{newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			[]*v1.Pod{testPods[1]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		), newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		)},
	}}

	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddPod(podToAdd); err != nil {
				t.Fatalf("AddPod failed: %v", err)
			}
		}

		for i := range tt.podsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdatePod(tt.podsToUpdate[i-1], tt.podsToUpdate[i]); err != nil {
				t.Fatalf("UpdatePod failed: %v", err)
			}
			// check after expiration. confirmed pods shouldn't be expired.
			n := cache.nodes[nodeName]
			deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo[i-1])
		}
	}
}

// TestUpdatePodAndGet tests get always return latest pod state
func TestUpdatePodAndGet(t *testing.T) {
	nodeName := "node"
	ttl := 10 * time.Second
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	tests := []struct {
		pod *v1.Pod

		podToUpdate *v1.Pod
		handler     func(cache Cache, pod *v1.Pod) error

		assumePod bool
	}{
		{
			pod: testPods[0],

			podToUpdate: testPods[0],
			handler: func(cache Cache, pod *v1.Pod) error {
				return cache.AssumePod(pod)
			},
			assumePod: true,
		},
		{
			pod: testPods[0],

			podToUpdate: testPods[1],
			handler: func(cache Cache, pod *v1.Pod) error {
				return cache.AddPod(pod)
			},
			assumePod: false,
		},
	}

	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)

		if err := tt.handler(cache, tt.pod); err != nil {
			t.Fatalf("unexpected err: %v", err)
		}

		if !tt.assumePod {
			if err := cache.UpdatePod(tt.pod, tt.podToUpdate); err != nil {
				t.Fatalf("UpdatePod failed: %v", err)
			}
		}

		cachedPod, err := cache.GetPod(tt.pod)
		if err != nil {
			t.Fatalf("GetPod failed: %v", err)
		}
		if !reflect.DeepEqual(tt.podToUpdate, cachedPod) {
			t.Fatalf("pod get=%s, want=%s", cachedPod, tt.podToUpdate)
		}
	}
}

// TestExpireAddUpdatePod test the sequence that a pod is expired, added, then updated
func TestExpireAddUpdatePod(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	ttl := 10 * time.Second
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	tests := []struct {
		podsToAssume []*v1.Pod
		podsToAdd    []*v1.Pod
		podsToUpdate []*v1.Pod

		wNodeInfo []*schedulernodeinfo.NodeInfo
	}{{ // Pod is assumed, expired, and added. Then it would be updated twice.
		podsToAssume: []*v1.Pod{testPods[0]},
		podsToAdd:    []*v1.Pod{testPods[0]},
		podsToUpdate: []*v1.Pod{testPods[0], testPods[1], testPods[0]},
		wNodeInfo: []*schedulernodeinfo.NodeInfo{newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			[]*v1.Pod{testPods[1]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		), newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		)},
	}}

	now := time.Now()
	for _, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, podToAssume := range tt.podsToAssume {
			if err := assumeAndFinishBinding(cache, podToAssume, now); err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))

		for _, podToAdd := range tt.podsToAdd {
			if err := cache.AddPod(podToAdd); err != nil {
				t.Fatalf("AddPod failed: %v", err)
			}
		}

		for i := range tt.podsToUpdate {
			if i == 0 {
				continue
			}
			if err := cache.UpdatePod(tt.podsToUpdate[i-1], tt.podsToUpdate[i]); err != nil {
				t.Fatalf("UpdatePod failed: %v", err)
			}
			// check after expiration. confirmed pods shouldn't be expired.
			n := cache.nodes[nodeName]
			deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo[i-1])
		}
	}
}

func makePodWithEphemeralStorage(nodeName, ephemeralStorage string) *v1.Pod {
	req := v1.ResourceList{
		v1.ResourceEphemeralStorage: resource.MustParse(ephemeralStorage),
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default-namespace",
			Name:      "pod-with-ephemeral-storage",
			UID:       types.UID("pod-with-ephemeral-storage"),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Resources: v1.ResourceRequirements{
					Requests: req,
				},
			}},
			NodeName: nodeName,
		},
	}
}

func TestEphemeralStorageResource(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	podE := makePodWithEphemeralStorage(nodeName, "500")
	tests := []struct {
		pod       *v1.Pod
		wNodeInfo *schedulernodeinfo.NodeInfo
	}{
		{
			pod: podE,
			wNodeInfo: newNodeInfo(
				&schedulernodeinfo.Resource{
					EphemeralStorage: 500,
				},
				&schedulernodeinfo.Resource{
					MilliCPU: priorityutil.DefaultMilliCPURequest,
					Memory:   priorityutil.DefaultMemoryRequest,
				},
				[]*v1.Pod{podE},
				schedulernodeinfo.HostPortInfo{},
				make(map[string]*schedulernodeinfo.ImageStateSummary),
			),
		},
	}
	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		if err := cache.AddPod(tt.pod); err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		n := cache.nodes[nodeName]
		deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo)

		if err := cache.RemovePod(tt.pod); err != nil {
			t.Fatalf("RemovePod failed: %v", err)
		}

		n = cache.nodes[nodeName]
		if n != nil {
			t.Errorf("#%d: expecting pod deleted and nil node info, get=%s", i, n.info)
		}
	}
}

// TestRemovePod tests after added pod is removed, its information should also be subtracted.
func TestRemovePod(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	nodeName := "node"
	basePod := makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}})
	tests := []struct {
		pod       *v1.Pod
		wNodeInfo *schedulernodeinfo.NodeInfo
	}{{
		pod: basePod,
		wNodeInfo: newNodeInfo(
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&schedulernodeinfo.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{basePod},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*schedulernodeinfo.ImageStateSummary),
		),
	}}

	for i, tt := range tests {
		cache := newSchedulerCache(time.Second, time.Second, nil)
		if err := cache.AddPod(tt.pod); err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		n := cache.nodes[nodeName]
		deepEqualWithoutGeneration(t, i, n, tt.wNodeInfo)

		if err := cache.RemovePod(tt.pod); err != nil {
			t.Fatalf("RemovePod failed: %v", err)
		}

		n = cache.nodes[nodeName]
		if n != nil {
			t.Errorf("#%d: expecting pod deleted and nil node info, get=%s", i, n.info)
		}
	}
}

func TestForgetPod(t *testing.T) {
	nodeName := "node"
	basePod := makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}})
	tests := []struct {
		pods []*v1.Pod
	}{{
		pods: []*v1.Pod{basePod},
	}}
	now := time.Now()
	ttl := 10 * time.Second

	for i, tt := range tests {
		cache := newSchedulerCache(ttl, time.Second, nil)
		for _, pod := range tt.pods {
			if err := assumeAndFinishBinding(cache, pod, now); err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
			isAssumed, err := cache.IsAssumedPod(pod)
			if err != nil {
				t.Fatalf("IsAssumedPod failed: %v.", err)
			}
			if !isAssumed {
				t.Fatalf("Pod is expected to be assumed.")
			}
			assumedPod, err := cache.GetPod(pod)
			if err != nil {
				t.Fatalf("GetPod failed: %v.", err)
			}
			if assumedPod.Namespace != pod.Namespace {
				t.Errorf("assumedPod.Namespace != pod.Namespace (%s != %s)", assumedPod.Namespace, pod.Namespace)
			}
			if assumedPod.Name != pod.Name {
				t.Errorf("assumedPod.Name != pod.Name (%s != %s)", assumedPod.Name, pod.Name)
			}
		}
		for _, pod := range tt.pods {
			if err := cache.ForgetPod(pod); err != nil {
				t.Fatalf("ForgetPod failed: %v", err)
			}
			isAssumed, err := cache.IsAssumedPod(pod)
			if err != nil {
				t.Fatalf("IsAssumedPod failed: %v.", err)
			}
			if isAssumed {
				t.Fatalf("Pod is expected to be unassumed.")
			}
		}
		cache.cleanupAssumedPods(now.Add(2 * ttl))
		if n := cache.nodes[nodeName]; n != nil {
			t.Errorf("#%d: expecting pod deleted and nil node info, get=%s", i, n.info)
		}
	}
}

// getResourceRequest returns the resource request of all containers in Pods;
// excluding initContainers.
func getResourceRequest(pod *v1.Pod) v1.ResourceList {
	result := &schedulernodeinfo.Resource{}
	for _, container := range pod.Spec.Containers {
		result.Add(container.Resources.Requests)
	}

	return result.ResourceList()
}

// buildNodeInfo creates a NodeInfo by simulating node operations in cache.
func buildNodeInfo(node *v1.Node, pods []*v1.Pod) *schedulernodeinfo.NodeInfo {
	expected := schedulernodeinfo.NewNodeInfo()

	// Simulate SetNode.
	expected.SetNode(node)

	expected.SetAllocatableResource(schedulernodeinfo.NewResource(node.Status.Allocatable))
	expected.SetTaints(node.Spec.Taints)
	expected.SetGeneration(expected.GetGeneration() + 1)

	for _, pod := range pods {
		// Simulate AddPod
		pods := append(expected.Pods(), pod)
		expected.SetPods(pods)
		requestedResource := expected.RequestedResource()
		newRequestedResource := &requestedResource
		newRequestedResource.Add(getResourceRequest(pod))
		expected.SetRequestedResource(newRequestedResource)
		nonZeroRequest := expected.NonZeroRequest()
		newNonZeroRequest := &nonZeroRequest
		newNonZeroRequest.Add(getResourceRequest(pod))
		expected.SetNonZeroRequest(newNonZeroRequest)
		expected.UpdateUsedPorts(pod, true)
		expected.SetGeneration(expected.GetGeneration() + 1)
	}

	return expected
}

// TestNodeOperators tests node operations of cache, including add, update
// and remove.
func TestNodeOperators(t *testing.T) {
	// Test datas
	nodeName := "test-node"
	cpu1 := resource.MustParse("1000m")
	mem100m := resource.MustParse("100m")
	cpuHalf := resource.MustParse("500m")
	mem50m := resource.MustParse("50m")
	resourceFooName := "example.com/foo"
	resourceFoo := resource.MustParse("1")

	tests := []struct {
		node *v1.Node
		pods []*v1.Pod
	}{
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
				},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:                   cpu1,
						v1.ResourceMemory:                mem100m,
						v1.ResourceName(resourceFooName): resourceFoo,
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "test-key",
							Value:  "test-value",
							Effect: v1.TaintEffectPreferNoSchedule,
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod1",
						UID:  types.UID("pod1"),
					},
					Spec: v1.PodSpec{
						NodeName: nodeName,
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    cpuHalf,
										v1.ResourceMemory: mem50m,
									},
								},
								Ports: []v1.ContainerPort{
									{
										Name:          "http",
										HostPort:      80,
										ContainerPort: 80,
									},
								},
							},
						},
					},
				},
			},
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
				},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:                   cpu1,
						v1.ResourceMemory:                mem100m,
						v1.ResourceName(resourceFooName): resourceFoo,
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "test-key",
							Value:  "test-value",
							Effect: v1.TaintEffectPreferNoSchedule,
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod1",
						UID:  types.UID("pod1"),
					},
					Spec: v1.PodSpec{
						NodeName: nodeName,
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    cpuHalf,
										v1.ResourceMemory: mem50m,
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod2",
						UID:  types.UID("pod2"),
					},
					Spec: v1.PodSpec{
						NodeName: nodeName,
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    cpuHalf,
										v1.ResourceMemory: mem50m,
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		expected := buildNodeInfo(test.node, test.pods)
		node := test.node

		cache := newSchedulerCache(time.Second, time.Second, nil)
		cache.AddNode(node)
		for _, pod := range test.pods {
			cache.AddPod(pod)
		}

		// Case 1: the node was added into cache successfully.
		got, found := cache.nodes[node.Name]
		if !found {
			t.Errorf("Failed to find node %v in internalcache.", node.Name)
		}
		if cache.nodeTree.NumNodes() != 1 || cache.nodeTree.Next() != node.Name {
			t.Errorf("cache.nodeTree is not updated correctly after adding node: %v", node.Name)
		}

		// Generations are globally unique. We check in our unit tests that they are incremented correctly.
		expected.SetGeneration(got.info.GetGeneration())
		if !reflect.DeepEqual(got.info, expected) {
			t.Errorf("Failed to add node into schedulercache:\n got: %+v \nexpected: %+v", got, expected)
		}

		// Case 2: dump cached nodes successfully.
		cachedNodes := NewNodeInfoSnapshot()
		cache.UpdateNodeInfoSnapshot(cachedNodes)
		newNode, found := cachedNodes.NodeInfoMap[node.Name]
		if !found || len(cachedNodes.NodeInfoMap) != 1 {
			t.Errorf("failed to dump cached nodes:\n got: %v \nexpected: %v", cachedNodes, cache.nodes)
		}
		expected.SetGeneration(newNode.GetGeneration())
		if !reflect.DeepEqual(newNode, expected) {
			t.Errorf("Failed to clone node:\n got: %+v, \n expected: %+v", newNode, expected)
		}

		// Case 3: update node attribute successfully.
		node.Status.Allocatable[v1.ResourceMemory] = mem50m
		allocatableResource := expected.AllocatableResource()
		newAllocatableResource := &allocatableResource
		newAllocatableResource.Memory = mem50m.Value()
		expected.SetAllocatableResource(newAllocatableResource)

		cache.UpdateNode(nil, node)
		got, found = cache.nodes[node.Name]
		if !found {
			t.Errorf("Failed to find node %v in schedulernodeinfo after UpdateNode.", node.Name)
		}
		if got.info.GetGeneration() <= expected.GetGeneration() {
			t.Errorf("Generation is not incremented. got: %v, expected: %v", got.info.GetGeneration(), expected.GetGeneration())
		}
		expected.SetGeneration(got.info.GetGeneration())

		if !reflect.DeepEqual(got.info, expected) {
			t.Errorf("Failed to update node in schedulernodeinfo:\n got: %+v \nexpected: %+v", got, expected)
		}
		// Check nodeTree after update
		if cache.nodeTree.NumNodes() != 1 || cache.nodeTree.Next() != node.Name {
			t.Errorf("unexpected cache.nodeTree after updating node: %v", node.Name)
		}

		// Case 4: the node can not be removed if pods is not empty.
		cache.RemoveNode(node)
		if _, found := cache.nodes[node.Name]; !found {
			t.Errorf("The node %v should not be removed if pods is not empty.", node.Name)
		}
		// Check nodeTree after remove. The node should be removed from the nodeTree even if there are
		// still pods on it.
		if cache.nodeTree.NumNodes() != 0 || cache.nodeTree.Next() != "" {
			t.Errorf("unexpected cache.nodeTree after removing node: %v", node.Name)
		}
	}
}

// TestSchedulerCache_UpdateNodeInfoSnapshot tests UpdateNodeInfoSnapshot function of cache.
func TestSchedulerCache_UpdateNodeInfoSnapshot(t *testing.T) {
	// Create a few nodes to be used in tests.
	nodes := []*v1.Node{}
	for i := 0; i < 10; i++ {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("test-node%v", i),
			},
			Status: v1.NodeStatus{
				Allocatable: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("1000m"),
					v1.ResourceMemory: resource.MustParse("100m"),
				},
			},
		}
		nodes = append(nodes, node)
	}
	// Create a few nodes as updated versions of the above nodes
	updatedNodes := []*v1.Node{}
	for _, n := range nodes {
		updatedNode := n.DeepCopy()
		updatedNode.Status.Allocatable = v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("2000m"),
			v1.ResourceMemory: resource.MustParse("500m"),
		}
		updatedNodes = append(updatedNodes, updatedNode)
	}

	// Create a few pods for tests.
	pods := []*v1.Pod{}
	for i := 0; i < 10; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("test-pod%v", i),
				Namespace: "test-ns",
				UID:       types.UID(fmt.Sprintf("test-puid%v", i)),
			},
			Spec: v1.PodSpec{
				NodeName: fmt.Sprintf("test-node%v", i),
			},
		}
		pods = append(pods, pod)
	}
	// Create a few pods as updated versions of the above pods.
	updatedPods := []*v1.Pod{}
	for _, p := range pods {
		updatedPod := p.DeepCopy()
		priority := int32(1000)
		updatedPod.Spec.Priority = &priority
		updatedPods = append(updatedPods, updatedPod)
	}

	var cache *schedulerCache
	var snapshot *NodeInfoSnapshot
	type operation = func()

	addNode := func(i int) operation {
		return func() {
			cache.AddNode(nodes[i])
		}
	}
	removeNode := func(i int) operation {
		return func() {
			cache.RemoveNode(nodes[i])
		}
	}
	updateNode := func(i int) operation {
		return func() {
			cache.UpdateNode(nodes[i], updatedNodes[i])
		}
	}
	addPod := func(i int) operation {
		return func() {
			cache.AddPod(pods[i])
		}
	}
	removePod := func(i int) operation {
		return func() {
			cache.RemovePod(pods[i])
		}
	}
	updatePod := func(i int) operation {
		return func() {
			cache.UpdatePod(pods[i], updatedPods[i])
		}
	}
	updateSnapshot := func() operation {
		return func() {
			cache.UpdateNodeInfoSnapshot(snapshot)
			if err := compareCacheWithNodeInfoSnapshot(cache, snapshot); err != nil {
				t.Error(err)
			}
		}
	}

	tests := []struct {
		name       string
		operations []operation
		expected   []*v1.Node
	}{
		{
			name:       "Empty cache",
			operations: []operation{},
			expected:   []*v1.Node{},
		},
		{
			name:       "Single node",
			operations: []operation{addNode(1)},
			expected:   []*v1.Node{nodes[1]},
		},
		{
			name: "Add node, remove it, add it again",
			operations: []operation{
				addNode(1), updateSnapshot(), removeNode(1), addNode(1),
			},
			expected: []*v1.Node{nodes[1]},
		},
		{
			name: "Add a few nodes, and snapshot in the middle",
			operations: []operation{
				addNode(0), updateSnapshot(), addNode(1), updateSnapshot(), addNode(2),
				updateSnapshot(), addNode(3),
			},
			expected: []*v1.Node{nodes[3], nodes[2], nodes[1], nodes[0]},
		},
		{
			name: "Add a few nodes, and snapshot in the end",
			operations: []operation{
				addNode(0), addNode(2), addNode(5), addNode(6),
			},
			expected: []*v1.Node{nodes[6], nodes[5], nodes[2], nodes[0]},
		},
		{
			name: "Remove non-existing node",
			operations: []operation{
				addNode(0), addNode(1), updateSnapshot(), removeNode(8),
			},
			expected: []*v1.Node{nodes[1], nodes[0]},
		},
		{
			name: "Update some nodes",
			operations: []operation{
				addNode(0), addNode(1), addNode(5), updateSnapshot(), updateNode(1),
			},
			expected: []*v1.Node{nodes[1], nodes[5], nodes[0]},
		},
		{
			name: "Add a few nodes, and remove all of them",
			operations: []operation{
				addNode(0), addNode(2), addNode(5), addNode(6), updateSnapshot(),
				removeNode(0), removeNode(2), removeNode(5), removeNode(6),
			},
			expected: []*v1.Node{},
		},
		{
			name: "Add a few nodes, and remove some of them",
			operations: []operation{
				addNode(0), addNode(2), addNode(5), addNode(6), updateSnapshot(),
				removeNode(0), removeNode(6),
			},
			expected: []*v1.Node{nodes[5], nodes[2]},
		},
		{
			name: "Add a few nodes, remove all of them, and add more",
			operations: []operation{
				addNode(2), addNode(5), addNode(6), updateSnapshot(),
				removeNode(2), removeNode(5), removeNode(6), updateSnapshot(),
				addNode(7), addNode(9),
			},
			expected: []*v1.Node{nodes[9], nodes[7]},
		},
		{
			name: "Update nodes in particular order",
			operations: []operation{
				addNode(8), updateNode(2), updateNode(8), updateSnapshot(),
				addNode(1),
			},
			expected: []*v1.Node{nodes[1], nodes[8], nodes[2]},
		},
		{
			name: "Add some nodes and some pods",
			operations: []operation{
				addNode(0), addNode(2), addNode(8), updateSnapshot(),
				addPod(8), addPod(2),
			},
			expected: []*v1.Node{nodes[2], nodes[8], nodes[0]},
		},
		{
			name: "Updating a pod moves its node to the head",
			operations: []operation{
				addNode(0), addPod(0), addNode(2), addNode(4), updatePod(0),
			},
			expected: []*v1.Node{nodes[0], nodes[4], nodes[2]},
		},
		{
			name: "Remove pod from non-existing node",
			operations: []operation{
				addNode(0), addPod(0), addNode(2), updateSnapshot(), removePod(3),
			},
			expected: []*v1.Node{nodes[2], nodes[0]},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache = newSchedulerCache(time.Second, time.Second, nil)
			snapshot = NewNodeInfoSnapshot()

			for _, op := range test.operations {
				op()
			}

			if len(test.expected) != len(cache.nodes) {
				t.Errorf("unexpected number of nodes. Expected: %v, got: %v", len(test.expected), len(cache.nodes))
			}
			var i int
			// Check that cache is in the expected state.
			for node := cache.headNode; node != nil; node = node.next {
				if node.info.Node().Name != test.expected[i].Name {
					t.Errorf("unexpected node. Expected: %v, got: %v, index: %v", test.expected[i].Name, node.info.Node().Name, i)
				}
				i++
			}
			// Make sure we visited all the cached nodes in the above for loop.
			if i != len(cache.nodes) {
				t.Errorf("Not all the nodes were visited by following the NodeInfo linked list. Expected to see %v nodes, saw %v.", len(cache.nodes), i)
			}

			// Always update the snapshot at the end of operations and compare it.
			cache.UpdateNodeInfoSnapshot(snapshot)
			if err := compareCacheWithNodeInfoSnapshot(cache, snapshot); err != nil {
				t.Error(err)
			}
		})
	}
}

func compareCacheWithNodeInfoSnapshot(cache *schedulerCache, snapshot *NodeInfoSnapshot) error {
	if len(snapshot.NodeInfoMap) != len(cache.nodes) {
		return fmt.Errorf("unexpected number of nodes in the snapshot. Expected: %v, got: %v", len(cache.nodes), len(snapshot.NodeInfoMap))
	}
	for name, ni := range cache.nodes {
		if !reflect.DeepEqual(snapshot.NodeInfoMap[name], ni.info) {
			return fmt.Errorf("unexpected node info. Expected: %v, got: %v", ni.info, snapshot.NodeInfoMap[name])
		}
	}
	return nil
}

func BenchmarkList1kNodes30kPods(b *testing.B) {
	cache := setupCacheOf1kNodes30kPods(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cache.List(labels.Everything())
	}
}

func BenchmarkUpdate1kNodes30kPods(b *testing.B) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(nil, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	cache := setupCacheOf1kNodes30kPods(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cachedNodes := NewNodeInfoSnapshot()
		cache.UpdateNodeInfoSnapshot(cachedNodes)
	}
}

func BenchmarkExpirePods(b *testing.B) {
	podNums := []int{
		100,
		1000,
		10000,
	}
	for _, podNum := range podNums {
		name := fmt.Sprintf("%dPods", podNum)
		b.Run(name, func(b *testing.B) {
			benchmarkExpire(b, podNum)
		})
	}
}

func benchmarkExpire(b *testing.B, podNum int) {
	now := time.Now()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		cache := setupCacheWithAssumedPods(b, podNum, now)
		b.StartTimer()
		cache.cleanupAssumedPods(now.Add(2 * time.Second))
	}
}

type testingMode interface {
	Fatalf(format string, args ...interface{})
}

func makeBasePod(t testingMode, nodeName, objName, cpu, mem, extended string, ports []v1.ContainerPort) *v1.Pod {
	req := v1.ResourceList{}
	if cpu != "" {
		req = v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse(cpu),
			v1.ResourceMemory: resource.MustParse(mem),
		}
		if extended != "" {
			parts := strings.Split(extended, ":")
			if len(parts) != 2 {
				t.Fatalf("Invalid extended resource string: \"%s\"", extended)
			}
			req[v1.ResourceName(parts[0])] = resource.MustParse(parts[1])
		}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID(objName),
			Namespace: "node_info_cache_test",
			Name:      objName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Resources: v1.ResourceRequirements{
					Requests: req,
				},
				Ports: ports,
			}},
			NodeName: nodeName,
		},
	}
}

func setupCacheOf1kNodes30kPods(b *testing.B) Cache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < 1000; i++ {
		nodeName := fmt.Sprintf("node-%d", i)
		for j := 0; j < 30; j++ {
			objName := fmt.Sprintf("%s-pod-%d", nodeName, j)
			pod := makeBasePod(b, nodeName, objName, "0", "0", "", nil)

			if err := cache.AddPod(pod); err != nil {
				b.Fatalf("AddPod failed: %v", err)
			}
		}
	}
	return cache
}

func setupCacheWithAssumedPods(b *testing.B, podNum int, assumedTime time.Time) *schedulerCache {
	cache := newSchedulerCache(time.Second, time.Second, nil)
	for i := 0; i < podNum; i++ {
		nodeName := fmt.Sprintf("node-%d", i/10)
		objName := fmt.Sprintf("%s-pod-%d", nodeName, i%10)
		pod := makeBasePod(b, nodeName, objName, "0", "0", "", nil)

		err := assumeAndFinishBinding(cache, pod, assumedTime)
		if err != nil {
			b.Fatalf("assumePod failed: %v", err)
		}
	}
	return cache
}
