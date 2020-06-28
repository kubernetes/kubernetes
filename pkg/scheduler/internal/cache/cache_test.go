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
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

func deepEqualWithoutGeneration(actual *nodeInfoListItem, expected *framework.NodeInfo) error {
	if (actual == nil) != (expected == nil) {
		return errors.New("one of the actual or expected is nil and the other is not")
	}
	// Ignore generation field.
	if actual != nil {
		actual.info.Generation = 0
	}
	if expected != nil {
		expected.Generation = 0
	}
	if actual != nil && !reflect.DeepEqual(actual.info, expected) {
		return fmt.Errorf("got node info %s, want %s", actual.info, expected)
	}
	return nil
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

func (b *hostPortInfoBuilder) build() framework.HostPortInfo {
	res := make(framework.HostPortInfo)
	for _, param := range b.inputs {
		res.Add(param.ip, param.protocol, param.port)
	}
	return res
}

func newNodeInfo(requestedResource *framework.Resource,
	nonzeroRequest *framework.Resource,
	pods []*v1.Pod,
	usedPorts framework.HostPortInfo,
	imageStates map[string]*framework.ImageStateSummary,
) *framework.NodeInfo {
	nodeInfo := framework.NewNodeInfo(pods...)
	nodeInfo.Requested = requestedResource
	nodeInfo.NonZeroRequested = nonzeroRequest
	nodeInfo.UsedPorts = usedPorts
	nodeInfo.ImageStates = imageStates
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

		wNodeInfo *framework.NodeInfo
	}{{
		pods: []*v1.Pod{testPods[0]},
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[1], testPods[2]},
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			&framework.Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			[]*v1.Pod{testPods[1], testPods[2]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}, { // test non-zero request
		pods: []*v1.Pod{testPods[3]},
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 0,
				Memory:   0,
			},
			&framework.Resource{
				MilliCPU: schedutil.DefaultMilliCPURequest,
				Memory:   schedutil.DefaultMemoryRequest,
			},
			[]*v1.Pod{testPods[3]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[4]},
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU:        100,
				Memory:          500,
				ScalarResources: map[v1.ResourceName]int64{"example.com/foo": 3},
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[4]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[4], testPods[5]},
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU:        300,
				Memory:          1524,
				ScalarResources: map[v1.ResourceName]int64{"example.com/foo": 8},
			},
			&framework.Resource{
				MilliCPU: 300,
				Memory:   1524,
			},
			[]*v1.Pod{testPods[4], testPods[5]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}, {
		pods: []*v1.Pod{testPods[6]},
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[6]},
			newHostPortInfoBuilder().build(),
			make(map[string]*framework.ImageStateSummary),
		),
	},
	}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			cache := newSchedulerCache(time.Second, time.Second, nil)
			for _, pod := range tt.pods {
				if err := cache.AssumePod(pod); err != nil {
					t.Fatalf("AssumePod failed: %v", err)
				}
			}
			n := cache.nodes[nodeName]
			if err := deepEqualWithoutGeneration(n, tt.wNodeInfo); err != nil {
				t.Error(err)
			}

			for _, pod := range tt.pods {
				if err := cache.ForgetPod(pod); err != nil {
					t.Fatalf("ForgetPod failed: %v", err)
				}
				if err := isForgottenFromCache(pod, cache); err != nil {
					t.Errorf("pod %s: %v", pod.Name, err)
				}
			}
		})
	}
}

type testExpirePodStruct struct {
	pod         *v1.Pod
	finishBind  bool
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
		makeBasePod(t, nodeName, "test-3", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	now := time.Now()
	ttl := 10 * time.Second
	tests := []struct {
		pods        []*testExpirePodStruct
		cleanupTime time.Time

		wNodeInfo *framework.NodeInfo
	}{{ // assumed pod would expires
		pods: []*testExpirePodStruct{
			{pod: testPods[0], finishBind: true, assumedTime: now},
		},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo:   framework.NewNodeInfo(),
	}, { // first one would expire, second and third would not.
		pods: []*testExpirePodStruct{
			{pod: testPods[0], finishBind: true, assumedTime: now},
			{pod: testPods[1], finishBind: true, assumedTime: now.Add(3 * ttl / 2)},
			{pod: testPods[2]},
		},
		cleanupTime: now.Add(2 * ttl),
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 400,
				Memory:   2048,
			},
			&framework.Resource{
				MilliCPU: 400,
				Memory:   2048,
			},
			// Order gets altered when removing pods.
			[]*v1.Pod{testPods[2], testPods[1]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			cache := newSchedulerCache(ttl, time.Second, nil)

			for _, pod := range tt.pods {
				if err := cache.AssumePod(pod.pod); err != nil {
					t.Fatal(err)
				}
				if !pod.finishBind {
					continue
				}
				if err := cache.finishBinding(pod.pod, pod.assumedTime); err != nil {
					t.Fatal(err)
				}
			}
			// pods that got bound and have assumedTime + ttl < cleanupTime will get
			// expired and removed
			cache.cleanupAssumedPods(tt.cleanupTime)
			n := cache.nodes[nodeName]
			if err := deepEqualWithoutGeneration(n, tt.wNodeInfo); err != nil {
				t.Error(err)
			}
		})
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

		wNodeInfo *framework.NodeInfo
	}{{ // two pod were assumed at same time. But first one is called Add() and gets confirmed.
		podsToAssume: []*v1.Pod{testPods[0], testPods[1]},
		podsToAdd:    []*v1.Pod{testPods[0]},
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
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
			if err := deepEqualWithoutGeneration(n, tt.wNodeInfo); err != nil {
				t.Error(err)
			}
		})
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

		snapshot := cache.Dump()
		if len(snapshot.Nodes) != len(cache.nodes) {
			t.Errorf("Unequal number of nodes in the cache and its snapshot. expected: %v, got: %v", len(cache.nodes), len(snapshot.Nodes))
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

		wNodeInfo map[string]*framework.NodeInfo
	}{{
		podsToAssume: []*v1.Pod{assumedPod.DeepCopy()},
		podsToAdd:    []*v1.Pod{addedPod.DeepCopy()},
		podsToUpdate: [][]*v1.Pod{{addedPod.DeepCopy(), updatedPod.DeepCopy()}},
		wNodeInfo: map[string]*framework.NodeInfo{
			"assumed-node": nil,
			"actual-node": newNodeInfo(
				&framework.Resource{
					MilliCPU: 200,
					Memory:   500,
				},
				&framework.Resource{
					MilliCPU: 200,
					Memory:   500,
				},
				[]*v1.Pod{updatedPod.DeepCopy()},
				newHostPortInfoBuilder().add("TCP", "0.0.0.0", 90).build(),
				make(map[string]*framework.ImageStateSummary),
			),
		},
	}}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
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
				n := cache.nodes[nodeName]
				if err := deepEqualWithoutGeneration(n, expected); err != nil {
					t.Errorf("node %q: %v", nodeName, err)
				}
			}
		})
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

		wNodeInfo *framework.NodeInfo
	}{{
		pod: basePod,
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{basePod},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			now := time.Now()
			cache := newSchedulerCache(ttl, time.Second, nil)
			if err := assumeAndFinishBinding(cache, tt.pod, now); err != nil {
				t.Fatalf("assumePod failed: %v", err)
			}
			cache.cleanupAssumedPods(now.Add(2 * ttl))
			// It should be expired and removed.
			if err := isForgottenFromCache(tt.pod, cache); err != nil {
				t.Error(err)
			}
			if err := cache.AddPod(tt.pod); err != nil {
				t.Fatalf("AddPod failed: %v", err)
			}
			// check after expiration. confirmed pods shouldn't be expired.
			n := cache.nodes[nodeName]
			if err := deepEqualWithoutGeneration(n, tt.wNodeInfo); err != nil {
				t.Error(err)
			}
		})
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

		wNodeInfo []*framework.NodeInfo
	}{{ // add a pod and then update it twice
		podsToAdd:    []*v1.Pod{testPods[0]},
		podsToUpdate: []*v1.Pod{testPods[0], testPods[1], testPods[0]},
		wNodeInfo: []*framework.NodeInfo{newNodeInfo(
			&framework.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			&framework.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			[]*v1.Pod{testPods[1]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*framework.ImageStateSummary),
		), newNodeInfo(
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		)},
	}}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			cache := newSchedulerCache(ttl, time.Second, nil)
			for _, podToAdd := range tt.podsToAdd {
				if err := cache.AddPod(podToAdd); err != nil {
					t.Fatalf("AddPod failed: %v", err)
				}
			}

			for j := range tt.podsToUpdate {
				if j == 0 {
					continue
				}
				if err := cache.UpdatePod(tt.podsToUpdate[j-1], tt.podsToUpdate[j]); err != nil {
					t.Fatalf("UpdatePod failed: %v", err)
				}
				// check after expiration. confirmed pods shouldn't be expired.
				n := cache.nodes[nodeName]
				if err := deepEqualWithoutGeneration(n, tt.wNodeInfo[j-1]); err != nil {
					t.Errorf("update %d: %v", j, err)
				}
			}
		})
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

		wNodeInfo []*framework.NodeInfo
	}{{ // Pod is assumed, expired, and added. Then it would be updated twice.
		podsToAssume: []*v1.Pod{testPods[0]},
		podsToAdd:    []*v1.Pod{testPods[0]},
		podsToUpdate: []*v1.Pod{testPods[0], testPods[1], testPods[0]},
		wNodeInfo: []*framework.NodeInfo{newNodeInfo(
			&framework.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			&framework.Resource{
				MilliCPU: 200,
				Memory:   1024,
			},
			[]*v1.Pod{testPods[1]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 8080).build(),
			make(map[string]*framework.ImageStateSummary),
		), newNodeInfo(
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{testPods[0]},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		)},
	}}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			now := time.Now()
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

			for j := range tt.podsToUpdate {
				if j == 0 {
					continue
				}
				if err := cache.UpdatePod(tt.podsToUpdate[j-1], tt.podsToUpdate[j]); err != nil {
					t.Fatalf("UpdatePod failed: %v", err)
				}
				// check after expiration. confirmed pods shouldn't be expired.
				n := cache.nodes[nodeName]
				if err := deepEqualWithoutGeneration(n, tt.wNodeInfo[j-1]); err != nil {
					t.Errorf("update %d: %v", j, err)
				}
			}
		})
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
		wNodeInfo *framework.NodeInfo
	}{
		{
			pod: podE,
			wNodeInfo: newNodeInfo(
				&framework.Resource{
					EphemeralStorage: 500,
				},
				&framework.Resource{
					MilliCPU: schedutil.DefaultMilliCPURequest,
					Memory:   schedutil.DefaultMemoryRequest,
				},
				[]*v1.Pod{podE},
				framework.HostPortInfo{},
				make(map[string]*framework.ImageStateSummary),
			),
		},
	}
	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			cache := newSchedulerCache(time.Second, time.Second, nil)
			if err := cache.AddPod(tt.pod); err != nil {
				t.Fatalf("AddPod failed: %v", err)
			}
			n := cache.nodes[nodeName]
			if err := deepEqualWithoutGeneration(n, tt.wNodeInfo); err != nil {
				t.Error(err)
			}

			if err := cache.RemovePod(tt.pod); err != nil {
				t.Fatalf("RemovePod failed: %v", err)
			}
			if _, err := cache.GetPod(tt.pod); err == nil {
				t.Errorf("pod was not deleted")
			}
		})
	}
}

// TestRemovePod tests after added pod is removed, its information should also be subtracted.
func TestRemovePod(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	basePod := makeBasePod(t, "node-1", "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}})
	tests := []struct {
		nodes     []*v1.Node
		pod       *v1.Pod
		wNodeInfo *framework.NodeInfo
	}{{
		nodes: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node-2"},
			},
		},
		pod: basePod,
		wNodeInfo: newNodeInfo(
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			&framework.Resource{
				MilliCPU: 100,
				Memory:   500,
			},
			[]*v1.Pod{basePod},
			newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
			make(map[string]*framework.ImageStateSummary),
		),
	}}

	for i, tt := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			nodeName := tt.pod.Spec.NodeName
			cache := newSchedulerCache(time.Second, time.Second, nil)
			// Add pod succeeds even before adding the nodes.
			if err := cache.AddPod(tt.pod); err != nil {
				t.Fatalf("AddPod failed: %v", err)
			}
			n := cache.nodes[nodeName]
			if err := deepEqualWithoutGeneration(n, tt.wNodeInfo); err != nil {
				t.Error(err)
			}
			for _, n := range tt.nodes {
				if err := cache.AddNode(n); err != nil {
					t.Error(err)
				}
			}

			if err := cache.RemovePod(tt.pod); err != nil {
				t.Fatalf("RemovePod failed: %v", err)
			}

			if _, err := cache.GetPod(tt.pod); err == nil {
				t.Errorf("pod was not deleted")
			}

			// Node that owned the Pod should be at the head of the list.
			if cache.headNode.info.Node().Name != nodeName {
				t.Errorf("node %q is not at the head of the list", nodeName)
			}
		})
	}
}

func TestForgetPod(t *testing.T) {
	nodeName := "node"
	basePod := makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}})
	pods := []*v1.Pod{basePod}
	now := time.Now()
	ttl := 10 * time.Second

	cache := newSchedulerCache(ttl, time.Second, nil)
	for _, pod := range pods {
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
	for _, pod := range pods {
		if err := cache.ForgetPod(pod); err != nil {
			t.Fatalf("ForgetPod failed: %v", err)
		}
		if err := isForgottenFromCache(pod, cache); err != nil {
			t.Errorf("pod %q: %v", pod.Name, err)
		}
	}
}

// buildNodeInfo creates a NodeInfo by simulating node operations in cache.
func buildNodeInfo(node *v1.Node, pods []*v1.Pod) *framework.NodeInfo {
	expected := framework.NewNodeInfo()
	expected.SetNode(node)
	expected.Allocatable = framework.NewResource(node.Status.Allocatable)
	expected.Generation++
	for _, pod := range pods {
		expected.AddPod(pod)
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

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			expected := buildNodeInfo(test.node, test.pods)
			node := test.node

			cache := newSchedulerCache(time.Second, time.Second, nil)
			if err := cache.AddNode(node); err != nil {
				t.Fatal(err)
			}
			for _, pod := range test.pods {
				if err := cache.AddPod(pod); err != nil {
					t.Fatal(err)
				}
			}

			// Step 1: the node was added into cache successfully.
			got, found := cache.nodes[node.Name]
			if !found {
				t.Errorf("Failed to find node %v in internalcache.", node.Name)
			}
			if cache.nodeTree.numNodes != 1 || cache.nodeTree.next() != node.Name {
				t.Errorf("cache.nodeTree is not updated correctly after adding node: %v", node.Name)
			}

			// Generations are globally unique. We check in our unit tests that they are incremented correctly.
			expected.Generation = got.info.Generation
			if !reflect.DeepEqual(got.info, expected) {
				t.Errorf("Failed to add node into schedulercache:\n got: %+v \nexpected: %+v", got, expected)
			}

			// Step 2: dump cached nodes successfully.
			cachedNodes := NewEmptySnapshot()
			if err := cache.UpdateSnapshot(cachedNodes); err != nil {
				t.Error(err)
			}
			newNode, found := cachedNodes.nodeInfoMap[node.Name]
			if !found || len(cachedNodes.nodeInfoMap) != 1 {
				t.Errorf("failed to dump cached nodes:\n got: %v \nexpected: %v", cachedNodes, cache.nodes)
			}
			expected.Generation = newNode.Generation
			if !reflect.DeepEqual(newNode, expected) {
				t.Errorf("Failed to clone node:\n got: %+v, \n expected: %+v", newNode, expected)
			}

			// Step 3: update node attribute successfully.
			node.Status.Allocatable[v1.ResourceMemory] = mem50m
			expected.Allocatable.Memory = mem50m.Value()

			if err := cache.UpdateNode(nil, node); err != nil {
				t.Error(err)
			}
			got, found = cache.nodes[node.Name]
			if !found {
				t.Errorf("Failed to find node %v in schedulertypes after UpdateNode.", node.Name)
			}
			if got.info.Generation <= expected.Generation {
				t.Errorf("Generation is not incremented. got: %v, expected: %v", got.info.Generation, expected.Generation)
			}
			expected.Generation = got.info.Generation

			if !reflect.DeepEqual(got.info, expected) {
				t.Errorf("Failed to update node in schedulertypes:\n got: %+v \nexpected: %+v", got, expected)
			}
			// Check nodeTree after update
			if cache.nodeTree.numNodes != 1 || cache.nodeTree.next() != node.Name {
				t.Errorf("unexpected cache.nodeTree after updating node: %v", node.Name)
			}

			// Step 4: the node can be removed even if it still has pods.
			if err := cache.RemoveNode(node); err != nil {
				t.Error(err)
			}
			if _, err := cache.GetNodeInfo(node.Name); err == nil {
				t.Errorf("The node %v should be removed.", node.Name)
			}
			// Check node is removed from nodeTree as well.
			if cache.nodeTree.numNodes != 0 || cache.nodeTree.next() != "" {
				t.Errorf("unexpected cache.nodeTree after removing node: %v", node.Name)
			}
			// Pods are still in the pods cache.
			for _, p := range test.pods {
				if _, err := cache.GetPod(p); err != nil {
					t.Error(err)
				}
			}

			// Step 5: removing pods for the removed node still succeeds.
			for _, p := range test.pods {
				if err := cache.RemovePod(p); err != nil {
					t.Error(err)
				}
				if _, err := cache.GetPod(p); err == nil {
					t.Errorf("pod %q still in cache", p.Name)
				}
			}
		})
	}
}

func TestSchedulerCache_UpdateSnapshot(t *testing.T) {
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
	for i := 0; i < 20; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("test-pod%v", i),
				Namespace: "test-ns",
				UID:       types.UID(fmt.Sprintf("test-puid%v", i)),
			},
			Spec: v1.PodSpec{
				NodeName: fmt.Sprintf("test-node%v", i%10),
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

	// Add a couple of pods with affinity, on the first and seconds nodes.
	podsWithAffinity := []*v1.Pod{}
	for i := 0; i < 2; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("test-pod%v", i),
				Namespace: "test-ns",
				UID:       types.UID(fmt.Sprintf("test-puid%v", i)),
			},
			Spec: v1.PodSpec{
				NodeName: fmt.Sprintf("test-node%v", i),
				Affinity: &v1.Affinity{
					PodAffinity: &v1.PodAffinity{},
				},
			},
		}
		podsWithAffinity = append(podsWithAffinity, pod)
	}

	var cache *schedulerCache
	var snapshot *Snapshot
	type operation = func()

	addNode := func(i int) operation {
		return func() {
			if err := cache.AddNode(nodes[i]); err != nil {
				t.Error(err)
			}
		}
	}
	removeNode := func(i int) operation {
		return func() {
			if err := cache.RemoveNode(nodes[i]); err != nil {
				t.Error(err)
			}
		}
	}
	updateNode := func(i int) operation {
		return func() {
			if err := cache.UpdateNode(nodes[i], updatedNodes[i]); err != nil {
				t.Error(err)
			}
		}
	}
	addPod := func(i int) operation {
		return func() {
			if err := cache.AddPod(pods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	addPodWithAffinity := func(i int) operation {
		return func() {
			if err := cache.AddPod(podsWithAffinity[i]); err != nil {
				t.Error(err)
			}
		}
	}
	removePod := func(i int) operation {
		return func() {
			if err := cache.RemovePod(pods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	removePodWithAffinity := func(i int) operation {
		return func() {
			if err := cache.RemovePod(podsWithAffinity[i]); err != nil {
				t.Error(err)
			}
		}
	}
	updatePod := func(i int) operation {
		return func() {
			if err := cache.UpdatePod(pods[i], updatedPods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	updateSnapshot := func() operation {
		return func() {
			cache.UpdateSnapshot(snapshot)
			if err := compareCacheWithNodeInfoSnapshot(cache, snapshot); err != nil {
				t.Error(err)
			}
		}
	}

	tests := []struct {
		name                         string
		operations                   []operation
		expected                     []*v1.Node
		expectedHavePodsWithAffinity int
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
			name: "Add node and remove it in the same cycle, add it again",
			operations: []operation{
				addNode(1), updateSnapshot(), addNode(2), removeNode(1),
			},
			expected: []*v1.Node{nodes[2]},
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
			name: "Add pod before its node",
			operations: []operation{
				addNode(0), addPod(1), updatePod(1), addNode(1),
			},
			expected: []*v1.Node{nodes[1], nodes[0]},
		},
		{
			name: "Remove node before its pods",
			operations: []operation{
				addNode(0), addNode(1), addPod(1), addPod(11),
				removeNode(1), updatePod(1), updatePod(11), removePod(1), removePod(11),
			},
			expected: []*v1.Node{nodes[0]},
		},
		{
			name: "Add Pods with affinity",
			operations: []operation{
				addNode(0), addPodWithAffinity(0), updateSnapshot(), addNode(1),
			},
			expected:                     []*v1.Node{nodes[1], nodes[0]},
			expectedHavePodsWithAffinity: 1,
		},
		{
			name: "Add multiple nodes with pods with affinity",
			operations: []operation{
				addNode(0), addPodWithAffinity(0), updateSnapshot(), addNode(1), addPodWithAffinity(1), updateSnapshot(),
			},
			expected:                     []*v1.Node{nodes[1], nodes[0]},
			expectedHavePodsWithAffinity: 2,
		},
		{
			name: "Add then Remove pods with affinity",
			operations: []operation{
				addNode(0), addNode(1), addPodWithAffinity(0), updateSnapshot(), removePodWithAffinity(0), updateSnapshot(),
			},
			expected:                     []*v1.Node{nodes[0], nodes[1]},
			expectedHavePodsWithAffinity: 0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache = newSchedulerCache(time.Second, time.Second, nil)
			snapshot = NewEmptySnapshot()

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

			// Check number of nodes with pods with affinity
			if len(snapshot.havePodsWithAffinityNodeInfoList) != test.expectedHavePodsWithAffinity {
				t.Errorf("unexpected number of HavePodsWithAffinity nodes. Expected: %v, got: %v", test.expectedHavePodsWithAffinity, len(snapshot.havePodsWithAffinityNodeInfoList))
			}

			// Always update the snapshot at the end of operations and compare it.
			if err := cache.UpdateSnapshot(snapshot); err != nil {
				t.Error(err)
			}
			if err := compareCacheWithNodeInfoSnapshot(cache, snapshot); err != nil {
				t.Error(err)
			}
		})
	}
}

func compareCacheWithNodeInfoSnapshot(cache *schedulerCache, snapshot *Snapshot) error {
	// Compare the map.
	if len(snapshot.nodeInfoMap) != len(cache.nodes) {
		return fmt.Errorf("unexpected number of nodes in the snapshot. Expected: %v, got: %v", len(cache.nodes), len(snapshot.nodeInfoMap))
	}
	for name, ni := range cache.nodes {
		if !reflect.DeepEqual(snapshot.nodeInfoMap[name], ni.info) {
			return fmt.Errorf("unexpected node info for node %q. Expected: %v, got: %v", name, ni.info, snapshot.nodeInfoMap[name])
		}
	}

	// Compare the lists.
	if len(snapshot.nodeInfoList) != len(cache.nodes) {
		return fmt.Errorf("unexpected number of nodes in NodeInfoList. Expected: %v, got: %v", len(cache.nodes), len(snapshot.nodeInfoList))
	}

	expectedNodeInfoList := make([]*framework.NodeInfo, 0, cache.nodeTree.numNodes)
	expectedHavePodsWithAffinityNodeInfoList := make([]*framework.NodeInfo, 0, cache.nodeTree.numNodes)
	for i := 0; i < cache.nodeTree.numNodes; i++ {
		nodeName := cache.nodeTree.next()
		if n := snapshot.nodeInfoMap[nodeName]; n != nil {
			expectedNodeInfoList = append(expectedNodeInfoList, n)
			if len(n.PodsWithAffinity) > 0 {
				expectedHavePodsWithAffinityNodeInfoList = append(expectedHavePodsWithAffinityNodeInfoList, n)
			}
		} else {
			return fmt.Errorf("node %q exist in nodeTree but not in NodeInfoMap, this should not happen", nodeName)
		}
	}

	for i, expected := range expectedNodeInfoList {
		got := snapshot.nodeInfoList[i]
		if expected != got {
			return fmt.Errorf("unexpected NodeInfo pointer in NodeInfoList. Expected: %p, got: %p", expected, got)
		}
	}

	for i, expected := range expectedHavePodsWithAffinityNodeInfoList {
		got := snapshot.havePodsWithAffinityNodeInfoList[i]
		if expected != got {
			return fmt.Errorf("unexpected NodeInfo pointer in HavePodsWithAffinityNodeInfoList. Expected: %p, got: %p", expected, got)
		}
	}

	return nil
}

func BenchmarkUpdate1kNodes30kPods(b *testing.B) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer featuregatetesting.SetFeatureGateDuringTest(nil, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	cache := setupCacheOf1kNodes30kPods(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cachedNodes := NewEmptySnapshot()
		cache.UpdateSnapshot(cachedNodes)
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

func isForgottenFromCache(p *v1.Pod, c *schedulerCache) error {
	if assumed, err := c.IsAssumedPod(p); err != nil {
		return err
	} else if assumed {
		return errors.New("still assumed")
	}
	if _, err := c.GetPod(p); err == nil {
		return errors.New("still in cache")
	}
	return nil
}
