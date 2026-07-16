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
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

var nodeInfoCmpOpts = []cmp.Option{
	cmp.AllowUnexported(framework.NodeInfo{}),
	// This field needs to be ignored because we can't call AllowUnexported for type framework.podResource (it's not visible in this package).
	cmpopts.IgnoreFields(framework.PodInfo{}, "cachedResource"),
}

var podGroupStateCmpOpts = []cmp.Option{
	cmp.AllowUnexported(podGroupStateSnapshot{}, podGroupStateData{}, podGroupKey{}),
	cmpopts.IgnoreFields(podGroupStateData{}, "generation"),
	cmpopts.EquateEmpty(),
}

func init() {
	metrics.Register()
}

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
	if actual != nil {
		if diff := cmp.Diff(expected, actual.info, nodeInfoCmpOpts...); diff != "" {
			return fmt.Errorf("Unexpected node info (-want,+got):\n%s", diff)
		}
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

func (b *hostPortInfoBuilder) build() fwk.HostPortInfo {
	res := make(fwk.HostPortInfo)
	for _, param := range b.inputs {
		res.Add(param.ip, param.protocol, param.port)
	}
	return res
}

func newNodeInfo(requestedResource *framework.Resource,
	nonzeroRequest *framework.Resource,
	pods []*v1.Pod,
	usedPorts fwk.HostPortInfo,
	imageStates map[string]*fwk.ImageStateSummary,
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
	nodeName := "node"
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test-resource-request-and-port-0", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-resource-request-and-port-1", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-resource-request-and-port-2", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-nonzero-request", "", "", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-extended-resource-1", "100m", "500", "example.com/foo:3", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-extended-resource-2", "200m", "1Ki", "example.com/foo:5", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-extended-key", "100m", "500", "random-invalid-extended-key:100", []v1.ContainerPort{{}}),
	}

	tests := []struct {
		name string
		pods []*v1.Pod

		wNodeInfo *framework.NodeInfo
	}{{
		name: "assumed one pod with resource request and used ports",
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
			make(map[string]*fwk.ImageStateSummary),
		),
	}, {
		name: "node requested resource are equal to the sum of the assumed pods requested resource, node contains host ports defined by pods",
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
			make(map[string]*fwk.ImageStateSummary),
		),
	}, { // test non-zero request
		name: "assumed pod without resource request",
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
			make(map[string]*fwk.ImageStateSummary),
		),
	}, {
		name: "assumed one pod with extended resource",
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
			make(map[string]*fwk.ImageStateSummary),
		),
	}, {
		name: "assumed two pods with extended resources",
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
			make(map[string]*fwk.ImageStateSummary),
		),
	}, {
		name: "assumed pod with random invalid extended resource key",
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
			make(map[string]*fwk.ImageStateSummary),
		),
	},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			cache := newCache(ctx, time.Second, nil, false)
			for _, pod := range tc.pods {
				if err := cache.AssumePod(logger, pod); err != nil {
					t.Fatalf("AssumePod failed: %v", err)
				}
				// pod already in cache so can't be assumed
				if err := cache.AssumePod(logger, pod); err == nil {
					t.Error("expected error, no error found")
				}
			}
			n := cache.nodes[nodeName]
			if err := deepEqualWithoutGeneration(n, tc.wNodeInfo); err != nil {
				t.Error(err)
			}

			for _, pod := range tc.pods {
				if err := cache.ForgetPod(logger, pod); err != nil {
					t.Fatalf("ForgetPod failed: %v", err)
				}
				if err := isForgottenFromCache(pod, cache); err != nil {
					t.Errorf("pod %s: %v", pod.Name, err)
				}
			}
		})
	}
}

// TestAddPodWillConfirm tests that a pod being Add()ed will be confirmed if assumed.
func TestAddPodWillConfirm(t *testing.T) {
	nodeName := "node"

	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test-1", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-2", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	test := struct {
		podsToAssume []*v1.Pod
		podsToAdd    []*v1.Pod

		wNodeInfo *framework.NodeInfo
	}{ // two pod were assumed at same time. But first one is called Add() and gets confirmed.
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
			make(map[string]*fwk.ImageStateSummary),
		),
	}

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cache := newCache(ctx, time.Second, nil, false)
	for _, podToAssume := range test.podsToAssume {
		if err := cache.AssumePod(logger, podToAssume); err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
	}
	for _, podToAdd := range test.podsToAdd {
		if err := cache.AddPod(logger, podToAdd); err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
		// pod already in added state
		if err := cache.AddPod(logger, podToAdd); err == nil {
			t.Error("expected error, no error found")
		}
	}

	for _, podToAssume := range test.podsToAssume {
		assumed, err := cache.IsAssumedPod(podToAssume)
		if err != nil {
			t.Fatalf("IsAssumedPod failed: %v", err)
		}
		if !assumed {
			continue
		}
		if err = cache.ForgetPod(logger, podToAssume); err != nil {
			t.Fatalf("ForgetPod failed: %v", err)
		}
	}

	n := cache.nodes[nodeName]
	if err := deepEqualWithoutGeneration(n, test.wNodeInfo); err != nil {
		t.Error(err)
	}
}

func TestDump(t *testing.T) {
	nodeName := "node"

	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test-1", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test-2", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
	}
	test := struct {
		podsToAssume []*v1.Pod
		podsToAdd    []*v1.Pod
	}{ // two pod were assumed at same time. But first one is called Add() and gets confirmed.
		podsToAssume: []*v1.Pod{testPods[0], testPods[1]},
		podsToAdd:    []*v1.Pod{testPods[0]},
	}

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cache := newCache(ctx, time.Second, nil, false)
	for _, podToAssume := range test.podsToAssume {
		if err := cache.AssumePod(logger, podToAssume); err != nil {
			t.Errorf("assumePod failed: %v", err)
		}
	}
	for _, podToAdd := range test.podsToAdd {
		if err := cache.AddPod(logger, podToAdd); err != nil {
			t.Errorf("AddPod failed: %v", err)
		}
	}

	snapshot := cache.Dump()
	if len(snapshot.Nodes) != len(cache.nodes) {
		t.Errorf("Unequal number of nodes in the cache and its snapshot. expected: %v, got: %v", len(cache.nodes), len(snapshot.Nodes))
	}
	for name, ni := range snapshot.Nodes {
		nItem := cache.nodes[name]
		if diff := cmp.Diff(nItem.info, ni, nodeInfoCmpOpts...); diff != "" {
			t.Errorf("Unexpected node info (-want,+got):\n%s", diff)
		}
	}
	if diff := cmp.Diff(cache.assumedPods, snapshot.AssumedPods); diff != "" {
		t.Errorf("Unexpected assumedPods (-want,+got):\n%s", diff)
	}

}

// TestAddPodAlwaysUpdatePodInfoInNodeInfo tests that AddPod method always updates PodInfo in NodeInfo,
// even when the Pod is assumed one.
func TestAddPodAlwaysUpdatesPodInfoInNodeInfo(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	p1 := makeBasePod(t, "node1", "test-1", "100m", "500", "", []v1.ContainerPort{{HostPort: 80}})

	p2 := p1.DeepCopy()
	p2.Status.Conditions = append(p1.Status.Conditions, v1.PodCondition{
		Type:   v1.PodScheduled,
		Status: v1.ConditionTrue,
	})

	test := struct {
		podsToAssume         []*v1.Pod
		podsToAddAfterAssume []*v1.Pod
		nodeInfo             map[string]*framework.NodeInfo
	}{
		podsToAssume:         []*v1.Pod{p1},
		podsToAddAfterAssume: []*v1.Pod{p2},
		nodeInfo: map[string]*framework.NodeInfo{
			"node1": newNodeInfo(
				&framework.Resource{
					MilliCPU: 100,
					Memory:   500,
				},
				&framework.Resource{
					MilliCPU: 100,
					Memory:   500,
				},
				[]*v1.Pod{p2},
				newHostPortInfoBuilder().add("TCP", "0.0.0.0", 80).build(),
				make(map[string]*fwk.ImageStateSummary),
			),
		},
	}

	cache := newCache(ctx, time.Second, nil, false)
	for _, podToAssume := range test.podsToAssume {
		if err := cache.AssumePod(logger, podToAssume); err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
	}
	for _, podToAdd := range test.podsToAddAfterAssume {
		if err := cache.AddPod(logger, podToAdd); err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
	}
	for nodeName, expected := range test.nodeInfo {
		n := cache.nodes[nodeName]
		if err := deepEqualWithoutGeneration(n, expected); err != nil {
			t.Errorf("node %q: %v", nodeName, err)
		}
	}
}

// TestAddPodWillReplaceAssumed tests that a pod being Add()ed will replace any assumed pod.
func TestAddPodWillReplaceAssumed(t *testing.T) {
	assumedPod := makeBasePod(t, "assumed-node-1", "test-1", "100m", "500", "", []v1.ContainerPort{{HostPort: 80}})
	addedPod := makeBasePod(t, "actual-node", "test-1", "100m", "500", "", []v1.ContainerPort{{HostPort: 80}})
	updatedPod := makeBasePod(t, "actual-node", "test-1", "200m", "500", "", []v1.ContainerPort{{HostPort: 90}})

	test := struct {
		podsToAssume []*v1.Pod
		podsToAdd    []*v1.Pod
		podsToUpdate [][]*v1.Pod

		wNodeInfo map[string]*framework.NodeInfo
	}{
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
				make(map[string]*fwk.ImageStateSummary),
			),
		},
	}

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cache := newCache(ctx, time.Second, nil, false)
	for _, podToAssume := range test.podsToAssume {
		if err := cache.AssumePod(logger, podToAssume); err != nil {
			t.Fatalf("assumePod failed: %v", err)
		}
	}
	for _, podToAdd := range test.podsToAdd {
		if err := cache.AddPod(logger, podToAdd); err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
	}
	for _, podToUpdate := range test.podsToUpdate {
		if err := cache.UpdatePod(logger, podToUpdate[0], podToUpdate[1]); err != nil {
			t.Fatalf("UpdatePod failed: %v", err)
		}
	}
	for nodeName, expected := range test.wNodeInfo {
		n := cache.nodes[nodeName]
		if err := deepEqualWithoutGeneration(n, expected); err != nil {
			t.Errorf("node %q: %v", nodeName, err)
		}
	}
}

// TestUpdatePod tests that a pod will be updated if added before.
func TestUpdatePod(t *testing.T) {
	nodeName := "node"
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	test := struct {
		podsToAdd    []*v1.Pod
		podsToUpdate []*v1.Pod

		wNodeInfo []*framework.NodeInfo
	}{ // add a pod and then update it twice
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
			make(map[string]*fwk.ImageStateSummary),
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
			make(map[string]*fwk.ImageStateSummary),
		)},
	}

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cache := newCache(ctx, time.Second, nil, false)
	for _, podToAdd := range test.podsToAdd {
		if err := cache.AddPod(logger, podToAdd); err != nil {
			t.Fatalf("AddPod failed: %v", err)
		}
	}

	for j := range test.podsToUpdate {
		if j == 0 {
			continue
		}
		if err := cache.UpdatePod(logger, test.podsToUpdate[j-1], test.podsToUpdate[j]); err != nil {
			t.Fatalf("UpdatePod failed: %v", err)
		}
		n := cache.nodes[nodeName]
		if err := deepEqualWithoutGeneration(n, test.wNodeInfo[j-1]); err != nil {
			t.Errorf("update %d: %v", j, err)
		}
	}
}

// TestUpdatePodAndGet tests get always return latest pod state
func TestUpdatePodAndGet(t *testing.T) {
	nodeName := "node"
	testPods := []*v1.Pod{
		makeBasePod(t, nodeName, "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}}),
		makeBasePod(t, nodeName, "test", "200m", "1Ki", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 8080, Protocol: "TCP"}}),
	}
	tests := []struct {
		name        string
		pod         *v1.Pod
		podToUpdate *v1.Pod
		handler     func(logger klog.Logger, cache Cache, pod *v1.Pod) error
		assumePod   bool
	}{
		{
			name:        "do not update pod when pod information has not changed",
			pod:         testPods[0],
			podToUpdate: testPods[0],
			handler: func(logger klog.Logger, cache Cache, pod *v1.Pod) error {
				return cache.AssumePod(logger, pod)
			},
			assumePod: true,
		},
		{
			name:        "update  pod when pod information changed",
			pod:         testPods[0],
			podToUpdate: testPods[1],
			handler: func(logger klog.Logger, cache Cache, pod *v1.Pod) error {
				return cache.AddPod(logger, pod)
			},
			assumePod: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			cache := newCache(ctx, time.Second, nil, false)
			// trying to get an unknown pod should return an error
			// podToUpdate has not been added yet
			if _, err := cache.GetPod(tc.podToUpdate); err == nil {
				t.Error("expected error, no error found")
			}

			// trying to update an unknown pod should return an error
			// pod has not been added yet
			if err := cache.UpdatePod(logger, tc.pod, tc.podToUpdate); err == nil {
				t.Error("expected error, no error found")
			}

			if err := tc.handler(logger, cache, tc.pod); err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			if !tc.assumePod {
				if err := cache.UpdatePod(logger, tc.pod, tc.podToUpdate); err != nil {
					t.Fatalf("UpdatePod failed: %v", err)
				}
			}

			cachedPod, err := cache.GetPod(tc.pod)
			if err != nil {
				t.Fatalf("GetPod failed: %v", err)
			}
			if diff := cmp.Diff(tc.podToUpdate, cachedPod); diff != "" {
				t.Fatalf("Unexpected pod (-want, +got):\n%s", diff)
			}
		})
	}
}

func makePodWithEphemeralStorage(nodeName, ephemeralStorage string) *v1.Pod {
	return st.MakePod().Name("pod-with-ephemeral-storage").Namespace("default-namespace").UID("pod-with-ephemeral-storage").Req(
		map[v1.ResourceName]string{
			v1.ResourceEphemeralStorage: ephemeralStorage,
		},
	).Node(nodeName).Obj()
}

func TestEphemeralStorageResource(t *testing.T) {
	nodeName := "node"
	podE := makePodWithEphemeralStorage(nodeName, "500")
	test := struct {
		pod       *v1.Pod
		wNodeInfo *framework.NodeInfo
	}{
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
			fwk.HostPortInfo{},
			make(map[string]*fwk.ImageStateSummary),
		),
	}
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cache := newCache(ctx, time.Second, nil, false)
	if err := cache.AddPod(logger, test.pod); err != nil {
		t.Fatalf("AddPod failed: %v", err)
	}
	n := cache.nodes[nodeName]
	if err := deepEqualWithoutGeneration(n, test.wNodeInfo); err != nil {
		t.Error(err)
	}

	if err := cache.RemovePod(logger, test.pod); err != nil {
		t.Fatalf("RemovePod failed: %v", err)
	}
	if _, err := cache.GetPod(test.pod); err == nil {
		t.Errorf("pod was not deleted")
	}
}

func Test_AddPodGroupMember(t *testing.T) {
	podGroupName := "pg"
	// Pod with no pod group name.
	pod1 := st.MakePod().Namespace("namespace").Name("non-workload-pod").Obj()
	// Unscheduled pod with a pod group name.
	pod2 := st.MakePod().Namespace("namespace").Name("unscheduled-pod").PodGroupName(podGroupName).Obj()
	// Assigned pod with the same pod group name.
	pod3 := st.MakePod().Namespace("namespace").Name("assigned-pod").Node("node1").PodGroupName(podGroupName).Obj()

	tests := []struct {
		name                    string
		pod                     *v1.Pod
		genericWorkloadEnabled  bool
		expectInUnscheduledPods bool
		expectInAssignedPods    bool
	}{
		{
			name:                   "generic workload disabled",
			pod:                    pod2,
			genericWorkloadEnabled: false,
		},
		{
			name:                   "pod with no pod group name",
			pod:                    pod1,
			genericWorkloadEnabled: true,
		},
		{
			name:                    "unscheduled pod with a pod group name",
			pod:                     pod2,
			genericWorkloadEnabled:  true,
			expectInUnscheduledPods: true,
		},
		{
			name:                   "assigned pod with a pod group name",
			pod:                    pod3,
			genericWorkloadEnabled: true,
			expectInAssignedPods:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := newCache(context.Background(), time.Second, nil, tt.genericWorkloadEnabled)
			cache.AddPodGroupMember(tt.pod)

			if tt.pod.Spec.SchedulingGroup == nil {
				if tt.expectInAssignedPods || tt.expectInUnscheduledPods {
					t.Errorf("Expected pod group to exist, but pod has no pod group")
				}
				return
			}

			podGroupState, err := cache.PodGroupStates().Get(tt.pod.Namespace, *tt.pod.Spec.SchedulingGroup.PodGroupName)
			if err != nil {
				if tt.genericWorkloadEnabled {
					t.Errorf("Expected pod group to exist, but got error: %v", err)
				}
				return
			}

			_, inUnscheduledPods := podGroupState.UnscheduledPods()[tt.pod.Name]
			if inUnscheduledPods != tt.expectInUnscheduledPods {
				t.Errorf("expected pod in UnscheduledPods: %v, got %v", tt.expectInUnscheduledPods, inUnscheduledPods)
			}

			if inAssignedPods := podGroupState.AssignedPods().Has(tt.pod.UID); inAssignedPods != tt.expectInAssignedPods {
				t.Errorf("expected pod in AssignedPods: %v, got %v", tt.expectInAssignedPods, inAssignedPods)
			}
		})
	}
}

func Test_UpdatePodGroupMember(t *testing.T) {
	podGroupName := "pg"
	// unscheduled pod with a pod group name
	pod := st.MakePod().Namespace("namespace").Name("unscheduled-pod").UID("pod1").
		PodGroupName(podGroupName).Obj()
	// updated unscheduled pod with a pod group name
	updatedPod := st.MakePod().Namespace("namespace").Name("unscheduled-pod").UID("pod1").
		Labels(map[string]string{"foo": "bar"}).PodGroupName(podGroupName).Obj()
	// assigned pod with a pod group name
	assignedPod := st.MakePod().Namespace("namespace").Name("assigned-pod").UID("pod2").Node("node").PodGroupName(podGroupName).Obj()
	// pod with no pod group name
	noPodGroupPod := st.MakePod().Namespace("namespace").Name("no-pod-group-pod").UID("pod3").Obj()
	// updated pod with no pod group name
	updatedNoPodGroupPod := st.MakePod().Namespace("namespace").Name("no-pod-group-pod").UID("pod3").
		Labels(map[string]string{"foo": "bar"}).Obj()

	tests := []struct {
		name                    string
		isAssumedPod            bool
		oldPod                  *v1.Pod
		newPod                  *v1.Pod
		genericWorkloadEnabled  bool
		expectInAssumedPods     bool
		expectInUnscheduledPods bool
		expectInAssignedPods    bool
	}{
		{
			name:                    "updating a pod with genericWorkload disabled should be a no-op",
			oldPod:                  pod,
			newPod:                  updatedPod,
			genericWorkloadEnabled:  false,
			expectInUnscheduledPods: true,
		},
		{
			name:                   "update a pod with no pod group name should be a no-op",
			oldPod:                 noPodGroupPod,
			newPod:                 updatedNoPodGroupPod,
			genericWorkloadEnabled: true,
		},
		{
			name:                    "update a pod",
			isAssumedPod:            true,
			oldPod:                  pod,
			newPod:                  updatedPod,
			genericWorkloadEnabled:  true,
			expectInUnscheduledPods: true,
		},
		{
			name:                   "update a pod, move to assigned",
			isAssumedPod:           true,
			oldPod:                 pod,
			newPod:                 assignedPod,
			genericWorkloadEnabled: true,
			expectInAssignedPods:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			cache := newCache(context.Background(), time.Second, nil, true)
			cache.AddPodGroupMember(tt.oldPod)
			cache.genericWorkloadEnabled = tt.genericWorkloadEnabled

			newPod := tt.newPod
			if newPod == nil {
				newPod = tt.oldPod
			}
			cache.UpdatePodGroupMember(logger, tt.oldPod, newPod)

			if newPod.Spec.SchedulingGroup == nil {
				if tt.expectInAssumedPods || tt.expectInUnscheduledPods || tt.expectInAssignedPods {
					t.Errorf("Expected pod group to exist, but pod has no SchedulingGroup")
				}
				return
			}

			podGroupState, err := cache.PodGroupStates().Get(newPod.Namespace, *newPod.Spec.SchedulingGroup.PodGroupName)
			if err != nil {
				return
			}

			_, inUnscheduledPods := podGroupState.UnscheduledPods()[newPod.Name]
			if inUnscheduledPods != tt.expectInUnscheduledPods {
				t.Errorf("expected pod in UnscheduledPods: %v, got %v", tt.expectInUnscheduledPods, inUnscheduledPods)
			}

			if inAssignedPods := podGroupState.AssignedPods().Has(newPod.UID); inAssignedPods != tt.expectInAssignedPods {
				t.Errorf("expected pod in AssignedPods: %v, got %v", tt.expectInAssignedPods, inAssignedPods)
			}

			if inAssumedPods := podGroupState.AssumedPods().Has(newPod.UID); inAssumedPods != tt.expectInAssumedPods {
				t.Errorf("expected pod in AssumedPods: %v, got %v", tt.expectInAssumedPods, inAssumedPods)
			}

			if !tt.genericWorkloadEnabled {
				return
			}

			podGroupKey := newPodGroupKey(newPod.Namespace, *newPod.Spec.SchedulingGroup.PodGroupName)
			gotPod := cache.podGroupStates[podGroupKey].allPods[newPod.UID]
			if diff := cmp.Diff(tt.newPod, gotPod); diff != "" {
				t.Errorf("stored pod does not match newPod (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_RemovePodGroupMember(t *testing.T) {
	podGroupName := "pg"
	pod1 := st.MakePod().Namespace("namespace").Name("unscheduled-pod").UID("pod1").
		PodGroupName(podGroupName).Obj()
	pod2 := st.MakePod().Namespace("namespace").Name("assigned-pod").UID("pod2").Node("node").
		PodGroupName(podGroupName).Obj()

	tests := []struct {
		name                     string
		initPods                 []*v1.Pod
		podToDelete              *v1.Pod
		expectPodGroupStateCount int
		genericWorkloadEnabled   bool
	}{
		{
			name:                     "remove a pod from a group with multiple pods",
			initPods:                 []*v1.Pod{pod1, pod2},
			podToDelete:              pod1,
			expectPodGroupStateCount: 1,
			genericWorkloadEnabled:   true,
		},
		{
			name:                     "remove a last pod from a group",
			initPods:                 []*v1.Pod{pod1},
			podToDelete:              pod1,
			expectPodGroupStateCount: 0,
			genericWorkloadEnabled:   true,
		},
		{
			name:                     "remove a non-existent pod from a group should be a no-op",
			podToDelete:              pod1,
			expectPodGroupStateCount: 0,
			genericWorkloadEnabled:   true,
		},
		{
			name:                     "remove a non-existent pod from a group should be a no-op",
			podToDelete:              pod1,
			expectPodGroupStateCount: 0,
			genericWorkloadEnabled:   true,
		},
		{
			name:                     "remove a pod while generic workload disabled should be a no-op",
			initPods:                 []*v1.Pod{pod1},
			expectPodGroupStateCount: 0,
			podToDelete:              pod1,
			genericWorkloadEnabled:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := newCache(context.Background(), time.Second, nil, tt.genericWorkloadEnabled)

			for _, pod := range tt.initPods {
				cache.AddPodGroupMember(pod)
			}

			cache.RemovePodGroupMember(tt.podToDelete)

			podGroupStateCount := len(cache.podGroupStates)
			if podGroupStateCount != tt.expectPodGroupStateCount {
				t.Errorf("expected %d pod groups remaining, got %d", tt.expectPodGroupStateCount, podGroupStateCount)
			}

			if podGroupStateCount == 0 {
				return
			}

			podGroupState, err := cache.PodGroupStates().Get(tt.podToDelete.Namespace, *tt.podToDelete.Spec.SchedulingGroup.PodGroupName)
			if err != nil {
				t.Fatalf("Unexpected error getting pod group state: %v", err)
			}

			if podGroupState.AllPods().Has(tt.podToDelete.UID) {
				t.Errorf("Expected pod %s to be deleted from pod group but it still exists", tt.podToDelete.UID)
			}
		})
	}
}

// TestUpdatePodGroupStateSnapshot tests that pod group states of the snapshot have
// their data and generations updated properly.
func TestUpdatePodGroupStateSnapshot(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	cache := newCache(ctx, time.Second, nil, true)

	podGroupName1 := "pg1"
	podGroupName2 := "pg2"
	pod1 := st.MakePod().Namespace("ns").Name("pod1").UID("uid1").PodGroupName(podGroupName1).Obj()
	pod2 := st.MakePod().Namespace("ns").Name("pod2").UID("uid2").PodGroupName(podGroupName1).Obj()
	pod3 := st.MakePod().Namespace("ns").Name("pod3").UID("uid3").PodGroupName(podGroupName2).Obj()

	snapshot := NewEmptySnapshot()

	tests := []struct {
		name         string
		action       func()
		expectedPods []*v1.Pod
	}{
		{
			name:         "add a pod group member and update snapshot",
			action:       func() { cache.AddPodGroupMember(pod1) },
			expectedPods: []*v1.Pod{pod1},
		},
		{
			name:         "add a pod with different pod group and update snapshot",
			action:       func() { cache.AddPodGroupMember(pod3) },
			expectedPods: []*v1.Pod{pod1, pod3},
		},
		{
			name:         "remove a last pod group member and update snapshot",
			action:       func() { cache.RemovePodGroupMember(pod1) },
			expectedPods: []*v1.Pod{pod3},
		},
		{
			name:         "add a pod to a recently deleted pod group and update snapshot",
			action:       func() { cache.AddPodGroupMember(pod2) },
			expectedPods: []*v1.Pod{pod2, pod3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Capture cache generations before snapshot update to detect which pod groups are going to be modified.
			prevCacheGenerations := make(map[podGroupKey]int64, len(cache.podGroupStates))
			for key, pgs := range cache.podGroupStates {
				prevCacheGenerations[key] = pgs.generation
			}

			tt.action()
			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("UpdateSnapshot failed: %v", err)
			}

			// For each pod group that the action modified (its cache generation advanced), the snapshot generation must have advanced too.
			// Unmodified pod groups keep their previous generation.
			for key, pgs := range snapshot.podGroupStates {
				cachePgs, ok := cache.podGroupStates[key]
				if !ok {
					continue
				}
				if cachePgs.generation > prevCacheGenerations[key] {
					if pgs.generation <= prevCacheGenerations[key] {
						t.Errorf("pod group %s was modified but snapshot generation (%d) was not incremented (%d)", key, pgs.generation, prevCacheGenerations[key])
					}
				}
			}

			expectedPodGroupStatesSnapshot := createPodGroupStates(tt.expectedPods)
			if diff := cmp.Diff(expectedPodGroupStatesSnapshot, snapshot.podGroupStates, podGroupStateCmpOpts...); diff != "" {
				t.Errorf("snapshot data mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestBindingPodGroupMember simulates binding and tests that when an assumed pod
// gets bound, its state within pod group transitions from assumed to assigned.
func TestBindingPodGroupMember(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	cache := newCache(ctx, time.Second, nil, true)
	podGroupName := "pg"
	pod := st.MakePod().Namespace("namespace").Name("pod1").UID("pod1-uid").
		PodGroupName(podGroupName).Obj()

	// Simulate the informer firing an Add event for an unscheduled
	// pod (no NodeName set) reflecting on PodGroupStates.
	cache.AddPodGroupMember(pod)

	// Simulate the scheduler assuming the pod on a node.
	assumedPod := pod.DeepCopy()
	assumedPod.Spec.NodeName = "node1"
	if err := cache.AssumePod(logger, assumedPod); err != nil {
		t.Fatalf("AssumePod failed: %v", err)
	}

	podGroupState, err := cache.PodGroupStates().Get(pod.Namespace, podGroupName)
	if err != nil {
		t.Fatalf("Unexpected error getting pod group state after AssumePod: %v", err)
	}
	if !podGroupState.AssumedPods().Has(assumedPod.UID) {
		t.Errorf("Expected pod to be in AssumedPods after AssumePod")
	}
	if podGroupState.AssignedPods().Has(assumedPod.UID) {
		t.Errorf("Expected pod NOT to be in AssignedPods after AssumePod")
	}

	// Simulate binding confirmation: the informer fires an Add event with NodeName set.
	if err := cache.AddPod(logger, assumedPod); err != nil {
		t.Fatalf("AddPod (binding confirmation) failed: %v", err)
	}

	podGroupState, err = cache.PodGroupStates().Get(pod.Namespace, podGroupName)
	if err != nil {
		t.Fatalf("Unexpected error getting pod group state after AddPod: %v", err)
	}
	if podGroupState.AssumedPods().Has(assumedPod.UID) {
		t.Errorf("Expected pod not to be in AssumedPods after binding confirmation")
	}
	if !podGroupState.AssignedPods().Has(assumedPod.UID) {
		t.Errorf("Expected pod to be in AssignedPods after binding confirmation")
	}
}

// TestRemovePod tests after added pod is removed, its information should also be subtracted.
func TestRemovePod(t *testing.T) {
	pod := makeBasePod(t, "node-1", "test", "100m", "500", "", []v1.ContainerPort{{HostIP: "127.0.0.1", HostPort: 80, Protocol: "TCP"}})
	nodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "node-2"},
		},
	}
	wNodeInfo := newNodeInfo(
		&framework.Resource{
			MilliCPU: 100,
			Memory:   500,
		},
		&framework.Resource{
			MilliCPU: 100,
			Memory:   500,
		},
		[]*v1.Pod{pod},
		newHostPortInfoBuilder().add("TCP", "127.0.0.1", 80).build(),
		make(map[string]*fwk.ImageStateSummary),
	)
	tests := map[string]struct {
		assume bool
	}{
		"bound":   {},
		"assumed": {assume: true},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			nodeName := pod.Spec.NodeName
			cache := newCache(ctx, time.Second, nil, false)
			// Add/Assume pod succeeds even before adding the nodes.
			if tt.assume {
				if err := cache.AddPod(logger, pod); err != nil {
					t.Fatalf("AddPod failed: %v", err)
				}
			} else {
				if err := cache.AssumePod(logger, pod); err != nil {
					t.Fatalf("AssumePod failed: %v", err)
				}
			}
			n := cache.nodes[nodeName]
			if err := deepEqualWithoutGeneration(n, wNodeInfo); err != nil {
				t.Error(err)
			}
			for _, n := range nodes {
				cache.AddNode(logger, n)
			}

			if err := cache.RemovePod(logger, pod); err != nil {
				t.Fatalf("RemovePod failed: %v", err)
			}

			if _, err := cache.GetPod(pod); err == nil {
				t.Errorf("pod was not deleted")
			}

			// trying to remove a pod already removed should return an error
			if err := cache.RemovePod(logger, pod); err == nil {
				t.Error("expected error, no error found")
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
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	cache := newCache(ctx, time.Second, nil, false)
	for _, pod := range pods {
		if err := cache.AssumePod(logger, pod); err != nil {
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
		if err := cache.ForgetPod(logger, pod); err != nil {
			t.Fatalf("ForgetPod failed: %v", err)
		}
		if err := isForgottenFromCache(pod, cache); err != nil {
			t.Errorf("pod %q: %v", pod.Name, err)
		}
		// trying to forget a pod already forgotten should return nil
		if err := cache.ForgetPod(logger, pod); err != nil {
			t.Error("expected no error, error found")
		}
	}
}

// buildNodeInfo creates a NodeInfo by simulating node operations in cache.
func buildNodeInfo(node *v1.Node, pods []*v1.Pod, imageStates map[string]*fwk.ImageStateSummary) *framework.NodeInfo {
	expected := framework.NewNodeInfo()
	expected.SetNode(node)
	expected.Allocatable = framework.NewResource(node.Status.Allocatable)
	expected.Generation++
	for _, pod := range pods {
		expected.AddPod(pod)
	}
	for _, image := range node.Status.Images {
		for _, name := range image.Names {
			if state, ok := imageStates[name]; ok {
				expected.ImageStates[name] = state
			}
		}
	}
	return expected
}

// buildImageStates creates ImageStateSummary of image from nodes that will be added in cache.
func buildImageStates(nodes []*v1.Node) map[string]*fwk.ImageStateSummary {
	imageStates := make(map[string]*fwk.ImageStateSummary)
	for _, item := range nodes {
		for _, image := range item.Status.Images {
			for _, name := range image.Names {
				if state, ok := imageStates[name]; !ok {
					state = &fwk.ImageStateSummary{
						Size:  image.SizeBytes,
						Nodes: sets.New[string](item.Name),
					}
					imageStates[name] = state
				} else {
					state.Nodes.Insert(item.Name)
				}
			}
		}
	}
	return imageStates
}

// TestNodeOperators tests node operations of cache, including add, update
// and remove.
func TestNodeOperators(t *testing.T) {
	// Test data
	cpuHalf := resource.MustParse("500m")
	mem50m := resource.MustParse("50m")
	resourceList1 := map[v1.ResourceName]string{
		v1.ResourceCPU:                     "1000m",
		v1.ResourceMemory:                  "100m",
		v1.ResourceName("example.com/foo"): "1",
	}
	resourceList2 := map[v1.ResourceName]string{
		v1.ResourceCPU:                     "500m",
		v1.ResourceMemory:                  "50m",
		v1.ResourceName("example.com/foo"): "2",
	}
	taints := []v1.Taint{
		{
			Key:    "test-key",
			Value:  "test-value",
			Effect: v1.TaintEffectPreferNoSchedule,
		},
	}
	imageStatus1 := map[string]int64{
		"gcr.io/80:latest":  80 * mb,
		"gcr.io/80:v1":      80 * mb,
		"gcr.io/300:latest": 300 * mb,
		"gcr.io/300:v1":     300 * mb,
	}
	imageStatus2 := map[string]int64{
		"gcr.io/600:latest": 600 * mb,
		"gcr.io/80:latest":  80 * mb,
		"gcr.io/900:latest": 900 * mb,
	}
	tests := []struct {
		name  string
		nodes []*v1.Node
		pods  []*v1.Pod
	}{
		{
			name: "operate the node with one pod",
			nodes: []*v1.Node{
				&st.MakeNode().Name("test-node-1").Capacity(resourceList1).Taints(taints).Images(imageStatus1).Node,
				&st.MakeNode().Name("test-node-2").Capacity(resourceList2).Taints(taints).Images(imageStatus2).Node,
				&st.MakeNode().Name("test-node-3").Capacity(resourceList1).Taints(taints).Images(imageStatus1).Node,
				&st.MakeNode().Name("test-node-4").Capacity(resourceList2).Taints(taints).Images(imageStatus2).Node,
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod1",
						UID:  types.UID("pod1"),
					},
					Spec: v1.PodSpec{
						NodeName: "test-node-1",
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
			name: "operate the node with two pods",
			nodes: []*v1.Node{
				&st.MakeNode().Name("test-node-1").Capacity(resourceList1).Taints(taints).Images(imageStatus1).Node,
				&st.MakeNode().Name("test-node-2").Capacity(resourceList2).Taints(taints).Images(imageStatus2).Node,
				&st.MakeNode().Name("test-node-3").Capacity(resourceList1).Taints(taints).Images(imageStatus1).Node,
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod1",
						UID:  types.UID("pod1"),
					},
					Spec: v1.PodSpec{
						NodeName: "test-node-1",
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
						NodeName: "test-node-1",
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

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			node := tc.nodes[0]

			imageStates := buildImageStates(tc.nodes)
			expected := buildNodeInfo(node, tc.pods, imageStates)

			cache := newCache(ctx, time.Second, nil, false)
			for _, nodeItem := range tc.nodes {
				cache.AddNode(logger, nodeItem)
			}
			for _, pod := range tc.pods {
				if err := cache.AddPod(logger, pod); err != nil {
					t.Fatal(err)
				}
			}
			nodes := map[string]*framework.NodeInfo{}
			for nodeItem := cache.headNode; nodeItem != nil; nodeItem = nodeItem.next {
				nodes[nodeItem.info.Node().Name] = nodeItem.info
			}

			// Step 1: the node was added into cache successfully.
			got, found := cache.nodes[node.Name]
			if !found {
				t.Errorf("Failed to find node %v in internalcache.", node.Name)
			}
			nodesList, err := cache.nodeTree.list()
			if err != nil {
				t.Fatal(err)
			}
			if cache.nodeTree.numNodes != len(tc.nodes) || len(nodesList) != len(tc.nodes) {
				t.Errorf("cache.nodeTree is not updated correctly after adding node got: %d, expected: %d",
					cache.nodeTree.numNodes, len(tc.nodes))
			}

			// Generations are globally unique. We check in our unit tests that they are incremented correctly.
			expected.Generation = got.info.Generation
			if diff := cmp.Diff(expected, got.info, nodeInfoCmpOpts...); diff != "" {
				t.Errorf("Failed to add node into scheduler cache (-want,+got):\n%s", diff)
			}

			// check imageState of NodeInfo with specific image when node added
			if !checkImageStateSummary(nodes, "gcr.io/80:latest", "gcr.io/300:latest") {
				t.Error("image have different ImageStateSummary")
			}

			// Step 2: dump cached nodes successfully.
			cachedNodes := NewEmptySnapshot()
			if err := cache.UpdateSnapshot(logger, cachedNodes); err != nil {
				t.Error(err)
			}
			newNode, found := cachedNodes.nodeInfoMap[node.Name]
			if !found || len(cachedNodes.nodeInfoMap) != len(tc.nodes) {
				t.Errorf("failed to dump cached nodes:\n got: %v \nexpected: %v", cachedNodes.nodeInfoMap, tc.nodes)
			}
			expected.Generation = newNode.Generation
			if diff := cmp.Diff(expected.Snapshot(), newNode, nodeInfoCmpOpts...); diff != "" {
				t.Errorf("Failed to clone node (-want,+got):\n%s", diff)
			}
			// check imageState of NodeInfo with specific image when update snapshot
			if !checkImageStateSummary(cachedNodes.nodeInfoMap, "gcr.io/80:latest", "gcr.io/300:latest") {
				t.Error("image have different ImageStateSummary")
			}

			// Step 3: update node attribute successfully.
			node.Status.Allocatable[v1.ResourceMemory] = mem50m
			expected.Allocatable.Memory = mem50m.Value()

			cache.UpdateNode(logger, nil, node)
			got, found = cache.nodes[node.Name]
			if !found {
				t.Errorf("Failed to find node %v in schedulertypes after UpdateNode.", node.Name)
			}
			if got.info.Generation <= expected.Generation {
				t.Errorf("Generation is not incremented. got: %v, expected: %v", got.info.Generation, expected.Generation)
			}
			expected.Generation = got.info.Generation

			if diff := cmp.Diff(expected, got.info, nodeInfoCmpOpts...); diff != "" {
				t.Errorf("Unexpected schedulertypes after updating node (-want, +got):\n%s", diff)
			}
			// check imageState of NodeInfo with specific image when update node
			if !checkImageStateSummary(nodes, "gcr.io/80:latest", "gcr.io/300:latest") {
				t.Error("image have different ImageStateSummary")
			}
			// Check nodeTree after update
			nodesList, err = cache.nodeTree.list()
			if err != nil {
				t.Fatal(err)
			}
			if cache.nodeTree.numNodes != len(tc.nodes) || len(nodesList) != len(tc.nodes) {
				t.Errorf("unexpected cache.nodeTree after updating node")
			}

			// Step 4: the node can be removed even if it still has pods.
			if err := cache.RemoveNode(logger, node); err != nil {
				t.Error(err)
			}
			if n, err := cache.getNodeInfo(node.Name); err != nil {
				t.Errorf("The node %v should still have a ghost entry: %v", node.Name, err)
			} else if n != nil {
				t.Errorf("The node object for %v should be nil", node.Name)
			}

			// trying to remove a node already removed should return an error
			if err := cache.RemoveNode(logger, node); err == nil {
				t.Error("expected error, no error found")
			}

			// Check node is removed from nodeTree as well.
			nodesList, err = cache.nodeTree.list()
			if err != nil {
				t.Fatal(err)
			}
			if cache.nodeTree.numNodes != len(tc.nodes)-1 || len(nodesList) != len(tc.nodes)-1 {
				t.Errorf("unexpected cache.nodeTree after removing node: %v", node.Name)
			}
			// check imageState of NodeInfo with specific image when delete node
			if !checkImageStateSummary(nodes, "gcr.io/80:latest", "gcr.io/300:latest") {
				t.Error("image have different ImageStateSummary after removing node")
			}
			// Pods are still in the pods cache.
			for _, p := range tc.pods {
				if _, err := cache.GetPod(p); err != nil {
					t.Error(err)
				}
			}

			// Step 5: removing pods for the removed node still succeeds.
			for _, p := range tc.pods {
				if err := cache.RemovePod(logger, p); err != nil {
					t.Error(err)
				}
				if _, err := cache.GetPod(p); err == nil {
					t.Errorf("pod %q still in cache", p.Name)
				}
			}
		})
	}
}

// TestPodGroupPodOperations tests that operations (Add, Update, Remove, Assume, Forget) on
// pods with pod group name properly update PodGroupStates only when GenericWorkload feature gate is enabled.
func TestPodGroupPodOperations(t *testing.T) {
	groupName := "pg"
	pod := st.MakePod().Namespace("test-ns").Name("pod-0").UID("uid-0").
		PodGroupName(groupName).Obj()

	type state struct {
		podGroupStatesCount int
		assignedCount       int
		unscheduledCount    int
		assumedCount        int
	}

	tests := []struct {
		name                   string
		genericWorkloadEnabled bool
		setup                  func(*testing.T, *cacheImpl, context.Context)
		operation              func(*testing.T, *cacheImpl, context.Context)
		expected               state
	}{
		{
			name:                   "AddPod with GenericWorkload disabled",
			genericWorkloadEnabled: false,
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AddPod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AddPod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 0},
		},
		{
			name:                   "AddPod with GenericWorkload enabled",
			genericWorkloadEnabled: true,
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AddPod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AddPod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 1, unscheduledCount: 1, assignedCount: 0, assumedCount: 0},
		},
		{
			name:                   "AssumePod with GenericWorkload disabled",
			genericWorkloadEnabled: false,
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AssumePod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AssumePod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 0},
		},
		{
			name:                   "AssumePod with GenericWorkload enabled",
			genericWorkloadEnabled: true,
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AssumePod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AssumePod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 1, assignedCount: 0, unscheduledCount: 0, assumedCount: 1},
		},
		{
			name:                   "ForgetPod with GenericWorkload disabled",
			genericWorkloadEnabled: false,
			setup: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AssumePod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AssumePod failed: %v", err)
				}
			},
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.ForgetPod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("ForgetPod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 1, assignedCount: 0, unscheduledCount: 0, assumedCount: 1},
		},
		{
			name:                   "ForgetPod with GenericWorkload enabled",
			genericWorkloadEnabled: true,
			setup: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AssumePod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AssumePod failed: %v", err)
				}
			},
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.ForgetPod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("ForgetPod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 1, unscheduledCount: 1},
		},
		{
			name:                   "RemovePod with GenericWorkload disabled",
			genericWorkloadEnabled: false,
			setup: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AddPod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AddPod failed: %v", err)
				}
			},
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.RemovePod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("RemovePod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 1, assignedCount: 0, unscheduledCount: 1, assumedCount: 0},
		},
		{
			name:                   "RemovePod with GenericWorkload enabled",
			genericWorkloadEnabled: true,
			setup: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.AddPod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("AddPod failed: %v", err)
				}
			},
			operation: func(t *testing.T, cache *cacheImpl, ctx context.Context) {
				if err := cache.RemovePod(klog.FromContext(ctx), pod); err != nil {
					t.Fatalf("RemovePod failed: %v", err)
				}
			},
			expected: state{podGroupStatesCount: 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			// Initialize cache with feature gate enabled to ensure group state is
			// properly established for operations that require it.
			cache := newCache(ctx, time.Second, nil, true)
			if tt.setup != nil {
				tt.setup(t, cache, ctx)
			}
			cache.genericWorkloadEnabled = tt.genericWorkloadEnabled
			tt.operation(t, cache, ctx)

			if count := len(cache.podGroupStates); count != tt.expected.podGroupStatesCount {
				t.Errorf("expected %d pod group states, got %d", tt.expected.podGroupStatesCount, count)
			}

			if tt.expected.podGroupStatesCount == 0 {
				return
			}

			pgs, err := cache.PodGroupStates().Get("test-ns", groupName)
			if err != nil {
				t.Fatalf("unexpected error getting pod group state: %v", err)
			}

			assignedCount := pgs.AssignedPods().Len()
			if assignedCount != tt.expected.assignedCount {
				t.Errorf("expected %d pods in assignedPods, got %d", tt.expected.assignedCount, assignedCount)
			}

			unscheduledCount := len(pgs.UnscheduledPods())
			if unscheduledCount != tt.expected.unscheduledCount {
				t.Errorf("expected %d pods in unscheduledPods, got %d", tt.expected.unscheduledCount, unscheduledCount)
			}

			assumedCount := pgs.AssumedPods().Len()
			if assumedCount != tt.expected.assumedCount {
				t.Errorf("expected %d pods in assumedPods, got %d", tt.expected.assumedCount, assumedCount)
			}
		})
	}
}

func TestSchedulerCache_UpdateSnapshot(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	var podGroupName = "pg"

	// Create a few nodes to be used in tests.
	var nodes []*v1.Node
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
	var updatedNodes []*v1.Node
	for _, n := range nodes {
		updatedNode := n.DeepCopy()
		updatedNode.Status.Allocatable = v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("2000m"),
			v1.ResourceMemory: resource.MustParse("500m"),
		}
		updatedNodes = append(updatedNodes, updatedNode)
	}

	// Create a few pods for tests.
	var pods []*v1.Pod
	for i := 0; i < 20; i++ {
		pod := st.MakePod().Name(fmt.Sprintf("test-pod%v", i)).Namespace("test-ns").UID(fmt.Sprintf("test-puid%v", i)).
			Node(fmt.Sprintf("test-node%v", i%10)).Obj()
		pods = append(pods, pod)
	}

	// Create a few pods as updated versions of the above pods.
	var updatedPods []*v1.Pod
	for _, p := range pods {
		updatedPod := p.DeepCopy()
		priority := int32(1000)
		updatedPod.Spec.Priority = &priority
		updatedPods = append(updatedPods, updatedPod)
	}

	// Add a few of pods with affinity
	var podsWithAffinity []*v1.Pod
	for i := range 20 {
		pod := st.MakePod().Name(fmt.Sprintf("p-affinity-%v", i)).Namespace("test-ns").UID(fmt.Sprintf("puid-affinity-%v", i)).
			PodAffinityExists("foo", "", st.PodAffinityWithRequiredReq).Node(fmt.Sprintf("test-node%v", i)).Obj()
		podsWithAffinity = append(podsWithAffinity, pod)
	}

	makePodWithPVC := func(podID int, node int, pvcID int) *v1.Pod {
		return st.MakePod().Name(fmt.Sprintf("p-pvc-%v", podID)).Namespace("test-ns").UID(fmt.Sprintf("puid-pvc-%v", podID)).
			PVC(fmt.Sprintf("test-pvc%v", pvcID)).Node(fmt.Sprintf("test-node%v", node)).Obj()
	}

	// Add a few pods with a pod group name
	var podsWithPodGroupName []*v1.Pod
	for i := range 20 {
		pod := st.MakePod().Name(fmt.Sprintf("p-podgroup-%v", i)).Namespace("test-ns").UID(fmt.Sprintf("puid-podgroup-%v", i)).
			PodGroupName(fmt.Sprintf("%s-%v", podGroupName, i)).
			Node(fmt.Sprintf("test-node%v", i)).Obj()
		podsWithPodGroupName = append(podsWithPodGroupName, pod)
	}

	var cache *cacheImpl
	var snapshot *Snapshot
	type operation = func(t *testing.T)

	addNode := func(i int) operation {
		return func(t *testing.T) {
			cache.AddNode(logger, nodes[i])
		}
	}
	removeNode := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.RemoveNode(logger, nodes[i]); err != nil {
				t.Error(err)
			}
		}
	}
	updateNode := func(i int) operation {
		return func(t *testing.T) {
			cache.UpdateNode(logger, nodes[i], updatedNodes[i])
		}
	}
	addPod := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.AddPod(logger, pods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	addPodWithAffinity := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.AddPod(logger, podsWithAffinity[i]); err != nil {
				t.Error(err)
			}
		}
	}
	addPodWithPVC := func(podID int, node int, pvcID int) operation {
		return func(t *testing.T) {
			pod := makePodWithPVC(podID, node, pvcID)
			if err := cache.AddPod(logger, pod); err != nil {
				t.Error(err)
			}
		}
	}
	addPodWithPodGroupName := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.AddPod(logger, podsWithPodGroupName[i]); err != nil {
				t.Error(err)
			}
		}
	}
	removePod := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.RemovePod(logger, pods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	removePodWithAffinity := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.RemovePod(logger, podsWithAffinity[i]); err != nil {
				t.Error(err)
			}
		}
	}
	removePodWithPVC := func(podID int, node int, pvcID int) operation {
		return func(t *testing.T) {
			pod := makePodWithPVC(podID, node, pvcID)
			if err := cache.RemovePod(logger, pod); err != nil {
				t.Error(err)
			}
		}
	}
	removePodWithPodGroupName := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.RemovePod(logger, podsWithPodGroupName[i]); err != nil {
				t.Error(err)
			}
		}
	}
	updatePod := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.UpdatePod(logger, pods[i], updatedPods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	updatePodWithPodGroupName := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.UpdatePod(logger, podsWithPodGroupName[i], podsWithPodGroupName[i]); err != nil {
				t.Error(err)
			}
		}
	}
	assumePod := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.AssumePod(logger, pods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	assumePodWithAffinity := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.AssumePod(logger, podsWithAffinity[i]); err != nil {
				t.Error(err)
			}
		}
	}
	assumePodWithPVC := func(podID int, node int, pvcID int) operation {
		return func(t *testing.T) {
			pod := makePodWithPVC(podID, node, pvcID)
			if err := cache.AssumePod(logger, pod); err != nil {
				t.Error(err)
			}
		}
	}
	assumePodInSnapshot := func(i int) operation {
		return func(t *testing.T) {
			podInfo, _ := framework.NewPodInfo(pods[i])
			if err := snapshot.AssumePod(podInfo); err != nil {
				t.Error(err)
			}
		}
	}
	assumePodWithAffinityInSnapshot := func(i int) operation {
		return func(t *testing.T) {
			podInfo, _ := framework.NewPodInfo(podsWithAffinity[i])
			if err := snapshot.AssumePod(podInfo); err != nil {
				t.Error(err)
			}
		}
	}
	assumePodWithPVCInSnapshot := func(podID int, node int, pvcID int) operation {
		return func(t *testing.T) {
			pod := makePodWithPVC(podID, node, pvcID)
			podInfo, _ := framework.NewPodInfo(pod)
			if err := snapshot.AssumePod(podInfo); err != nil {
				t.Error(err)
			}
		}
	}
	forgetPod := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.ForgetPod(logger, pods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	forgetPodWithAffinity := func(i int) operation {
		return func(t *testing.T) {
			if err := cache.ForgetPod(logger, podsWithAffinity[i]); err != nil {
				t.Error(err)
			}
		}
	}
	forgetPodWithPVC := func(podID int, node int, pvcID int) operation {
		return func(t *testing.T) {
			pod := makePodWithPVC(podID, node, pvcID)
			if err := cache.ForgetPod(logger, pod); err != nil {
				t.Error(err)
			}
		}
	}
	forgetPodInSnapshot := func(i int) operation {
		return func(t *testing.T) {
			if err := snapshot.ForgetPod(logger, pods[i]); err != nil {
				t.Error(err)
			}
		}
	}
	forgetPodWithAffinityInSnapshot := func(i int) operation {
		return func(t *testing.T) {
			if err := snapshot.ForgetPod(logger, podsWithAffinity[i]); err != nil {
				t.Error(err)
			}
		}
	}
	forgetPodWithPVCInSnapshot := func(podID int, node int, pvcID int) operation {
		return func(t *testing.T) {
			pod := makePodWithPVC(podID, node, pvcID)
			if err := snapshot.ForgetPod(logger, pod); err != nil {
				t.Error(err)
			}
		}
	}
	updateSnapshot := func() operation {
		return func(t *testing.T) {
			cache.UpdateSnapshot(logger, snapshot)
			if err := compareCacheWithNodeInfoSnapshot(t, cache, snapshot); err != nil {
				t.Error(err)
			}
		}
	}

	tests := []struct {
		name                           string
		operations                     []operation
		expected                       []*v1.Node
		expectedHavePodsWithAffinity   int
		expectedPodGroupStatesSnapshot map[podGroupKey]*podGroupStateSnapshot
		expectedUsedPVCSet             sets.Set[string]
	}{
		{
			name:               "Empty cache",
			operations:         []operation{},
			expected:           []*v1.Node{},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name:               "Single node",
			operations:         []operation{addNode(1)},
			expected:           []*v1.Node{nodes[1]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add node, remove it, add it again",
			operations: []operation{
				addNode(1), updateSnapshot(), removeNode(1), addNode(1),
			},
			expected:           []*v1.Node{nodes[1]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add node and remove it in the same cycle, add it again",
			operations: []operation{
				addNode(1), updateSnapshot(), addNode(2), removeNode(1),
			},
			expected:           []*v1.Node{nodes[2]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add a few nodes, and snapshot in the middle",
			operations: []operation{
				addNode(0), updateSnapshot(), addNode(1), updateSnapshot(), addNode(2),
				updateSnapshot(), addNode(3),
			},
			expected:           []*v1.Node{nodes[3], nodes[2], nodes[1], nodes[0]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add a few nodes, and snapshot in the end",
			operations: []operation{
				addNode(0), addNode(2), addNode(5), addNode(6),
			},
			expected:           []*v1.Node{nodes[6], nodes[5], nodes[2], nodes[0]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Update some nodes",
			operations: []operation{
				addNode(0), addNode(1), addNode(5), updateSnapshot(), updateNode(1),
			},
			expected:           []*v1.Node{nodes[1], nodes[5], nodes[0]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add a few nodes, and remove all of them",
			operations: []operation{
				addNode(0), addNode(2), addNode(5), addNode(6), updateSnapshot(),
				removeNode(0), removeNode(2), removeNode(5), removeNode(6),
			},
			expected:           []*v1.Node{},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add a few nodes, and remove some of them",
			operations: []operation{
				addNode(0), addNode(2), addNode(5), addNode(6), updateSnapshot(),
				removeNode(0), removeNode(6),
			},
			expected:           []*v1.Node{nodes[5], nodes[2]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add a few nodes, remove all of them, and add more",
			operations: []operation{
				addNode(2), addNode(5), addNode(6), updateSnapshot(),
				removeNode(2), removeNode(5), removeNode(6), updateSnapshot(),
				addNode(7), addNode(9),
			},
			expected:           []*v1.Node{nodes[9], nodes[7]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Update nodes in particular order",
			operations: []operation{
				addNode(8), updateNode(2), updateNode(8), updateSnapshot(),
				addNode(1),
			},
			expected:           []*v1.Node{nodes[1], nodes[8], nodes[2]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add some nodes and some pods",
			operations: []operation{
				addNode(0), addNode(2), addNode(8), updateSnapshot(),
				addPod(8), addPod(2),
			},
			expected:           []*v1.Node{nodes[2], nodes[8], nodes[0]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Updating a pod moves its node to the head",
			operations: []operation{
				addNode(0), addPod(0), addNode(2), addNode(4), updatePod(0),
			},
			expected:           []*v1.Node{nodes[0], nodes[4], nodes[2]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add pod before its node",
			operations: []operation{
				addNode(0), addPod(1), updatePod(1), addNode(1),
			},
			expected:           []*v1.Node{nodes[1], nodes[0]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Remove node before its pods",
			operations: []operation{
				addNode(0), addNode(1), addPod(1), addPod(11), updateSnapshot(),
				removeNode(1), updateSnapshot(),
				updatePod(1), updatePod(11), removePod(1), removePod(11),
			},
			expected:           []*v1.Node{nodes[0]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Add Pods with affinity",
			operations: []operation{
				addNode(0), addPodWithAffinity(0), updateSnapshot(), addNode(1),
			},
			expected:                     []*v1.Node{nodes[1], nodes[0]},
			expectedHavePodsWithAffinity: 1,
			expectedUsedPVCSet:           sets.New[string](),
		},
		{
			name: "Add Pods with PVC",
			operations: []operation{
				addNode(0), addPodWithPVC(0, 0, 0), updateSnapshot(), addNode(1),
			},
			expected:           []*v1.Node{nodes[1], nodes[0]},
			expectedUsedPVCSet: sets.New("test-ns/test-pvc0"),
		},
		{
			name: "Add multiple nodes with pods with affinity",
			operations: []operation{
				addNode(0), addPodWithAffinity(0), updateSnapshot(), addNode(1), addPodWithAffinity(1), updateSnapshot(),
			},
			expected:                     []*v1.Node{nodes[1], nodes[0]},
			expectedHavePodsWithAffinity: 2,
			expectedUsedPVCSet:           sets.New[string](),
		},
		{
			name: "Add multiple nodes with pods with PVC",
			operations: []operation{
				addNode(0), addPodWithPVC(0, 0, 0), updateSnapshot(), addNode(1), addPodWithPVC(1, 1, 1), updateSnapshot(),
			},
			expected:           []*v1.Node{nodes[1], nodes[0]},
			expectedUsedPVCSet: sets.New("test-ns/test-pvc0", "test-ns/test-pvc1"),
		},
		{
			name: "Add then Remove pods with affinity",
			operations: []operation{
				addNode(0), addNode(1), addPodWithAffinity(0), updateSnapshot(), removePodWithAffinity(0), updateSnapshot(),
			},
			expected:                     []*v1.Node{nodes[0], nodes[1]},
			expectedHavePodsWithAffinity: 0,
			expectedUsedPVCSet:           sets.New[string](),
		},
		{
			name: "Add then Remove pod with PVC",
			operations: []operation{
				addNode(0), addPodWithPVC(0, 0, 0), updateSnapshot(), removePodWithPVC(0, 0, 0), addPodWithPVC(2, 0, 2), updateSnapshot(),
			},
			expected:           []*v1.Node{nodes[0]},
			expectedUsedPVCSet: sets.New("test-ns/test-pvc2"),
		},
		{
			name: "Add then Remove pod with PVC and add same pod again",
			operations: []operation{
				addNode(0), addPodWithPVC(0, 0, 0), updateSnapshot(), removePodWithPVC(0, 0, 0), addPodWithPVC(0, 0, 0), updateSnapshot(),
			},
			expected:           []*v1.Node{nodes[0]},
			expectedUsedPVCSet: sets.New("test-ns/test-pvc0"),
		},
		{
			name: "Add and Remove multiple pods with PVC with same ref count length different content",
			operations: []operation{
				addNode(0), addNode(1), addPodWithPVC(0, 0, 0), addPodWithPVC(1, 1, 1), updateSnapshot(),
				removePodWithPVC(0, 0, 0), removePodWithPVC(1, 1, 1), addPodWithPVC(2, 0, 2), addPodWithPVC(3, 1, 3), updateSnapshot(),
			},
			expected:           []*v1.Node{nodes[1], nodes[0]},
			expectedUsedPVCSet: sets.New("test-ns/test-pvc2", "test-ns/test-pvc3"),
		},
		{
			name: "Add, Update and Remove multiple pods with SchedulingGroup",
			operations: []operation{
				addNode(0), addNode(1), addNode(2), addPodWithPodGroupName(0), addPodWithPodGroupName(1),
				addPodWithPodGroupName(2), updateSnapshot(),
				updatePodWithPodGroupName(0), removePodWithPodGroupName(1), updateSnapshot(),
			},
			expected: []*v1.Node{nodes[1], nodes[0], nodes[2]},
			expectedPodGroupStatesSnapshot: map[podGroupKey]*podGroupStateSnapshot{
				newPodGroupKey("test-ns", "pg-0"): {
					podGroupStateData: podGroupStateData{
						allPods:         map[types.UID]*v1.Pod{"puid-podgroup-0": podsWithPodGroupName[0]},
						assignedPods:    sets.New[types.UID]("puid-podgroup-0"),
						unscheduledPods: sets.New[types.UID](),
						assumedPods:     make(map[types.UID]*v1.Pod),
					},
				},
				newPodGroupKey("test-ns", "pg-2"): {
					podGroupStateData: podGroupStateData{
						allPods:         map[types.UID]*v1.Pod{"puid-podgroup-2": podsWithPodGroupName[2]},
						assignedPods:    sets.New[types.UID]("puid-podgroup-2"),
						unscheduledPods: sets.New[types.UID](),
						assumedPods:     make(map[types.UID]*v1.Pod),
					},
				},
			},
		},
		{
			name: "Add and Remove multiple pods with PVC",
			operations: []operation{
				addNode(0), addNode(1), addPodWithPVC(0, 0, 0), addPodWithPVC(1, 1, 1), addPodWithPVC(2, 0, 2), updateSnapshot(),
				removePodWithPVC(0, 0, 0), removePodWithPVC(1, 1, 1), updateSnapshot(), addPodWithPVC(0, 0, 0), updateSnapshot(),
				addPodWithPVC(3, 1, 3), addPodWithPVC(4, 0, 0), addPodWithPVC(5, 1, 1), updateSnapshot(),
				removePodWithPVC(0, 0, 0), removePodWithPVC(3, 1, 3), removePodWithPVC(4, 0, 0), updateSnapshot(),
			},
			expected:           []*v1.Node{nodes[0], nodes[1]},
			expectedUsedPVCSet: sets.New("test-ns/test-pvc1", "test-ns/test-pvc2"),
		},
		{
			name: "Assume and forget in cache, and in snapshot",
			operations: []operation{
				addNode(0), addNode(2), addNode(4), addNode(8), updateSnapshot(),
				assumePod(8), assumePodInSnapshot(4), assumePod(0), forgetPod(0),
				assumePodInSnapshot(2), forgetPodInSnapshot(4), updateSnapshot(),
			},
			expected:           []*v1.Node{nodes[0], nodes[8], nodes[4], nodes[2]},
			expectedUsedPVCSet: sets.New[string](),
		},
		{
			name: "Assume and forget in cache, and in snapshot, with affinity",
			operations: []operation{
				addNode(0), addNode(2), addNode(4), addNode(8), updateSnapshot(),
				assumePodWithAffinity(8), assumePodWithAffinityInSnapshot(4), assumePodWithAffinity(0), forgetPodWithAffinity(0),
				assumePodWithAffinityInSnapshot(2), forgetPodWithAffinityInSnapshot(4), updateSnapshot(),
			},
			expected:                     []*v1.Node{nodes[0], nodes[8], nodes[4], nodes[2]},
			expectedHavePodsWithAffinity: 1,
			expectedUsedPVCSet:           sets.New[string](),
		},
		{
			name: "Assume and forget in cache, and in snapshot, with PVC",
			operations: []operation{
				addNode(0), addNode(2), addNode(4), addNode(8), updateSnapshot(),
				assumePodWithPVC(8, 8, 8), assumePodWithPVCInSnapshot(4, 4, 4), assumePodWithPVC(0, 0, 0), forgetPodWithPVC(0, 0, 0),
				assumePodWithPVCInSnapshot(2, 2, 2), forgetPodWithPVCInSnapshot(4, 4, 4), updateSnapshot(),
			},
			expected:           []*v1.Node{nodes[0], nodes[8], nodes[4], nodes[2]},
			expectedUsedPVCSet: sets.New("test-ns/test-pvc8"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			cache = newCache(ctx, time.Second, nil, true)
			snapshot = NewEmptySnapshot()

			for _, op := range test.operations {
				op(t)
			}

			if len(test.expected) != len(cache.nodes) {
				t.Errorf("unexpected number of nodes. Expected: %v, got: %v", len(test.expected), len(cache.nodes))
			}
			var i int
			// Check that cache is in the expected state.
			for node := cache.headNode; node != nil; node = node.next {
				if node.info.Node() != nil && node.info.Node().Name != test.expected[i].Name {
					t.Errorf("unexpected node. Expected: %v, got: %v, index: %v", test.expected[i].Name, node.info.Node().Name, i)
				}
				i++
			}
			// Make sure we visited all the cached nodes in the above for loop.
			if i != len(cache.nodes) {
				t.Errorf("Not all the nodes were visited by following the NodeInfo linked list. Expected to see %v nodes, saw %v.", len(cache.nodes), i)
			}

			// Check pod group states in the snapshot.
			if diff := cmp.Diff(test.expectedPodGroupStatesSnapshot, snapshot.podGroupStates, podGroupStateCmpOpts...); diff != "" {
				t.Errorf("unexpected podGroupStates in snapshot (-want, +got):\n%s", diff)
			}

			// Check number of nodes with pods with affinity
			if len(snapshot.havePodsWithAffinityNodeInfoList) != test.expectedHavePodsWithAffinity {
				t.Errorf("unexpected number of HavePodsWithAffinity nodes. Expected: %v, got: %v", test.expectedHavePodsWithAffinity, len(snapshot.havePodsWithAffinityNodeInfoList))
			}

			// Compare content of the used PVC set
			if diff := cmp.Diff(test.expectedUsedPVCSet, snapshot.usedPVCSet); diff != "" {
				t.Errorf("Unexpected usedPVCSet (-want +got):\n%s", diff)
			}

			// Always update the snapshot at the end of operations and compare it.
			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Error(err)
			}
			if err := compareCacheWithNodeInfoSnapshot(t, cache, snapshot); err != nil {
				t.Error(err)
			}
		})
	}
}

func compareCacheWithNodeInfoSnapshot(t *testing.T, cache *cacheImpl, snapshot *Snapshot) error {
	// Compare the map.
	if len(snapshot.nodeInfoMap) != cache.nodeTree.numNodes {
		return fmt.Errorf("unexpected number of nodes in the snapshot. Expected: %v, got: %v", cache.nodeTree.numNodes, len(snapshot.nodeInfoMap))
	}
	for name, ni := range cache.nodes {
		want := ni.info
		if want.Node() == nil {
			want = nil
		}
		if diff := cmp.Diff(want, snapshot.nodeInfoMap[name], nodeInfoCmpOpts...); diff != "" {
			return fmt.Errorf("Unexpected node info for node (-want, +got):\n%s", diff)
		}
	}
	// Compare the lists.
	if len(snapshot.nodeInfoList) != cache.nodeTree.numNodes {
		return fmt.Errorf("unexpected number of nodes in NodeInfoList. Expected: %v, got: %v", cache.nodeTree.numNodes, len(snapshot.nodeInfoList))
	}

	expectedNodeInfoList := make([]fwk.NodeInfo, 0, cache.nodeTree.numNodes)
	expectedHavePodsWithAffinityNodeInfoList := make([]fwk.NodeInfo, 0, cache.nodeTree.numNodes)
	expectedUsedPVCSet := sets.New[string]()
	nodesList, err := cache.nodeTree.list()
	if err != nil {
		t.Fatal(err)
	}
	for _, nodeName := range nodesList {
		if n := snapshot.nodeInfoMap[nodeName]; n != nil {
			expectedNodeInfoList = append(expectedNodeInfoList, n)
			if len(n.PodsWithAffinity) > 0 {
				expectedHavePodsWithAffinityNodeInfoList = append(expectedHavePodsWithAffinityNodeInfoList, n)
			}
			for key := range n.PVCRefCounts {
				expectedUsedPVCSet.Insert(key)
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

	for key := range expectedUsedPVCSet {
		if !snapshot.usedPVCSet.Has(key) {
			return fmt.Errorf("expected PVC %s to exist in UsedPVCSet but it is not found", key)
		}
	}

	return nil
}

func TestSchedulerCache_updateNodeInfoSnapshotList(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Create a few nodes to be used in tests.
	var nodes []*v1.Node
	i := 0
	// List of number of nodes per zone, zone 0 -> 2, zone 1 -> 6
	for zone, nb := range []int{2, 6} {
		for j := 0; j < nb; j++ {
			nodes = append(nodes, &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: fmt.Sprintf("node-%d", i),
					Labels: map[string]string{
						v1.LabelTopologyRegion: fmt.Sprintf("region-%d", zone),
						v1.LabelTopologyZone:   fmt.Sprintf("zone-%d", zone),
					},
				},
			})
			i++
		}
	}

	var cache *cacheImpl
	var snapshot *Snapshot

	addNode := func(t *testing.T, i int) {
		cache.AddNode(logger, nodes[i])
		_, ok := snapshot.nodeInfoMap[nodes[i].Name]
		if !ok {
			snapshot.nodeInfoMap[nodes[i].Name] = cache.nodes[nodes[i].Name].info
		}
	}

	updateSnapshot := func(t *testing.T) {
		cache.updateNodeInfoSnapshotList(logger, snapshot, true)
		if err := compareCacheWithNodeInfoSnapshot(t, cache, snapshot); err != nil {
			t.Error(err)
		}
	}

	tests := []struct {
		name       string
		operations func(t *testing.T)
		expected   []string
	}{
		{
			name:       "Empty cache",
			operations: func(t *testing.T) {},
			expected:   []string{},
		},
		{
			name: "Single node",
			operations: func(t *testing.T) {
				addNode(t, 0)
			},
			expected: []string{"node-0"},
		},
		{
			name: "Two nodes",
			operations: func(t *testing.T) {
				addNode(t, 0)
				updateSnapshot(t)
				addNode(t, 1)
			},
			expected: []string{"node-0", "node-1"},
		},
		{
			name: "bug 91601, two nodes, update the snapshot and add two nodes in different zones",
			operations: func(t *testing.T) {
				addNode(t, 2)
				addNode(t, 3)
				updateSnapshot(t)
				addNode(t, 4)
				addNode(t, 0)
			},
			expected: []string{"node-2", "node-0", "node-3", "node-4"},
		},
		{
			name: "bug 91601, 6 nodes, one in a different zone",
			operations: func(t *testing.T) {
				addNode(t, 2)
				addNode(t, 3)
				addNode(t, 4)
				addNode(t, 5)
				updateSnapshot(t)
				addNode(t, 6)
				addNode(t, 0)
			},
			expected: []string{"node-2", "node-0", "node-3", "node-4", "node-5", "node-6"},
		},
		{
			name: "bug 91601, 7 nodes, two in a different zone",
			operations: func(t *testing.T) {
				addNode(t, 2)
				updateSnapshot(t)
				addNode(t, 3)
				addNode(t, 4)
				updateSnapshot(t)
				addNode(t, 5)
				addNode(t, 6)
				addNode(t, 0)
				addNode(t, 1)
			},
			expected: []string{"node-2", "node-0", "node-3", "node-1", "node-4", "node-5", "node-6"},
		},
		{
			name: "bug 91601, 7 nodes, two in a different zone, different zone order",
			operations: func(t *testing.T) {
				addNode(t, 2)
				addNode(t, 1)
				updateSnapshot(t)
				addNode(t, 3)
				addNode(t, 4)
				updateSnapshot(t)
				addNode(t, 5)
				addNode(t, 6)
				addNode(t, 0)
			},
			expected: []string{"node-2", "node-1", "node-3", "node-0", "node-4", "node-5", "node-6"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			cache = newCache(ctx, time.Second, nil, false)
			snapshot = NewEmptySnapshot()

			test.operations(t)

			// Always update the snapshot at the end of operations and compare it.
			cache.updateNodeInfoSnapshotList(logger, snapshot, true)
			if err := compareCacheWithNodeInfoSnapshot(t, cache, snapshot); err != nil {
				t.Error(err)
			}
			nodeNames := make([]string, len(snapshot.nodeInfoList))
			for i, nodeInfo := range snapshot.nodeInfoList {
				nodeNames[i] = nodeInfo.Node().Name
			}
			if diff := cmp.Diff(test.expected, nodeNames); diff != "" {
				t.Errorf("Unexpected nodeInfoList (-want, +got):\n%s", diff)
			}
		})
	}
}

func BenchmarkUpdate1kNodes30kPods(b *testing.B) {
	logger, _ := ktesting.NewTestContext(b)
	cache := setupCacheOf1kNodes30kPods(b)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		cachedNodes := NewEmptySnapshot()
		cache.UpdateSnapshot(logger, cachedNodes)
	}
}

type testingMode interface {
	Fatalf(format string, args ...interface{})
}

func makeBasePod(t testingMode, nodeName, objName, cpu, mem, extended string, ports []v1.ContainerPort) *v1.Pod {
	req := make(map[v1.ResourceName]string)
	if cpu != "" {
		req[v1.ResourceCPU] = cpu
		req[v1.ResourceMemory] = mem

		if extended != "" {
			parts := strings.Split(extended, ":")
			if len(parts) != 2 {
				t.Fatalf("Invalid extended resource string: \"%s\"", extended)
			}
			req[v1.ResourceName(parts[0])] = parts[1]
		}
	}
	podWrapper := st.MakePod().Name(objName).Namespace("node_info_cache_test").UID(objName).Node(nodeName).Containers([]v1.Container{
		st.MakeContainer().Name("container").Image("pause").Resources(req).ContainerPort(ports).Obj(),
	})
	return podWrapper.Obj()
}

// checkImageStateSummary collect ImageStateSummary of image traverse nodes,
// the collected ImageStateSummary should be equal
func checkImageStateSummary(nodes map[string]*framework.NodeInfo, imageNames ...string) bool {
	for _, imageName := range imageNames {
		var imageState *fwk.ImageStateSummary
		for _, node := range nodes {
			state, ok := node.ImageStates[imageName]
			if !ok {
				continue
			}
			if imageState == nil {
				imageState = state
				continue
			}
			if diff := cmp.Diff(imageState, state); diff != "" {
				return false
			}
		}
	}
	return true
}

func setupCacheOf1kNodes30kPods(b *testing.B) Cache {
	logger, ctx := ktesting.NewTestContext(b)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cache := newCache(ctx, time.Second, nil, false)
	for i := 0; i < 1000; i++ {
		nodeName := fmt.Sprintf("node-%d", i)
		cache.AddNode(logger, st.MakeNode().Name(nodeName).Obj())
		for j := 0; j < 30; j++ {
			objName := fmt.Sprintf("%s-pod-%d", nodeName, j)
			pod := makeBasePod(b, nodeName, objName, "0", "0", "", nil)

			if err := cache.AddPod(logger, pod); err != nil {
				b.Fatalf("AddPod failed: %v", err)
			}
		}
	}
	return cache
}

func isForgottenFromCache(p *v1.Pod, c *cacheImpl) error {
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

// getNodeInfo returns cached data for the node name.
func (cache *cacheImpl) getNodeInfo(nodeName string) (*v1.Node, error) {
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	n, ok := cache.nodes[nodeName]
	if !ok {
		return nil, fmt.Errorf("node %q not found in cache", nodeName)
	}

	return n.info.Node(), nil
}
