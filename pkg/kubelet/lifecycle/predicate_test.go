/*
Copyright 2018 The Kubernetes Authors.

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

package lifecycle

import (
	goruntime "runtime"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/kubelet/types"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/utils/ptr"
)

var (
	quantity = *resource.NewQuantity(1, resource.DecimalSI)
)

func TestRemoveMissingExtendedResources(t *testing.T) {
	for _, test := range []struct {
		desc string
		pod  *v1.Pod
		node *v1.Node

		expectedPod *v1.Pod
	}{
		{
			desc: "requests in Limits should be ignored",
			pod: makeTestPod(
				v1.ResourceList{},                        // Requests
				v1.ResourceList{"foo.com/bar": quantity}, // Limits
			),
			node: makeTestNode(
				v1.ResourceList{"foo.com/baz": quantity}, // Allocatable
			),
			expectedPod: makeTestPod(
				v1.ResourceList{},                        // Requests
				v1.ResourceList{"foo.com/bar": quantity}, // Limits
			),
		},
		{
			desc: "requests for resources available in node should not be removed",
			pod: makeTestPod(
				v1.ResourceList{"foo.com/bar": quantity}, // Requests
				v1.ResourceList{},                        // Limits
			),
			node: makeTestNode(
				v1.ResourceList{"foo.com/bar": quantity}, // Allocatable
			),
			expectedPod: makeTestPod(
				v1.ResourceList{"foo.com/bar": quantity}, // Requests
				v1.ResourceList{}),                       // Limits
		},
		{
			desc: "requests for resources unavailable in node should be removed",
			pod: makeTestPod(
				v1.ResourceList{"foo.com/bar": quantity}, // Requests
				v1.ResourceList{},                        // Limits
			),
			node: makeTestNode(
				v1.ResourceList{"foo.com/baz": quantity}, // Allocatable
			),
			expectedPod: makeTestPod(
				v1.ResourceList{}, // Requests
				v1.ResourceList{}, // Limits
			),
		},
	} {
		nodeInfo := schedulerframework.NewNodeInfo()
		nodeInfo.SetNode(test.node)
		pod := removeMissingExtendedResources(test.pod, nodeInfo)
		if diff := cmp.Diff(test.expectedPod, pod); diff != "" {
			t.Errorf("unexpected pod (-want, +got):\n%s", diff)
		}
	}
}

func makeTestPod(requests, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

func makeTestNode(allocatable v1.ResourceList) *v1.Node {
	return &v1.Node{
		Status: v1.NodeStatus{
			Allocatable: allocatable,
		},
	}
}

var (
	extendedResourceA = v1.ResourceName("example.com/aaa")
	hugePageResourceA = v1helper.HugePageResourceName(resource.MustParse("2Mi"))
)

func makeResources(milliCPU, memory, pods, extendedA, storage, hugePageA int64) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
		v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
		extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
		v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
		hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
	}
}

func makeAllocatableResources(milliCPU, memory, pods, extendedA, storage, hugePageA int64) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
		v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
		extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
		v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
		hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
	}
}

func newResourcePod(containerResources ...v1.ResourceList) *v1.Pod {
	containers := []v1.Container{}
	for _, rl := range containerResources {
		containers = append(containers, v1.Container{
			Resources: v1.ResourceRequirements{Requests: rl},
		})
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}
}

func newPodWithPort(hostPorts ...int) *v1.Pod {
	networkPorts := []v1.ContainerPort{}
	for _, port := range hostPorts {
		networkPorts = append(networkPorts, v1.ContainerPort{HostPort: int32(port)})
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Ports: networkPorts,
				},
			},
		},
	}
}

func TestGeneralPredicates(t *testing.T) {
	resourceTests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulerframework.NodeInfo
		node     *v1.Node
		name     string
		reasons  []PredicateFailureReason
	}{
		{
			pod: &v1.Pod{},
			nodeInfo: schedulerframework.NewNodeInfo(
				newResourcePod(v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(9, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(19, resource.BinarySI),
				})),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			name: "no resources/port/host requested always fits",
		},
		{
			pod: newResourcePod(v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(8, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(10, resource.BinarySI),
			}),
			nodeInfo: schedulerframework.NewNodeInfo(
				newResourcePod(v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(5, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(19, resource.BinarySI),
				})),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			reasons: []PredicateFailureReason{
				&InsufficientResourceError{ResourceName: v1.ResourceCPU, Requested: 8, Used: 5, Capacity: 10},
				&InsufficientResourceError{ResourceName: v1.ResourceMemory, Requested: 10, Used: 19, Capacity: 20},
			},
			name: "not enough cpu and memory resource",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeName: "machine2",
				},
			},
			nodeInfo: schedulerframework.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			reasons: []PredicateFailureReason{&PredicateFailureError{nodename.Name, nodename.ErrReason}},
			name:    "host not match",
		},
		{
			pod:      newPodWithPort(123),
			nodeInfo: schedulerframework.NewNodeInfo(newPodWithPort(123)),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			reasons: []PredicateFailureReason{&PredicateFailureError{nodeports.Name, nodeports.ErrReason}},
			name:    "hostport conflict",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Tolerations: []v1.Toleration{
						{Key: "foo"},
						{Key: "bar"},
					},
				},
			},
			nodeInfo: schedulerframework.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "foo", Effect: v1.TaintEffectNoSchedule},
						{Key: "bar", Effect: v1.TaintEffectNoExecute},
					},
				},
				Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			name: "taint/toleration match",
		},
		{
			pod:      &v1.Pod{},
			nodeInfo: schedulerframework.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "foo", Effect: v1.TaintEffectNoSchedule},
					},
				},
				Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			name: "NoSchedule taint/toleration not match",
		},
		{
			pod:      &v1.Pod{},
			nodeInfo: schedulerframework.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "bar", Effect: v1.TaintEffectNoExecute},
					},
				},
				Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			reasons: []PredicateFailureReason{&PredicateFailureError{tainttoleration.Name, tainttoleration.ErrReasonNotMatch}},
			name:    "NoExecute taint/toleration not match",
		},
		{
			pod:      &v1.Pod{},
			nodeInfo: schedulerframework.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "baz", Effect: v1.TaintEffectPreferNoSchedule},
					},
				},
				Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			name: "PreferNoSchedule taint/toleration not match",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						types.ConfigSourceAnnotationKey: types.FileSource,
					},
				},
			},
			nodeInfo: schedulerframework.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "foo", Effect: v1.TaintEffectNoSchedule},
						{Key: "bar", Effect: v1.TaintEffectNoExecute},
					},
				},
				Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0), Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			name: "static pods ignore taints",
		},
	}
	for _, test := range resourceTests {
		t.Run(test.name, func(t *testing.T) {
			test.nodeInfo.SetNode(test.node)
			reasons := generalFilter(test.pod, test.nodeInfo)
			if diff := cmp.Diff(test.reasons, reasons); diff != "" {
				t.Errorf("unexpected failure reasons (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestRejectPodAdmissionBasedOnOSSelector(t *testing.T) {
	tests := []struct {
		name            string
		pod             *v1.Pod
		node            *v1.Node
		expectRejection bool
	}{
		{
			name:            "OS label match",
			pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: goruntime.GOOS}}},
			node:            &v1.Node{Spec: v1.NodeSpec{}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: goruntime.GOOS}}},
			expectRejection: false,
		},
		{
			name:            "dummyOS label, but the underlying OS matches",
			pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: goruntime.GOOS}}},
			node:            &v1.Node{Spec: v1.NodeSpec{}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			expectRejection: false,
		},
		{
			name:            "dummyOS label, but the underlying OS doesn't match",
			pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			node:            &v1.Node{Spec: v1.NodeSpec{}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			expectRejection: true,
		},
		{
			name:            "dummyOS label, but the underlying OS doesn't match",
			pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			node:            &v1.Node{Spec: v1.NodeSpec{}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			expectRejection: true,
		},
		{
			name:            "OS field mismatch, OS label on node object would be reset to correct value",
			pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			node:            &v1.Node{Spec: v1.NodeSpec{}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			expectRejection: true,
		},
		{
			name:            "No label selector on the pod, should be admitted",
			pod:             &v1.Pod{},
			node:            &v1.Node{Spec: v1.NodeSpec{}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelOSStable: "dummyOS"}}},
			expectRejection: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualResult := rejectPodAdmissionBasedOnOSSelector(test.pod, test.node)
			if test.expectRejection != actualResult {
				t.Errorf("unexpected result, expected %v but got %v", test.expectRejection, actualResult)
			}
		})
	}
}

func TestRejectPodAdmissionBasedOnOSField(t *testing.T) {
	tests := []struct {
		name            string
		pod             *v1.Pod
		expectRejection bool
	}{
		{
			name:            "OS field match",
			pod:             &v1.Pod{Spec: v1.PodSpec{OS: &v1.PodOS{Name: v1.OSName(goruntime.GOOS)}}},
			expectRejection: false,
		},
		{
			name:            "OS field mismatch",
			pod:             &v1.Pod{Spec: v1.PodSpec{OS: &v1.PodOS{Name: "dummyOS"}}},
			expectRejection: true,
		},
		{
			name:            "no OS field",
			pod:             &v1.Pod{Spec: v1.PodSpec{}},
			expectRejection: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualResult := rejectPodAdmissionBasedOnOSField(test.pod)
			if test.expectRejection != actualResult {
				t.Errorf("unexpected result, expected %v but got %v", test.expectRejection, actualResult)
			}
		})
	}
}

func TestPodAdmissionBasedOnSupplementalGroupsPolicy(t *testing.T) {
	nodeWithFeature := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Status: v1.NodeStatus{
			Features: &v1.NodeFeatures{
				SupplementalGroupsPolicy: ptr.To(true),
			},
		},
	}
	nodeWithoutFeature := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
	}
	podNotUsingFeature := &v1.Pod{}
	podUsingFeature := &v1.Pod{Spec: v1.PodSpec{
		SecurityContext: &v1.PodSecurityContext{
			SupplementalGroupsPolicy: ptr.To(v1.SupplementalGroupsPolicyStrict),
		},
	}}
	tests := []struct {
		name             string
		emulationVersion *utilversion.Version
		node             *v1.Node
		pod              *v1.Pod
		expectRejection  bool
	}{
		// The feature is Beta in v1.33
		{
			name:             "feature=Beta, node=feature not supported, pod=in use: it should REJECT",
			emulationVersion: utilversion.MustParse("1.33"),
			node:             nodeWithoutFeature,
			pod:              podUsingFeature,
			expectRejection:  true,
		},
		{
			name:             "feature=Beta, node=feature supported, pod=in use: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.33"),
			node:             nodeWithFeature,
			pod:              podUsingFeature,
			expectRejection:  false,
		},
		{
			name:             "feature=Beta, node=feature not supported, pod=not in use: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.33"),
			node:             nodeWithoutFeature,
			pod:              podNotUsingFeature,
			expectRejection:  false,
		},
		{
			name:             "feature=Beta, node=feature supported, pod=not in use: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.33"),
			node:             nodeWithFeature,
			pod:              podNotUsingFeature,
			expectRejection:  false,
		},
		// The feature is Alpha(v1.31, v1.32) in emulated version
		// Note: When the feature is alpha in emulated version, it should always admit for backward compatibility
		{
			name:             "feature=Alpha, node=feature not supported, pod=feature used: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.32"),
			node:             nodeWithoutFeature,
			pod:              podUsingFeature,
			expectRejection:  false,
		},
		{
			name:             "feature=Alpha, node=feature not supported, pod=feature not used: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.32"),
			node:             nodeWithoutFeature,
			pod:              podNotUsingFeature,
			expectRejection:  false,
		},
		{
			name:             "feature=Alpha, node=feature supported, pod=feature used: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.32"),
			node:             nodeWithFeature,
			pod:              podUsingFeature,
			expectRejection:  false,
		},
		{
			name:             "feature=Alpha, node=feature supported, pod=feature not used: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.32"),
			node:             nodeWithFeature,
			pod:              podNotUsingFeature,
			expectRejection:  false,
		},
		// The feature is not yet released (< v1.31) in emulated version (this can happen when only kubelet downgraded).
		// Note: When the feature is not yet released in emulated version, it should always admit for backward compatibility
		{
			name:             "feature=NotReleased, node=feature not supported, pod=feature used: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.30"),
			node:             nodeWithoutFeature,
			pod:              podUsingFeature,
			expectRejection:  false,
		},
		{
			name:             "feature=NotReleased, node=feature not supported, pod=feature not used: it should ADMIT",
			emulationVersion: utilversion.MustParse("1.30"),
			node:             nodeWithoutFeature,
			pod:              podNotUsingFeature,
			expectRejection:  false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, test.emulationVersion)
			actualResult := rejectPodAdmissionBasedOnSupplementalGroupsPolicy(test.pod, test.node)
			if test.expectRejection != actualResult {
				t.Errorf("unexpected result, expected %v but got %v", test.expectRejection, actualResult)
			}
		})
	}
}
