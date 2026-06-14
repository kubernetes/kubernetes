/*
Copyright 2019 The Kubernetes Authors.

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

package topologymanager

import (
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func NewTestBitMask(sockets ...int) bitmask.BitMask {
	s, _ := bitmask.NewBitMask(sockets...)
	return s
}

func TestNewManager(t *testing.T) {
	tcases := []struct {
		description    string
		policyName     string
		expectedPolicy string
		expectedError  error
		topologyError  error
		policyOptions  map[string]string
		topology       []cadvisorapi.Node
	}{
		{
			description:    "Policy is set to none",
			policyName:     "none",
			expectedPolicy: "none",
		},
		{
			description:    "Policy is set to best-effort",
			policyName:     "best-effort",
			expectedPolicy: "best-effort",
		},
		{
			description:    "Policy is set to restricted",
			policyName:     "restricted",
			expectedPolicy: "restricted",
		},
		{
			description:    "Policy is set to single-numa-node",
			policyName:     "single-numa-node",
			expectedPolicy: "single-numa-node",
		},
		{
			description:   "Policy is set to unknown",
			policyName:    "unknown",
			expectedError: fmt.Errorf("unknown policy: \"unknown\""),
		},
		{
			description:    "Unknown policy name best-effort policy",
			policyName:     "best-effort",
			expectedPolicy: "best-effort",
			expectedError:  fmt.Errorf("unknown Topology Manager Policy option:"),
			policyOptions: map[string]string{
				"unknown-option": "true",
			},
		},
		{
			description:    "Unknown policy name restricted policy",
			policyName:     "restricted",
			expectedPolicy: "restricted",
			expectedError:  fmt.Errorf("unknown Topology Manager Policy option:"),
			policyOptions: map[string]string{
				"unknown-option": "true",
			},
		},
		{
			description:    "can't get NUMA distances",
			policyName:     "best-effort",
			expectedPolicy: "best-effort",
			policyOptions: map[string]string{
				PreferClosestNUMANodes: "true",
			},
			expectedError: fmt.Errorf("error getting NUMA distances from cadvisor"),
			topology: []cadvisorapi.Node{
				{
					Id: 0,
				},
			},
		},
		{
			description:    "more than 8 NUMA nodes",
			policyName:     "best-effort",
			expectedPolicy: "best-effort",
			expectedError:  fmt.Errorf("unsupported on machines with more than %v NUMA Nodes", defaultMaxAllowableNUMANodes),
			topology: []cadvisorapi.Node{
				{
					Id: 0,
				},
				{
					Id: 1,
				},
				{
					Id: 2,
				},
				{
					Id: 3,
				},
				{
					Id: 4,
				},
				{
					Id: 5,
				},
				{
					Id: 6,
				},
				{
					Id: 7,
				},
				{
					Id: 8,
				},
			},
		},
	}

	for _, tc := range tcases {
		topology := tc.topology

		mngr, err := NewManager(topology, tc.policyName, "container", tc.policyOptions)
		if tc.expectedError != nil {
			if !strings.Contains(err.Error(), tc.expectedError.Error()) {
				t.Errorf("Unexpected error message. Have: %s wants %s", err.Error(), tc.expectedError.Error())
			}
		} else {
			rawMgr := mngr.(*manager)
			var policyName string
			if rawScope, ok := rawMgr.scope.(*containerScope); ok {
				policyName = rawScope.policy.Name()
			} else if rawScope, ok := rawMgr.scope.(*noneScope); ok {
				policyName = rawScope.policy.Name()
			}
			if policyName != tc.expectedPolicy {
				t.Errorf("Unexpected policy name. Have: %q wants %q", policyName, tc.expectedPolicy)
			}
		}
	}
}

func TestManagerScope(t *testing.T) {
	tcases := []struct {
		description   string
		scopeName     string
		expectedScope string
		expectedError error
	}{
		{
			description:   "Topology Manager Scope is set to container",
			scopeName:     "container",
			expectedScope: "container",
		},
		{
			description:   "Topology Manager Scope is set to pod",
			scopeName:     "pod",
			expectedScope: "pod",
		},
		{
			description:   "Topology Manager Scope is set to unknown",
			scopeName:     "unknown",
			expectedError: fmt.Errorf("unknown scope: \"unknown\""),
		},
	}

	for _, tc := range tcases {
		mngr, err := NewManager(nil, "best-effort", tc.scopeName, nil)

		if tc.expectedError != nil {
			if !strings.Contains(err.Error(), tc.expectedError.Error()) {
				t.Errorf("Unexpected error message. Have: %s wants %s", err.Error(), tc.expectedError.Error())
			}
		} else {
			rawMgr := mngr.(*manager)
			if rawMgr.scope.Name() != tc.expectedScope {
				t.Errorf("Unexpected scope name. Have: %q wants %q", rawMgr.scope, tc.expectedScope)
			}
		}
	}
}

type mockHintProvider struct {
	th map[string][]TopologyHint
	//TODO: Add this field and add some tests to make sure things error out
	//appropriately on allocation errors.
	//allocateError error
}

func (m *mockHintProvider) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]TopologyHint {
	return m.th
}

func (m *mockHintProvider) GetPodTopologyHints(pod *v1.Pod) map[string][]TopologyHint {
	return m.th
}

func (m *mockHintProvider) AllocatePod(pod *v1.Pod) error {
	return nil
}

func (m *mockHintProvider) Allocate(pod *v1.Pod, container *v1.Container) error {
	//return allocateError
	return nil
}

type mockPolicy struct {
	nonePolicy
	ph []map[string][]TopologyHint
}

func (p *mockPolicy) Merge(logger klog.Logger, providersHints []map[string][]TopologyHint) (TopologyHint, bool) {
	p.ph = providersHints
	return TopologyHint{}, true
}

func TestAddHintProvider(t *testing.T) {
	tcases := []struct {
		name string
		hp   []HintProvider
	}{
		{
			name: "Add HintProvider",
			hp: []HintProvider{
				&mockHintProvider{},
				&mockHintProvider{},
				&mockHintProvider{},
			},
		},
	}
	mngr := manager{}
	mngr.scope = NewContainerScope(NewNonePolicy())
	logger, _ := ktesting.NewTestContext(t)
	for _, tc := range tcases {
		for _, hp := range tc.hp {
			mngr.AddHintProvider(logger, hp)
		}
		if len(tc.hp) != len(mngr.scope.(*containerScope).hintProviders) {
			t.Errorf("error")
		}
	}
}

func TestAdmit(t *testing.T) {
	tCtx := ktesting.Init(t)
	numaInfo := &NUMAInfo{
		Nodes: []int{0, 1},
		NUMADistances: NUMADistances{
			0: {10, 11},
			1: {11, 10},
		},
	}

	opts := PolicyOptions{}
	bePolicy := NewBestEffortPolicy(numaInfo, opts)
	restrictedPolicy := NewRestrictedPolicy(numaInfo, opts)
	singleNumaPolicy := NewSingleNumaNodePolicy(numaInfo, opts)

	tcases := []struct {
		name     string
		result   lifecycle.PodAdmitResult
		qosClass v1.PodQOSClass
		policy   Policy
		hp       []HintProvider
		expected bool
	}{
		{
			name:     "QOSClass set as BestEffort. None Policy. No Hints.",
			qosClass: v1.PodQOSBestEffort,
			policy:   NewNonePolicy(),
			hp:       []HintProvider{},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. None Policy. No Hints.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   NewNonePolicy(),
			hp:       []HintProvider{},
			expected: true,
		},
		{
			name:     "QOSClass set as BestEffort. single-numa-node Policy. No Hints.",
			qosClass: v1.PodQOSBestEffort,
			policy:   singleNumaPolicy,
			hp: []HintProvider{
				&mockHintProvider{},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as BestEffort. Restricted Policy. No Hints.",
			qosClass: v1.PodQOSBestEffort,
			policy:   restrictedPolicy,
			hp: []HintProvider{
				&mockHintProvider{},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. BestEffort Policy. Preferred Affinity.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   bePolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. BestEffort Policy. More than one Preferred Affinity.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   bePolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Burstable. BestEffort Policy. More than one Preferred Affinity.",
			qosClass: v1.PodQOSBurstable,
			policy:   bePolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. BestEffort Policy. No Preferred Affinity.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   bePolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. Restricted Policy. Preferred Affinity.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   restrictedPolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Burstable. Restricted Policy. Preferred Affinity.",
			qosClass: v1.PodQOSBurstable,
			policy:   restrictedPolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. Restricted Policy. More than one Preferred affinity.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   restrictedPolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Burstable. Restricted Policy. More than one Preferred affinity.",
			qosClass: v1.PodQOSBurstable,
			policy:   restrictedPolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. Restricted Policy. No Preferred affinity.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   restrictedPolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name:     "QOSClass set as Burstable. Restricted Policy. No Preferred affinity.",
			qosClass: v1.PodQOSBurstable,
			policy:   restrictedPolicy,
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: false,
		},
	}
	for _, tc := range tcases {
		ctnScopeManager := manager{}
		ctnScopeManager.scope = NewContainerScope(tc.policy)
		ctnScopeManager.scope.(*containerScope).hintProviders = tc.hp

		podScopeManager := manager{}
		podScopeManager.scope = NewPodScope(tc.policy)
		podScopeManager.scope.(*podScope).hintProviders = tc.hp

		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{},
					},
				},
			},
			Status: v1.PodStatus{
				QOSClass: tc.qosClass,
			},
		}

		podAttr := lifecycle.PodAdmitAttributes{
			Pod: pod,
		}

		// Container scope Admit
		ctnActual := ctnScopeManager.Admit(tCtx, &podAttr)
		if ctnActual.Admit != tc.expected {
			t.Errorf("Error occurred, expected Admit in result to be %v got %v", tc.expected, ctnActual.Admit)
		}
		if !ctnActual.Admit && ctnActual.Reason != ErrorTopologyAffinity {
			t.Errorf("Error occurred, expected Reason in result to be %v got %v", ErrorTopologyAffinity, ctnActual.Reason)
		}

		// Pod scope Admit
		podActual := podScopeManager.Admit(tCtx, &podAttr)
		if podActual.Admit != tc.expected {
			t.Errorf("Error occurred, expected Admit in result to be %v got %v", tc.expected, podActual.Admit)
		}
		if !ctnActual.Admit && ctnActual.Reason != ErrorTopologyAffinity {
			t.Errorf("Error occurred, expected Reason in result to be %v got %v", ErrorTopologyAffinity, ctnActual.Reason)
		}
	}
}

type trackingHintProvider struct {
	podHintsCalled          bool
	containerHintsCalled    bool
	allocatePodCalled       bool
	allocateContainerCalled bool
	hints                   map[string][]TopologyHint
}

func (m *trackingHintProvider) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]TopologyHint {
	m.containerHintsCalled = true
	return m.hints
}

func (m *trackingHintProvider) GetPodTopologyHints(pod *v1.Pod) map[string][]TopologyHint {
	m.podHintsCalled = true
	return m.hints
}

func (m *trackingHintProvider) AllocatePod(pod *v1.Pod) error {
	m.allocatePodCalled = true
	return nil
}

func (m *trackingHintProvider) Allocate(pod *v1.Pod, container *v1.Container) error {
	m.allocateContainerCalled = true
	return nil
}

func TestAdmitWithPodLevelResources(t *testing.T) {
	numaInfo := &NUMAInfo{
		Nodes: []int{0, 1},
		NUMADistances: NUMADistances{
			0: {10, 11},
			1: {11, 10},
		},
	}
	opts := PolicyOptions{}
	restrictedPolicy := NewRestrictedPolicy(numaInfo, opts)

	tcases := []struct {
		name                            string
		podLevelResourcesEnabled        bool
		podLevelResourceManagersEnabled bool
		pod                             *v1.Pod
		expectedAdmit                   bool
		expectedPodHintsCalled          bool
		expectedContainerHintsCalled    bool
		expectedAllocatePodCalled       bool
		expectedAllocateContainerCalled bool
		scope                           Scope
	}{
		{
			name:                            "pod scope, feature disabled, falls back to container level flow",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: false,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources:  &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}},
					Containers: []v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}}}},
				},
				Status: v1.PodStatus{QOSClass: v1.PodQOSGuaranteed},
			},
			expectedAdmit:                   true,
			expectedPodHintsCalled:          true,
			expectedContainerHintsCalled:    false,
			expectedAllocatePodCalled:       false,
			expectedAllocateContainerCalled: true,
			scope:                           NewPodScope(restrictedPolicy),
		},
		{
			name:                            "pod scope, feature enabled, uses pod-level flow",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources:  &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}},
					Containers: []v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}}}},
				},
				Status: v1.PodStatus{QOSClass: v1.PodQOSGuaranteed},
			},
			expectedAdmit:                   true,
			expectedPodHintsCalled:          true,
			expectedContainerHintsCalled:    false,
			expectedAllocatePodCalled:       true,
			expectedAllocateContainerCalled: false,
			scope:                           NewPodScope(restrictedPolicy),
		},
		{
			name:                            "container scope, feature enabled, uses container-level flow",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources:  &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}},
					Containers: []v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}}}},
				},
				Status: v1.PodStatus{QOSClass: v1.PodQOSGuaranteed},
			},
			expectedAdmit:                   true,
			expectedPodHintsCalled:          false,
			expectedContainerHintsCalled:    true,
			expectedAllocatePodCalled:       false,
			expectedAllocateContainerCalled: true,
			scope:                           NewContainerScope(restrictedPolicy),
		},
		{
			name:                            "container scope, feature disabled, uses container-level flow",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: false,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources:  &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}},
					Containers: []v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}}}},
				},
				Status: v1.PodStatus{QOSClass: v1.PodQOSGuaranteed},
			},
			expectedAdmit:                   true,
			expectedPodHintsCalled:          false,
			expectedContainerHintsCalled:    true,
			expectedAllocatePodCalled:       false,
			expectedAllocateContainerCalled: true,
			scope:                           NewContainerScope(restrictedPolicy),
		},
	}

	for _, tc := range tcases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, tc.podLevelResourcesEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, tc.podLevelResourceManagersEnabled)

			tracker := &trackingHintProvider{
				hints: map[string][]TopologyHint{
					"resource": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
			}

			m := manager{scope: tc.scope}
			switch scope := tc.scope.(type) {
			case *podScope:
				scope.hintProviders = []HintProvider{tracker}
			case *containerScope:
				scope.hintProviders = []HintProvider{tracker}
			}

			podAttr := lifecycle.PodAdmitAttributes{Pod: tc.pod}
			actual := m.Admit(tCtx, &podAttr)

			if actual.Admit != tc.expectedAdmit {
				t.Errorf("Expected Admit to be %v got %v", tc.expectedAdmit, actual.Admit)
			}
			if tracker.podHintsCalled != tc.expectedPodHintsCalled {
				t.Errorf("Expected podHintsCalled to be %v got %v", tc.expectedPodHintsCalled, tracker.podHintsCalled)
			}
			if tracker.containerHintsCalled != tc.expectedContainerHintsCalled {
				t.Errorf("Expected containerHintsCalled to be %v got %v", tc.expectedContainerHintsCalled, tracker.containerHintsCalled)
			}
			if tracker.allocatePodCalled != tc.expectedAllocatePodCalled {
				t.Errorf("Expected allocatePodCalled to be %v got %v", tc.expectedAllocatePodCalled, tracker.allocatePodCalled)
			}
			if tracker.allocateContainerCalled != tc.expectedAllocateContainerCalled {
				t.Errorf("Expected allocateContainerCalled to be %v got %v", tc.expectedAllocateContainerCalled, tracker.allocateContainerCalled)
			}
		})
	}
}
