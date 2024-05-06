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

	"k8s.io/api/core/v1"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
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
			expectedError:  fmt.Errorf("unsupported on machines with more than %v NUMA Nodes", maxAllowableNUMANodes),
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

func (m *mockHintProvider) Allocate(pod *v1.Pod, container *v1.Container) error {
	//return allocateError
	return nil
}

type mockPolicy struct {
	nonePolicy
	ph []map[string][]TopologyHint
}

func (p *mockPolicy) Merge(providersHints []map[string][]TopologyHint) (TopologyHint, bool) {
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
	for _, tc := range tcases {
		for _, hp := range tc.hp {
			mngr.AddHintProvider(hp)
		}
		if len(tc.hp) != len(mngr.scope.(*containerScope).hintProviders) {
			t.Errorf("error")
		}
	}
}

func TestAdmit(t *testing.T) {
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
		ctnActual := ctnScopeManager.Admit(&podAttr)
		if ctnActual.Admit != tc.expected {
			t.Errorf("Error occurred, expected Admit in result to be %v got %v", tc.expected, ctnActual.Admit)
		}
		if !ctnActual.Admit && ctnActual.Reason != ErrorTopologyAffinity {
			t.Errorf("Error occurred, expected Reason in result to be %v got %v", ErrorTopologyAffinity, ctnActual.Reason)
		}

		// Pod scope Admit
		podActual := podScopeManager.Admit(&podAttr)
		if podActual.Admit != tc.expected {
			t.Errorf("Error occurred, expected Admit in result to be %v got %v", tc.expected, podActual.Admit)
		}
		if !ctnActual.Admit && ctnActual.Reason != ErrorTopologyAffinity {
			t.Errorf("Error occurred, expected Reason in result to be %v got %v", ErrorTopologyAffinity, ctnActual.Reason)
		}
	}
}
