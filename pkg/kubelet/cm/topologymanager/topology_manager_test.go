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
	"context"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"

	cadvisorapi "github.com/google/cadvisor/lib/model"

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
	logger, _ := ktesting.NewTestContext(t)
	numaDistanceErr := "error getting NUMA distances from cadvisor"
	if runtime.GOOS == "windows" {
		numaDistanceErr = fmt.Sprintf("the %q policy option is not supported on Windows because NUMA node distances are not available", PreferClosestNUMANodes)
	}

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
			expectedError: fmt.Errorf("%s", numaDistanceErr),
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

		mngr, err := NewManager(logger, topology, tc.policyName, "container", tc.policyOptions)
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

// policyOptionsOf returns the PolicyOptions a policy was built with.
func policyOptionsOf(t *testing.T, policy Policy) PolicyOptions {
	t.Helper()
	switch p := policy.(type) {
	case *bestEffortPolicy:
		return p.opts
	case *restrictedPolicy:
		return p.opts
	case *singleNumaNodePolicy:
		return p.opts
	default:
		t.Fatalf("policy %q has an unexpected type %T", policy.Name(), policy)
		return PolicyOptions{}
	}
}

// TestNewManagerPolicyOptionPropagation checks that the policy option map
// reaches the policy the manager builds, for every policy and every scope.
// The container manager hands NodeConfig.TopologyManagerPolicyOptions to
// NewManager verbatim on both Linux and Windows, so NewPolicyOptions is the
// only place the map is decoded: whatever lands in PolicyOptions here is what
// hint selection sees.
func TestNewManagerPolicyOptionPropagation(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tcases := []struct {
		description     string
		policyOptions   map[string]string
		expectedOptions PolicyOptions
	}{
		{
			description: "no policy options",
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  defaultMaxAllowableNUMANodes,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
		},
		{
			description: "numa-allocation-strategy set to most-allocated",
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  defaultMaxAllowableNUMANodes,
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
			},
		},
		{
			description: "numa-allocation-strategy set to least-allocated",
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  defaultMaxAllowableNUMANodes,
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
			},
		},
		{
			description: "numa-allocation-strategy set to none",
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  defaultMaxAllowableNUMANodes,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
		},
		{
			description: "numa-allocation-strategy alongside another option",
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
				MaxAllowableNUMANodes:  "9",
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  9,
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
			},
		},
		{
			description: "numa-score-weights reaches the policy parsed",
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=3,memory=1,nvidia.com/gpu=6",
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  defaultMaxAllowableNUMANodes,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
				NUMAScoreWeights:       map[string]int{"cpu": 3, "memory": 1, "nvidia.com/gpu": 6},
			},
		},
		{
			description: "numa-score-weights alongside numa-allocation-strategy",
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
				NUMAScoreWeights:       "intel.com/sriov-nic=10,cpu=0,memory=0",
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  defaultMaxAllowableNUMANodes,
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
				NUMAScoreWeights:       map[string]int{"intel.com/sriov-nic": 10, "cpu": 0, "memory": 0},
			},
		},
	}

	for _, policyName := range []string{PolicyBestEffort, PolicyRestricted, PolicySingleNumaNode} {
		for _, scopeName := range []string{ContainerTopologyScope, PodTopologyScope} {
			for _, tc := range tcases {
				t.Run(fmt.Sprintf("%s/%s/%s", policyName, scopeName, tc.description), func(t *testing.T) {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyManagerPolicyAlphaOptions, true)

					mngr, err := NewManager(logger, nil, policyName, scopeName, tc.policyOptions)
					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}

					opts := policyOptionsOf(t, mngr.GetPolicy())
					if !reflect.DeepEqual(opts, tc.expectedOptions) {
						t.Errorf("Unexpected policy options. Have: %v wants %v", opts, tc.expectedOptions)
					}
				})
			}
		}
	}
}

// TestNewManagerInvalidNUMAAllocationStrategy checks that a bad
// numa-allocation-strategy value aborts manager creation. NewContainerManager
// propagates that error, so kubelet startup fails rather than silently running
// with the option ignored.
func TestNewManagerInvalidNUMAAllocationStrategy(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tcases := []struct {
		description       string
		alphaOptionsGate  bool
		policyOptions     map[string]string
		expectedErrSubstr string
	}{
		{
			description:      "unknown numa-allocation-strategy value",
			alphaOptionsGate: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: "most-allocated-ish",
			},
			expectedErrSubstr: `bad value for option "numa-allocation-strategy"`,
		},
		{
			description:      "numa-allocation-strategy value differing only in case",
			alphaOptionsGate: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: "Most-Allocated",
			},
			expectedErrSubstr: `bad value for option "numa-allocation-strategy"`,
		},
		{
			description:      "numa-allocation-strategy without the alpha options gate",
			alphaOptionsGate: false,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
			},
			expectedErrSubstr: `topology manager policy alpha-level options not enabled`,
		},
	}

	for _, policyName := range []string{PolicyBestEffort, PolicyRestricted, PolicySingleNumaNode} {
		for _, tc := range tcases {
			t.Run(fmt.Sprintf("%s/%s", policyName, tc.description), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyManagerPolicyAlphaOptions, tc.alphaOptionsGate)

				_, err := NewManager(logger, nil, policyName, ContainerTopologyScope, tc.policyOptions)
				if err == nil {
					t.Fatalf("expected an error containing %q, got none", tc.expectedErrSubstr)
				}
				if !strings.Contains(err.Error(), tc.expectedErrSubstr) {
					t.Errorf("Unexpected error message. Have: %s wants a message containing %s", err.Error(), tc.expectedErrSubstr)
				}
			})
		}
	}
}

// TestNewManagerInvalidNUMAScoreWeights checks that a bad numa-score-weights
// value aborts manager creation, for the same reason as the
// numa-allocation-strategy equivalent above: NewContainerManager propagates
// the error, so kubelet startup fails rather than silently running with the
// weights ignored.
func TestNewManagerInvalidNUMAScoreWeights(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tcases := []struct {
		description       string
		alphaOptionsGate  bool
		policyOptions     map[string]string
		expectedErrSubstr string
	}{
		{
			description:      "numa-score-weights entry without a separator",
			alphaOptionsGate: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu:3",
			},
			expectedErrSubstr: `bad value for option "numa-score-weights"`,
		},
		{
			description:      "numa-score-weights with a fractional weight",
			alphaOptionsGate: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=3.5",
			},
			expectedErrSubstr: `bad value for option "numa-score-weights"`,
		},
		{
			description:      "numa-score-weights with a negative weight",
			alphaOptionsGate: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=-5",
			},
			expectedErrSubstr: "must be in range [0, 100]",
		},
		{
			description:      "numa-score-weights above the accepted range",
			alphaOptionsGate: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=101",
			},
			expectedErrSubstr: "must be in range [0, 100]",
		},
		{
			description:      "numa-score-weights without the alpha options gate",
			alphaOptionsGate: false,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=3,memory=1",
			},
			expectedErrSubstr: `topology manager policy alpha-level options not enabled`,
		},
	}

	for _, policyName := range []string{PolicyBestEffort, PolicyRestricted, PolicySingleNumaNode} {
		for _, tc := range tcases {
			t.Run(fmt.Sprintf("%s/%s", policyName, tc.description), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyManagerPolicyAlphaOptions, tc.alphaOptionsGate)

				_, err := NewManager(logger, nil, policyName, ContainerTopologyScope, tc.policyOptions)
				if err == nil {
					t.Fatalf("expected an error containing %q, got none", tc.expectedErrSubstr)
				}
				if !strings.Contains(err.Error(), tc.expectedErrSubstr) {
					t.Errorf("Unexpected error message. Have: %s wants a message containing %s", err.Error(), tc.expectedErrSubstr)
				}
			})
		}
	}
}

func TestManagerScope(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
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
		mngr, err := NewManager(logger, nil, "best-effort", tc.scopeName, nil)

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

func (m *mockHintProvider) GetTopologyHints(_ klog.Logger, _ *v1.Pod, _ *v1.Container, _ lifecycle.Operation) map[string][]TopologyHint {
	return m.th
}

func (m *mockHintProvider) GetPodTopologyHints(_ klog.Logger, _ *v1.Pod, _ lifecycle.Operation) map[string][]TopologyHint {
	return m.th
}

func (m *mockHintProvider) AllocatePod(_ klog.Logger, _ *v1.Pod, _ lifecycle.Operation) error {
	return nil
}

func (m *mockHintProvider) Allocate(_ context.Context, _ *v1.Pod, _ *v1.Container, _ lifecycle.Operation) error {
	//return allocateError
	return nil
}

type mockPolicy struct {
	nonePolicy
	ph []map[string][]TopologyHint
}

func (p *mockPolicy) Merge(_ klog.Logger, providersHints []map[string][]TopologyHint) (TopologyHint, bool) {
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

func (m *trackingHintProvider) GetTopologyHints(_ klog.Logger, _ *v1.Pod, _ *v1.Container, _ lifecycle.Operation) map[string][]TopologyHint {
	m.containerHintsCalled = true
	return m.hints
}

func (m *trackingHintProvider) GetPodTopologyHints(_ klog.Logger, _ *v1.Pod, _ lifecycle.Operation) map[string][]TopologyHint {
	m.podHintsCalled = true
	return m.hints
}

func (m *trackingHintProvider) AllocatePod(_ klog.Logger, _ *v1.Pod, _ lifecycle.Operation) error {
	m.allocatePodCalled = true
	return nil
}

func (m *trackingHintProvider) Allocate(_ context.Context, _ *v1.Pod, _ *v1.Container, _ lifecycle.Operation) error {
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
			tc.scope.AddHintProvider(tCtx.Logger(), tracker)

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
