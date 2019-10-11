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
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

func NewTestSocketMask(sockets ...int) socketmask.SocketMask {
	s, _ := socketmask.NewSocketMask(sockets...)
	return s
}

func TestNewManager(t *testing.T) {
	tcases := []struct {
		description    string
		policyName     string
		expectedPolicy string
		expectedError  error
	}{
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
			description:   "Policy is set to unknown",
			policyName:    "unknown",
			expectedError: fmt.Errorf("unknown policy: \"unknown\""),
		},
	}

	for _, tc := range tcases {
		mngr, err := NewManager(nil, tc.policyName)

		if tc.expectedError != nil {
			if !strings.Contains(err.Error(), tc.expectedError.Error()) {
				t.Errorf("Unexpected error message. Have: %s wants %s", err.Error(), tc.expectedError.Error())
			}
		} else {
			rawMgr := mngr.(*manager)
			if rawMgr.policy.Name() != tc.expectedPolicy {
				t.Errorf("Unexpected policy name. Have: %q wants %q", rawMgr.policy.Name(), tc.expectedPolicy)
			}
		}
	}
}

type mockHintProvider struct {
	th map[string][]TopologyHint
}

func (m *mockHintProvider) GetTopologyHints(pod v1.Pod, container v1.Container) map[string][]TopologyHint {
	return m.th
}

func TestGetAffinity(t *testing.T) {
	tcases := []struct {
		name          string
		containerName string
		podUID        string
		expected      TopologyHint
	}{
		{
			name:          "case1",
			containerName: "nginx",
			podUID:        "0aafa4c4-38e8-11e9-bcb1-a4bf01040474",
			expected:      TopologyHint{},
		},
	}
	for _, tc := range tcases {
		mngr := manager{}
		actual := mngr.GetAffinity(tc.podUID, tc.containerName)
		if !reflect.DeepEqual(actual, tc.expected) {
			t.Errorf("Expected Affinity in result to be %v, got %v", tc.expected, actual)
		}
	}
}

func TestCalculateAffinity(t *testing.T) {
	numaNodes := []int{0, 1}

	tcases := []struct {
		name     string
		hp       []HintProvider
		expected TopologyHint
		policy   Policy
	}{
		{
			name: "TopologyHint not set",
			hp:   []HintProvider{},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns -nil map[string][]TopologyHint from provider",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": nil,
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint from provider",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Single TopologyHint with Preferred as true and NUMANodeAffinity as nil",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: nil,
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "Single TopologyHint with Preferred as false and NUMANodeAffinity as nil",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: nil,
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, both preferred 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, both preferred 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 hint each, 1 wider mask, both preferred 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 hint each, 1 wider mask, both preferred 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 hint each, no common mask",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, 1 preferred, 1 not 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, 1 preferred, 1 not 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(1),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 no hints, 1 single hint preferred 1/2",
			hp: []HintProvider{
				&mockHintProvider{},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 no hints, 1 single hint preferred 2/2",
			hp: []HintProvider{
				&mockHintProvider{},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 with 2 hints, 1 with single hint matching 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 with 2 hints, 1 with single hint matching 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 with 2 hints, 1 with single non-preferred hint matching",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, both with 2 hints, matching narrower preferred hint from both",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Ensure less narrow preferred hints are chosen over narrower non-preferred hints",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Multiple resources, same provider",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(1),
				Preferred:        true,
			},
		},
		{
			name:   "Special cased PolicySingleNumaNode for single NUMA hint generation",
			policy: NewSingleNumaNodePolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        true,
							},
						},
						"resource2": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        false,
			},
		},
		{
			name:   "Special cased PolicySingleNumaNode with one no-preference provider",
			policy: NewSingleNumaNodePolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
								Preferred:        false,
							},
						},
					},
				},
				&mockHintProvider{
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestSocketMask(0),
				Preferred:        true,
			},
		},
	}

	for _, tc := range tcases {
		mngr := manager{
			policy:        tc.policy,
			hintProviders: tc.hp,
			numaNodes:     numaNodes,
		}
		actual := mngr.calculateAffinity(v1.Pod{}, v1.Container{})
		if !actual.NUMANodeAffinity.IsEqual(tc.expected.NUMANodeAffinity) {
			t.Errorf("Expected NUMANodeAffinity in result to be %v, got %v", tc.expected.NUMANodeAffinity, actual.NUMANodeAffinity)
		}
		if actual.Preferred != tc.expected.Preferred {
			t.Errorf("Expected Affinity preference in result to be %v, got %v", tc.expected.Preferred, actual.Preferred)
		}
	}
}

func TestAddContainer(t *testing.T) {
	testCases := []struct {
		name        string
		containerID string
		podUID      types.UID
	}{
		{
			name:        "Case1",
			containerID: "nginx",
			podUID:      "0aafa4c4-38e8-11e9-bcb1-a4bf01040474",
		},
		{
			name:        "Case2",
			containerID: "Busy_Box",
			podUID:      "b3ee37fc-39a5-11e9-bcb1-a4bf01040474",
		},
	}
	mngr := manager{}
	mngr.podMap = make(map[string]string)
	for _, tc := range testCases {
		pod := v1.Pod{}
		pod.UID = tc.podUID
		err := mngr.AddContainer(&pod, tc.containerID)
		if err != nil {
			t.Errorf("Expected error to be nil but got: %v", err)
		}
		if val, ok := mngr.podMap[tc.containerID]; ok {
			if reflect.DeepEqual(val, pod.UID) {
				t.Errorf("Error occurred")
			}
		} else {
			t.Errorf("Error occurred, Pod not added to podMap")
		}
	}
}

func TestRemoveContainer(t *testing.T) {
	testCases := []struct {
		name        string
		containerID string
		podUID      types.UID
	}{
		{
			name:        "Case1",
			containerID: "nginx",
			podUID:      "0aafa4c4-38e8-11e9-bcb1-a4bf01040474",
		},
		{
			name:        "Case2",
			containerID: "Busy_Box",
			podUID:      "b3ee37fc-39a5-11e9-bcb1-a4bf01040474",
		},
	}
	var len1, len2 int
	mngr := manager{}
	mngr.podMap = make(map[string]string)
	for _, tc := range testCases {
		mngr.podMap[tc.containerID] = string(tc.podUID)
		len1 = len(mngr.podMap)
		err := mngr.RemoveContainer(tc.containerID)
		len2 = len(mngr.podMap)
		if err != nil {
			t.Errorf("Expected error to be nil but got: %v", err)
		}
		if len1-len2 != 1 {
			t.Errorf("Remove Pod resulted in error")
		}
	}

}
func TestAddHintProvider(t *testing.T) {
	var len1 int
	tcases := []struct {
		name string
		hp   []HintProvider
	}{
		{
			name: "Add HintProvider",
			hp: []HintProvider{
				&mockHintProvider{},
			},
		},
	}
	mngr := manager{}
	for _, tc := range tcases {
		mngr.hintProviders = []HintProvider{}
		len1 = len(mngr.hintProviders)
		mngr.AddHintProvider(tc.hp[0])
	}
	len2 := len(mngr.hintProviders)
	if len2-len1 != 1 {
		t.Errorf("error")
	}
}

func TestAdmit(t *testing.T) {
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
			policy:   NewRestrictedPolicy(),
			hp: []HintProvider{
				&mockHintProvider{},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as BestEffort. Restricted Policy. No Hints.",
			qosClass: v1.PodQOSBestEffort,
			policy:   NewRestrictedPolicy(),
			hp: []HintProvider{
				&mockHintProvider{},
			},
			expected: true,
		},
		{
			name:     "QOSClass set as Guaranteed. BestEffort Policy. Preferred Affinity.",
			qosClass: v1.PodQOSGuaranteed,
			policy:   NewBestEffortPolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
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
			policy:   NewBestEffortPolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
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
			policy:   NewBestEffortPolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
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
			policy:   NewRestrictedPolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
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
			policy:   NewRestrictedPolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(0),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
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
			policy:   NewRestrictedPolicy(),
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestSocketMask(0, 1),
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
		man := manager{
			policy:           tc.policy,
			podTopologyHints: make(map[string]map[string]TopologyHint),
			hintProviders:    tc.hp,
			numaNodes:        []int{0, 1},
		}

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

		actual := man.Admit(&podAttr)
		if actual.Admit != tc.expected {
			t.Errorf("Error occurred, expected Admit in result to be %v got %v", tc.expected, actual.Admit)
		}
	}
}
