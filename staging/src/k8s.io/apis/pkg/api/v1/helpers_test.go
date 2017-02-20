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

package v1

import (
	"reflect"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

func TestAddToNodeAddresses(t *testing.T) {
	testCases := []struct {
		existing []NodeAddress
		toAdd    []NodeAddress
		expected []NodeAddress
	}{
		{
			existing: []NodeAddress{},
			toAdd:    []NodeAddress{},
			expected: []NodeAddress{},
		},
		{
			existing: []NodeAddress{},
			toAdd: []NodeAddress{
				{Type: NodeExternalIP, Address: "1.1.1.1"},
				{Type: NodeHostName, Address: "localhost"},
			},
			expected: []NodeAddress{
				{Type: NodeExternalIP, Address: "1.1.1.1"},
				{Type: NodeHostName, Address: "localhost"},
			},
		},
		{
			existing: []NodeAddress{},
			toAdd: []NodeAddress{
				{Type: NodeExternalIP, Address: "1.1.1.1"},
				{Type: NodeExternalIP, Address: "1.1.1.1"},
			},
			expected: []NodeAddress{
				{Type: NodeExternalIP, Address: "1.1.1.1"},
			},
		},
		{
			existing: []NodeAddress{
				{Type: NodeExternalIP, Address: "1.1.1.1"},
				{Type: NodeInternalIP, Address: "10.1.1.1"},
			},
			toAdd: []NodeAddress{
				{Type: NodeExternalIP, Address: "1.1.1.1"},
				{Type: NodeHostName, Address: "localhost"},
			},
			expected: []NodeAddress{
				{Type: NodeExternalIP, Address: "1.1.1.1"},
				{Type: NodeInternalIP, Address: "10.1.1.1"},
				{Type: NodeHostName, Address: "localhost"},
			},
		},
	}

	for i, tc := range testCases {
		AddToNodeAddresses(&tc.existing, tc.toAdd...)
		if !apiequality.Semantic.DeepEqual(tc.expected, tc.existing) {
			t.Errorf("case[%d], expected: %v, got: %v", i, tc.expected, tc.existing)
		}
	}
}

func TestGetAccessModesFromString(t *testing.T) {
	modes := GetAccessModesFromString("ROX")
	if !containsAccessMode(modes, ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", ReadOnlyMany, modes)
	}

	modes = GetAccessModesFromString("ROX,RWX")
	if !containsAccessMode(modes, ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", ReadOnlyMany, modes)
	}
	if !containsAccessMode(modes, ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", ReadWriteMany, modes)
	}

	modes = GetAccessModesFromString("RWO,ROX,RWX")
	if !containsAccessMode(modes, ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", ReadOnlyMany, modes)
	}
	if !containsAccessMode(modes, ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", ReadWriteMany, modes)
	}
}

func TestRemoveDuplicateAccessModes(t *testing.T) {
	modes := []PersistentVolumeAccessMode{
		ReadWriteOnce, ReadOnlyMany, ReadOnlyMany, ReadOnlyMany,
	}
	modes = removeDuplicateAccessModes(modes)
	if len(modes) != 2 {
		t.Errorf("Expected 2 distinct modes in set but found %v", len(modes))
	}
}

func TestNodeSelectorRequirementsAsSelector(t *testing.T) {
	matchExpressions := []NodeSelectorRequirement{{
		Key:      "foo",
		Operator: NodeSelectorOpIn,
		Values:   []string{"bar", "baz"},
	}}
	mustParse := func(s string) labels.Selector {
		out, e := labels.Parse(s)
		if e != nil {
			panic(e)
		}
		return out
	}
	tc := []struct {
		in        []NodeSelectorRequirement
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: []NodeSelectorRequirement{}, out: labels.Nothing()},
		{
			in:  matchExpressions,
			out: mustParse("foo in (baz,bar)"),
		},
		{
			in: []NodeSelectorRequirement{{
				Key:      "foo",
				Operator: NodeSelectorOpExists,
				Values:   []string{"bar", "baz"},
			}},
			expectErr: true,
		},
		{
			in: []NodeSelectorRequirement{{
				Key:      "foo",
				Operator: NodeSelectorOpGt,
				Values:   []string{"1"},
			}},
			out: mustParse("foo>1"),
		},
		{
			in: []NodeSelectorRequirement{{
				Key:      "bar",
				Operator: NodeSelectorOpLt,
				Values:   []string{"7"},
			}},
			out: mustParse("bar<7"),
		},
	}

	for i, tc := range tc {
		out, err := NodeSelectorRequirementsAsSelector(tc.in)
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
		if !reflect.DeepEqual(out, tc.out) {
			t.Errorf("[%v]expected:\n\t%+v\nbut got:\n\t%+v", i, tc.out, out)
		}
	}
}

func TestTaintToString(t *testing.T) {
	testCases := []struct {
		taint          *Taint
		expectedString string
	}{
		{
			taint: &Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectedString: "foo=bar:NoSchedule",
		},
		{
			taint: &Taint{
				Key:    "foo",
				Effect: TaintEffectNoSchedule,
			},
			expectedString: "foo:NoSchedule",
		},
	}

	for i, tc := range testCases {
		if tc.expectedString != tc.taint.ToString() {
			t.Errorf("[%v] expected taint %v converted to %s, got %s", i, tc.taint, tc.expectedString, tc.taint.ToString())
		}
	}
}

func TestMatchTaint(t *testing.T) {
	testCases := []struct {
		description  string
		taint        *Taint
		taintToMatch Taint
		expectMatch  bool
	}{
		{
			description: "two taints with the same key,value,effect should match",
			taint: &Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			taintToMatch: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectMatch: true,
		},
		{
			description: "two taints with the same key,effect but different value should match",
			taint: &Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			taintToMatch: Taint{
				Key:    "foo",
				Value:  "different-value",
				Effect: TaintEffectNoSchedule,
			},
			expectMatch: true,
		},
		{
			description: "two taints with the different key cannot match",
			taint: &Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			taintToMatch: Taint{
				Key:    "different-key",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectMatch: false,
		},
		{
			description: "two taints with the different effect cannot match",
			taint: &Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			taintToMatch: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectPreferNoSchedule,
			},
			expectMatch: false,
		},
	}

	for _, tc := range testCases {
		if tc.expectMatch != tc.taint.MatchTaint(&tc.taintToMatch) {
			t.Errorf("[%s] expect taint %s match taint %s", tc.description, tc.taint.ToString(), tc.taintToMatch.ToString())
		}
	}
}

func TestTolerationToleratesTaint(t *testing.T) {

	testCases := []struct {
		description     string
		toleration      Toleration
		taint           Taint
		expectTolerated bool
	}{
		{
			description: "toleration and taint have the same key and effect, and operator is Exists, and taint has no value, expect tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpExists,
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key and effect, and operator is Exists, and taint has some value, expect tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpExists,
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same effect, toleration has empty key and operator is Exists, means match all taints, expect tolerated",
			toleration: Toleration{
				Key:      "",
				Operator: TolerationOpExists,
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key, effect and value, and operator is Equal, expect tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key and effect, but different values, and operator is Equal, expect not tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "value1",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "value2",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
		},
		{
			description: "toleration and taint have the same key and value, but different effects, and operator is Equal, expect not tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoExecute,
			},
			expectTolerated: false,
		},
	}
	for _, tc := range testCases {
		if tolerated := tc.toleration.ToleratesTaint(&tc.taint); tc.expectTolerated != tolerated {
			t.Errorf("[%s] expect %v, got %v: toleration %+v, taint %s", tc.description, tc.expectTolerated, tolerated, tc.toleration, tc.taint.ToString())
		}
	}
}

func TestTolerationsTolerateTaintsWithFilter(t *testing.T) {
	testCases := []struct {
		description     string
		tolerations     []Toleration
		taints          []Taint
		applyFilter     taintsFilterFunc
		expectTolerated bool
	}{
		{
			description:     "empty tolerations tolerate empty taints",
			tolerations:     []Toleration{},
			taints:          []Taint{},
			applyFilter:     func(t *Taint) bool { return true },
			expectTolerated: true,
		},
		{
			description: "non-empty tolerations tolerate empty taints",
			tolerations: []Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   TaintEffectNoSchedule,
				},
			},
			taints:          []Taint{},
			applyFilter:     func(t *Taint) bool { return true },
			expectTolerated: true,
		},
		{
			description: "tolerations match all taints, expect tolerated",
			tolerations: []Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   TaintEffectNoSchedule,
				},
			},
			taints: []Taint{
				{
					Key:    "foo",
					Effect: TaintEffectNoSchedule,
				},
			},
			applyFilter:     func(t *Taint) bool { return true },
			expectTolerated: true,
		},
		{
			description: "tolerations don't match taints, but no taints apply to the filter, expect tolerated",
			tolerations: []Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   TaintEffectNoSchedule,
				},
			},
			taints: []Taint{
				{
					Key:    "bar",
					Effect: TaintEffectNoSchedule,
				},
			},
			applyFilter:     func(t *Taint) bool { return false },
			expectTolerated: true,
		},
		{
			description: "no filterFunc indicated, means all taints apply to the filter, tolerations don't match taints, expect untolerated",
			tolerations: []Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   TaintEffectNoSchedule,
				},
			},
			taints: []Taint{
				{
					Key:    "bar",
					Effect: TaintEffectNoSchedule,
				},
			},
			applyFilter:     nil,
			expectTolerated: false,
		},
		{
			description: "tolerations match taints, expect tolerated",
			tolerations: []Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   TaintEffectNoExecute,
				},
			},
			taints: []Taint{
				{
					Key:    "foo",
					Effect: TaintEffectNoExecute,
				},
				{
					Key:    "bar",
					Effect: TaintEffectNoSchedule,
				},
			},
			applyFilter:     func(t *Taint) bool { return t.Effect == TaintEffectNoExecute },
			expectTolerated: true,
		},
	}

	for _, tc := range testCases {
		if tc.expectTolerated != TolerationsTolerateTaintsWithFilter(tc.tolerations, tc.taints, tc.applyFilter) {
			filteredTaints := []Taint{}
			for _, taint := range tc.taints {
				if tc.applyFilter != nil && !tc.applyFilter(&taint) {
					continue
				}
				filteredTaints = append(filteredTaints, taint)
			}
			t.Errorf("[%s] expect tolerations %+v tolerate filtered taints %+v in taints %+v", tc.description, tc.tolerations, filteredTaints, tc.taints)
		}
	}
}

func TestGetAvoidPodsFromNode(t *testing.T) {
	controllerFlag := true
	testCases := []struct {
		node        *Node
		expectValue AvoidPods
		expectErr   bool
	}{
		{
			node:        &Node{},
			expectValue: AvoidPods{},
			expectErr:   false,
		},
		{
			node: &Node{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
						                            "apiVersion": "v1",
						                            "kind": "ReplicationController",
						                            "name": "foo",
						                            "uid": "abcdef123456",
						                            "controller": true
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
					},
				},
			},
			expectValue: AvoidPods{
				PreferAvoidPods: []PreferAvoidPodsEntry{
					{
						PodSignature: PodSignature{
							PodController: &metav1.OwnerReference{
								APIVersion: "v1",
								Kind:       "ReplicationController",
								Name:       "foo",
								UID:        "abcdef123456",
								Controller: &controllerFlag,
							},
						},
						Reason:  "some reason",
						Message: "some message",
					},
				},
			},
			expectErr: false,
		},
		{
			node: &Node{
				// Missing end symbol of "podController" and "podSignature"
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "kind": "ReplicationController",
							                    "apiVersion": "v1"
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
					},
				},
			},
			expectValue: AvoidPods{},
			expectErr:   true,
		},
	}

	for i, tc := range testCases {
		v, err := GetAvoidPodsFromNodeAnnotations(tc.node.Annotations)
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
		if !reflect.DeepEqual(tc.expectValue, v) {
			t.Errorf("[%v]expect value %v but got %v with %v", i, tc.expectValue, v, v.PreferAvoidPods[0].PodSignature.PodController.Controller)
		}
	}
}

func TestSysctlsFromPodAnnotation(t *testing.T) {
	type Test struct {
		annotation  string
		expectValue []Sysctl
		expectErr   bool
	}
	for i, test := range []Test{
		{
			annotation:  "",
			expectValue: nil,
		},
		{
			annotation: "foo.bar",
			expectErr:  true,
		},
		{
			annotation: "=123",
			expectErr:  true,
		},
		{
			annotation:  "foo.bar=",
			expectValue: []Sysctl{{Name: "foo.bar", Value: ""}},
		},
		{
			annotation:  "foo.bar=42",
			expectValue: []Sysctl{{Name: "foo.bar", Value: "42"}},
		},
		{
			annotation: "foo.bar=42,",
			expectErr:  true,
		},
		{
			annotation:  "foo.bar=42,abc.def=1",
			expectValue: []Sysctl{{Name: "foo.bar", Value: "42"}, {Name: "abc.def", Value: "1"}},
		},
	} {
		sysctls, err := SysctlsFromPodAnnotation(test.annotation)
		if test.expectErr && err == nil {
			t.Errorf("[%v]expected error but got none", i)
		} else if !test.expectErr && err != nil {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		} else if !reflect.DeepEqual(sysctls, test.expectValue) {
			t.Errorf("[%v]expect value %v but got %v", i, test.expectValue, sysctls)
		}
	}
}

// TODO: remove when alpha support for affinity is removed
func TestGetAffinityFromPodAnnotations(t *testing.T) {
	testCases := []struct {
		pod       *Pod
		expectErr bool
	}{
		{
			pod:       &Pod{},
			expectErr: false,
		},
		{
			pod: &Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "foo",
									"operator": "In",
									"values": ["value1", "value2"]
								}]
							}]
						}}}`,
					},
				},
			},
			expectErr: false,
		},
		{
			pod: &Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "foo",
						`,
					},
				},
			},
			expectErr: true,
		},
	}

	for i, tc := range testCases {
		_, err := GetAffinityFromPodAnnotations(tc.pod.Annotations)
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
	}
}
