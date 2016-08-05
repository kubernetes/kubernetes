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

package api

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/labels"
)

func TestConversionError(t *testing.T) {
	var i int
	var s string
	i = 3
	s = "foo"
	c := ConversionError{
		In: &i, Out: &s,
		Message: "Can't make x into y, silly",
	}
	var e error
	e = &c // ensure it implements error
	msg := e.Error()
	t.Logf("Message is %v", msg)
	for _, part := range []string{"3", "int", "string", "Can't"} {
		if !strings.Contains(msg, part) {
			t.Errorf("didn't find %v", part)
		}
	}
}

func TestSemantic(t *testing.T) {
	table := []struct {
		a, b        interface{}
		shouldEqual bool
	}{
		{resource.MustParse("0"), resource.Quantity{}, true},
		{resource.Quantity{}, resource.MustParse("0"), true},
		{resource.Quantity{}, resource.MustParse("1m"), false},
		{
			resource.NewQuantity(5, resource.BinarySI),
			resource.NewQuantity(5, resource.DecimalSI),
			true,
		},
		{resource.MustParse("2m"), resource.MustParse("1m"), false},
	}

	for index, item := range table {
		if e, a := item.shouldEqual, Semantic.DeepEqual(item.a, item.b); e != a {
			t.Errorf("case[%d], expected %v, got %v.", index, e, a)
		}
	}
}

func TestIsStandardResource(t *testing.T) {
	testCases := []struct {
		input  string
		output bool
	}{
		{"cpu", true},
		{"memory", true},
		{"disk", false},
		{"blah", false},
		{"x.y.z", false},
	}
	for i, tc := range testCases {
		if IsStandardResourceName(tc.input) != tc.output {
			t.Errorf("case[%d], expected: %t, got: %t", i, tc.output, !tc.output)
		}
	}
}

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
		if !Semantic.DeepEqual(tc.expected, tc.existing) {
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

func TestGetAffinityFromPod(t *testing.T) {
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
				ObjectMeta: ObjectMeta{
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
				ObjectMeta: ObjectMeta{
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
				ObjectMeta: ObjectMeta{
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
							PodController: &OwnerReference{
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
				ObjectMeta: ObjectMeta{
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
