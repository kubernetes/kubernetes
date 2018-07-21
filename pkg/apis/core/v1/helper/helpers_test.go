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

package helper

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
)

func TestIsNativeResource(t *testing.T) {
	testCases := []struct {
		resourceName v1.ResourceName
		expectVal    bool
	}{
		{
			resourceName: "pod.alpha.kubernetes.io/opaque-int-resource-foo",
			expectVal:    true,
		},
		{
			resourceName: "kubernetes.io/resource-foo",
			expectVal:    true,
		},
		{
			resourceName: "foo",
			expectVal:    true,
		},
		{
			resourceName: "a/b",
			expectVal:    false,
		},
		{
			resourceName: "",
			expectVal:    true,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("resourceName input=%s, expected value=%v", tc.resourceName, tc.expectVal), func(t *testing.T) {
			t.Parallel()
			v := IsNativeResource(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestHugePageSizeFromResourceName(t *testing.T) {
	expected100m, _ := resource.ParseQuantity("100m")
	testCases := []struct {
		resourceName v1.ResourceName
		expectVal    resource.Quantity
		expectErr    bool
	}{
		{
			resourceName: "pod.alpha.kubernetes.io/opaque-int-resource-foo",
			expectVal:    resource.Quantity{},
			expectErr:    true,
		},
		{
			resourceName: "hugepages-",
			expectVal:    resource.Quantity{},
			expectErr:    true,
		},
		{
			resourceName: "hugepages-100m",
			expectVal:    expected100m,
			expectErr:    false,
		},
		{
			resourceName: "",
			expectVal:    resource.Quantity{},
			expectErr:    true,
		},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("resourceName input=%s, expected value=%v", tc.resourceName, tc.expectVal), func(t *testing.T) {
			t.Parallel()
			v, err := HugePageSizeFromResourceName(tc.resourceName)
			if err == nil && tc.expectErr {
				t.Errorf("[%v]expected error but got none.", i)
			}
			if err != nil && !tc.expectErr {
				t.Errorf("[%v]did not expect error but got: %v", i, err)
			}
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestIsOvercommitAllowed(t *testing.T) {
	testCases := []struct {
		resourceName v1.ResourceName
		expectVal    bool
	}{
		{
			resourceName: "pod.alpha.kubernetes.io/opaque-int-resource-foo",
			expectVal:    true,
		},
		{
			resourceName: "kubernetes.io/resource-foo",
			expectVal:    true,
		},
		{
			resourceName: "hugepages-100m",
			expectVal:    false,
		},
		{
			resourceName: "",
			expectVal:    true,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("resourceName input=%s, expected value=%v", tc.resourceName, tc.expectVal), func(t *testing.T) {
			t.Parallel()
			v := IsOvercommitAllowed(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestAddToNodeAddresses(t *testing.T) {
	testCases := []struct {
		existing []v1.NodeAddress
		toAdd    []v1.NodeAddress
		expected []v1.NodeAddress
	}{
		{
			existing: []v1.NodeAddress{},
			toAdd:    []v1.NodeAddress{},
			expected: []v1.NodeAddress{},
		},
		{
			existing: []v1.NodeAddress{},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
		},
		{
			existing: []v1.NodeAddress{},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
		},
		{
			existing: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
			},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
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
	if !containsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadOnlyMany, modes)
	}

	modes = GetAccessModesFromString("ROX,RWX")
	if !containsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadOnlyMany, modes)
	}
	if !containsAccessMode(modes, v1.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteMany, modes)
	}

	modes = GetAccessModesFromString("RWO,ROX,RWX")
	if !containsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadOnlyMany, modes)
	}
	if !containsAccessMode(modes, v1.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteMany, modes)
	}
}

func TestRemoveDuplicateAccessModes(t *testing.T) {
	modes := []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce, v1.ReadOnlyMany, v1.ReadOnlyMany, v1.ReadOnlyMany,
	}
	modes = removeDuplicateAccessModes(modes)
	if len(modes) != 2 {
		t.Errorf("Expected 2 distinct modes in set but found %v", len(modes))
	}
}

func TestNodeSelectorRequirementsAsSelector(t *testing.T) {
	matchExpressions := []v1.NodeSelectorRequirement{{
		Key:      "foo",
		Operator: v1.NodeSelectorOpIn,
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
		in        []v1.NodeSelectorRequirement
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: []v1.NodeSelectorRequirement{}, out: labels.Nothing()},
		{
			in:  matchExpressions,
			out: mustParse("foo in (baz,bar)"),
		},
		{
			in: []v1.NodeSelectorRequirement{{
				Key:      "foo",
				Operator: v1.NodeSelectorOpExists,
				Values:   []string{"bar", "baz"},
			}},
			expectErr: true,
		},
		{
			in: []v1.NodeSelectorRequirement{{
				Key:      "foo",
				Operator: v1.NodeSelectorOpGt,
				Values:   []string{"1"},
			}},
			out: mustParse("foo>1"),
		},
		{
			in: []v1.NodeSelectorRequirement{{
				Key:      "bar",
				Operator: v1.NodeSelectorOpLt,
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

func TestTopologySelectorRequirementsAsSelector(t *testing.T) {
	mustParse := func(s string) labels.Selector {
		out, e := labels.Parse(s)
		if e != nil {
			panic(e)
		}
		return out
	}
	tc := []struct {
		in        []v1.TopologySelectorLabelRequirement
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: []v1.TopologySelectorLabelRequirement{}, out: labels.Nothing()},
		{
			in: []v1.TopologySelectorLabelRequirement{{
				Key:    "foo",
				Values: []string{"bar", "baz"},
			}},
			out: mustParse("foo in (baz,bar)"),
		},
		{
			in: []v1.TopologySelectorLabelRequirement{{
				Key:    "foo",
				Values: []string{},
			}},
			expectErr: true,
		},
		{
			in: []v1.TopologySelectorLabelRequirement{
				{
					Key:    "foo",
					Values: []string{"bar", "baz"},
				},
				{
					Key:    "invalid",
					Values: []string{},
				},
			},
			expectErr: true,
		},
		{
			in: []v1.TopologySelectorLabelRequirement{{
				Key:    "/invalidkey",
				Values: []string{"bar", "baz"},
			}},
			expectErr: true,
		},
	}

	for i, tc := range tc {
		out, err := TopologySelectorRequirementsAsSelector(tc.in)
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

func TestTolerationsTolerateTaintsWithFilter(t *testing.T) {
	testCases := []struct {
		description     string
		tolerations     []v1.Toleration
		taints          []v1.Taint
		applyFilter     taintsFilterFunc
		expectTolerated bool
	}{
		{
			description:     "empty tolerations tolerate empty taints",
			tolerations:     []v1.Toleration{},
			taints:          []v1.Taint{},
			applyFilter:     func(t *v1.Taint) bool { return true },
			expectTolerated: true,
		},
		{
			description: "non-empty tolerations tolerate empty taints",
			tolerations: []v1.Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   v1.TaintEffectNoSchedule,
				},
			},
			taints:          []v1.Taint{},
			applyFilter:     func(t *v1.Taint) bool { return true },
			expectTolerated: true,
		},
		{
			description: "tolerations match all taints, expect tolerated",
			tolerations: []v1.Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   v1.TaintEffectNoSchedule,
				},
			},
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			applyFilter:     func(t *v1.Taint) bool { return true },
			expectTolerated: true,
		},
		{
			description: "tolerations don't match taints, but no taints apply to the filter, expect tolerated",
			tolerations: []v1.Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   v1.TaintEffectNoSchedule,
				},
			},
			taints: []v1.Taint{
				{
					Key:    "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			applyFilter:     func(t *v1.Taint) bool { return false },
			expectTolerated: true,
		},
		{
			description: "no filterFunc indicated, means all taints apply to the filter, tolerations don't match taints, expect untolerated",
			tolerations: []v1.Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   v1.TaintEffectNoSchedule,
				},
			},
			taints: []v1.Taint{
				{
					Key:    "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			applyFilter:     nil,
			expectTolerated: false,
		},
		{
			description: "tolerations match taints, expect tolerated",
			tolerations: []v1.Toleration{
				{
					Key:      "foo",
					Operator: "Exists",
					Effect:   v1.TaintEffectNoExecute,
				},
			},
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			applyFilter:     func(t *v1.Taint) bool { return t.Effect == v1.TaintEffectNoExecute },
			expectTolerated: true,
		},
	}

	for _, tc := range testCases {
		if tc.expectTolerated != TolerationsTolerateTaintsWithFilter(tc.tolerations, tc.taints, tc.applyFilter) {
			filteredTaints := []v1.Taint{}
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
		node        *v1.Node
		expectValue v1.AvoidPods
		expectErr   bool
	}{
		{
			node:        &v1.Node{},
			expectValue: v1.AvoidPods{},
			expectErr:   false,
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.PreferAvoidPodsAnnotationKey: `
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
			expectValue: v1.AvoidPods{
				PreferAvoidPods: []v1.PreferAvoidPodsEntry{
					{
						PodSignature: v1.PodSignature{
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
			node: &v1.Node{
				// Missing end symbol of "podController" and "podSignature"
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.PreferAvoidPodsAnnotationKey: `
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
			expectValue: v1.AvoidPods{},
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

func TestMatchNodeSelectorTerms(t *testing.T) {
	type args struct {
		nodeSelectorTerms []v1.NodeSelectorTerm
		nodeLabels        labels.Set
		nodeFields        fields.Set
	}

	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "nil terms",
			args: args{
				nodeSelectorTerms: nil,
				nodeLabels:        nil,
				nodeFields:        nil,
			},
			want: false,
		},
		{
			name: "node label matches matchExpressions terms",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val"},
						}},
					},
				},
				nodeLabels: map[string]string{"label_1": "label_1_val"},
				nodeFields: nil,
			},
			want: true,
		},
		{
			name: "node field matches matchFields terms",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1"},
						}},
					},
				},
				nodeLabels: nil,
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: true,
		},
		{
			name: "invalid node field requirement",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1, host_2"},
						}},
					},
				},
				nodeLabels: nil,
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: false,
		},
		{
			name: "fieldSelectorTerm with node labels",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"metadata.name": "host_1",
				},
				nodeFields: nil,
			},
			want: false,
		},
		{
			name: "labelSelectorTerm with node fields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1"},
						}},
					},
				},
				nodeLabels: nil,
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: false,
		},
		{
			name: "labelSelectorTerm and fieldSelectorTerm was set, but only node fields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val"},
						}},
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1"},
						}},
					},
				},
				nodeLabels: nil,
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: false,
		},
		{
			name: "labelSelectorTerm and fieldSelectorTerm was set, both node fields and labels (both matched)",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val"},
						}},
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"label_1": "label_1_val",
				},
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: true,
		},
		{
			name: "labelSelectorTerm and fieldSelectorTerm was set, both node fields and labels (one mismatched)",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val"},
						}},
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"label_1": "label_1_val-failed",
				},
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: false,
		},
		{
			name: "multi-selector was set, both node fields and labels (one mismatched)",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val"},
						}},
					},
					{
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"label_1": "label_1_val-failed",
				},
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchNodeSelectorTerms(tt.args.nodeSelectorTerms, tt.args.nodeLabels, tt.args.nodeFields); got != tt.want {
				t.Errorf("MatchNodeSelectorTermsORed() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestMatchNodeSelectorTermsStateless ensures MatchNodeSelectorTerms()
// is invoked in a "stateless" manner, i.e. nodeSelectorTerms should NOT
// be deeply modifed after invoking
func TestMatchNodeSelectorTermsStateless(t *testing.T) {
	type args struct {
		nodeSelectorTerms []v1.NodeSelectorTerm
		nodeLabels        labels.Set
		nodeFields        fields.Set
	}

	tests := []struct {
		name string
		args args
		want []v1.NodeSelectorTerm
	}{
		{
			name: "nil terms",
			args: args{
				nodeSelectorTerms: nil,
				nodeLabels:        nil,
				nodeFields:        nil,
			},
			want: nil,
		},
		{
			name: "nodeLabels: preordered matchExpressions and nil matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val", "label_2_val"},
						}},
					},
				},
				nodeLabels: map[string]string{"label_1": "label_1_val"},
				nodeFields: nil,
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{{
						Key:      "label_1",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"label_1_val", "label_2_val"},
					}},
				},
			},
		},
		{
			name: "nodeLabels: unordered matchExpressions and nil matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_2_val", "label_1_val"},
						}},
					},
				},
				nodeLabels: map[string]string{"label_1": "label_1_val"},
				nodeFields: nil,
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{{
						Key:      "label_1",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"label_2_val", "label_1_val"},
					}},
				},
			},
		},
		{
			name: "nodeFields: nil matchExpressions and preordered matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1", "host_2"},
						}},
					},
				},
				nodeLabels: nil,
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"host_1", "host_2"},
					}},
				},
			},
		},
		{
			name: "nodeFields: nil matchExpressions and unordered matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_2", "host_1"},
						}},
					},
				},
				nodeLabels: nil,
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"host_2", "host_1"},
					}},
				},
			},
		},
		{
			name: "nodeLabels and nodeFields: ordered matchExpressions and ordered matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val", "label_2_val"},
						}},
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1", "host_2"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"label_1": "label_1_val",
				},
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{{
						Key:      "label_1",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"label_1_val", "label_2_val"},
					}},
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"host_1", "host_2"},
					}},
				},
			},
		},
		{
			name: "nodeLabels and nodeFields: ordered matchExpressions and unordered matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_1_val", "label_2_val"},
						}},
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_2", "host_1"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"label_1": "label_1_val",
				},
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{{
						Key:      "label_1",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"label_1_val", "label_2_val"},
					}},
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"host_2", "host_1"},
					}},
				},
			},
		},
		{
			name: "nodeLabels and nodeFields: unordered matchExpressions and ordered matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_2_val", "label_1_val"},
						}},
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_1", "host_2"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"label_1": "label_1_val",
				},
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{{
						Key:      "label_1",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"label_2_val", "label_1_val"},
					}},
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"host_1", "host_2"},
					}},
				},
			},
		},
		{
			name: "nodeLabels and nodeFields: unordered matchExpressions and unordered matchFields",
			args: args{
				nodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "label_1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"label_2_val", "label_1_val"},
						}},
						MatchFields: []v1.NodeSelectorRequirement{{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"host_2", "host_1"},
						}},
					},
				},
				nodeLabels: map[string]string{
					"label_1": "label_1_val",
				},
				nodeFields: map[string]string{
					"metadata.name": "host_1",
				},
			},
			want: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{{
						Key:      "label_1",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"label_2_val", "label_1_val"},
					}},
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"host_2", "host_1"},
					}},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			MatchNodeSelectorTerms(tt.args.nodeSelectorTerms, tt.args.nodeLabels, tt.args.nodeFields)
			if !apiequality.Semantic.DeepEqual(tt.args.nodeSelectorTerms, tt.want) {
				// fail when tt.args.nodeSelectorTerms is deeply modified
				t.Errorf("MatchNodeSelectorTerms() got = %v, want %v", tt.args.nodeSelectorTerms, tt.want)
			}
		})
	}
}

func TestMatchTopologySelectorTerms(t *testing.T) {
	type args struct {
		topologySelectorTerms []v1.TopologySelectorTerm
		labels                labels.Set
	}

	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "nil term list",
			args: args{
				topologySelectorTerms: nil,
				labels:                nil,
			},
			want: true,
		},
		{
			name: "nil term",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{},
					},
				},
				labels: nil,
			},
			want: false,
		},
		{
			name: "label matches MatchLabelExpressions terms",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val"},
						}},
					},
				},
				labels: map[string]string{"label_1": "label_1_val"},
			},
			want: true,
		},
		{
			name: "label does not match MatchLabelExpressions terms",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val"},
						}},
					},
				},
				labels: map[string]string{"label_1": "label_1_val-failed"},
			},
			want: false,
		},
		{
			name: "multi-values in one requirement, one matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val1", "label_1_val2"},
						}},
					},
				},
				labels: map[string]string{"label_1": "label_1_val2"},
			},
			want: true,
		},
		{
			name: "multi-terms was set, one matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val"},
						}},
					},
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_2",
							Values: []string{"label_2_val"},
						}},
					},
				},
				labels: map[string]string{
					"label_2": "label_2_val",
				},
			},
			want: true,
		},
		{
			name: "multi-requirement in one term, fully matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
							{
								Key:    "label_1",
								Values: []string{"label_1_val"},
							},
							{
								Key:    "label_2",
								Values: []string{"label_2_val"},
							},
						},
					},
				},
				labels: map[string]string{
					"label_1": "label_1_val",
					"label_2": "label_2_val",
				},
			},
			want: true,
		},
		{
			name: "multi-requirement in one term, partial matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
							{
								Key:    "label_1",
								Values: []string{"label_1_val"},
							},
							{
								Key:    "label_2",
								Values: []string{"label_2_val"},
							},
						},
					},
				},
				labels: map[string]string{
					"label_1": "label_1_val-failed",
					"label_2": "label_2_val",
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchTopologySelectorTerms(tt.args.topologySelectorTerms, tt.args.labels); got != tt.want {
				t.Errorf("MatchTopologySelectorTermsORed() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodeSelectorRequirementKeyExistsInNodeSelectorTerms(t *testing.T) {
	tests := []struct {
		name   string
		reqs   []v1.NodeSelectorRequirement
		terms  []v1.NodeSelectorTerm
		exists bool
	}{
		{
			name:   "empty set of keys in empty set of terms",
			reqs:   []v1.NodeSelectorRequirement{},
			terms:  []v1.NodeSelectorTerm{},
			exists: false,
		},
		{
			name: "key existence in terms with all keys specified",
			reqs: []v1.NodeSelectorRequirement{
				{
					Key:      "key1",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value1"},
				},
				{
					Key:      "key2",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value2"},
				},
			},
			terms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "key2",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value2"},
						},
						{
							Key:      "key3",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value3"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "key1",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value11, test-value12"},
						},
						{
							Key:      "key4",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value41, test-value42"},
						},
					},
				},
			},
			exists: true,
		},
		{
			name: "key existence in terms with one of the keys specfied",
			reqs: []v1.NodeSelectorRequirement{
				{
					Key:      "key1",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value1"},
				},
				{
					Key:      "key2",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value2"},
				},
				{
					Key:      "key3",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value3"},
				},
				{
					Key:      "key6",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value6"},
				},
			},
			terms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "key2",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value2"},
						}, {
							Key:      "key4",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value4"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "key5",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value5"},
						},
					},
				},
			},
			exists: true,
		},
		{
			name: "key existence in terms without any of the keys specified",
			reqs: []v1.NodeSelectorRequirement{
				{
					Key:      "key2",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value2"},
				},
				{
					Key:      "key3",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value3"},
				},
			},
			terms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "key4",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value"},
						},
						{
							Key:      "key5",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "key6",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "key7",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value"},
						},
						{
							Key:      "key8",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"test-value"},
						},
					},
				},
			},
			exists: false,
		},
		{
			name: "key existence in empty set of terms",
			reqs: []v1.NodeSelectorRequirement{
				{
					Key:      "key2",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value2"},
				},
				{
					Key:      "key3",
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"test-value3"},
				},
			},
			terms:  []v1.NodeSelectorTerm{},
			exists: false,
		},
	}
	for _, test := range tests {
		keyExists := NodeSelectorRequirementKeysExistInNodeSelectorTerms(test.reqs, test.terms)
		if test.exists != keyExists {
			t.Errorf("test %s failed. Expected %v but got %v", test.name, test.exists, keyExists)
		}
	}
}
