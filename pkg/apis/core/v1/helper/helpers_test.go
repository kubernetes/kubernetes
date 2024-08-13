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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
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
		tc := tc
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
		i := i
		tc := tc
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

func TestHugePageSizeFromMedium(t *testing.T) {
	testCases := []struct {
		description string
		medium      v1.StorageMedium
		expectVal   resource.Quantity
		expectErr   bool
	}{
		{
			description: "Invalid hugepages medium",
			medium:      "Memory",
			expectVal:   resource.Quantity{},
			expectErr:   true,
		},
		{
			description: "Invalid hugepages medium",
			medium:      "Memory",
			expectVal:   resource.Quantity{},
			expectErr:   true,
		},
		{
			description: "Invalid: HugePages without size",
			medium:      "HugePages",
			expectVal:   resource.Quantity{},
			expectErr:   true,
		},
		{
			description: "Invalid: HugePages without size",
			medium:      "HugePages",
			expectVal:   resource.Quantity{},
			expectErr:   true,
		},
		{
			description: "Valid: HugePages-1Gi",
			medium:      "HugePages-1Gi",
			expectVal:   resource.MustParse("1Gi"),
			expectErr:   false,
		},
		{
			description: "Valid: HugePages-2Mi",
			medium:      "HugePages-2Mi",
			expectVal:   resource.MustParse("2Mi"),
			expectErr:   false,
		},
		{
			description: "Valid: HugePages-64Ki",
			medium:      "HugePages-64Ki",
			expectVal:   resource.MustParse("64Ki"),
			expectErr:   false,
		},
	}
	for i, tc := range testCases {
		i := i
		tc := tc
		t.Run(tc.description, func(t *testing.T) {
			t.Parallel()
			v, err := HugePageSizeFromMedium(tc.medium)
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
		tc := tc
		t.Run(fmt.Sprintf("resourceName input=%s, expected value=%v", tc.resourceName, tc.expectVal), func(t *testing.T) {
			t.Parallel()
			v := IsOvercommitAllowed(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestGetAccessModesFromString(t *testing.T) {
	modes := GetAccessModesFromString("ROX")
	if !ContainsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadOnlyMany, modes)
	}

	modes = GetAccessModesFromString("ROX,RWX")
	if !ContainsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadOnlyMany, modes)
	}
	if !ContainsAccessMode(modes, v1.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteMany, modes)
	}

	modes = GetAccessModesFromString("RWO,ROX,RWX")
	if !ContainsAccessMode(modes, v1.ReadWriteOnce) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteOnce, modes)
	}
	if !ContainsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadOnlyMany, modes)
	}
	if !ContainsAccessMode(modes, v1.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteMany, modes)
	}

	modes = GetAccessModesFromString("RWO,ROX,RWX,RWOP")
	if !ContainsAccessMode(modes, v1.ReadWriteOnce) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteOnce, modes)
	}
	if !ContainsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadOnlyMany, modes)
	}
	if !ContainsAccessMode(modes, v1.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteMany, modes)
	}
	if !ContainsAccessMode(modes, v1.ReadWriteOncePod) {
		t.Errorf("Expected mode %s, but got %+v", v1.ReadWriteOncePod, modes)
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
			name: "key existence in terms with one of the keys specified",
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

func TestHugePageUnitSizeFromByteSize(t *testing.T) {
	tests := []struct {
		size     int64
		expected string
		wantErr  bool
	}{
		{
			size:     1024,
			expected: "1KB",
			wantErr:  false,
		},
		{
			size:     33554432,
			expected: "32MB",
			wantErr:  false,
		},
		{
			size:     3221225472,
			expected: "3GB",
			wantErr:  false,
		},
		{
			size:     1024 * 1024 * 1023 * 3,
			expected: "3069MB",
			wantErr:  true,
		},
	}
	for _, test := range tests {
		size := test.size
		result, err := HugePageUnitSizeFromByteSize(size)
		if err != nil {
			if test.wantErr {
				t.Logf("HugePageUnitSizeFromByteSize() expected error = %v", err)
			} else {
				t.Errorf("HugePageUnitSizeFromByteSize() error = %v, wantErr %v", err, test.wantErr)
			}
			continue
		}
		if test.expected != result {
			t.Errorf("HugePageUnitSizeFromByteSize() expected %v but got %v", test.expected, result)
		}
	}
}
