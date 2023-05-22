/*
Copyright 2022 The Kubernetes Authors.

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

package node

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
)

func TestWarnings(t *testing.T) {
	testcases := []struct {
		name     string
		template *node.RuntimeClass
		expected []string
	}{
		{
			name:     "null",
			template: nil,
			expected: nil,
		},
		{
			name: "no warning",
			template: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			expected: nil,
		},
		{
			name: "warning",
			template: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Scheduling: &node.Scheduling{
					NodeSelector: map[string]string{
						"beta.kubernetes.io/arch": "amd64",
						"beta.kubernetes.io/os":   "linux",
					},
				},
			},
			expected: []string{
				`scheduling.nodeSelector: deprecated since v1.14; use "kubernetes.io/arch" instead`,
				`scheduling.nodeSelector: deprecated since v1.14; use "kubernetes.io/os" instead`,
			},
		},
	}

	for _, tc := range testcases {
		t.Run("podspec_"+tc.name, func(t *testing.T) {
			actual := sets.NewString(GetWarningsForRuntimeClass(tc.template)...)
			expected := sets.NewString(tc.expected...)
			for _, missing := range expected.Difference(actual).List() {
				t.Errorf("missing: %s", missing)
			}
			for _, extra := range actual.Difference(expected).List() {
				t.Errorf("extra: %s", extra)
			}
		})

	}
}

func TestGetWarningsForNodeSelector(t *testing.T) {
	type args struct {
		nodeSelector *metav1.LabelSelector
		fieldPath    *field.Path
	}
	tests := []struct {
		name string
		args args
		want []string
	}{
		{
			name: "nil nodeSelector",
			args: args{
				nodeSelector: nil,
			},
			want: nil,
		},
		{
			name: "test empty matchExpressions and matchLabels",
			args: args{
				nodeSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{},
					MatchLabels:      map[string]string{},
				},
			},
			want: nil,
		},
		{
			name: "test matchExpressions and matchLabels",
			args: args{
				nodeSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "beta.kubernetes.io/arch",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"amd64"},
						},
					},
					MatchLabels: map[string]string{
						"beta.kubernetes.io/os": "linux",
					},
				},
				fieldPath: field.NewPath("scheduling", "nodeSelector"),
			},
			want: []string{
				`scheduling.nodeSelector.matchExpressions[0].key: beta.kubernetes.io/arch is deprecated since v1.14; use "kubernetes.io/arch" instead`,
				`scheduling.nodeSelector.matchLabels.beta.kubernetes.io/os: deprecated since v1.14; use "kubernetes.io/os" instead`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetWarningsForNodeSelector(tt.args.nodeSelector, tt.args.fieldPath)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetWarningsForNodeSelector() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetWarningsForNodeSelectorTerm(t *testing.T) {
	type args struct {
		nodeSelectorTerm core.NodeSelectorTerm
		fieldPath        *field.Path
	}
	tests := []struct {
		name string
		args args
		want []string
	}{
		{
			name: "test filedPath is nil",
			args: args{
				nodeSelectorTerm: core.NodeSelectorTerm{},
				fieldPath:        nil,
			},
			want: nil,
		},
		{
			name: "test matchExpressions not contains deprecated label",
			args: args{
				nodeSelectorTerm: core.NodeSelectorTerm{
					MatchExpressions: []core.NodeSelectorRequirement{
						{
							Key:      "kubernetes.io/arch",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"amd64"},
						},
					},
				},
				fieldPath: field.NewPath("scheduling", "nodeSelector"),
			},
			want: nil,
		},
		{
			name: "test matchExpressions contains deprecated label",
			args: args{
				nodeSelectorTerm: core.NodeSelectorTerm{
					MatchExpressions: []core.NodeSelectorRequirement{
						{
							Key:      "beta.kubernetes.io/arch",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"amd64"},
						},
					},
				},
				fieldPath: field.NewPath("scheduling", "nodeSelector"),
			},
			want: []string{
				`scheduling.nodeSelector.matchExpressions[0].key: beta.kubernetes.io/arch is deprecated since v1.14; use "kubernetes.io/arch" instead`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetWarningsForNodeSelectorTerm(tt.args.nodeSelectorTerm, tt.args.fieldPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetWarningsForNodeSelectorTerm() = %v, want %v", got, tt.want)
			}
		})
	}
}
