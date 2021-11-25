/*
Copyright 2021 The Kubernetes Authors.

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

package v1beta1

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/policy"
)

func TestConversion(t *testing.T) {
	testcases := []struct {
		Name      string
		In        runtime.Object
		Out       runtime.Object
		ExpectOut runtime.Object
		ExpectErr string
	}{
		{
			Name: "v1beta1 to internal with empty selector",
			In: &v1beta1.PodDisruptionBudget{
				Spec: v1beta1.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{},
				},
			},
			Out: &policy.PodDisruptionBudget{},
			ExpectOut: &policy.PodDisruptionBudget{
				Spec: policy.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "pdb.kubernetes.io/deprecated-v1beta1-empty-selector-match",
								Operator: metav1.LabelSelectorOpExists,
							},
						},
					},
				},
			},
		},
		{
			Name:      "v1beta1 to internal with nil selector",
			In:        &v1beta1.PodDisruptionBudget{},
			Out:       &policy.PodDisruptionBudget{},
			ExpectOut: &policy.PodDisruptionBudget{},
		},
		{
			Name: "v1 to internal with existing selector",
			In: &v1beta1.PodDisruptionBudget{
				Spec: v1beta1.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
			Out: &policy.PodDisruptionBudget{},
			ExpectOut: &policy.PodDisruptionBudget{
				Spec: policy.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
		},
		{
			Name: "v1beta1 to internal with existing pdb selector",
			In: &v1beta1.PodDisruptionBudget{
				Spec: v1beta1.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"foo": "bar",
						},
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "pdb.kubernetes.io/deprecated-v1beta1-empty-selector-match",
								Operator: metav1.LabelSelectorOpDoesNotExist,
							},
						},
					},
				},
			},
			Out: &policy.PodDisruptionBudget{},
			ExpectOut: &policy.PodDisruptionBudget{
				Spec: policy.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"foo": "bar",
						},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
		},
		{
			Name: "internal to v1beta1 with empty selector",
			In: &policy.PodDisruptionBudget{
				Spec: policy.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{},
				},
			},
			Out: &v1beta1.PodDisruptionBudget{},
			ExpectOut: &v1beta1.PodDisruptionBudget{
				Spec: v1beta1.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "pdb.kubernetes.io/deprecated-v1beta1-empty-selector-match",
								Operator: metav1.LabelSelectorOpDoesNotExist,
							},
						},
					},
				},
			},
		},
		{
			Name:      "internal to v1beta1 with nil selector",
			In:        &policy.PodDisruptionBudget{},
			Out:       &v1beta1.PodDisruptionBudget{},
			ExpectOut: &v1beta1.PodDisruptionBudget{},
		},
		{
			Name: "internal to v1beta1 with existing selector",
			In: &policy.PodDisruptionBudget{
				Spec: policy.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
			Out: &v1beta1.PodDisruptionBudget{},
			ExpectOut: &v1beta1.PodDisruptionBudget{
				Spec: v1beta1.PodDisruptionBudgetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	if err := policy.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			err := scheme.Convert(tc.In, tc.Out, nil)
			if err != nil {
				if len(tc.ExpectErr) == 0 {
					t.Fatalf("unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tc.ExpectErr) {
					t.Fatalf("expected error %s, got %v", tc.ExpectErr, err)
				}
				return
			}
			if len(tc.ExpectErr) > 0 {
				t.Fatalf("expected error %s, got none", tc.ExpectErr)
			}
			if !reflect.DeepEqual(tc.Out, tc.ExpectOut) {
				t.Fatalf("unexpected result:\n %s", cmp.Diff(tc.ExpectOut, tc.Out))
			}
		})
	}
}
