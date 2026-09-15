/*
Copyright 2026 The Kubernetes Authors.

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

package poddisruptionbudget

import (
	"testing"

	policy "k8s.io/api/policy/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestUpdateDisruptionAllowedCondition(t *testing.T) {
	tests := []struct {
		name               string
		disruptionsAllowed int32
		wantStatus         metav1.ConditionStatus
		wantReason         string
	}{
		{
			name:               "disruptions allowed",
			disruptionsAllowed: 1,
			wantStatus:         metav1.ConditionTrue,
			wantReason:         policy.SufficientPodsReason,
		},
		{
			name:               "no disruptions allowed",
			disruptionsAllowed: 0,
			wantStatus:         metav1.ConditionFalse,
			wantReason:         policy.InsufficientPodsReason,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pdb := &policy.PodDisruptionBudget{
				Status: policy.PodDisruptionBudgetStatus{
					DisruptionsAllowed: tt.disruptionsAllowed,
					ObservedGeneration: 3,
				},
			}

			UpdateDisruptionAllowedCondition(pdb)

			cond := apimeta.FindStatusCondition(pdb.Status.Conditions, policy.DisruptionAllowedCondition)
			if cond == nil {
				t.Fatal("expected DisruptionAllowed condition to be set")
			}
			if cond.Status != tt.wantStatus {
				t.Errorf("Status = %v, want %v", cond.Status, tt.wantStatus)
			}
			if cond.Reason != tt.wantReason {
				t.Errorf("Reason = %v, want %v", cond.Reason, tt.wantReason)
			}
			if cond.ObservedGeneration != 3 {
				t.Errorf("ObservedGeneration = %d, want 3", cond.ObservedGeneration)
			}
		})
	}
}

func TestUpdateDisruptionAllowedConditionOverwritesExisting(t *testing.T) {
	pdb := &policy.PodDisruptionBudget{
		Status: policy.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 0,
			Conditions: []metav1.Condition{
				{
					Type:   policy.DisruptionAllowedCondition,
					Status: metav1.ConditionTrue,
					Reason: policy.SufficientPodsReason,
				},
			},
		},
	}

	UpdateDisruptionAllowedCondition(pdb)

	cond := apimeta.FindStatusCondition(pdb.Status.Conditions, policy.DisruptionAllowedCondition)
	if cond == nil {
		t.Fatal("expected DisruptionAllowed condition to remain set")
	}
	if cond.Status != metav1.ConditionFalse {
		t.Errorf("Status = %v, want %v", cond.Status, metav1.ConditionFalse)
	}
	if cond.Reason != policy.InsufficientPodsReason {
		t.Errorf("Reason = %v, want %v", cond.Reason, policy.InsufficientPodsReason)
	}
	if len(pdb.Status.Conditions) != 1 {
		t.Errorf("expected condition to be updated in place, got %d conditions", len(pdb.Status.Conditions))
	}
}

func TestConditionsAreUpToDate(t *testing.T) {
	tests := []struct {
		name string
		pdb  *policy.PodDisruptionBudget
		want bool
	}{
		{
			name: "no conditions set",
			pdb:  &policy.PodDisruptionBudget{},
			want: false,
		},
		{
			name: "observed generation stale",
			pdb: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Generation: 2},
				Status: policy.PodDisruptionBudgetStatus{
					ObservedGeneration: 1,
					Conditions: []metav1.Condition{
						{
							Type:   policy.DisruptionAllowedCondition,
							Status: metav1.ConditionFalse,
							Reason: policy.InsufficientPodsReason,
						},
					},
				},
			},
			want: false,
		},
		{
			name: "up to date with disruptions allowed",
			pdb: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Generation: 2},
				Status: policy.PodDisruptionBudgetStatus{
					ObservedGeneration: 2,
					DisruptionsAllowed: 1,
					Conditions: []metav1.Condition{
						{
							Type:   policy.DisruptionAllowedCondition,
							Status: metav1.ConditionTrue,
							Reason: policy.SufficientPodsReason,
						},
					},
				},
			},
			want: true,
		},
		{
			name: "up to date with no disruptions allowed",
			pdb: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Generation: 2},
				Status: policy.PodDisruptionBudgetStatus{
					ObservedGeneration: 2,
					DisruptionsAllowed: 0,
					Conditions: []metav1.Condition{
						{
							Type:   policy.DisruptionAllowedCondition,
							Status: metav1.ConditionFalse,
							Reason: policy.InsufficientPodsReason,
						},
					},
				},
			},
			want: true,
		},
		{
			name: "stale reason for allowed disruptions",
			pdb: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Generation: 2},
				Status: policy.PodDisruptionBudgetStatus{
					ObservedGeneration: 2,
					DisruptionsAllowed: 1,
					Conditions: []metav1.Condition{
						{
							Type:   policy.DisruptionAllowedCondition,
							Status: metav1.ConditionTrue,
							Reason: policy.InsufficientPodsReason,
						},
					},
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ConditionsAreUpToDate(tt.pdb); got != tt.want {
				t.Errorf("ConditionsAreUpToDate() = %v, want %v", got, tt.want)
			}
		})
	}
}
