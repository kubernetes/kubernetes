/*
Copyright 2017 The Kubernetes Authors.

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

package apiextensions

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestSetCRDCondition(t *testing.T) {
	tests := []struct {
		name                 string
		crdCondition         []CustomResourceDefinitionCondition
		newCondition         CustomResourceDefinitionCondition
		expectedcrdCondition []CustomResourceDefinitionCondition
	}{
		{
			name: "test setCRDcondition when one condition",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: CustomResourceDefinitionCondition{
				Type:               Established,
				Status:             ConditionFalse,
				Reason:             "NotAccepted",
				Message:            "Not accepted",
				LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionFalse,
					Reason:             "NotAccepted",
					Message:            "Not accepted",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test setCRDcondition when two condition",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               NamesAccepted,
					Status:             ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: CustomResourceDefinitionCondition{
				Type:               NamesAccepted,
				Status:             ConditionFalse,
				Reason:             "Conflicts",
				Message:            "conflicts found",
				LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               NamesAccepted,
					Status:             ConditionFalse,
					Reason:             "Conflicts",
					Message:            "conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test setCRDcondition when condition needs to be appended",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: CustomResourceDefinitionCondition{
				Type:               Terminating,
				Status:             ConditionFalse,
				Reason:             "NeverEstablished",
				Message:            "resource was never established",
				LastTransitionTime: metav1.Date(2018, 2, 1, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               Terminating,
					Status:             ConditionFalse,
					Reason:             "NeverEstablished",
					Message:            "resource was never established",
					LastTransitionTime: metav1.Date(2018, 2, 1, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "set new condition which doesn't have lastTransitionTime set",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: CustomResourceDefinitionCondition{
				Type:    Established,
				Status:  ConditionFalse,
				Reason:  "NotAccepted",
				Message: "Not accepted",
			},
			expectedcrdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionFalse,
					Reason:             "NotAccepted",
					Message:            "Not accepted",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "append new condition which doesn't have lastTransitionTime set",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: CustomResourceDefinitionCondition{
				Type:    Terminating,
				Status:  ConditionFalse,
				Reason:  "NeverEstablished",
				Message: "resource was never established",
			},
			expectedcrdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               Terminating,
					Status:             ConditionFalse,
					Reason:             "NeverEstablished",
					Message:            "resource was never established",
					LastTransitionTime: metav1.Date(2018, 2, 1, 0, 0, 0, 0, time.UTC),
				},
			},
		},
	}
	for _, tc := range tests {
		crd := generateCRDwithCondition(tc.crdCondition)
		SetCRDCondition(crd, tc.newCondition)
		if len(tc.expectedcrdCondition) != len(crd.Status.Conditions) {
			t.Errorf("%v expected %v, got %v", tc.name, tc.expectedcrdCondition, crd.Status.Conditions)
		}
		for i := range tc.expectedcrdCondition {
			if !isCRDConditionEquivalent(&tc.expectedcrdCondition[i], &crd.Status.Conditions[i]) {
				t.Errorf("%v expected %v, got %v", tc.name, tc.expectedcrdCondition[i], crd.Status.Conditions[i])
			}
			if crd.Status.Conditions[i].LastTransitionTime.IsZero() {
				t.Errorf("%q/%d lastTransitionTime should not be null: %v", tc.name, i, crd.Status.Conditions[i])
			}
		}
	}
}

// isCRDConditionEquivalent returns true if the lhs and rhs are equivalent except for times.
func isCRDConditionEquivalent(lhs, rhs *CustomResourceDefinitionCondition) bool {
	if lhs == nil && rhs == nil {
		return true
	}
	if lhs == nil || rhs == nil {
		return false
	}

	return lhs.Message == rhs.Message && lhs.Reason == rhs.Reason && lhs.Status == rhs.Status && lhs.Type == rhs.Type
}

func TestIsCRDConditionPresentAndEqual(t *testing.T) {
	tests := []struct {
		name          string
		crdCondition  []CustomResourceDefinitionCondition
		conditionType CustomResourceDefinitionConditionType
		status        ConditionStatus
		expectresult  bool
	}{
		{
			name: "test CRDCondition is not Present",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               NamesAccepted,
					Status:             ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: Terminating,
			status:        ConditionTrue,
			expectresult:  false,
		},
		{
			name: "test CRDCondition is Present but not Equal",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               NamesAccepted,
					Status:             ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: Established,
			status:        ConditionFalse,
			expectresult:  false,
		},
		{
			name: "test CRDCondition is Present and Equal",
			crdCondition: []CustomResourceDefinitionCondition{
				{
					Type:               Established,
					Status:             ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               NamesAccepted,
					Status:             ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: NamesAccepted,
			status:        ConditionTrue,
			expectresult:  true,
		},
	}
	for _, tc := range tests {
		crd := generateCRDwithCondition(tc.crdCondition)
		res := IsCRDConditionPresentAndEqual(crd, tc.conditionType, tc.status)
		if res != tc.expectresult {
			t.Errorf("%v expected %t, got %t", tc.name, tc.expectresult, res)
		}
	}
}

func generateCRDwithCondition(conditions []CustomResourceDefinitionCondition) *CustomResourceDefinition {
	testCRDObjectMeta := metav1.ObjectMeta{
		Name:            "plural.group.com",
		ResourceVersion: "12",
	}
	testCRDSpec := CustomResourceDefinitionSpec{
		Group:   "group.com",
		Version: "version",
		Scope:   ResourceScope("Cluster"),
		Names: CustomResourceDefinitionNames{
			Plural:   "plural",
			Singular: "singular",
			Kind:     "kind",
			ListKind: "listkind",
		},
	}
	testCRDAcceptedNames := CustomResourceDefinitionNames{
		Plural:   "plural",
		Singular: "singular",
		Kind:     "kind",
		ListKind: "listkind",
	}
	return &CustomResourceDefinition{
		ObjectMeta: testCRDObjectMeta,
		Spec:       testCRDSpec,
		Status: CustomResourceDefinitionStatus{
			AcceptedNames: testCRDAcceptedNames,
			Conditions:    conditions,
		},
	}
}
