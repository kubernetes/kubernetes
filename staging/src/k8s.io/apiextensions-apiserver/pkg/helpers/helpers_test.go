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

package helpers

import (
	"reflect"
	"testing"
	"time"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCRDHasFinalizer(t *testing.T) {
	tests := []struct {
		name             string
		crd              *apiextensions.CustomResourceDefinition
		finalizerToCheck string

		expected bool
	}{
		{
			name: "missing",
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it"}},
			},
			finalizerToCheck: "it",
			expected:         false,
		},
		{
			name: "present",
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it", "it"}},
			},
			finalizerToCheck: "it",
			expected:         true,
		},
	}
	for _, tc := range tests {
		actual := CRDHasFinalizer(tc.crd, tc.finalizerToCheck)
		if tc.expected != actual {
			t.Errorf("%v expected %v, got %v", tc.name, tc.expected, actual)
		}
	}
}

func TestCRDRemoveFinalizer(t *testing.T) {
	tests := []struct {
		name             string
		crd              *apiextensions.CustomResourceDefinition
		finalizerToCheck string

		expected []string
	}{
		{
			name: "missing",
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it"}},
			},
			finalizerToCheck: "it",
			expected:         []string{"not-it"},
		},
		{
			name: "present",
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it", "it"}},
			},
			finalizerToCheck: "it",
			expected:         []string{"not-it"},
		},
	}
	for _, tc := range tests {
		CRDRemoveFinalizer(tc.crd, tc.finalizerToCheck)
		if !reflect.DeepEqual(tc.expected, tc.crd.Finalizers) {
			t.Errorf("%v expected %v, got %v", tc.name, tc.expected, tc.crd.Finalizers)
		}
	}
}

func TestSetCRDCondition(t *testing.T) {
	tests := []struct {
		name                 string
		crdCondition         []apiextensions.CustomResourceDefinitionCondition
		newCondition         apiextensions.CustomResourceDefinitionCondition
		expectedcrdCondition []apiextensions.CustomResourceDefinitionCondition
	}{
		{
			name: "test setCRDcondition when one condition",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensions.CustomResourceDefinitionCondition{
				Type:               apiextensions.Established,
				Status:             apiextensions.ConditionFalse,
				Reason:             "NotAccepted",
				Message:            "Not accepted",
				LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionFalse,
					Reason:             "NotAccepted",
					Message:            "Not accepted",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test setCRDcondition when two condition",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensions.CustomResourceDefinitionCondition{
				Type:               apiextensions.NamesAccepted,
				Status:             apiextensions.ConditionFalse,
				Reason:             "Conflicts",
				Message:            "conflicts found",
				LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionFalse,
					Reason:             "Conflicts",
					Message:            "conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test setCRDcondition when condition needs to be appended",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensions.CustomResourceDefinitionCondition{
				Type:               apiextensions.Terminating,
				Status:             apiextensions.ConditionFalse,
				Reason:             "Neverapiextensions.Established",
				Message:            "resource was never apiextensions.Established",
				LastTransitionTime: metav1.Date(2018, 2, 1, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.Terminating,
					Status:             apiextensions.ConditionFalse,
					Reason:             "Neverapiextensions.Established",
					Message:            "resource was never apiextensions.Established",
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
			if !IsCRDConditionEquivalent(&tc.expectedcrdCondition[i], &crd.Status.Conditions[i]) {
				t.Errorf("%v expected %v, got %v", tc.name, tc.expectedcrdCondition, crd.Status.Conditions)
			}
		}
	}
}

func TestRemoveCRDCondition(t *testing.T) {
	tests := []struct {
		name                 string
		crdCondition         []apiextensions.CustomResourceDefinitionCondition
		conditionType        apiextensions.CustomResourceDefinitionConditionType
		expectedcrdCondition []apiextensions.CustomResourceDefinitionCondition
	}{
		{
			name: "test remove CRDCondition when the conditionType meets",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensions.NamesAccepted,
			expectedcrdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2011, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test remove CRDCondition when the conditionType not meets",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensions.Terminating,
			expectedcrdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
		},
	}
	for _, tc := range tests {
		crd := generateCRDwithCondition(tc.crdCondition)
		RemoveCRDCondition(crd, tc.conditionType)
		if len(tc.expectedcrdCondition) != len(crd.Status.Conditions) {
			t.Errorf("%v expected %v, got %v", tc.name, tc.expectedcrdCondition, crd.Status.Conditions)
		}
		for i := range tc.expectedcrdCondition {
			if !IsCRDConditionEquivalent(&tc.expectedcrdCondition[i], &crd.Status.Conditions[i]) {
				t.Errorf("%v expected %v, got %v", tc.name, tc.expectedcrdCondition, crd.Status.Conditions)
			}
		}
	}
}

func TestIsCRDConditionPresentAndEqual(t *testing.T) {
	tests := []struct {
		name          string
		crdCondition  []apiextensions.CustomResourceDefinitionCondition
		conditionType apiextensions.CustomResourceDefinitionConditionType
		status        apiextensions.ConditionStatus
		expectresult  bool
	}{
		{
			name: "test CRDCondition is not Present",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensions.Terminating,
			status:        apiextensions.ConditionTrue,
			expectresult:  false,
		},
		{
			name: "test CRDCondition is Present but not Equal",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensions.Established,
			status:        apiextensions.ConditionFalse,
			expectresult:  false,
		},
		{
			name: "test CRDCondition is Present and Equal",
			crdCondition: []apiextensions.CustomResourceDefinitionCondition{
				{
					Type:               apiextensions.Established,
					Status:             apiextensions.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensions.NamesAccepted,
					Status:             apiextensions.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensions.NamesAccepted,
			status:        apiextensions.ConditionTrue,
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

func generateCRDwithCondition(conditions []apiextensions.CustomResourceDefinitionCondition) *apiextensions.CustomResourceDefinition {
	testCRDObjectMeta := metav1.ObjectMeta{
		Name:            "plural.group.com",
		ResourceVersion: "12",
	}
	testCRDSpec := apiextensions.CustomResourceDefinitionSpec{
		Group:   "group.com",
		Version: "version",
		Scope:   apiextensions.ResourceScope("Cluster"),
		Names: apiextensions.CustomResourceDefinitionNames{
			Plural:   "plural",
			Singular: "singular",
			Kind:     "kind",
			ListKind: "listkind",
		},
	}
	testCRDAcceptedNames := apiextensions.CustomResourceDefinitionNames{
		Plural:   "plural",
		Singular: "singular",
		Kind:     "kind",
		ListKind: "listkind",
	}
	return &apiextensions.CustomResourceDefinition{
		ObjectMeta: testCRDObjectMeta,
		Spec:       testCRDSpec,
		Status: apiextensions.CustomResourceDefinitionStatus{
			AcceptedNames: testCRDAcceptedNames,
			Conditions:    conditions,
		},
	}
}
