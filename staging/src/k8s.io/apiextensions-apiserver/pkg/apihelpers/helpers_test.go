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

package apihelpers

import (
	"reflect"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsProtectedCommunityGroup(t *testing.T) {
	tests := []struct {
		name string

		group    string
		expected bool
	}{
		{
			name:     "bare k8s",
			group:    "k8s.io",
			expected: true,
		},
		{
			name:     "bare kube",
			group:    "kubernetes.io",
			expected: true,
		},
		{
			name:     "nested k8s",
			group:    "sigs.k8s.io",
			expected: true,
		},
		{
			name:     "nested kube",
			group:    "sigs.kubernetes.io",
			expected: true,
		},
		{
			name:     "alternative",
			group:    "different.io",
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := IsProtectedCommunityGroup(test.group)

			if actual != test.expected {
				t.Fatalf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}

func TestGetAPIApprovalState(t *testing.T) {
	tests := []struct {
		name string

		annotations map[string]string
		expected    APIApprovalState
	}{
		{
			name:        "bare unapproved",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "unapproved"},
			expected:    APIApprovalBypassed,
		},
		{
			name:        "unapproved with message",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "unapproved, experimental-only"},
			expected:    APIApprovalBypassed,
		},
		{
			name:        "mismatched case",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "Unapproved"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "empty",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: ""},
			expected:    APIApprovalMissing,
		},
		{
			name:        "missing",
			annotations: map[string]string{},
			expected:    APIApprovalMissing,
		},
		{
			name:        "url",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "https://github.com/kubernetes/kubernetes/pull/78458"},
			expected:    APIApproved,
		},
		{
			name:        "url - no scheme",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "github.com/kubernetes/kubernetes/pull/78458"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "url - no host",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "http:///kubernetes/kubernetes/pull/78458"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "url - just path",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "/"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "missing scheme",
			annotations: map[string]string{apiextensionsv1.KubeAPIApprovedAnnotation: "github.com/kubernetes/kubernetes/pull/78458"},
			expected:    APIApprovalInvalid,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual, _ := GetAPIApprovalState(test.annotations)

			if actual != test.expected {
				t.Fatalf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}

func TestCRDHasFinalizer(t *testing.T) {
	tests := []struct {
		name             string
		crd              *apiextensionsv1.CustomResourceDefinition
		finalizerToCheck string

		expected bool
	}{
		{
			name: "missing",
			crd: &apiextensionsv1.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it"}},
			},
			finalizerToCheck: "it",
			expected:         false,
		},
		{
			name: "present",
			crd: &apiextensionsv1.CustomResourceDefinition{
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
		crd              *apiextensionsv1.CustomResourceDefinition
		finalizerToCheck string

		expected []string
	}{
		{
			name: "missing",
			crd: &apiextensionsv1.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it"}},
			},
			finalizerToCheck: "it",
			expected:         []string{"not-it"},
		},
		{
			name: "present",
			crd: &apiextensionsv1.CustomResourceDefinition{
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
		crdCondition         []apiextensionsv1.CustomResourceDefinitionCondition
		newCondition         apiextensionsv1.CustomResourceDefinitionCondition
		expectedcrdCondition []apiextensionsv1.CustomResourceDefinitionCondition
	}{
		{
			name: "test setCRDcondition when one condition",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensionsv1.CustomResourceDefinitionCondition{
				Type:               apiextensionsv1.Established,
				Status:             apiextensionsv1.ConditionFalse,
				Reason:             "NotAccepted",
				Message:            "Not accepted",
				LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionFalse,
					Reason:             "NotAccepted",
					Message:            "Not accepted",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test setCRDcondition when two condition",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensionsv1.CustomResourceDefinitionCondition{
				Type:               apiextensionsv1.NamesAccepted,
				Status:             apiextensionsv1.ConditionFalse,
				Reason:             "Conflicts",
				Message:            "conflicts found",
				LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionFalse,
					Reason:             "Conflicts",
					Message:            "conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test setCRDcondition when condition needs to be appended",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensionsv1.CustomResourceDefinitionCondition{
				Type:               apiextensionsv1.Terminating,
				Status:             apiextensionsv1.ConditionFalse,
				Reason:             "Neverapiextensionsv1.Established",
				Message:            "resource was never apiextensionsv1.Established",
				LastTransitionTime: metav1.Date(2018, 2, 1, 0, 0, 0, 0, time.UTC),
			},
			expectedcrdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.Terminating,
					Status:             apiextensionsv1.ConditionFalse,
					Reason:             "Neverapiextensionsv1.Established",
					Message:            "resource was never apiextensionsv1.Established",
					LastTransitionTime: metav1.Date(2018, 2, 1, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "set new condition which doesn't have lastTransitionTime set",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensionsv1.CustomResourceDefinitionCondition{
				Type:    apiextensionsv1.Established,
				Status:  apiextensionsv1.ConditionFalse,
				Reason:  "NotAccepted",
				Message: "Not accepted",
			},
			expectedcrdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionFalse,
					Reason:             "NotAccepted",
					Message:            "Not accepted",
					LastTransitionTime: metav1.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "append new condition which doesn't have lastTransitionTime set",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			newCondition: apiextensionsv1.CustomResourceDefinitionCondition{
				Type:    apiextensionsv1.Terminating,
				Status:  apiextensionsv1.ConditionFalse,
				Reason:  "Neverapiextensionsv1.Established",
				Message: "resource was never apiextensionsv1.Established",
			},
			expectedcrdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.Terminating,
					Status:             apiextensionsv1.ConditionFalse,
					Reason:             "Neverapiextensionsv1.Established",
					Message:            "resource was never apiextensionsv1.Established",
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
				t.Errorf("%v expected %v, got %v", tc.name, tc.expectedcrdCondition[i], crd.Status.Conditions[i])
			}
			if crd.Status.Conditions[i].LastTransitionTime.IsZero() {
				t.Errorf("%q/%d lastTransitionTime should not be null: %v", tc.name, i, crd.Status.Conditions[i])
			}
		}
	}
}

func TestRemoveCRDCondition(t *testing.T) {
	tests := []struct {
		name                 string
		crdCondition         []apiextensionsv1.CustomResourceDefinitionCondition
		conditionType        apiextensionsv1.CustomResourceDefinitionConditionType
		expectedcrdCondition []apiextensionsv1.CustomResourceDefinitionCondition
	}{
		{
			name: "test remove CRDCondition when the conditionType meets",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensionsv1.NamesAccepted,
			expectedcrdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2011, 1, 2, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "test remove CRDCondition when the conditionType not meets",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensionsv1.Terminating,
			expectedcrdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionTrue,
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
		crdCondition  []apiextensionsv1.CustomResourceDefinitionCondition
		conditionType apiextensionsv1.CustomResourceDefinitionConditionType
		status        apiextensionsv1.ConditionStatus
		expectresult  bool
	}{
		{
			name: "test CRDCondition is not Present",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensionsv1.Terminating,
			status:        apiextensionsv1.ConditionTrue,
			expectresult:  false,
		},
		{
			name: "test CRDCondition is Present but not Equal",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensionsv1.Established,
			status:        apiextensionsv1.ConditionFalse,
			expectresult:  false,
		},
		{
			name: "test CRDCondition is Present and Equal",
			crdCondition: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:               apiextensionsv1.Established,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "Accepted",
					Message:            "the initial names have been accepted",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				{
					Type:               apiextensionsv1.NamesAccepted,
					Status:             apiextensionsv1.ConditionTrue,
					Reason:             "NoConflicts",
					Message:            "no conflicts found",
					LastTransitionTime: metav1.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			conditionType: apiextensionsv1.NamesAccepted,
			status:        apiextensionsv1.ConditionTrue,
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

func generateCRDwithCondition(conditions []apiextensionsv1.CustomResourceDefinitionCondition) *apiextensionsv1.CustomResourceDefinition {
	testCRDObjectMeta := metav1.ObjectMeta{
		Name:            "plural.group.com",
		ResourceVersion: "12",
	}
	testCRDSpec := apiextensionsv1.CustomResourceDefinitionSpec{
		Group: "group.com",
		Names: apiextensionsv1.CustomResourceDefinitionNames{
			Plural:   "plural",
			Singular: "singular",
			Kind:     "kind",
			ListKind: "listkind",
		},
	}
	testCRDAcceptedNames := apiextensionsv1.CustomResourceDefinitionNames{
		Plural:   "plural",
		Singular: "singular",
		Kind:     "kind",
		ListKind: "listkind",
	}
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: testCRDObjectMeta,
		Spec:       testCRDSpec,
		Status: apiextensionsv1.CustomResourceDefinitionStatus{
			AcceptedNames: testCRDAcceptedNames,
			Conditions:    conditions,
		},
	}
}
