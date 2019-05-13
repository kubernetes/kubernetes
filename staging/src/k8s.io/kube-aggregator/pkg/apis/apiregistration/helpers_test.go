/*
Copyright 2018 The Kubernetes Authors.

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

package apiregistration

import (
	"reflect"
	"testing"
)

var (
	a APIServiceConditionType = "A"
	b APIServiceConditionType = "B"
	c APIServiceConditionType = "C"
)

func TestGetAPIServiceConditionByType(t *testing.T) {
	conditionA := makeNewAPIServiceCondition(a, "a reason", "a message", ConditionTrue)
	conditionB := makeNewAPIServiceCondition(b, "b reason", "b message", ConditionTrue)
	tests := []*struct {
		name              string
		apiService        *APIService
		conditionType     APIServiceConditionType
		expectedCondition *APIServiceCondition
	}{
		{
			name:              "Should find a matching condition from apiService",
			apiService:        makeNewAPIService("v1", 100, conditionA, conditionB),
			conditionType:     a,
			expectedCondition: &conditionA,
		},
		{
			name:              "Should not find a matching condition",
			apiService:        makeNewAPIService("v1", 100, conditionA),
			conditionType:     b,
			expectedCondition: nil,
		},
	}

	for _, tc := range tests {
		actual := GetAPIServiceConditionByType(tc.apiService, tc.conditionType)
		if !reflect.DeepEqual(tc.expectedCondition, actual) {
			t.Errorf("expected %s, actual %s", tc.expectedCondition, actual)
		}
	}
}

func TestIsAPIServiceConditionTrue(t *testing.T) {
	conditionATrue := makeNewAPIServiceCondition(a, "a reason", "a message", ConditionTrue)
	conditionAFalse := makeNewAPIServiceCondition(a, "a reason", "a message", ConditionFalse)
	tests := []*struct {
		name          string
		apiService    *APIService
		conditionType APIServiceConditionType
		expected      bool
	}{
		{
			name:          "Should return false when condition of type is not present",
			apiService:    makeNewAPIService("v1", 100),
			conditionType: a,
			expected:      false,
		},
		{
			name:          "Should return false when condition of type is present but status is not ConditionTrue",
			apiService:    makeNewAPIService("v1", 100, conditionAFalse),
			conditionType: a,
			expected:      false,
		},
		{
			name:          "Should return false when condition of type is present but status is not ConditionTrue",
			apiService:    makeNewAPIService("v1", 100, conditionATrue),
			conditionType: a,
			expected:      true,
		},
	}

	for _, tc := range tests {
		if isConditionTrue := IsAPIServiceConditionTrue(tc.apiService, tc.conditionType); isConditionTrue != tc.expected {
			t.Errorf("expected condition of type %v to be %v, actually was %v",
				tc.conditionType, isConditionTrue, tc.expected)

		}
	}
}

func TestSetAPIServiceCondition(t *testing.T) {
	conditionA1 := makeNewAPIServiceCondition(a, "a1 reason", "a1 message", ConditionTrue)
	conditionA2 := makeNewAPIServiceCondition(a, "a2 reason", "a2 message", ConditionTrue)
	tests := []*struct {
		name              string
		apiService        *APIService
		conditionType     APIServiceConditionType
		initialCondition  *APIServiceCondition
		setCondition      APIServiceCondition
		expectedCondition *APIServiceCondition
	}{
		{
			name:              "Should set a new condition with type where previously there was no condition of that type",
			apiService:        makeNewAPIService("v1", 100),
			conditionType:     a,
			initialCondition:  nil,
			setCondition:      conditionA1,
			expectedCondition: &conditionA1,
		},
		{
			name:              "Should override a condition of type, when a condition of that type existed previously",
			apiService:        makeNewAPIService("v1", 100, conditionA1),
			conditionType:     a,
			initialCondition:  &conditionA1,
			setCondition:      conditionA2,
			expectedCondition: &conditionA2,
		},
	}

	for _, tc := range tests {
		startingCondition := GetAPIServiceConditionByType(tc.apiService, tc.conditionType)
		if !reflect.DeepEqual(startingCondition, tc.initialCondition) {
			t.Errorf("expected to find condition %s initially, actual was %s", tc.initialCondition, startingCondition)

		}
		SetAPIServiceCondition(tc.apiService, tc.setCondition)
		actual := GetAPIServiceConditionByType(tc.apiService, tc.setCondition.Type)
		if !reflect.DeepEqual(actual, tc.expectedCondition) {
			t.Errorf("expected %s, actual %s", tc.expectedCondition, actual)
		}
	}
}

func TestSortedAPIServicesByVersion(t *testing.T) {
	tests := []*struct {
		name     string
		versions []string
		expected []string
	}{
		{
			name:     "case1",
			versions: []string{"v1", "v2"},
			expected: []string{"v2", "v1"},
		},
		{
			name:     "case2",
			versions: []string{"v2", "v10"},
			expected: []string{"v10", "v2"},
		},
		{
			name:     "case3",
			versions: []string{"v2", "v2beta1", "v10beta2", "v10beta1", "v10alpha1", "v1"},
			expected: []string{"v2", "v1", "v10beta2", "v10beta1", "v2beta1", "v10alpha1"},
		},
		{
			name:     "case4",
			versions: []string{"v1", "v2", "test", "foo10", "final", "foo2", "foo1"},
			expected: []string{"v2", "v1", "final", "foo1", "foo10", "foo2", "test"},
		},
		{
			name:     "case5_from_documentation",
			versions: []string{"v12alpha1", "v10", "v11beta2", "v10beta3", "v3beta1", "v2", "v11alpha2", "foo1", "v1", "foo10"},
			expected: []string{"v10", "v2", "v1", "v11beta2", "v10beta3", "v3beta1", "v12alpha1", "v11alpha2", "foo1", "foo10"},
		},
	}

	for _, tc := range tests {
		apiServices := make([]*APIService, 0)
		for _, v := range tc.versions {
			apiServices = append(apiServices, makeNewAPIService(v, 100))
		}
		sortedServices := SortedByGroupAndVersion(apiServices)
		actual := make([]string, 0)
		for _, s := range sortedServices[0] {
			actual = append(actual, s.Spec.Version)
		}
		if !reflect.DeepEqual(tc.expected, actual) {
			t.Errorf("expected %s, actual %s", tc.expected, actual)
		}
	}
}

func makeNewAPIService(version string, priority int32, conditions ...APIServiceCondition) *APIService {
	status := APIServiceStatus{Conditions: conditions}
	return &APIService{Spec: APIServiceSpec{Version: version, VersionPriority: priority}, Status: status}
}

func makeNewAPIServiceCondition(conditionType APIServiceConditionType, reason string, message string, status ConditionStatus) APIServiceCondition {
	return APIServiceCondition{Type: conditionType, Reason: reason, Message: message, Status: status}
}
