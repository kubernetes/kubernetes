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

package deletion

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestUpdateConditions(t *testing.T) {
	tests := []struct {
		name string

		newConditions  []v1.NamespaceCondition
		startingStatus *v1.NamespaceStatus

		expecteds []v1.NamespaceCondition
	}{
		{
			name: "leave unknown",

			newConditions: []v1.NamespaceCondition{},
			startingStatus: &v1.NamespaceStatus{
				Conditions: []v1.NamespaceCondition{
					{Type: "unknown", Status: v1.ConditionTrue},
				},
			},
			expecteds: []v1.NamespaceCondition{
				{Type: "unknown", Status: v1.ConditionTrue},
				*newSuccessfulCondition(v1.NamespaceDeletionDiscoveryFailure),
				*newSuccessfulCondition(v1.NamespaceDeletionGVParsingFailure),
				*newSuccessfulCondition(v1.NamespaceDeletionContentFailure),
				*newSuccessfulCondition(v1.NamespaceContentRemaining),
				*newSuccessfulCondition(v1.NamespaceFinalizersRemaining),
			},
		},
		{
			name: "replace with success",

			newConditions: []v1.NamespaceCondition{},
			startingStatus: &v1.NamespaceStatus{
				Conditions: []v1.NamespaceCondition{
					{Type: v1.NamespaceDeletionDiscoveryFailure, Status: v1.ConditionTrue},
				},
			},
			expecteds: []v1.NamespaceCondition{
				*newSuccessfulCondition(v1.NamespaceDeletionDiscoveryFailure),
				*newSuccessfulCondition(v1.NamespaceDeletionGVParsingFailure),
				*newSuccessfulCondition(v1.NamespaceDeletionContentFailure),
				*newSuccessfulCondition(v1.NamespaceContentRemaining),
				*newSuccessfulCondition(v1.NamespaceFinalizersRemaining),
			},
		},
		{
			name: "leave different order",

			newConditions: []v1.NamespaceCondition{},
			startingStatus: &v1.NamespaceStatus{
				Conditions: []v1.NamespaceCondition{
					{Type: v1.NamespaceDeletionGVParsingFailure, Status: v1.ConditionTrue},
					{Type: v1.NamespaceDeletionDiscoveryFailure, Status: v1.ConditionTrue},
				},
			},
			expecteds: []v1.NamespaceCondition{
				*newSuccessfulCondition(v1.NamespaceDeletionGVParsingFailure),
				*newSuccessfulCondition(v1.NamespaceDeletionDiscoveryFailure),
				*newSuccessfulCondition(v1.NamespaceDeletionContentFailure),
				*newSuccessfulCondition(v1.NamespaceContentRemaining),
				*newSuccessfulCondition(v1.NamespaceFinalizersRemaining),
			},
		},
		{
			name: "overwrite with failure",

			newConditions: []v1.NamespaceCondition{
				{Type: v1.NamespaceDeletionGVParsingFailure, Status: v1.ConditionTrue, Reason: "foo", Message: "bar"},
			},
			startingStatus: &v1.NamespaceStatus{
				Conditions: []v1.NamespaceCondition{
					{Type: v1.NamespaceDeletionGVParsingFailure, Status: v1.ConditionFalse},
					{Type: v1.NamespaceDeletionDiscoveryFailure, Status: v1.ConditionTrue},
				},
			},
			expecteds: []v1.NamespaceCondition{
				{Type: v1.NamespaceDeletionGVParsingFailure, Status: v1.ConditionTrue, Reason: "foo", Message: "bar"},
				*newSuccessfulCondition(v1.NamespaceDeletionDiscoveryFailure),
				*newSuccessfulCondition(v1.NamespaceDeletionContentFailure),
				*newSuccessfulCondition(v1.NamespaceContentRemaining),
				*newSuccessfulCondition(v1.NamespaceFinalizersRemaining),
			},
		},
		{
			name: "write new failure",

			newConditions: []v1.NamespaceCondition{
				{Type: v1.NamespaceDeletionGVParsingFailure, Status: v1.ConditionTrue, Reason: "foo", Message: "bar"},
			},
			startingStatus: &v1.NamespaceStatus{
				Conditions: []v1.NamespaceCondition{
					{Type: v1.NamespaceDeletionDiscoveryFailure, Status: v1.ConditionTrue},
				},
			},
			expecteds: []v1.NamespaceCondition{
				*newSuccessfulCondition(v1.NamespaceDeletionDiscoveryFailure),
				{Type: v1.NamespaceDeletionGVParsingFailure, Status: v1.ConditionTrue, Reason: "foo", Message: "bar"},
				*newSuccessfulCondition(v1.NamespaceDeletionContentFailure),
				*newSuccessfulCondition(v1.NamespaceContentRemaining),
				*newSuccessfulCondition(v1.NamespaceFinalizersRemaining),
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			updateConditions(test.startingStatus, test.newConditions)

			actuals := test.startingStatus.Conditions
			if len(actuals) != len(test.expecteds) {
				t.Fatal(actuals)
			}
			for i := range actuals {
				actual := actuals[i]
				expected := test.expecteds[i]
				expected.LastTransitionTime = actual.LastTransitionTime
				if !reflect.DeepEqual(expected, actual) {
					t.Error(actual)
				}
			}
		})
	}
}

func TestProcessContentTotals(t *testing.T) {
	tests := []struct {
		name string

		contentTotals allGVRDeletionMetadata
		expecteds     []v1.NamespaceCondition
	}{
		{
			name: "nothing",

			contentTotals: allGVRDeletionMetadata{
				gvrToNumRemaining:        map[schema.GroupVersionResource]int{},
				finalizersToNumRemaining: map[string]int{},
			},
			expecteds: []v1.NamespaceCondition{},
		},
		{
			name: "just remaining",

			contentTotals: allGVRDeletionMetadata{
				gvrToNumRemaining: map[schema.GroupVersionResource]int{
					{Group: "apps.k8s.io", Resource: "daemonsets"}:  5,
					{Group: "apps.k8s.io", Resource: "deployments"}: 5,
				},
				finalizersToNumRemaining: map[string]int{},
			},
			expecteds: []v1.NamespaceCondition{
				{Type: v1.NamespaceContentRemaining, Status: v1.ConditionTrue, Reason: "SomeResourcesRemain", Message: `Some resources are remaining: daemonsets.apps.k8s.io has 5 resource instances, deployments.apps.k8s.io has 5 resource instances`},
			},
		},
		{
			name: "just finalizers", // this shouldn't happen

			contentTotals: allGVRDeletionMetadata{
				gvrToNumRemaining: map[schema.GroupVersionResource]int{},
				finalizersToNumRemaining: map[string]int{
					"service-catalog": 6,
					"kubedb":          5,
				},
			},
			expecteds: []v1.NamespaceCondition{
				{Type: v1.NamespaceFinalizersRemaining, Status: v1.ConditionTrue, Reason: "SomeFinalizersRemain", Message: `Some content in the namespace has finalizers remaining: kubedb in 5 resource instances, service-catalog in 6 resource instances`},
			},
		},
		{
			name: "both",

			contentTotals: allGVRDeletionMetadata{
				gvrToNumRemaining: map[schema.GroupVersionResource]int{
					{Group: "apps.k8s.io", Resource: "daemonsets"}:  5,
					{Group: "apps.k8s.io", Resource: "deployments"}: 5,
				},
				finalizersToNumRemaining: map[string]int{
					"service-catalog": 6,
					"kubedb":          5,
				},
			},
			expecteds: []v1.NamespaceCondition{
				{Type: v1.NamespaceContentRemaining, Status: v1.ConditionTrue, Reason: "SomeResourcesRemain", Message: `Some resources are remaining: daemonsets.apps.k8s.io has 5 resource instances, deployments.apps.k8s.io has 5 resource instances`},
				{Type: v1.NamespaceFinalizersRemaining, Status: v1.ConditionTrue, Reason: "SomeFinalizersRemain", Message: `Some content in the namespace has finalizers remaining: kubedb in 5 resource instances, service-catalog in 6 resource instances`},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			u := namespaceConditionUpdater{}

			u.ProcessContentTotals(test.contentTotals)

			actuals := u.newConditions
			if len(actuals) != len(test.expecteds) {
				t.Fatal(actuals)
			}
			for i := range actuals {
				actual := actuals[i]
				expected := test.expecteds[i]
				expected.LastTransitionTime = actual.LastTransitionTime
				if !reflect.DeepEqual(expected, actual) {
					t.Error(actual)
				}
			}
		})
	}
}
