/*
Copyright 2024 The Kubernetes Authors.

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

package typetracker

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestMappingTracking is an open-box testing of the update & deletion tracking.
func TestMappingTracking(t *testing.T) {
	gv := schema.GroupVersion{Group: "apis.example.com", Version: "v1"}

	gvks := []schema.GroupVersionKind{
		gv.WithKind("Zero"),
		gv.WithKind("One"),
		gv.WithKind("Two"),
		gv.WithKind("Three"),
		gv.WithKind("Four"),
	}

	tracker := New()

	for _, step := range []struct {
		isDeletion                  bool
		policyName                  string
		gvks                        []schema.GroupVersionKind
		expectedPolicyToGVKsMapping map[string]sets.Set[schema.GroupVersionKind]
	}{
		// insert test1 -> [gvk0]
		{
			policyName: "test1",
			gvks:       []schema.GroupVersionKind{gvks[0]},
			expectedPolicyToGVKsMapping: map[string]sets.Set[schema.GroupVersionKind]{
				"test1": sets.New[schema.GroupVersionKind](gvks[0]),
			},
		},
		// update test1 -> [gvk0] -> [gvk1, gvk2]
		{
			policyName: "test1",
			gvks:       []schema.GroupVersionKind{gvks[1], gvks[2]},
			expectedPolicyToGVKsMapping: map[string]sets.Set[schema.GroupVersionKind]{
				"test1": sets.New[schema.GroupVersionKind](gvks[1], gvks[2]),
			},
		},
		// insert test2 -> [gvk0, gvk1, gvk2, gvk3, gvk4]
		{
			policyName: "test2",
			gvks:       []schema.GroupVersionKind{gvks[0], gvks[1], gvks[2], gvks[3], gvks[4]},
			expectedPolicyToGVKsMapping: map[string]sets.Set[schema.GroupVersionKind]{
				"test1": sets.New[schema.GroupVersionKind](gvks[1], gvks[2]),
				"test2": sets.New[schema.GroupVersionKind](gvks[0], gvks[1], gvks[2], gvks[3], gvks[4]),
			},
		},
		// delete test1
		{
			isDeletion: true,
			policyName: "test1",
			expectedPolicyToGVKsMapping: map[string]sets.Set[schema.GroupVersionKind]{
				"test2": sets.New[schema.GroupVersionKind](gvks[0], gvks[1], gvks[2], gvks[3], gvks[4]),
			},
		},
	} {
		invertedMapping := make(map[schema.GroupVersionKind]sets.Set[string])
		for policyName, gvkSet := range step.expectedPolicyToGVKsMapping {
			for gvk := range gvkSet {
				s, ok := invertedMapping[gvk]
				if !ok {
					s = sets.New[string]()
					invertedMapping[gvk] = s
				}
				s.Insert(policyName)
			}
		}
		if step.isDeletion {
			tracker.ObservedDeletion(step.policyName)
		} else {
			tracker.ObserveChange(step.policyName, step.gvks)
		}
		if expected, got := sets.KeySet(step.expectedPolicyToGVKsMapping),
			sets.KeySet(tracker.policyToGVKs); !expected.Equal(got) {
			t.Fatalf("wrong tracked policies, expected %v but got %v", expected, got)
		}
		allGVKs := sets.New[schema.GroupVersionKind]()
		for _, gvkSet := range step.expectedPolicyToGVKsMapping {
			allGVKs = allGVKs.Union(gvkSet)
		}
		if got := sets.KeySet(tracker.gvkToPolicies); !allGVKs.Equal(got) {
			t.Fatalf("wrong tracked GVKs, expected %v but got %v", allGVKs, got)
		}
		for policyName, gvkSet := range tracker.policyToGVKs {
			expected := step.expectedPolicyToGVKsMapping[policyName]
			if !expected.Equal(gvkSet) {
				t.Fatalf("wrong tracked GVKs for %q, expeceted %v but got %v", policyName, expected, gvkSet)
			}
		}
		for gvk, policySet := range tracker.gvkToPolicies {
			expected := invertedMapping[gvk]
			if !expected.Equal(policySet) {
				t.Fatalf("wrong tracked policies for %v, expeceted %v but got %v", gvk, expected, policySet)
			}
		}
		for gvk, expectedPolicies := range invertedMapping {
			policies := sets.New[string](tracker.AffectedPolicies([]schema.GroupVersionKind{gvk})...)
			if !expectedPolicies.Equal(policies) {
				t.Fatalf("wrong AffectedPolicies result: expected %v but got %v", expectedPolicies, policies)
			}
		}
	}
}
