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

package job

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	batch "k8s.io/api/batch/v1"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestMatchSuccessPolicy(t *testing.T) {
	testCases := map[string]struct {
		successPolicy          *batch.SuccessPolicy
		completions            int32
		succeededIndexes       orderedIntervals
		wantMessage            string
		wantMetSuccessPolicy   bool
		enableJobSuccessPolicy bool
	}{
		"JobSuccessPolicy is disabled": {
			completions:      10,
			succeededIndexes: orderedIntervals{{0, 0}},
		},
		"successPolicy is null": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 0}},
		},
		"any rules are nothing": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 0}},
			successPolicy:          &batch.SuccessPolicy{Rules: []batch.SuccessPolicyRule{}},
		},
		"rules.succeededIndexes is invalid format": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 0}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("invalid-form"),
				}},
			},
		},
		"rules.succeededIndexes is specified; succeededIndexes matched rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}, {4, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-2"),
				}},
			},
			wantMessage:          "Matched rules at index 0",
			wantMetSuccessPolicy: true,
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("3"),
				}},
			},
		},
		"rules.succeededCount is specified; succeededIndexes matched rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededCount: ptr.To[int32](2),
				}},
			},
			wantMessage:          "Matched rules at index 0",
			wantMetSuccessPolicy: true,
		},
		"rules.succeededCount is specified; succeededIndexes didn't match rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededCount: ptr.To[int32](4),
				}},
			},
		},
		"multiple rules; rules.succeededIndexes is specified; succeededIndexes met one of rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}, {4, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{
					{
						SucceededIndexes: ptr.To("9"),
					},
					{
						SucceededIndexes: ptr.To("4,6"),
					},
				},
			},
			wantMessage:          "Matched rules at index 1",
			wantMetSuccessPolicy: true,
		},
		"multiple rules; rules.succeededIndexes is specified; succeededIndexes met all rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}, {4, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{
					{
						SucceededIndexes: ptr.To("0,1"),
					},
					{
						SucceededIndexes: ptr.To("5"),
					},
				},
			},
			wantMessage:          "Matched rules at index 0",
			wantMetSuccessPolicy: true,
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes met all rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}, {4, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("3-6"),
					SucceededCount:   ptr.To[int32](2),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes didn't match rules": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{0, 2}, {6, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("3-6"),
					SucceededCount:   ptr.To[int32](2),
				}},
			},
		},
		"rules.succeededIndexes is specified; succeededIndexes are nothing": {
			completions:      10,
			succeededIndexes: orderedIntervals{},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededCount: ptr.To[int32](4),
				}},
			},
		},

		"rules.succeededIndexes is specified; succeededIndexes matched rules; rules is proper subset of succeededIndexes": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 5}, {6, 9}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("2-4,6-8"),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes matched rules; rules equals succeededIndexes": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{2, 4}, {6, 9}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("2-4,6-9"),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes matched rules; rules is subset of succeededIndexes": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{2, 5}, {7, 15}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("2-4,8-12"),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; rules is an empty set": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To(""),
				}},
			},
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; succeededIndexes is an empty set": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To(""),
				}},
			},
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; rules and succeededIndexes are empty set": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To(""),
				}},
			},
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; all elements of rules.succeededIndexes aren't included in succeededIndexes": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}, {5, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("10-12,14-16"),
				}},
			},
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; rules overlaps succeededIndexes at first": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{2, 4}, {6, 8}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("1-3,5-7"),
				}},
			},
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes matched rules; rules overlaps succeededIndexes at first": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{2, 4}, {6, 8}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("1-3,5-7"),
					SucceededCount:   ptr.To[int32](4),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; rules overlaps succeededIndexes at last": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}, {5, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("2-4,6-9"),
				}},
			},
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes matched rules; rules overlaps succeededIndexes at last": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}, {5, 7}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("2-4,6-9"),
					SucceededCount:   ptr.To[int32](4),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; rules completely overlaps succeededIndexes": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}, {7, 8}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-4,6-9"),
				}},
			},
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes matched rules; rules completely overlaps succeededIndexes": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}, {7, 8}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-4,6-9"),
					SucceededCount:   ptr.To[int32](5),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; rules overlaps multiple succeededIndexes at last": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}, {5, 9}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-6,8-9"),
				}},
			},
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes matched rules; rules overlaps multiple succeededIndexes at last": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 3}, {5, 9}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-6,8-9"),
					SucceededCount:   ptr.To[int32](7),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; rules overlaps succeededIndexes at first, and rules equals succeededIndexes at last": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 5}, {7, 10}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-5,7-9"),
				}},
			},
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes matched rules; rules overlaps succeededIndexes at first, and rules equals succeededIndexes at last": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 5}, {7, 10}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-5,7-9"),
					SucceededCount:   ptr.To[int32](8),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
		"rules.succeededIndexes is specified; succeededIndexes didn't match rules; the first rules overlaps succeededIndexes at first": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 10}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-3,6-9"),
				}},
			},
		},
		"rules.succeededIndexes and rules.succeededCount are specified; succeededIndexes matched rules; the first rules overlaps succeededIndexes at first": {
			enableJobSuccessPolicy: true,
			completions:            10,
			succeededIndexes:       orderedIntervals{{1, 10}},
			successPolicy: &batch.SuccessPolicy{
				Rules: []batch.SuccessPolicyRule{{
					SucceededIndexes: ptr.To("0-3,6-9"),
					SucceededCount:   ptr.To[int32](7),
				}},
			},
			wantMetSuccessPolicy: true,
			wantMessage:          "Matched rules at index 0",
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobSuccessPolicy, tc.enableJobSuccessPolicy)
			logger := ktesting.NewLogger(t,
				ktesting.NewConfig(
					ktesting.BufferLogs(true),
				),
			)
			gotMessage, gotMetSuccessPolicy := matchSuccessPolicy(logger, tc.successPolicy, tc.completions, tc.succeededIndexes)
			if tc.wantMetSuccessPolicy != gotMetSuccessPolicy {
				t.Errorf("Unexpected bool from matchSuccessPolicy\nwant:%v\ngot:%v\n", tc.wantMetSuccessPolicy, gotMetSuccessPolicy)
			}
			if diff := cmp.Diff(tc.wantMessage, gotMessage); diff != "" {
				t.Errorf("Unexpected message from matchSuccessPolicy (-want,+got):\n%s", diff)
			}
		})
	}
}
