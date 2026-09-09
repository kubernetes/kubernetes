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

package v1beta1

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/api/flowcontrol/v1beta1"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

func TestConvert_v1beta1_LimitedPriorityLevelConfiguration_To_flowcontrol_LimitedPriorityLevelConfiguration(t *testing.T) {
	tests := []struct {
		name     string
		in       *v1beta1.LimitedPriorityLevelConfiguration
		expected *flowcontrol.LimitedPriorityLevelConfiguration
	}{
		{
			name: "nominal concurrency shares is set as expected",
			in: &v1beta1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: 100,
				LimitResponse: v1beta1.LimitResponse{
					Type: v1beta1.LimitResponseTypeReject,
				},
			},
			expected: &flowcontrol.LimitedPriorityLevelConfiguration{
				NominalConcurrencyShares: 100,
				LimitResponse: flowcontrol.LimitResponse{
					Type: flowcontrol.LimitResponseTypeReject,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			out := &flowcontrol.LimitedPriorityLevelConfiguration{}
			if err := Convert_v1beta1_LimitedPriorityLevelConfiguration_To_flowcontrol_LimitedPriorityLevelConfiguration(test.in, out, nil); err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}
			if !cmp.Equal(test.expected, out) {
				t.Errorf("Expected a match, diff: %s", cmp.Diff(test.expected, out))
			}
		})
	}
}

func TestConvert_flowcontrol_LimitedPriorityLevelConfiguration_To_v1beta1_LimitedPriorityLevelConfiguration(t *testing.T) {
	tests := []struct {
		name     string
		in       *flowcontrol.LimitedPriorityLevelConfiguration
		expected *v1beta1.LimitedPriorityLevelConfiguration
	}{
		{
			name: "assured concurrency shares is set as expected",
			in: &flowcontrol.LimitedPriorityLevelConfiguration{
				NominalConcurrencyShares: 100,
				LimitResponse: flowcontrol.LimitResponse{
					Type: flowcontrol.LimitResponseTypeReject,
				},
			},
			expected: &v1beta1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: 100,
				LimitResponse: v1beta1.LimitResponse{
					Type: v1beta1.LimitResponseTypeReject,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			out := &v1beta1.LimitedPriorityLevelConfiguration{}
			if err := Convert_flowcontrol_LimitedPriorityLevelConfiguration_To_v1beta1_LimitedPriorityLevelConfiguration(test.in, out, nil); err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}
			if !cmp.Equal(test.expected, out) {
				t.Errorf("Expected a match, diff: %s", cmp.Diff(test.expected, out))
			}
		})
	}
}
