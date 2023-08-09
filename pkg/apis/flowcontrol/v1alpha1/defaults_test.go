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

package v1alpha1

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/pointer"
)

func TestDefaultWithPriorityLevelConfiguration(t *testing.T) {
	tests := []struct {
		name     string
		original runtime.Object
		expected runtime.Object
	}{
		{
			name: "LendablePercent is not specified, should default to zero",
			original: &flowcontrolv1alpha1.PriorityLevelConfiguration{
				Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
					Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
					Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
						AssuredConcurrencyShares: 5,
						LimitResponse: flowcontrolv1alpha1.LimitResponse{
							Type: flowcontrolv1alpha1.LimitResponseTypeReject,
						},
					},
				},
			},
			expected: &flowcontrolv1alpha1.PriorityLevelConfiguration{
				Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
					Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
					Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
						AssuredConcurrencyShares: 5,
						LendablePercent:          pointer.Int32(0),
						LimitResponse: flowcontrolv1alpha1.LimitResponse{
							Type: flowcontrolv1alpha1.LimitResponseTypeReject,
						},
					},
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatalf("Failed to add to scheme: %v", err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := test.original
			expected := test.expected

			scheme.Default(original)
			if !reflect.DeepEqual(expected, original) {
				t.Errorf("Expected defaulting to work - diff: %s", cmp.Diff(expected, original))
			}
		})
	}
}
