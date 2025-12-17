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

package flowschema

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/flowcontrol" // Use internal type
)

func TestFlowSchemaValidation(t *testing.T) {
	tests := []struct {
		name        string
		tweak       func(*flowcontrol.FlowSchema)
		wantInvalid bool
	}{
		{
			name:  "valid flow schema",
			tweak: func(fs *flowcontrol.FlowSchema) {}, // No change, valid
		},
		{
			name: "empty priority level name",
			tweak: func(fs *flowcontrol.FlowSchema) {
				fs.Spec.PriorityLevelConfiguration.Name = ""
			},
			wantInvalid: true,
		},
		{
			name: "valid priority level name",
			tweak: func(fs *flowcontrol.FlowSchema) {
				fs.Spec.PriorityLevelConfiguration.Name = "workload-high"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fs := &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-flowschema",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "default",
					},
					MatchingPrecedence: 100,
					Rules:              []flowcontrol.PolicyRulesWithSubjects{},
				},
			}

			tt.tweak(fs)

			errs := Strategy.Validate(context.Background(), fs)

			if tt.wantInvalid && len(errs) == 0 {
				t.Errorf("Validate() = no errors, want validation error for empty priority level name")
			}
			if !tt.wantInvalid && len(errs) > 0 {
				t.Errorf("Validate() = %v, want no errors", errs)
			}
		})
	}
}
