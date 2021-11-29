/*
Copyright 2021 The Kubernetes Authors.

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

package flowcontrol

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	flowcontrol "k8s.io/api/flowcontrol/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func Test_configController_generatePatchBytes(t *testing.T) {
	now := time.Now().UTC()
	tests := []struct {
		name      string
		condition flowcontrol.FlowSchemaCondition
		want      []byte
	}{
		{
			name: "check if only condition is parsed",
			condition: flowcontrol.FlowSchemaCondition{
				Type:               flowcontrol.FlowSchemaConditionDangling,
				Status:             flowcontrol.ConditionTrue,
				Reason:             "test reason",
				Message:            "test none",
				LastTransitionTime: metav1.NewTime(now),
			},
			want: []byte(fmt.Sprintf(`{"status":{"conditions":[{"type":"Dangling","status":"True","lastTransitionTime":"%s","reason":"test reason","message":"test none"}]}}`, now.Format(time.RFC3339))),
		},
		{
			name: "check when message has double quotes",
			condition: flowcontrol.FlowSchemaCondition{
				Type:               flowcontrol.FlowSchemaConditionDangling,
				Status:             flowcontrol.ConditionTrue,
				Reason:             "test reason",
				Message:            `test ""none`,
				LastTransitionTime: metav1.NewTime(now),
			},
			want: []byte(fmt.Sprintf(`{"status":{"conditions":[{"type":"Dangling","status":"True","lastTransitionTime":"%s","reason":"test reason","message":"test \"\"none"}]}}`, now.Format(time.RFC3339))),
		},
		{
			name: "check when message has a whitespace character that can be escaped",
			condition: flowcontrol.FlowSchemaCondition{
				Type:   flowcontrol.FlowSchemaConditionDangling,
				Status: flowcontrol.ConditionTrue,
				Reason: "test reason",
				Message: `test 		none`,
				LastTransitionTime: metav1.NewTime(now),
			},
			want: []byte(fmt.Sprintf(`{"status":{"conditions":[{"type":"Dangling","status":"True","lastTransitionTime":"%s","reason":"test reason","message":"test \t\tnone"}]}}`, now.Format(time.RFC3339))),
		},
		{
			name: "check when a few fields (message & lastTransitionTime) are missing",
			condition: flowcontrol.FlowSchemaCondition{
				Type:   flowcontrol.FlowSchemaConditionDangling,
				Status: flowcontrol.ConditionTrue,
				Reason: "test reason",
			},
			want: []byte(`{"status":{"conditions":[{"type":"Dangling","status":"True","lastTransitionTime":null,"reason":"test reason"}]}}`),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := makeFlowSchemaConditionPatch(tt.condition)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("makeFlowSchemaConditionPatch() got = %s, want %s; diff is %s", got, tt.want, cmp.Diff(tt.want, got))
			}
		})
	}
}
