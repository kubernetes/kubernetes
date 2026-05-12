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

package cel

import (
	"context"
	"strings"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordAuthorizationMatchConditionEvaluationFailure(t *testing.T) {
	testCases := []struct {
		desc      string
		metrics   []string
		name      string
		authztype string
		want      string
	}{
		{
			desc: "evaluation failure total",
			metrics: []string{
				"apiserver_authorization_match_condition_evaluation_errors_total",
				"apiserver_authorization_match_condition_exclusions_total",
				"apiserver_authorization_match_condition_evaluation_seconds",
			},
			name:      "wh1.example.com",
			authztype: "Webhook",
			want: `
			# HELP apiserver_authorization_match_condition_evaluation_errors_total [ALPHA] Total number of errors when an authorization webhook encounters a match condition error split by authorizer type and name.
			# TYPE apiserver_authorization_match_condition_evaluation_errors_total counter
			apiserver_authorization_match_condition_evaluation_errors_total{name="wh1.example.com",type="Webhook"} 1
			# HELP apiserver_authorization_match_condition_evaluation_seconds [ALPHA] Authorization match condition evaluation time in seconds, split by authorizer type and name.
        	# TYPE apiserver_authorization_match_condition_evaluation_seconds histogram
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="0.001"} 0
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="0.005"} 0
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="0.01"} 0
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="0.025"} 0
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="0.1"} 0
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="0.2"} 0
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="0.25"} 0
        	apiserver_authorization_match_condition_evaluation_seconds_bucket{name="wh1.example.com",type="Webhook",le="+Inf"} 1
        	apiserver_authorization_match_condition_evaluation_seconds_sum{name="wh1.example.com",type="Webhook"} 1
        	apiserver_authorization_match_condition_evaluation_seconds_count{name="wh1.example.com",type="Webhook"} 1
			# HELP apiserver_authorization_match_condition_exclusions_total [ALPHA] Total number of exclusions when an authorization webhook is skipped because match conditions exclude it.
			# TYPE apiserver_authorization_match_condition_exclusions_total counter
			apiserver_authorization_match_condition_exclusions_total{name="wh1.example.com",type="Webhook"} 1
			`,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			ResetMetricsForTest()
			m := NewMatcherMetrics()
			m.RecordAuthorizationMatchConditionEvaluationFailure(context.Background(), tt.authztype, tt.name)
			m.RecordAuthorizationMatchConditionExclusion(context.Background(), tt.authztype, tt.name)
			m.RecordAuthorizationMatchConditionEvaluation(context.Background(), tt.authztype, tt.name, time.Duration(1*time.Second))
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
