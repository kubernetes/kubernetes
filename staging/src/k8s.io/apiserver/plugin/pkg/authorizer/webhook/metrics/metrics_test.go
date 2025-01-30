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

package metrics

import (
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordWebhookMetrics(t *testing.T) {
	testCases := []struct {
		desc     string
		metrics  []string
		name     string
		result   string
		duration float64
		want     string
	}{
		{
			desc: "evaluation failure total",
			metrics: []string{
				"apiserver_authorization_webhook_duration_seconds",
				"apiserver_authorization_webhook_evaluations_total",
				"apiserver_authorization_webhook_evaluations_fail_open_total",
			},
			name:     "wh1.example.com",
			result:   "timeout",
			duration: 1.5,
			want: `
			# HELP apiserver_authorization_webhook_duration_seconds [ALPHA] Request latency in seconds.
			# TYPE apiserver_authorization_webhook_duration_seconds histogram
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="0.005"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="0.01"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="0.025"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="0.05"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="0.1"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="0.25"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="0.5"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="1"} 0
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="2.5"} 1
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="5"} 1
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="10"} 1
            apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="timeout",le="+Inf"} 1
            apiserver_authorization_webhook_duration_seconds_sum{name="wh1.example.com",result="timeout"} 1.5
            apiserver_authorization_webhook_duration_seconds_count{name="wh1.example.com",result="timeout"} 1
            # HELP apiserver_authorization_webhook_evaluations_fail_open_total [ALPHA] NoOpinion results due to webhook timeout or error.
            # TYPE apiserver_authorization_webhook_evaluations_fail_open_total counter
            apiserver_authorization_webhook_evaluations_fail_open_total{name="wh1.example.com",result="timeout"} 1
            # HELP apiserver_authorization_webhook_evaluations_total [ALPHA] Round-trips to authorization webhooks.
            # TYPE apiserver_authorization_webhook_evaluations_total counter
            apiserver_authorization_webhook_evaluations_total{name="wh1.example.com",result="timeout"} 1
            `,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			ResetWebhookMetricsForTest()
			m := NewWebhookMetrics()
			m.RecordWebhookDuration(context.Background(), tt.name, tt.result, tt.duration)
			m.RecordWebhookEvaluation(context.Background(), tt.name, tt.result)
			m.RecordWebhookFailOpen(context.Background(), tt.name, tt.result)
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
