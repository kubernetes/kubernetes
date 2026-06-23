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

func TestAuthorizationWebhookEvaluationsTotalIsBeta(t *testing.T) {
	ResetWebhookMetricsForTest()
	defer ResetWebhookMetricsForTest()

	want := `
		# HELP apiserver_authorization_webhook_evaluations_total [BETA] Round-trips to authorization webhooks.
		# TYPE apiserver_authorization_webhook_evaluations_total counter
		apiserver_authorization_webhook_evaluations_total{name="wh1.example.com",result="success"} 1
	`

	m := NewWebhookMetrics()
	m.RecordWebhookEvaluation(context.Background(), "wh1.example.com", "success")

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "apiserver_authorization_webhook_evaluations_total"); err != nil {
		t.Fatal(err)
	}
}

func TestAuthorizationWebhookDurationSecondsIsBeta(t *testing.T) {
	ResetWebhookMetricsForTest()
	defer ResetWebhookMetricsForTest()

	want := `
		# HELP apiserver_authorization_webhook_duration_seconds [BETA] Request latency in seconds.
		# TYPE apiserver_authorization_webhook_duration_seconds histogram
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="0.005"} 0
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="0.01"} 0
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="0.025"} 0
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="0.05"} 0
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="0.1"} 0
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="0.25"} 1
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="0.5"} 1
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="1"} 1
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="2.5"} 1
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="5"} 1
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="10"} 1
		apiserver_authorization_webhook_duration_seconds_bucket{name="wh1.example.com",result="success",le="+Inf"} 1
		apiserver_authorization_webhook_duration_seconds_sum{name="wh1.example.com",result="success"} 0.2
		apiserver_authorization_webhook_duration_seconds_count{name="wh1.example.com",result="success"} 1
	`

	m := NewWebhookMetrics()
	m.RecordWebhookDuration(context.Background(), "wh1.example.com", "success", 0.2)

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "apiserver_authorization_webhook_duration_seconds"); err != nil {
		t.Fatal(err)
	}
}

func TestAuthorizationWebhookEvaluationsFailOpenTotalIsBeta(t *testing.T) {
	ResetWebhookMetricsForTest()
	defer ResetWebhookMetricsForTest()

	want := `
		# HELP apiserver_authorization_webhook_evaluations_fail_open_total [BETA] NoOpinion results due to webhook timeout or error.
		# TYPE apiserver_authorization_webhook_evaluations_fail_open_total counter
		apiserver_authorization_webhook_evaluations_fail_open_total{name="wh1.example.com",result="timeout"} 1
	`

	m := NewWebhookMetrics()
	m.RecordWebhookFailOpen(context.Background(), "wh1.example.com", "timeout")

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "apiserver_authorization_webhook_evaluations_fail_open_total"); err != nil {
		t.Fatal(err)
	}
}
