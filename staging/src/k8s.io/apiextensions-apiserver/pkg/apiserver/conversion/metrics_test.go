/*
Copyright 2023 The Kubernetes Authors.

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

package conversion

import (
	"context"
	"fmt"
	"testing"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestWebhookConversionMetrics_ObserveWebhookConversionSuccess(t *testing.T) {
	type fields struct {
		webhookConversionRequest *metrics.CounterVec
		webhookConversionLatency *metrics.HistogramVec
	}
	type args struct {
		elapsed time.Duration
	}
	tests := []struct {
		name                 string
		fields               fields
		args                 args
		wantLabels           map[string]string
		expectedRequestValue int
	}{
		{
			name: "test_conversion_success",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				elapsed: 2 * time.Second,
			},
			wantLabels: map[string]string{
				"result":       "success",
				"failure_type": "",
			},
			expectedRequestValue: 1,
		}, {
			name: "test_conversion_success_2",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				elapsed: 2 * time.Second,
			},
			wantLabels: map[string]string{
				"result":       "success",
				"failure_type": "",
			},
			expectedRequestValue: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &WebhookConversionMetrics{
				webhookConversionRequest: tt.fields.webhookConversionRequest,
				webhookConversionLatency: tt.fields.webhookConversionLatency,
			}
			m.ObserveWebhookConversionSuccess(context.TODO(), tt.args.elapsed)
			testutil.AssertVectorCount(t, fmt.Sprintf("%s_webhook_conversion_request_total", namespace), tt.wantLabels, tt.expectedRequestValue)
			testutil.AssertHistogramTotalCount(t, fmt.Sprintf("%s_webhook_conversion_duration_seconds", namespace), tt.wantLabels, tt.expectedRequestValue)
		})
	}
}

func TestWebhookConversionMetrics_ObserveWebhookConversionFailure(t *testing.T) {
	type fields struct {
		webhookConversionRequest *metrics.CounterVec
		webhookConversionLatency *metrics.HistogramVec
	}
	type args struct {
		elapsed   time.Duration
		errorType WebhookConversionErrorType
	}
	tests := []struct {
		name                 string
		fields               fields
		args                 args
		wantLabels           map[string]string
		expectedRequestValue int
		expectedLatencyCount int
	}{
		{
			name: "test_conversion_failure",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: WebhookConversionCallFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(WebhookConversionCallFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 1,
		}, {
			name: "test_conversion_failure_2",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: WebhookConversionMalformedResponseFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(WebhookConversionMalformedResponseFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 2,
		}, {
			name: "test_conversion_failure_3",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: WebhookConversionPartialResponseFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(WebhookConversionPartialResponseFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 3,
		}, {
			name: "test_conversion_failure_4",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: WebhookConversionInvalidConvertedObjectFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(WebhookConversionInvalidConvertedObjectFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 4,
		}, {
			name: "test_conversion_failure_5",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: WebhookConversionNoObjectsReturnedFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(WebhookConversionNoObjectsReturnedFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &WebhookConversionMetrics{
				webhookConversionRequest: tt.fields.webhookConversionRequest,
				webhookConversionLatency: tt.fields.webhookConversionLatency,
			}
			m.ObserveWebhookConversionFailure(context.TODO(), tt.args.elapsed, tt.args.errorType)
			testutil.AssertVectorCount(t, fmt.Sprintf("%s_webhook_conversion_request_total", namespace), tt.wantLabels, tt.expectedRequestValue)
			testutil.AssertHistogramTotalCount(t, fmt.Sprintf("%s_webhook_conversion_duration_seconds", namespace), tt.wantLabels, tt.expectedRequestValue)
		})
	}
}
