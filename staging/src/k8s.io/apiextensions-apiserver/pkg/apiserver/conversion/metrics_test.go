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

package conversion

import (
	"context"
	"testing"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestWebhookConversionMetrics_ObserveWebhookConversionSuccess(t *testing.T) {
	type fields struct {
		webhookConversionRequest *metrics.CounterVec
		webhookConversionLatency *metrics.HistogramVec
	}
	type args struct {
		ctx     context.Context
		elapsed time.Duration
	}
	tests := []struct {
		name                 string
		fields               fields
		args                 args
		wantLabels           map[string]string
		expectedRequestValue int
	}{
		// TODO: Add test cases.
		{
			name: "test_conversion_success",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				ctx:     context.TODO(),
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
				ctx:     context.TODO(),
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
			m.ObserveWebhookConversionSuccess(tt.args.ctx, tt.args.elapsed)
			expectCounterValue(t, "webhook_conversion_requests", tt.wantLabels, tt.expectedRequestValue)
			expectHistogramCountTotal(t, "webhook_conversion_duration_seconds", tt.wantLabels, tt.expectedRequestValue)
		})
	}
}

func TestWebhookConversionMetrics_ObserveWebhookConversionFailure(t *testing.T) {
	type fields struct {
		webhookConversionRequest *metrics.CounterVec
		webhookConversionLatency *metrics.HistogramVec
	}
	type args struct {
		ctx       context.Context
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
		// TODO: Add test cases.
		{
			name: "test_conversion_failure",
			fields: fields{
				webhookConversionRequest: Metrics.webhookConversionRequest,
				webhookConversionLatency: Metrics.webhookConversionLatency,
			},
			args: args{
				ctx:       context.TODO(),
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
				ctx:       context.TODO(),
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
				ctx:       context.TODO(),
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
				ctx:       context.TODO(),
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
				ctx:       context.TODO(),
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
			m.ObserveWebhookConversionFailure(tt.args.ctx, tt.args.elapsed, tt.args.errorType)
			expectCounterValue(t, "webhook_conversion_requests", tt.wantLabels, tt.expectedRequestValue)
			expectHistogramCountTotal(t, "webhook_conversion_duration_seconds", tt.wantLabels, tt.expectedRequestValue)
		})
	}
}

func expectCounterValue(t *testing.T, name string, labelFilter map[string]string, wantCount int) {
	metrics, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %s", err)
	}

	counterSum := 0
	for _, mf := range metrics {
		if mf.GetName() != name {
			continue // Ignore other metrics.
		}
		for _, metric := range mf.GetMetric() {
			if !testutil.LabelsMatch(metric, labelFilter) {
				continue
			}
			counterSum += int(metric.GetCounter().GetValue())
		}
	}
	if wantCount != counterSum {
		t.Errorf("Wanted count %d, got %d for metric %s with labels %#+v", wantCount, counterSum, name, labelFilter)
		for _, mf := range metrics {
			if mf.GetName() == name {
				for _, metric := range mf.GetMetric() {
					t.Logf("\tnear match: %s", metric.String())
				}
			}
		}
	}
}

func expectHistogramCountTotal(t *testing.T, name string, labelFilter map[string]string, wantCount int) {
	metrics, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %s", err)
	}

	counterSum := 0
	for _, mf := range metrics {
		if mf.GetName() != name {
			continue // Ignore other metrics.
		}
		for _, metric := range mf.GetMetric() {
			if !testutil.LabelsMatch(metric, labelFilter) {
				continue
			}
			counterSum += int(metric.GetHistogram().GetSampleCount())
		}
	}
	if wantCount != counterSum {
		t.Errorf("Wanted count %d, got %d for metric %s with labels %#+v", wantCount, counterSum, name, labelFilter)
		for _, mf := range metrics {
			if mf.GetName() == name {
				for _, metric := range mf.GetMetric() {
					t.Logf("\tnear match: %s\n", metric.String())
				}
			}
		}
	}
}
