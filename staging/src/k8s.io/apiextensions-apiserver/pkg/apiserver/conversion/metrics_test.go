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

func TestConversionWebhookMetrics_ObserveConversionWebhookSuccess(t *testing.T) {
	type fields struct {
		conversionWebhookRequest *metrics.CounterVec
		conversionWebhookLatency *metrics.HistogramVec
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
				conversionWebhookRequest: Metrics.conversionWebhookRequest,
				conversionWebhookLatency: Metrics.conversionWebhookLatency,
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
				conversionWebhookRequest: Metrics.conversionWebhookRequest,
				conversionWebhookLatency: Metrics.conversionWebhookLatency,
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
			m := &ConversionWebhookMetrics{
				conversionWebhookRequest: tt.fields.conversionWebhookRequest,
				conversionWebhookLatency: tt.fields.conversionWebhookLatency,
			}
			m.ObserveConversionWebhookSuccess(context.TODO(), tt.args.elapsed)
			testutil.AssertVectorCount(t, fmt.Sprintf("%s_conversion_webhook_request_total", namespace), tt.wantLabels, tt.expectedRequestValue)
			testutil.AssertHistogramTotalCount(t, fmt.Sprintf("%s_conversion_webhook_duration_seconds", namespace), tt.wantLabels, tt.expectedRequestValue)
		})
	}
}

func TestConversionWebhookMetrics_ObserveConversionWebhookFailure(t *testing.T) {
	type fields struct {
		conversionWebhookRequest *metrics.CounterVec
		conversionWebhookLatency *metrics.HistogramVec
	}
	type args struct {
		elapsed   time.Duration
		errorType ConversionWebhookErrorType
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
				conversionWebhookRequest: Metrics.conversionWebhookRequest,
				conversionWebhookLatency: Metrics.conversionWebhookLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: ConversionWebhookCallFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(ConversionWebhookCallFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 1,
		}, {
			name: "test_conversion_failure_2",
			fields: fields{
				conversionWebhookRequest: Metrics.conversionWebhookRequest,
				conversionWebhookLatency: Metrics.conversionWebhookLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: ConversionWebhookMalformedResponseFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(ConversionWebhookMalformedResponseFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 2,
		}, {
			name: "test_conversion_failure_3",
			fields: fields{
				conversionWebhookRequest: Metrics.conversionWebhookRequest,
				conversionWebhookLatency: Metrics.conversionWebhookLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: ConversionWebhookPartialResponseFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(ConversionWebhookPartialResponseFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 3,
		}, {
			name: "test_conversion_failure_4",
			fields: fields{
				conversionWebhookRequest: Metrics.conversionWebhookRequest,
				conversionWebhookLatency: Metrics.conversionWebhookLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: ConversionWebhookInvalidConvertedObjectFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(ConversionWebhookInvalidConvertedObjectFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 4,
		}, {
			name: "test_conversion_failure_5",
			fields: fields{
				conversionWebhookRequest: Metrics.conversionWebhookRequest,
				conversionWebhookLatency: Metrics.conversionWebhookLatency,
			},
			args: args{
				elapsed:   2 * time.Second,
				errorType: ConversionWebhookNoObjectsReturnedFailure,
			},
			wantLabels: map[string]string{
				"result":       "failure",
				"failure_type": string(ConversionWebhookNoObjectsReturnedFailure),
			},
			expectedRequestValue: 1,
			expectedLatencyCount: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &ConversionWebhookMetrics{
				conversionWebhookRequest: tt.fields.conversionWebhookRequest,
				conversionWebhookLatency: tt.fields.conversionWebhookLatency,
			}
			m.ObserveConversionWebhookFailure(context.TODO(), tt.args.elapsed, tt.args.errorType)
			testutil.AssertVectorCount(t, fmt.Sprintf("%s_conversion_webhook_request_total", namespace), tt.wantLabels, tt.expectedRequestValue)
			testutil.AssertHistogramTotalCount(t, fmt.Sprintf("%s_conversion_webhook_duration_seconds", namespace), tt.wantLabels, tt.expectedRequestValue)
		})
	}
}
