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

package metrics

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordKMSOperationLatency(t *testing.T) {
	testCases := []struct {
		name         string
		methodName   string
		duration     time.Duration
		operationErr error
		want         string
	}{
		{
			name:         "operation success",
			methodName:   "/v2alpha1.KeyManagementService/Encrypt",
			duration:     1 * time.Second,
			operationErr: nil,
			want: `
			# HELP apiserver_envelope_encryption_kms_operations_latency_seconds [ALPHA] KMS operation duration with gRPC error code status total.
			# TYPE apiserver_envelope_encryption_kms_operations_latency_seconds histogram
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.001"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.002"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.004"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.008"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.016"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.032"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.064"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.128"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.256"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.512"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="1.024"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="2.048"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="4.096"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="8.192"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="16.384"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="32.768"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="65.536"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="131.072"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="+Inf"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_sum{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_count{grpc_status_code="OK",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			`,
		},
		{
			name:         "operation error",
			methodName:   "/v2alpha1.KeyManagementService/Encrypt",
			duration:     1 * time.Second,
			operationErr: status.Error(codes.Internal, "some error"),
			want: `
			# HELP apiserver_envelope_encryption_kms_operations_latency_seconds [ALPHA] KMS operation duration with gRPC error code status total.
			# TYPE apiserver_envelope_encryption_kms_operations_latency_seconds histogram
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.001"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.002"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.004"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.008"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.016"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.032"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.064"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.128"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.256"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.512"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="1.024"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="2.048"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="4.096"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="8.192"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="16.384"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="32.768"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="65.536"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="131.072"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="+Inf"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_sum{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_count{grpc_status_code="Internal",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			`,
		},
		{
			name:         "wrapped not found error",
			methodName:   "/v2alpha1.KeyManagementService/Encrypt",
			duration:     1 * time.Second,
			operationErr: fmt.Errorf("some low level thing failed: %w", status.Error(codes.NotFound, "some error")),
			want: `
			# HELP apiserver_envelope_encryption_kms_operations_latency_seconds [ALPHA] KMS operation duration with gRPC error code status total.
			# TYPE apiserver_envelope_encryption_kms_operations_latency_seconds histogram
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.001"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.002"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.004"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.008"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.016"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.032"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.064"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.128"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.256"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.512"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="1.024"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="2.048"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="4.096"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="8.192"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="16.384"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="32.768"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="65.536"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="131.072"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="+Inf"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_sum{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_count{grpc_status_code="NotFound",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			`,
		},
		{
			name:         "non gRPC error",
			methodName:   "/v2alpha1.KeyManagementService/Encrypt",
			duration:     1 * time.Second,
			operationErr: fmt.Errorf("some bad thing happened"),
			want: `
			# HELP apiserver_envelope_encryption_kms_operations_latency_seconds [ALPHA] KMS operation duration with gRPC error code status total.
			# TYPE apiserver_envelope_encryption_kms_operations_latency_seconds histogram
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.001"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.002"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.004"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.008"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.016"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.032"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.064"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.128"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.256"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="0.512"} 0
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="1.024"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="2.048"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="4.096"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="8.192"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="16.384"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="32.768"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="65.536"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="131.072"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_bucket{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName",le="+Inf"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_sum{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			apiserver_envelope_encryption_kms_operations_latency_seconds_count{grpc_status_code="unknown-non-grpc",method_name="/v2alpha1.KeyManagementService/Encrypt",provider_name="providerName"} 1
			`,
		},
	}

	RegisterMetrics()

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			RecordKMSOperationLatency("providerName", tt.methodName, tt.duration, tt.operationErr)
			defer KMSOperationsLatencyMetric.Reset()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), "apiserver_envelope_encryption_kms_operations_latency_seconds"); err != nil {
				t.Fatal(err)
			}
		})
	}
}
