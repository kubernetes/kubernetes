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
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

const (
	testKeyHash1              = "sha256:6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"
	testKeyHash2              = "sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35"
	testKeyHash3              = "sha256:4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
	testProviderNameForMetric = "providerName"
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

func TestEnvelopeMetrics_Serial(t *testing.T) {
	testCases := []struct {
		desc               string
		keyID              string
		metrics            []string
		providerName       string
		transformationType string
		want               string
	}{
		{
			desc:  "keyIDHash total",
			keyID: "1",
			metrics: []string{
				"apiserver_envelope_encryption_key_id_hash_total",
			},
			providerName:       testProviderNameForMetric,
			transformationType: FromStorageLabel,
			want: fmt.Sprintf(`
			# HELP apiserver_envelope_encryption_key_id_hash_total [ALPHA] Number of times a keyID is used split by transformation type and provider.
			# TYPE apiserver_envelope_encryption_key_id_hash_total counter
			apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
			`, testKeyHash1, testProviderNameForMetric, FromStorageLabel),
		},
		{
			desc:  "keyIDHash total more labels",
			keyID: "2",
			metrics: []string{
				"apiserver_envelope_encryption_key_id_hash_total",
			},
			providerName:       testProviderNameForMetric,
			transformationType: FromStorageLabel,
			want: fmt.Sprintf(`
			# HELP apiserver_envelope_encryption_key_id_hash_total [ALPHA] Number of times a keyID is used split by transformation type and provider.
        	# TYPE apiserver_envelope_encryption_key_id_hash_total counter
        	apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
        	apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
			`, testKeyHash1, testProviderNameForMetric, FromStorageLabel, testKeyHash2, testProviderNameForMetric, FromStorageLabel),
		},
		{
			desc:  "keyIDHash total same labels",
			keyID: "2",
			metrics: []string{
				"apiserver_envelope_encryption_key_id_hash_total",
			},
			providerName:       testProviderNameForMetric,
			transformationType: FromStorageLabel,
			want: fmt.Sprintf(`
			# HELP apiserver_envelope_encryption_key_id_hash_total [ALPHA] Number of times a keyID is used split by transformation type and provider.
			# TYPE apiserver_envelope_encryption_key_id_hash_total counter
			apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
			apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 2
			`, testKeyHash1, testProviderNameForMetric, FromStorageLabel, testKeyHash2, testProviderNameForMetric, FromStorageLabel),
		},
		{
			desc:  "keyIDHash total exceeds limit, remove first label, and empty keyID",
			keyID: "",
			metrics: []string{
				"apiserver_envelope_encryption_key_id_hash_total",
			},
			providerName:       testProviderNameForMetric,
			transformationType: FromStorageLabel,
			want: fmt.Sprintf(`
			# HELP apiserver_envelope_encryption_key_id_hash_total [ALPHA] Number of times a keyID is used split by transformation type and provider.
			# TYPE apiserver_envelope_encryption_key_id_hash_total counter
			apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 2
			apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
			`, testKeyHash2, testProviderNameForMetric, FromStorageLabel, "", testProviderNameForMetric, FromStorageLabel),
		},
		{
			desc:  "keyIDHash total exceeds limit 2, remove first label",
			keyID: "1",
			metrics: []string{
				"apiserver_envelope_encryption_key_id_hash_total",
			},
			providerName:       testProviderNameForMetric,
			transformationType: FromStorageLabel,
			want: fmt.Sprintf(`
			# HELP apiserver_envelope_encryption_key_id_hash_total [ALPHA] Number of times a keyID is used split by transformation type and provider.
			# TYPE apiserver_envelope_encryption_key_id_hash_total counter
			apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
			apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
			`, "", testProviderNameForMetric, FromStorageLabel, testKeyHash1, testProviderNameForMetric, FromStorageLabel),
		},
	}

	KeyIDHashTotal.Reset()
	cacheSize = 2
	RegisterMetrics()
	registerLRUMetrics()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			RecordKeyID(tt.transformationType, tt.providerName, tt.keyID)
			// We are not resetting the metric here as each test is not independent in order to validate the behavior
			// when the metric labels exceed the limit to ensure the labels are not unbounded.
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEnvelopeMetricsLRUKey(t *testing.T) {
	RegisterMetrics()

	cacheSize = 3
	registerLRUMetrics()
	KeyIDHashTotal.Reset()
	defer KeyIDHashTotal.Reset()

	var wg sync.WaitGroup
	for i := 1; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			keyID := rand.String(32)
			key := metricLabels{
				transformationType: rand.String(32),
				providerName:       rand.String(32),
				keyIDHash:          getHash(keyID),
			}
			RecordKeyID(key.transformationType, key.providerName, keyID)
		}()
	}
	wg.Wait()

	validMetrics := 0
	metricFamilies, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range metricFamilies {
		if family.GetName() != "apiserver_envelope_encryption_key_id_hash_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metric.Counter.GetValue() != 1 {
				t.Errorf("invalid metric seen: %s", metric.String())
			} else {
				validMetrics++
			}
		}
	}
	if validMetrics != cacheSize {
		t.Fatalf("expected total valid metrics to be the same as cacheSize %d, got %d", cacheSize, validMetrics)
	}
}
