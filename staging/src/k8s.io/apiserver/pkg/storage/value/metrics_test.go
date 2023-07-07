/*
Copyright 2017 The Kubernetes Authors.

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

package value

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestTotals(t *testing.T) {
	nonStatusErr := errors.New("test error")
	failedPreconditionErr := status.Error(codes.FailedPrecondition, "test error")
	internalErr := status.Error(codes.Internal, "test error")
	wrappedErr := fmt.Errorf("some low level thing failed: %w", status.Error(codes.NotFound, "some error"))

	nonStatusErrTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{err: nonStatusErr}}
	failedPreconditionErrTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{err: failedPreconditionErr}}
	internalErrTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{err: internalErr}}
	okTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{from: []byte("value")}}
	wrappedErrTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{err: wrappedErr}}

	testCases := []struct {
		desc    string
		prefix  Transformer
		metrics []string
		want    string
	}{
		{
			desc:   "non-status error",
			prefix: NewPrefixTransformers(nil, nonStatusErrTransformer),
			metrics: []string{
				"apiserver_storage_transformation_operations_total",
			},
			want: `
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations. Successful transformation will have a status 'OK' and a varied status string when the transformation fails. This status and transformation_type fields may be used for alerting on encryption/decryption failure using transformation_type from_storage for decryption and to_storage for encryption
				# TYPE apiserver_storage_transformation_operations_total counter
				apiserver_storage_transformation_operations_total{status="unknown-non-grpc",transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				apiserver_storage_transformation_operations_total{status="unknown-non-grpc",transformation_type="to_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				`,
		},
		{
			desc:   "ok",
			prefix: NewPrefixTransformers(nil, okTransformer),
			metrics: []string{
				"apiserver_storage_transformation_operations_total",
			},
			want: `
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations. Successful transformation will have a status 'OK' and a varied status string when the transformation fails. This status and transformation_type fields may be used for alerting on encryption/decryption failure using transformation_type from_storage for decryption and to_storage for encryption
				# TYPE apiserver_storage_transformation_operations_total counter
				apiserver_storage_transformation_operations_total{status="OK",transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				apiserver_storage_transformation_operations_total{status="OK",transformation_type="to_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				`,
		},
		{
			desc:   "failed precondition",
			prefix: NewPrefixTransformers(nil, failedPreconditionErrTransformer),
			metrics: []string{
				"apiserver_storage_transformation_operations_total",
			},
			want: `
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations. Successful transformation will have a status 'OK' and a varied status string when the transformation fails. This status and transformation_type fields may be used for alerting on encryption/decryption failure using transformation_type from_storage for decryption and to_storage for encryption
				# TYPE apiserver_storage_transformation_operations_total counter
				apiserver_storage_transformation_operations_total{status="FailedPrecondition",transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				apiserver_storage_transformation_operations_total{status="FailedPrecondition",transformation_type="to_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				`,
		},
		{
			desc:   "internal",
			prefix: NewPrefixTransformers(nil, internalErrTransformer),
			metrics: []string{
				"apiserver_storage_transformation_operations_total",
			},
			want: `
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations. Successful transformation will have a status 'OK' and a varied status string when the transformation fails. This status and transformation_type fields may be used for alerting on encryption/decryption failure using transformation_type from_storage for decryption and to_storage for encryption
				# TYPE apiserver_storage_transformation_operations_total counter
				apiserver_storage_transformation_operations_total{status="Internal",transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				apiserver_storage_transformation_operations_total{status="Internal",transformation_type="to_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				`,
		},
		{
			desc:   "wrapped not found error",
			prefix: NewPrefixTransformers(nil, wrappedErrTransformer),
			metrics: []string{
				"apiserver_storage_transformation_operations_total",
			},
			want: `
			# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations. Successful transformation will have a status 'OK' and a varied status string when the transformation fails. This status and transformation_type fields may be used for alerting on encryption/decryption failure using transformation_type from_storage for decryption and to_storage for encryption
			# TYPE apiserver_storage_transformation_operations_total counter
			apiserver_storage_transformation_operations_total{status="NotFound",transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
			apiserver_storage_transformation_operations_total{status="NotFound",transformation_type="to_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
			`,
		},
	}

	RegisterMetrics()
	transformerOperationsTotal.Reset()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			tt.prefix.TransformToStorage(context.Background(), []byte("value"), nil)
			tt.prefix.TransformFromStorage(context.Background(), []byte("k8s:enc:kms:v1:value"), nil)
			defer transformerOperationsTotal.Reset()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestLatency(t *testing.T) {
	testCases := []struct {
		desc               string
		prefix             string
		transformationType string
		elapsed            time.Duration
		metrics            []string
		want               string
	}{
		{
			desc:               "transformation latency",
			prefix:             "k8s:enc:kms:v1:",
			transformationType: "from_storage",
			elapsed:            time.Duration(10) * time.Second,
			metrics: []string{
				"apiserver_storage_transformation_duration_seconds",
			},
			want: `
			# HELP apiserver_storage_transformation_duration_seconds [ALPHA] Latencies in seconds of value transformation operations.
			# TYPE apiserver_storage_transformation_duration_seconds histogram
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="5e-06"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="1e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="2e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="4e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="8e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.00016"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.00032"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.00064"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.00128"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.00256"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.00512"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.01024"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.02048"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.04096"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.08192"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.16384"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.32768"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="0.65536"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="1.31072"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="2.62144"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="5.24288"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="10.48576"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="20.97152"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="41.94304"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="83.88608"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:",le="+Inf"} 1
			apiserver_storage_transformation_duration_seconds_sum{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 10
			apiserver_storage_transformation_duration_seconds_count{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				`,
		},
		{
			desc:               "transformation latency 2",
			prefix:             "k8s:enc:kms:v2:",
			transformationType: "from_storage",
			elapsed:            time.Duration(5) * time.Second,
			metrics: []string{
				"apiserver_storage_transformation_duration_seconds",
			},
			want: `
			# HELP apiserver_storage_transformation_duration_seconds [ALPHA] Latencies in seconds of value transformation operations.
			# TYPE apiserver_storage_transformation_duration_seconds histogram
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="5e-06"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="1e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="2e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="4e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="8e-05"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.00016"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.00032"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.00064"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.00128"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.00256"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.00512"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.01024"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.02048"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.04096"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.08192"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.16384"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.32768"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="0.65536"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="1.31072"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="2.62144"} 0
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="5.24288"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="10.48576"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="20.97152"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="41.94304"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="83.88608"} 1
			apiserver_storage_transformation_duration_seconds_bucket{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:",le="+Inf"} 1
			apiserver_storage_transformation_duration_seconds_sum{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:"} 5
			apiserver_storage_transformation_duration_seconds_count{transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v2:"} 1
				`,
		},
	}

	RegisterMetrics()
	transformerLatencies.Reset()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			RecordTransformation(tt.transformationType, tt.prefix, tt.elapsed, nil)
			defer transformerLatencies.Reset()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
