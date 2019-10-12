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
	"errors"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestTotals(t *testing.T) {
	nonStatusErr := errors.New("test error")
	failedPreconditionErr := status.Error(codes.FailedPrecondition, "test error")
	internalErr := status.Error(codes.Internal, "test error")
	nonStatusErrTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{err: nonStatusErr}}
	failedPreconditionErrTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{err: failedPreconditionErr}}
	internalErrTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{err: internalErr}}
	okTransformer := PrefixTransformer{Prefix: []byte("k8s:enc:kms:v1:"), Transformer: &testTransformer{from: []byte("value")}}

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
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations.
				# TYPE apiserver_storage_transformation_operations_total counter
				apiserver_storage_transformation_operations_total{status="Unknown",transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				apiserver_storage_transformation_operations_total{status="Unknown",transformation_type="to_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				`,
		},
		{
			desc:   "ok",
			prefix: NewPrefixTransformers(nil, okTransformer),
			metrics: []string{
				"apiserver_storage_transformation_operations_total",
			},
			want: `
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations.
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
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations.
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
				# HELP apiserver_storage_transformation_operations_total [ALPHA] Total number of transformations.
				# TYPE apiserver_storage_transformation_operations_total counter
				apiserver_storage_transformation_operations_total{status="Internal",transformation_type="from_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				apiserver_storage_transformation_operations_total{status="Internal",transformation_type="to_storage",transformer_prefix="k8s:enc:kms:v1:"} 1
				`,
		},
	}

	RegisterMetrics()
	transformerOperationsTotal.Reset()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			tt.prefix.TransformToStorage([]byte("value"), nil)
			tt.prefix.TransformFromStorage([]byte("k8s:enc:kms:v1:value"), nil)
			defer transformerOperationsTotal.Reset()
			defer deprecatedTransformerFailuresTotal.Reset()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
