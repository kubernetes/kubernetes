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

package validation_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

type fakeMetrics struct {
	original validation.ValidationMetrics
	realSum  time.Duration
}

func (f *fakeMetrics) ObserveRatchetingTime(d time.Duration) {
	// Hardcode 1 ns duration for testing to exercise all buckets
	f.original.ObserveRatchetingTime(1 * time.Nanosecond)
	f.realSum += d
}

func (f *fakeMetrics) Reset() []metrics.Registerable {
	f.realSum = 0
	originalResettable, ok := f.original.(resettable)
	if !ok {
		panic("wrapped metrics must implement resettable")
	}
	return originalResettable.Reset()
}

type resettable interface {
	Reset() []metrics.Registerable
}

func TestMetrics(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiextensionsfeatures.CRDValidationRatcheting, true)()

	// Wrap metric to keep time constant
	testMetrics := &fakeMetrics{original: validation.Metrics}
	validation.Metrics = testMetrics
	defer func() {
		validation.Metrics = testMetrics.original
	}()

	metricNames := []string{
		"apiextensions_apiserver_validation_ratcheting_seconds",
	}

	testCases := []struct {
		desc   string
		obj    *unstructured.Unstructured
		old    *unstructured.Unstructured
		schema apiextensions.JSONSchemaProps
		iters  int // how many times to validate the same update before checking metric
		want   string
	}{
		{
			desc: "valid noop update",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "bar",
				},
			},
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "bar",
				},
			},
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"foo": {
						Type: "string",
					},
				},
			},
			want: `
			# HELP apiextensions_apiserver_validation_ratcheting_seconds [ALPHA] Time for comparison of old to new for the purposes of CRDValidationRatcheting during an UPDATE in seconds.
        	# TYPE apiextensions_apiserver_validation_ratcheting_seconds histogram
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="1e-05"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="4e-05"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00016"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00064"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00256"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.01024"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.04096"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.16384"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.65536"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="2.62144"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="Inf"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_sum 5e-09
        	apiextensions_apiserver_validation_ratcheting_seconds_count 5
			`,
			iters: 5,
		},
		{
			desc: "valid change yields no metrics",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "bar",
				},
			},
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "barx",
				},
			},
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"foo": {
						Type: "string",
						Enum: []apiextensions.JSON{
							"barx", "bar",
						},
					},
				},
			},
			want: `
			# HELP apiextensions_apiserver_validation_ratcheting_seconds [ALPHA] Time for comparison of old to new for the purposes of CRDValidationRatcheting during an UPDATE in seconds.
        	# TYPE apiextensions_apiserver_validation_ratcheting_seconds histogram
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="1e-05"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="4e-05"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00016"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00064"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00256"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.01024"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.04096"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.16384"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.65536"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="2.62144"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="Inf"} 3
        	apiextensions_apiserver_validation_ratcheting_seconds_sum 3.0000000000000004e-09
        	apiextensions_apiserver_validation_ratcheting_seconds_count 3
			`,
			iters: 3,
		},
		{
			desc: "invalid noop yields no metrics",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "bar",
				},
			},
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "bar",
				},
			},
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"foo": {
						Type: "string",
						Enum: []apiextensions.JSON{
							"incorrect",
						},
					},
				},
			},
			want: `
			# HELP apiextensions_apiserver_validation_ratcheting_seconds [ALPHA] Time for comparison of old to new for the purposes of CRDValidationRatcheting during an UPDATE in seconds.
        	# TYPE apiextensions_apiserver_validation_ratcheting_seconds histogram
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="1e-05"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="4e-05"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00016"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00064"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00256"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.01024"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.04096"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.16384"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.65536"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="2.62144"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="Inf"} 10
        	apiextensions_apiserver_validation_ratcheting_seconds_sum 1.0000000000000002e-08
        	apiextensions_apiserver_validation_ratcheting_seconds_count 10
			`,
			iters: 10,
		},
		{
			desc: "ratcheted change object yields metrics",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "bar",
				},
			},
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"foo": "barx",
				},
			},
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"foo": {
						Type: "string",
						Enum: []apiextensions.JSON{
							"incorrect",
						},
					},
				},
			},
			want: `
			# HELP apiextensions_apiserver_validation_ratcheting_seconds [ALPHA] Time for comparison of old to new for the purposes of CRDValidationRatcheting during an UPDATE in seconds.
        	# TYPE apiextensions_apiserver_validation_ratcheting_seconds histogram
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="1e-05"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="4e-05"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00016"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00064"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.00256"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.01024"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.04096"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.16384"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="0.65536"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="2.62144"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_bucket{le="Inf"} 5
        	apiextensions_apiserver_validation_ratcheting_seconds_sum 5e-09
        	apiextensions_apiserver_validation_ratcheting_seconds_count 5
			`,
			iters: 5,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			testRegistry := metrics.NewKubeRegistry()
			ms := testMetrics.Reset()
			testRegistry.MustRegister(ms...)

			schemaValidator, _, err := validation.NewSchemaValidator(&tt.schema)
			if err != nil {
				t.Fatal(err)
				return
			}
			sts, err := structuralschema.NewStructural(&tt.schema)
			if err != nil {
				t.Fatal(err)
			}
			gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Foo"}
			tt.obj.SetGroupVersionKind(gvk)
			tt.old.SetGroupVersionKind(gvk)
			strategy := customresource.NewStrategy(
				nil,
				true,
				gvk,
				schemaValidator,
				nil,
				sts,
				nil,
				nil,
				nil,
			)

			iters := 1
			if tt.iters > 0 {
				iters = tt.iters
			}
			for i := 0; i < iters; i++ {
				_ = strategy.ValidateUpdate(context.TODO(), tt.obj, tt.old)
			}

			if err := testutil.GatherAndCompare(testRegistry, strings.NewReader(tt.want), metricNames...); err != nil {
				t.Errorf("unexpected collecting result:\n%s", err)
			}

			// Ensure that the real durations is > 0 for all tests
			if testMetrics.realSum <= 0 {
				t.Errorf("realSum = %v, want > 0", testMetrics.realSum)
			}
		})
	}
}
