/*
Copyright 2026 The Kubernetes Authors.

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

package app

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/features"
)

func TestApplyFeatureGatesEnablesNativeHistograms(t *testing.T) {
	// Restore the metrics package flag after the feature gate itself is restored.
	t.Cleanup(func() {
		features.ApplyFeatureGates(utilfeature.DefaultFeatureGate)
	})
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NativeHistograms, true)
	features.ApplyFeatureGates(utilfeature.DefaultFeatureGate)

	reg := metrics.NewKubeRegistry()
	hist := metrics.NewHistogram(&metrics.HistogramOpts{
		Namespace:      "cloud_controller_manager",
		Name:           "test_native_histogram_gate_seconds",
		Help:           "histogram created after ApplyFeatureGates",
		Buckets:        []float64{0.1, 1, 5},
		StabilityLevel: metrics.ALPHA,
	})
	reg.MustRegister(hist)
	hist.Observe(0.2)

	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	if len(families) != 1 || len(families[0].GetMetric()) != 1 {
		t.Fatalf("unexpected gather result: %d families", len(families))
	}
	got := families[0].GetMetric()[0].GetHistogram()
	if got.GetSchema() == 0 {
		t.Fatal("histogram stayed classic after NativeHistograms was applied")
	}
}
