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

package dra

import (
	"fmt"
	"os"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	_ "k8s.io/component-base/logs/json/register"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/features"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
)

func TestMain(m *testing.M) {
	if err := perf.InitTests(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	m.Run()
}

func TestSchedulerPerf(t *testing.T) {
	// Verify correct behavior with all available allocators.
	for _, allocatorName := range []string{"stable", "incubating", "experimental"} {
		t.Run(allocatorName, func(t *testing.T) {
			structured.EnableAllocators(allocatorName)
			defer structured.EnableAllocators()

			// In order to run with the "stable" implementation, we have to disable
			// some features, something that isn't specified in the YAML
			// configuration because for other implementations we want the default
			// features. Using "AllAlpha" and "AllBeta" would be better here,
			// but interacts poorly with setting the emulated version to 1.33 later
			// on ("scheduler_perf.go:1117: failed to set emulation version to 1.33 during test:
			// cannot set feature gate NominatedNodeNameForExpectation to false, feature is PreAlpha at emulated version 1.33, ...")
			//
			// Once the current "incubating" becomes "stable", this can be replaced
			// with two sub tests:
			// - "ga-only": keep disabling optional features
			// - "default": don't change features
			if allocatorName == "stable" {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.DRAAdminAccess:     false,
					features.DRAPrioritizedList: false,
				})
			}

			perf.RunIntegrationPerfScheduling(t, "performance-config.yaml")
		})
	}
}

func BenchmarkPerfScheduling(b *testing.B) {
	// Restrict benchmarking to the default allocator.
	structured.EnableAllocators("incubating")
	defer structured.EnableAllocators()

	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "dra", nil)
}
