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

	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	_ "k8s.io/component-base/logs/json/register"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/features"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
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
			var options []perf.SchedulerPerfOption
			if allocatorName == "stable" {
				options = append(options, perf.WithPreRunFn(func(tCtx ktesting.TContext, _ *perf.Workload) (func(), error) {
					gate := utilfeature.DefaultFeatureGate.(featuregate.MutableVersionedFeatureGate)
					overrides := featuregatetesting.FeatureOverrides{
						features.DRAPrioritizedList: false,
					}
					// If version emulation already caused features to be off,
					// then we do not need and maybe even cannot turn them
					// off (pre-alpha = feature doesn't event exist).
					if gate.EmulationVersion().AtLeast(version.MustParse("1.34")) {
						overrides[features.DRAConsumableCapacity] = false
						overrides[features.DRADeviceBindingConditions] = false
					}
					featuregatetesting.SetFeatureGatesDuringTest(tCtx, utilfeature.DefaultFeatureGate, overrides)
					return nil, nil
				}))
			}

			perf.RunIntegrationPerfScheduling(t, "performance-config.yaml", options...)
		})
	}
}

// These benchmarks have to contain the word BenchmarkPerfScheduling to be picked up
// by benchmark jobs.
//
// Sub-tests should have worked, too, but gotestsum was unhappy.

func BenchmarkPerfScheduling(b *testing.B) {
	// Restrict benchmarking to the default allocator.
	structured.EnableAllocators("incubating")
	defer structured.EnableAllocators()

	// "dra" is how this was called traditionally.
	// It's kept to avoid changing perf-dash results.
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "dra", nil)
}

func BenchmarkPerfSchedulingExperimental(b *testing.B) {
	// And now the experimental allocator.
	structured.EnableAllocators("experimental")
	defer structured.EnableAllocators()

	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "dra_experimental", nil)
}
