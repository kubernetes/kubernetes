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

package benchmark

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestScheduling(t *testing.T) {
	testCases, err := getTestCases(configFile)
	if err != nil {
		t.Fatal(err)
	}
	if err = validateTestCases(testCases); err != nil {
		t.Fatal(err)
	}

	// Check for leaks at the very end.
	framework.GoleakCheck(t)

	// All integration test cases share the same etcd, similar to
	// https://github.com/kubernetes/kubernetes/blob/18d05b646d09b2971dc5400bc288062b0414e8cf/test/integration/framework/etcd.go#L186-L222.
	framework.StartEtcd(t, nil)

	// Workloads with the same configuration share the same apiserver. For that
	// we first need to determine what those different configs are.
	var configs []schedulerConfig
	for _, tc := range testCases {
		tcEnabled := false
		for _, w := range tc.Workloads {
			if enabled(*testSchedulingLabelFilter, append(tc.Labels, w.Labels...)...) {
				tcEnabled = true
				break
			}
		}
		if !tcEnabled {
			continue
		}
		exists := false
		for _, config := range configs {
			if config.equals(tc) {
				exists = true
				break
			}
		}
		if !exists {
			configs = append(configs, schedulerConfig{schedulerConfigPath: tc.SchedulerConfigPath, featureGates: tc.FeatureGates})
		}
	}
	for _, config := range configs {
		// Not a sub test because we don't have a good name for it.
		func() {
			tCtx := ktesting.Init(t)

			// No timeout here because the `go test -timeout` will ensure that
			// the test doesn't get stuck forever.

			for feature, flag := range config.featureGates {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, flag)()
			}
			informerFactory, tCtx := setupClusterForWorkload(tCtx, config.schedulerConfigPath, config.featureGates, nil)

			for _, tc := range testCases {
				if !config.equals(tc) {
					// Runs with some other config.
					continue
				}

				t.Run(tc.Name, func(t *testing.T) {
					for _, w := range tc.Workloads {
						t.Run(w.Name, func(t *testing.T) {
							if !enabled(*testSchedulingLabelFilter, append(tc.Labels, w.Labels...)...) {
								t.Skipf("disabled by label filter %q", *testSchedulingLabelFilter)
							}
							tCtx := ktesting.WithTB(tCtx, t)
							runWorkload(tCtx, tc, w, informerFactory)
						})
					}
				})
			}
		}()
	}
}
