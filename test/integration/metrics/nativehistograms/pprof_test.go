/*
Copyright The Kubernetes Authors.

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

package nativehistograms

import (
	"fmt"
	"net/http"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
	"time"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	metricsfeatures "k8s.io/component-base/metrics/features"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	apiservermetrics "k8s.io/apiserver/pkg/endpoints/metrics"
)

func TestProfileMemoryOverhead(t *testing.T) {
	// Look at the environment variable to determine if we should enable NativeHistograms.
	enableNH := os.Getenv("ENABLE_NATIVE_HISTOGRAMS") == "true"
	profileName := "heap_native_histograms_disabled.pprof"
	if enableNH {
		profileName = "heap_native_histograms_enabled.pprof"
	}

	t.Logf("Running profile memory test with NativeHistograms=%t, writing profile to %s", enableNH, profileName)

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, metricsfeatures.NativeHistograms, enableNH)

	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer s.TearDownFn()

	// Generate load to create many metric timeseries.
	// apiserver_request_duration_seconds has label values:
	// verb, dry_run, group, version, resource, subresource, scope, component
	// Let's create:
	// - 300 groups
	// - 15 resources per group
	// - 4 verbs per resource
	// - 3 scopes per resource
	// Total = 300 * 15 * 4 * 3 = 54000 unique timeseries vectors.
	// For each, we call MonitorRequest to record observations.

	t.Log("Generating metric observations...")
	req, _ := http.NewRequest(http.MethodGet, "https://127.0.0.1:443/apis/foo/v1/pods", nil)
	
	// Create some variety in the parameters
	verbs := []string{"GET", "LIST", "POST", "PUT"}
	scopes := []string{"resource", "namespace", "cluster"}
	latencies := []time.Duration{50 * time.Millisecond, 500 * time.Millisecond, 5000 * time.Millisecond}

	for g := 0; g < 300; g++ {
		group := fmt.Sprintf("group-%d", g)
		for r := 0; r < 15; r++ {
			resource := fmt.Sprintf("resource-%d", r)
			for _, verb := range verbs {
				for _, scope := range scopes {
					for _, latency := range latencies {
						// Call MonitorRequest
						apiservermetrics.MonitorRequest(
							req,
							verb,
							group,
							"v1",
							resource,
							"",
							scope,
							"apiserver",
							false,
							"",
							200,
							1024,
							latency,
						)
					}
				}
			}
		}
	}
	t.Log("Finished generating metric observations.")

	// Force Garbage Collection and wait for concurrent sweeps to complete to have a cleaner heap profile
	runtime.GC()
	time.Sleep(200 * time.Millisecond)
	runtime.GC()

	// Create output file for the heap profile in the current directory or workspace.
	outPath := fmt.Sprintf("/tmp/%s", profileName)
	f, err := os.Create(outPath)
	if err != nil {
		t.Fatalf("failed to create profile file: %v", err)
	}
	defer f.Close()

	if err := pprof.WriteHeapProfile(f); err != nil {
		t.Fatalf("failed to write heap profile: %v", err)
	}
	t.Logf("Successfully wrote heap profile to %s", outPath)
}
