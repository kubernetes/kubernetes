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

package features

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
)

func TestSchedulerFeatureGateDependencies(t *testing.T) {
	for _, tc := range []struct {
		name              string
		genericWorkload   bool
		topologyAware     bool
		compositePodGroup bool
		wantError         bool
	}{
		{name: "all disabled"},
		{name: "all enabled", genericWorkload: true, topologyAware: true, compositePodGroup: true},
		{name: "missing workload", topologyAware: true, compositePodGroup: true, wantError: true},
		{name: "missing topology", genericWorkload: true, compositePodGroup: true, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gate := featuregate.NewVersionedFeatureGate(version.MustParse("1.37"))
			if err := SetupCurrentKubernetesSpecificFeatureGates(gate); err != nil {
				t.Fatal(err)
			}
			err := gate.SetFromMapWithLogger(klog.Background(), map[string]bool{
				string(GenericWorkload):                 tc.genericWorkload,
				string(TopologyAwareWorkloadScheduling): tc.topologyAware,
				string(CompositePodGroup):               tc.compositePodGroup,
			})
			if (err != nil) != tc.wantError {
				t.Fatalf("setting feature gates returned %v, want error: %t", err, tc.wantError)
			}
		})
	}
}

func TestSchedulerFeatureGateSetupIsolation(t *testing.T) {
	const external featuregate.Feature = "ExternalFeature"
	gate := featuregate.NewFeatureGate()
	if err := gate.AddVersioned(map[featuregate.Feature]featuregate.VersionedSpecs{
		external: {{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gate.AddDependencies(map[featuregate.Feature][]featuregate.Feature{external: {}}); err != nil {
		t.Fatal(err)
	}
	for i := range 2 {
		if err := SetupCurrentKubernetesSpecificFeatureGates(gate); err != nil {
			t.Fatalf("registration %d failed: %v", i+1, err)
		}
	}
	if _, ok := gate.Dependencies()[external]; !ok {
		t.Fatal("scheduler registration removed the existing external dependency declaration")
	}

	// Registering into one gate must not leak its dependencies into subsequent gates.
	fresh := featuregate.NewFeatureGate()
	if err := SetupCurrentKubernetesSpecificFeatureGates(fresh); err != nil {
		t.Fatalf("registering into an independent gate failed: %v", err)
	}
	if _, ok := fresh.Dependencies()[external]; ok {
		t.Fatal("independent gate inherited the external dependency declaration")
	}
}
