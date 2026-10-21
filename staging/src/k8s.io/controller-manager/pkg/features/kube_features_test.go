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
)

func TestCloudNodeAdditionalLabelsReconciliationVersions(t *testing.T) {
	tests := []struct {
		name             string
		emulationVersion string
		flags            string
		wantEnabled      bool
		wantError        bool
	}{
		{
			name:             "disabled by default",
			emulationVersion: "1.38",
		},
		{
			name:             "can be enabled explicitly",
			emulationVersion: "1.38",
			flags:            "CloudNodeAdditionalLabelsReconciliation=true",
			wantEnabled:      true,
		},
		{
			name:             "can be disabled with AllAlpha enabled",
			emulationVersion: "1.38",
			flags:            "AllAlpha=true,CloudNodeAdditionalLabelsReconciliation=false",
		},
		{
			name:             "enabled by AllAlpha",
			emulationVersion: "1.38",
			flags:            "AllAlpha=true",
			wantEnabled:      true,
		},
		{
			name:             "disabled before introduction",
			emulationVersion: "1.37",
		},
		{
			name:             "AllAlpha does not enable it before introduction",
			emulationVersion: "1.37",
			flags:            "AllAlpha=true",
		},
		{
			name:             "cannot be enabled before introduction",
			emulationVersion: "1.37",
			flags:            "CloudNodeAdditionalLabelsReconciliation=true",
			wantError:        true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gate := featuregate.NewVersionedFeatureGate(version.MustParse(test.emulationVersion))
			if err := SetupCurrentKubernetesSpecificFeatureGates(gate); err != nil {
				t.Fatalf("registering controller manager feature gates: %v", err)
			}
			if err := gate.Set(test.flags); (err != nil) != test.wantError {
				t.Fatalf("setting feature gates: got error %v, want error %t", err, test.wantError)
			}
			if test.wantError {
				return
			}
			if got := gate.Enabled(CloudNodeAdditionalLabelsReconciliation); got != test.wantEnabled {
				t.Errorf("CloudNodeAdditionalLabelsReconciliation enabled = %t, want %t", got, test.wantEnabled)
			}
		})
	}
}
