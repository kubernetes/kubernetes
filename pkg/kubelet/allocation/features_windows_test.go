//go:build windows

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

package allocation

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestIsInPlacePodVerticalScalingAllowedWindows(t *testing.T) {
	tests := []struct {
		name        string
		windowsGate bool
		wantAllowed bool
		wantReason  string
	}{
		{name: "Windows gate disabled", windowsGate: false, wantReason: "windows_gate_off"},
		{name: "Windows gate enabled", windowsGate: true, wantAllowed: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsInPlacePodResize, tt.windowsGate)

			allowed, _, reason := IsInPlacePodVerticalScalingAllowed(&v1.Pod{})
			assert.Equal(t, tt.wantAllowed, allowed)
			assert.Equal(t, tt.wantReason, reason)
		})
	}
}
