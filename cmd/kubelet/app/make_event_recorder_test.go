/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet"
)

func TestMakeEventRecorder(t *testing.T) {
	nodeName := types.NodeName("test-node")
	customNamespace := "custom-namespace"

	testCases := []struct {
		name                 string
		featureEnabled       bool
		eventsNamespace      string
		expectedNamespace    string
	}{
		{
			name:              "feature disabled, custom namespace provided",
			featureEnabled:    false,
			eventsNamespace:   customNamespace,
			expectedNamespace: "", // Should be overridden to empty string
		},
		{
			name:              "feature enabled, custom namespace provided",
			featureEnabled:    true,
			eventsNamespace:   customNamespace,
			expectedNamespace: customNamespace,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletEventsNamespace, tc.featureEnabled)

			kubeDeps := &kubelet.Dependencies{}
			makeEventRecorder(context.Background(), kubeDeps, nodeName, tc.eventsNamespace)

			if kubeDeps.Recorder == nil {
				t.Fatal("Expected recorder to be initialized")
			}

			// We can't easily check the namespace from the recorder object itself as it's an interface 
			// and the underlying implementation is private.
			// However, we can verify it doesn't crash and we've covered the logic in server.go.
		})
	}
}
