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

package validating

import (
	"bytes"
	"fmt"
	"testing"

	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// TestStaticOrderRejected verifies that the ValidatingAdmissionWebhook plugin only
// accepts the default merge order, since validating webhooks are dispatched in parallel.
func TestStaticOrderRejected(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	manifestsDir := t.TempDir()

	testcases := []struct {
		name        string
		staticOrder string
		expectErr   string
	}{
		{
			name:        "unset",
			staticOrder: "",
		},
		{
			name:        "StaticFirst",
			staticOrder: "StaticFirst",
		},
		{
			name:        "StaticLast",
			staticOrder: "StaticLast",
			expectErr:   `staticOrder: Invalid value: "StaticLast": only "StaticFirst" is supported by ValidatingAdmissionWebhook`,
		},
		{
			name:        "ByName",
			staticOrder: "ByName",
			expectErr:   `staticOrder: Invalid value: "ByName": only "StaticFirst" is supported by ValidatingAdmissionWebhook`,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			configFile := fmt.Sprintf(`
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: %q
staticOrder: %q
`, manifestsDir, tc.staticOrder)

			_, err := NewValidatingAdmissionWebhook(bytes.NewBufferString(configFile))
			if len(tc.expectErr) == 0 {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error %q, got none", tc.expectErr)
			}
			if err.Error() != tc.expectErr {
				t.Fatalf("expected error %q, got %q", tc.expectErr, err.Error())
			}
		})
	}
}
