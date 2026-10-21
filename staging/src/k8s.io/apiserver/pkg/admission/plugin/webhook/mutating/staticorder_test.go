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

package mutating

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	webhooktesting "k8s.io/apiserver/pkg/admission/plugin/webhook/testing"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
)

// TestStaticOrderAccepted verifies that the MutatingAdmissionWebhook plugin accepts
// every supported merge order.
func TestStaticOrderAccepted(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	manifestsDir := t.TempDir()

	for _, staticOrder := range []webhookadmission.StaticOrder{"", webhookadmission.StaticFirst, webhookadmission.StaticLast, webhookadmission.ByName} {
		t.Run(string(staticOrder), func(t *testing.T) {
			configFile := fmt.Sprintf(`
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: %q
staticOrder: %q
`, manifestsDir, staticOrder)

			p, err := NewMutatingWebhook(bytes.NewBufferString(configFile))
			if err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
			want := staticOrder
			if len(want) == 0 {
				want = webhookadmission.StaticFirst
			}
			if got := p.Webhook.GetStaticOrder(); got != want {
				t.Fatalf("expected StaticOrder %q, got %q", want, got)
			}
		})
	}
}

// TestStaticOrderStartupLog verifies that the effective merge order is logged at startup
// only when manifest-based webhooks are configured. Without staticManifestsDir there is
// nothing to merge, so the line would be noise on every API server.
func TestStaticOrderStartupLog(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	manifestsDir := t.TempDir()

	testCases := []struct {
		name       string
		configFile io.Reader
		wantLog    []string
	}{
		{
			name: "no staticManifestsDir",
		},
		{
			name: "staticManifestsDir with staticOrder",
			configFile: bytes.NewBufferString(fmt.Sprintf(`
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: %q
staticOrder: ByName
`, manifestsDir)),
			wantLog: []string{
				`"Configured mutating webhook merge order"`,
				`plugin="MutatingAdmissionWebhook"`,
				fmt.Sprintf(`staticManifestsDir=%q`, manifestsDir),
				`staticOrder="ByName"`,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			klog.LogToStderr(false)
			t.Cleanup(func() {
				klog.SetOutput(io.Discard)
				klog.LogToStderr(true)
			})

			stopCh := make(chan struct{})
			t.Cleanup(func() { close(stopCh) })

			wh, err := NewMutatingWebhook(tc.configFile)
			if err != nil {
				t.Fatalf("failed to create mutating webhook: %v", err)
			}
			client, informer := webhooktesting.NewFakeMutatingDataSource("default", nil, stopCh)
			wh.SetExternalKubeClientSet(client)
			wh.SetExternalKubeInformerFactory(informer)
			wh.SetDrainedNotification(stopCh)
			wh.SetStaticSourceFactory(func(string) (generic.ReloadableSource, error) {
				return emptyReloadableSource{}, nil
			})
			if err := wh.ValidateInitialization(); err != nil {
				t.Fatalf("failed to validate initialization: %v", err)
			}
			klog.Flush()

			got := buf.String()
			if len(tc.wantLog) == 0 {
				if strings.Contains(got, "merge order") {
					t.Fatalf("expected no merge order log line, got:\n%s", got)
				}
				return
			}
			for _, want := range tc.wantLog {
				if !strings.Contains(got, want) {
					t.Errorf("expected log output to contain %s, got:\n%s", want, got)
				}
			}
		})
	}
}

// emptyReloadableSource is a manifest-based webhook source with no webhooks.
type emptyReloadableSource struct{}

func (emptyReloadableSource) Webhooks() []webhook.WebhookAccessor { return nil }
func (emptyReloadableSource) HasSynced() bool                     { return true }
func (emptyReloadableSource) RunReloadLoop(ctx context.Context)   { <-ctx.Done() }
