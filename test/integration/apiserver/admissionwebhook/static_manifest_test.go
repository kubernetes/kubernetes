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

package admissionwebhook

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	admissionreviewv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission/plugin/manifest/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/manifest/source"
	"k8s.io/apiserver/pkg/features"
	genericapiserver "k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"sigs.k8s.io/yaml"
)

// TestStaticWebhookBlocksAPICreation tests that resources with .static.k8s.io suffix
// cannot be created via the REST API when the feature gate is enabled.
func TestStaticWebhookBlocksAPICreation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	ctx := t.Context()

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	expectStaticSuffixRejection := func(resourceType string, err error) {
		t.Helper()
		if err == nil {
			t.Fatalf("Expected error creating %s with .static.k8s.io suffix, got nil", resourceType)
		}
		if !apierrors.IsInvalid(err) {
			t.Fatalf("Expected Invalid error for %s, got: %v", resourceType, err)
		}
		if !strings.Contains(err.Error(), "reserved for static manifest-based configurations") {
			t.Errorf("Expected error about reserved static suffix for %s, got: %v", resourceType, err)
		}
	}

	// ValidatingWebhookConfiguration
	_, err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(
		ctx,
		&admissionregistrationv1.ValidatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: "test-webhook.static.k8s.io"},
			Webhooks: []admissionregistrationv1.ValidatingWebhook{{
				Name: "test.webhook.io",
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					URL: new("https://example.com/webhook"),
				},
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
				}},
				AdmissionReviewVersions: []string{"v1"},
				SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
			}},
		},
		metav1.CreateOptions{})
	expectStaticSuffixRejection("ValidatingWebhookConfiguration", err)

	// MutatingWebhookConfiguration
	_, err = client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(
		ctx,
		&admissionregistrationv1.MutatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: "test-mutating.static.k8s.io"},
			Webhooks: []admissionregistrationv1.MutatingWebhook{{
				Name: "test.mutating.io",
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					URL: new("https://example.com/webhook"),
				},
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
				}},
				AdmissionReviewVersions: []string{"v1"},
				SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
			}},
		},
		metav1.CreateOptions{})
	expectStaticSuffixRejection("MutatingWebhookConfiguration", err)
}

// TestStaticWebhookComprehensive exercises static webhook manifest loading using a
// single shared API server with combined ValidatingAdmissionWebhook and
// MutatingAdmissionWebhook admission config.
//
// Scenarios tested (in order):
//  1. Startup loading — VWC manifest pre-populated before server start, verified active immediately
//  2. Hot reload — MWC manifest added after server start, verified via polling
//  3. Both webhooks active simultaneously — single configmap create triggers both
//  4. v1.List format manifest — VWC loaded from a JSON v1.List file targeting secrets
//  5. Static webhook intercepts admission config resources — VWC creation denied
//  6. Coexistence with API-based webhook — static MWC and API-based VWC enforce side by side
//  7. Hot reload removal — VWC manifest deleted, validating webhook stops
//  8. Independent lifecycle — MWC still active after VWC removed
//
// Reload metrics (count, hash, timestamp, apiserver_id_hash) are validated after every
// subtest that triggers a manifest reload to confirm the reload was successful.
func TestStaticWebhookComprehensive(t *testing.T) {
	genericapiserver.SetHostnameFuncForTests("testAPIServerID")
	metrics.ResetMetricsForTest()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	resetInterval := source.SetReloadIntervalForTests(10 * time.Millisecond)
	defer resetInterval()
	ctx := t.Context()

	// Create webhook servers (validating: allows configmaps but denies VWC, mutating: adds label)
	validatingServer, validatingCACert, validatingRequestCount := createStaticTestWebhookServer(t)
	defer validatingServer.Close()
	mutatingServer, mutatingCACert := createStaticMutatingTestWebhookServer(t)
	defer mutatingServer.Close()

	// Override the validating handler to deny VWC creation but allow everything else
	validatingServer.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		validatingRequestCount.Add(1)
		body, _ := readStaticBody(r)
		var req admissionreviewv1.AdmissionReview
		_ = json.Unmarshal(body, &req)
		uid := types.UID("test")
		allowed := true
		message := ""
		if req.Request != nil {
			uid = req.Request.UID
			if req.Request.Resource.Resource == "validatingwebhookconfigurations" {
				allowed = false
				message = "denied by static webhook"
			}
		}
		review := &admissionreviewv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
			Response: &admissionreviewv1.AdmissionResponse{
				UID:     uid,
				Allowed: allowed,
			},
		}
		if !allowed {
			review.Response.Result = &metav1.Status{Message: message}
		}
		w.Header().Set("Content-Type", "application/json")
		respBytes, _ := json.Marshal(review)
		_, _ = w.Write(respBytes)
	})

	// Separate directories for validating and mutating webhook manifests
	validatingManifestsDir := t.TempDir()
	mutatingManifestsDir := t.TempDir()

	// Webhook configurations
	vwcConfig := &admissionregistrationv1.ValidatingWebhookConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingWebhookConfiguration",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "comprehensive-validating-webhook.static.k8s.io",
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "comprehensive-validating-webhook.static.k8s.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &validatingServer.URL,
				CABundle: validatingCACert,
			},
			Rules: []admissionregistrationv1.RuleWithOperations{
				{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
				},
				{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"admissionregistration.k8s.io"},
						APIVersions: []string{"v1"},
						Resources:   []string{"validatingwebhookconfigurations"},
					},
				},
			},
			AdmissionReviewVersions: []string{"v1"},
			SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			FailurePolicy:           new(admissionregistrationv1.Fail),
		}},
	}

	mwcConfig := &admissionregistrationv1.MutatingWebhookConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "MutatingWebhookConfiguration",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "comprehensive-mutating-webhook.static.k8s.io",
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name: "comprehensive-mutating-webhook.static.k8s.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &mutatingServer.URL,
				CABundle: mutatingCACert,
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
				Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
			}},
			AdmissionReviewVersions: []string{"v1"},
			SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			FailurePolicy:           new(admissionregistrationv1.Fail),
		}},
	}

	// Pre-populate VWC manifest before server start (tests startup loading path)
	validatingManifestPath := writeManifestToFile(t, validatingManifestsDir, "validating.yaml", vwcConfig)

	// Combined admission config for both ValidatingAdmissionWebhook and MutatingAdmissionWebhook
	admissionConfigFile := createCombinedWebhookAdmissionConfig(t, validatingManifestsDir, mutatingManifestsDir)

	// Start server with both plugins — VWC manifest already present
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(rest.CopyConfig(server.ClientConfig))
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	ns := metav1.NamespaceDefault

	// Track reload metrics across subtests for both plugins
	var lastVWHMetrics, lastMWHMetrics *reloadMetrics
	expectReloadCountIncrease := func(pluginName string, prev **reloadMetrics, expectNonEmptyHash bool) {
		t.Helper()
		var m *reloadMetrics
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			m = getReloadMetrics(t, client, pluginName)
			if *prev == nil {
				return m.reloadSuccessCount > 0, nil
			}
			// Wait for both count increase AND hash change
			return m.reloadSuccessCount > (*prev).reloadSuccessCount && m.configHash != (*prev).configHash, nil
		})
		if err != nil {
			var prevCount int
			if *prev != nil {
				prevCount = (*prev).reloadSuccessCount
			}
			t.Fatalf("Timeout waiting for %s reload count to increase (was %d, now %d)", pluginName, prevCount, m.reloadSuccessCount)
		}
		validateReloadMetrics(t, client, pluginName)
		if expectNonEmptyHash && len(m.configHash) == 0 {
			t.Errorf("Expected non-empty config hash for %s", pluginName)
		}
		*prev = m
	}

	t.Run("validating webhook active at startup", func(t *testing.T) {
		// VWC was pre-populated — should be active from the first request
		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "startup-probe-cm",
					Namespace: ns,
				},
				Data: map[string]string{"key": "value"},
			}
			_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
			if lastErr == nil {
				_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
			}
			return validatingRequestCount.Load() > 0, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for validating webhook to be invoked: %v (last error: %v)", err, lastErr)
		}

		expectReloadCountIncrease("ValidatingAdmissionWebhook", &lastVWHMetrics, true)
	})

	t.Run("hot reload mutating webhook", func(t *testing.T) {
		writeManifestToFile(t, mutatingManifestsDir, "mutating.yaml", mwcConfig)

		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mutating-reload-cm",
					Namespace: ns,
					Labels:    map[string]string{"app": "test"},
				},
				Data: map[string]string{"key": "value"},
			}
			created, createErr := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
			if createErr != nil {
				lastErr = createErr
				return false, nil
			}
			_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
			if val, ok := created.Labels["static-mutated"]; ok && val == "true" {
				return true, nil
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for mutating webhook to take effect: %v (last error: %v)", err, lastErr)
		}

		expectReloadCountIncrease("MutatingAdmissionWebhook", &lastMWHMetrics, true)
	})

	t.Run("both webhooks active simultaneously", func(t *testing.T) {
		countBefore := validatingRequestCount.Load()
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "both-active-cm",
				Namespace: ns,
				Labels:    map[string]string{"app": "test"},
			},
			Data: map[string]string{"key": "value"},
		}
		created, err := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create configmap: %v", err)
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})

		if validatingRequestCount.Load() <= countBefore {
			t.Error("Expected validating webhook to be called")
		}
		if val, ok := created.Labels["static-mutated"]; !ok || val != "true" {
			t.Errorf("Expected label static-mutated=true, got labels: %v", created.Labels)
		}
	})

	t.Run("v1.List format manifest", func(t *testing.T) {
		// Write a second VWC using v1.List format
		listVWC := &admissionregistrationv1.ValidatingWebhookConfiguration{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "admissionregistration.k8s.io/v1",
				Kind:       "ValidatingWebhookConfiguration",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "list-webhook.static.k8s.io",
			},
			Webhooks: []admissionregistrationv1.ValidatingWebhook{{
				Name: "list-webhook.static.k8s.io",
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					URL:      &validatingServer.URL,
					CABundle: validatingCACert,
				},
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"secrets"}},
				}},
				AdmissionReviewVersions: []string{"v1"},
				SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           new(admissionregistrationv1.Fail),
			}},
		}
		itemBytes, err := json.Marshal(listVWC)
		if err != nil {
			t.Fatalf("Failed to marshal list VWC: %v", err)
		}
		listJSON, err := json.Marshal(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "List",
			"items":      []json.RawMessage{itemBytes},
		})
		if err != nil {
			t.Fatalf("Failed to marshal v1.List: %v", err)
		}
		listPath := filepath.Join(validatingManifestsDir, "list-webhook.json")
		tmp := listPath + ".tmp"
		if err := os.WriteFile(tmp, listJSON, 0644); err != nil {
			t.Fatalf("Failed to write v1.List manifest: %v", err)
		}
		if err := os.Rename(tmp, listPath); err != nil {
			t.Fatalf("Failed to rename v1.List manifest: %v", err)
		}

		// Wait for the v1.List webhook to be active by checking that secrets trigger the validating server
		countBefore := validatingRequestCount.Load()
		var lastErr error
		err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.CoreV1().Secrets(ns).Create(ctx, &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "list-webhook-probe", Namespace: ns},
			}, metav1.CreateOptions{})
			if lastErr == nil {
				_ = client.CoreV1().Secrets(ns).Delete(ctx, "list-webhook-probe", metav1.DeleteOptions{})
			}
			return validatingRequestCount.Load() > countBefore, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for v1.List webhook to be active: %v (last error: %v)", err, lastErr)
		}

		expectReloadCountIncrease("ValidatingAdmissionWebhook", &lastVWHMetrics, true)

		// Clean up the list manifest so it doesn't interfere with later subtests
		if err := os.Remove(listPath); err != nil {
			t.Fatalf("Failed to remove list manifest: %v", err)
		}

		expectReloadCountIncrease("ValidatingAdmissionWebhook", &lastVWHMetrics, true)
	})

	t.Run("static webhook protects admission config resources", func(t *testing.T) {
		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			testURL := "https://localhost:12345/validate"
			_, lastErr = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(
				ctx,
				&admissionregistrationv1.ValidatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{Name: "test-intercepted"},
					Webhooks: []admissionregistrationv1.ValidatingWebhook{{
						Name: "test.webhook.io",
						ClientConfig: admissionregistrationv1.WebhookClientConfig{
							URL: &testURL,
						},
						AdmissionReviewVersions: []string{"v1"},
						SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
					}},
				},
				metav1.CreateOptions{},
			)
			if lastErr != nil && strings.Contains(lastErr.Error(), "denied") {
				return true, nil
			}
			if lastErr == nil {
				_ = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(ctx, "test-intercepted", metav1.DeleteOptions{})
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for static webhook to deny VWC creation: %v (last error: %v)", err, lastErr)
		}
	})

	t.Run("coexists with API-based webhook", func(t *testing.T) {
		// Remove the static validating manifest so we can create an API-based VWC
		if err := os.Remove(validatingManifestPath); err != nil {
			t.Fatalf("Failed to remove validating manifest: %v", err)
		}

		expectReloadCountIncrease("ValidatingAdmissionWebhook", &lastVWHMetrics, false)

		// Wait for the static VWC to be unloaded, then create API-based VWC
		apiWebhookServer, apiCACert, apiRequestCount := createStaticTestWebhookServer(t)
		defer apiWebhookServer.Close()

		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(ctx,
				&admissionregistrationv1.ValidatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{Name: "api-webhook"},
					Webhooks: []admissionregistrationv1.ValidatingWebhook{{
						Name: "api-webhook.example.com",
						ClientConfig: admissionregistrationv1.WebhookClientConfig{
							URL:      &apiWebhookServer.URL,
							CABundle: apiCACert,
						},
						Rules: []admissionregistrationv1.RuleWithOperations{{
							Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
							Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"secrets"}},
						}},
						AdmissionReviewVersions: []string{"v1"},
						SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
					}},
				}, metav1.CreateOptions{})
			return lastErr == nil, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting to create API-based VWC: %v (last error: %v)", err, lastErr)
		}

		// Wait for API webhook to be active
		err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.CoreV1().Secrets(ns).Create(ctx, &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "api-webhook-probe", Namespace: ns},
			}, metav1.CreateOptions{})
			if lastErr == nil {
				_ = client.CoreV1().Secrets(ns).Delete(ctx, "api-webhook-probe", metav1.DeleteOptions{})
			}
			return apiRequestCount.Load() > 0, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for API webhook to be invoked: %v (last error: %v)", err, lastErr)
		}

		// Verify static mutating webhook still works on configmaps
		cm, err := client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "coexist-check-cm", Namespace: ns, Labels: map[string]string{"app": "test"}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create configmap: %v", err)
		}
		if cm.Labels == nil || cm.Labels["static-mutated"] != "true" {
			t.Errorf("Expected label static-mutated=true, got labels: %v", cm.Labels)
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})

		// Re-add the validating manifest for subsequent subtests
		validatingManifestPath = writeManifestToFile(t, validatingManifestsDir, "validating.yaml", vwcConfig)
		// Wait for it to reload
		err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			countBefore := validatingRequestCount.Load()
			_, _ = client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "reload-check-cm", Namespace: ns, Labels: map[string]string{"app": "test"}},
			}, metav1.CreateOptions{})
			_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, "reload-check-cm", metav1.DeleteOptions{})
			return validatingRequestCount.Load() > countBefore, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for static validating webhook to reload: %v", err)
		}

		expectReloadCountIncrease("ValidatingAdmissionWebhook", &lastVWHMetrics, true)

		// Clean up API webhook
		_ = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(ctx, "api-webhook", metav1.DeleteOptions{})
	})

	t.Run("hot reload removal stops validating webhook", func(t *testing.T) {
		if err := os.Remove(validatingManifestPath); err != nil {
			t.Fatalf("Failed to remove validating manifest: %v", err)
		}

		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			countBefore := validatingRequestCount.Load()
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "post-removal-cm",
					Namespace: ns,
					Labels:    map[string]string{"app": "test"},
				},
				Data: map[string]string{"key": "value"},
			}
			_, err := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
			if err != nil {
				return false, nil
			}
			_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
			return validatingRequestCount.Load() == countBefore, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for validating webhook to stop after removal: %v", err)
		}

		expectReloadCountIncrease("ValidatingAdmissionWebhook", &lastVWHMetrics, false)
	})

	t.Run("mutating webhook still active after validating removed", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "still-mutated-cm",
				Namespace: ns,
				Labels:    map[string]string{"app": "test"},
			},
			Data: map[string]string{"key": "value"},
		}
		created, err := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create configmap: %v", err)
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})

		if val, ok := created.Labels["static-mutated"]; !ok || val != "true" {
			t.Errorf("Expected label static-mutated=true after validating removal, got labels: %v", created.Labels)
		}
	})
}

// It returns the server, the CA certificate, and a counter that tracks how many
// admission requests the webhook has received.
func createStaticTestWebhookServer(t *testing.T) (*httptest.Server, []byte, *atomic.Int64) {
	// Use the test certificates already defined in this package
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to create key pair: %v", err)
	}

	var requestCount atomic.Int64

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		// Allow all requests
		review := &admissionreviewv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "admission.k8s.io/v1",
				Kind:       "AdmissionReview",
			},
			Response: &admissionreviewv1.AdmissionResponse{
				UID:     "test",
				Allowed: true,
			},
		}

		// Parse the request to get the UID
		body, _ := readStaticBody(r)
		if len(body) > 0 {
			var req admissionreviewv1.AdmissionReview
			if err := json.Unmarshal(body, &req); err == nil && req.Request != nil {
				review.Response.UID = req.Request.UID
			}
		}

		w.Header().Set("Content-Type", "application/json")
		respBytes, _ := json.Marshal(review)
		_, _ = w.Write(respBytes)
	})

	server := httptest.NewUnstartedServer(handler)
	server.TLS = &tls.Config{
		Certificates: []tls.Certificate{cert},
	}
	server.StartTLS()

	return server, localhostCert, &requestCount
}

func createCombinedWebhookAdmissionConfig(t *testing.T, validatingManifestsDir, mutatingManifestsDir string) string {
	t.Helper()
	admissionConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ValidatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    staticManifestsDir: %q
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    staticManifestsDir: %q
`, validatingManifestsDir, mutatingManifestsDir)
	configFile, err := os.CreateTemp("", "admission-config-*.yaml")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	t.Cleanup(func() { _ = os.Remove(configFile.Name()) })
	if _, err := configFile.WriteString(admissionConfig); err != nil {
		t.Fatalf("Failed to write config: %v", err)
	}
	if err := configFile.Close(); err != nil {
		t.Fatalf("Failed to close config file: %v", err)
	}
	return configFile.Name()
}

// createStaticMutatingTestWebhookServer creates an HTTPS webhook server that returns a JSON patch
// adding the label static-mutated=true to the resource.
// It returns the server and the CA certificate.
func createStaticMutatingTestWebhookServer(t *testing.T) (*httptest.Server, []byte) {
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to create key pair: %v", err)
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := readStaticBody(r)
		var req admissionreviewv1.AdmissionReview
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		patch := `[{"op":"add","path":"/metadata/labels/static-mutated","value":"true"}]`
		patchType := admissionreviewv1.PatchTypeJSONPatch
		review := &admissionreviewv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "admission.k8s.io/v1",
				Kind:       "AdmissionReview",
			},
			Response: &admissionreviewv1.AdmissionResponse{
				UID:       req.Request.UID,
				Allowed:   true,
				Patch:     []byte(patch),
				PatchType: &patchType,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		respBytes, _ := json.Marshal(review)
		_, _ = w.Write(respBytes)
	})

	server := httptest.NewUnstartedServer(handler)
	server.TLS = &tls.Config{
		Certificates: []tls.Certificate{cert},
	}
	server.StartTLS()

	return server, localhostCert
}

func readStaticBody(r *http.Request) ([]byte, error) {
	defer func() { _ = r.Body.Close() }()
	return io.ReadAll(r.Body)
}

// writeManifestToFile marshals the given objects as a multi-document YAML file.
// Uses atomic write (write to tmp file, then rename) to avoid partial reads.
func writeManifestToFile(t *testing.T, dir, filename string, objects ...interface{}) string {
	t.Helper()
	var parts []string
	for _, obj := range objects {
		bytes, err := yaml.Marshal(obj)
		if err != nil {
			t.Fatalf("Failed to marshal manifest object: %v", err)
		}
		parts = append(parts, string(bytes))
	}
	path := filepath.Join(dir, filename)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(strings.Join(parts, "\n---\n")), 0644); err != nil {
		t.Fatalf("Failed to write manifest file %s: %v", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		t.Fatalf("Failed to rename manifest file %s -> %s: %v", tmp, path, err)
	}
	return path
}

// reloadMetrics holds parsed reload metrics from /metrics endpoint.
type reloadMetrics struct {
	reloadSuccessCount int
	reloadSuccessTime  *time.Time
	configHash         string
	apiServerIDHash    string
}

var (
	reloadCountPrefix     = "apiserver_manifest_admission_config_controller_automatic_reloads_total{"
	reloadTimestampPrefix = "apiserver_manifest_admission_config_controller_automatic_reload_last_timestamp_seconds{"
	configInfoPrefix      = "apiserver_manifest_admission_config_controller_last_config_info{"
	labelValueRe          = regexp.MustCompile(`(\w+)="([^"]*)"`)
)

func validateReloadMetrics(t *testing.T, client clientset.Interface, pluginName string) *reloadMetrics {
	t.Helper()
	m := getReloadMetrics(t, client, pluginName)
	if m.reloadSuccessCount == 0 {
		t.Error("Expected reload success count > 0 after hot-reload")
	}
	if m.reloadSuccessTime == nil {
		t.Error("Expected reload success timestamp to be set after hot-reload")
	}
	if m.apiServerIDHash != "sha256:3c607df3b2bf22c9d9f01d5314b4bbf411c48ef43ff44ff29b1d55b41367c795" {
		t.Errorf("Expected apiserver_id_hash %q, got %q", "sha256:3c607df3b2bf22c9d9f01d5314b4bbf411c48ef43ff44ff29b1d55b41367c795", m.apiServerIDHash)
	}
	return m
}

func getReloadMetrics(t *testing.T, client clientset.Interface, pluginName string) *reloadMetrics {
	t.Helper()
	data, err := client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(t.Context())
	if err != nil {
		t.Fatalf("Failed to fetch metrics: %v", err)
	}

	m := &reloadMetrics{}
	for line := range strings.SplitSeq(string(data), "\n") {
		labels := parseLabels(line)
		if strings.HasPrefix(line, reloadCountPrefix) && labels["plugin"] == pluginName && labels["status"] == "success" {
			m.apiServerIDHash = labels["apiserver_id_hash"]
			parts := strings.Fields(line)
			if len(parts) > 0 {
				m.reloadSuccessCount, _ = strconv.Atoi(parts[len(parts)-1])
			}
		}
		if strings.HasPrefix(line, reloadTimestampPrefix) && labels["plugin"] == pluginName && labels["status"] == "success" {
			m.apiServerIDHash = labels["apiserver_id_hash"]
			parts := strings.Fields(line)
			if len(parts) > 0 {
				value, _ := strconv.ParseFloat(parts[len(parts)-1], 64)
				seconds := int64(value)
				nanoseconds := int64((value - float64(seconds)) * 1000000000)
				tm := time.Unix(seconds, nanoseconds)
				m.reloadSuccessTime = &tm
			}
		}
		if strings.HasPrefix(line, configInfoPrefix) && labels["plugin"] == pluginName {
			m.apiServerIDHash = labels["apiserver_id_hash"]
			m.configHash = labels["hash"]
		}
	}
	return m
}

func parseLabels(line string) map[string]string {
	labels := make(map[string]string)
	for _, match := range labelValueRe.FindAllStringSubmatch(line, -1) {
		if len(match) == 3 {
			labels[match[1]] = match[2]
		}
	}
	return labels
}
