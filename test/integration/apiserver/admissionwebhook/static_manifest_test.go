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
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/manifest/source"
	"k8s.io/apiserver/pkg/features"
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

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Try to create a ValidatingWebhookConfiguration with .static.k8s.io suffix
	webhookConfig := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-webhook.static.k8s.io",
		},
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
	}

	_, err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(
		context.Background(), webhookConfig, metav1.CreateOptions{})

	if err == nil {
		t.Fatal("Expected error when creating webhook with .static.k8s.io suffix, got nil")
	}

	if !apierrors.IsInvalid(err) {
		t.Fatalf("Expected Invalid error, got: %v", err)
	}

	// Also test MutatingWebhookConfiguration
	mutatingConfig := &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-mutating.static.k8s.io",
		},
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
	}

	_, err = client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(
		context.Background(), mutatingConfig, metav1.CreateOptions{})

	if err == nil {
		t.Fatal("Expected error when creating mutating webhook with .static.k8s.io suffix, got nil")
	}

	if !apierrors.IsInvalid(err) {
		t.Fatalf("Expected Invalid error, got: %v", err)
	}
}

// TestStaticWebhookManifestLoading tests that webhook configurations can be loaded
// from static manifest files at API server startup.
func TestStaticWebhookManifestLoading(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Create a test webhook server
	webhookServer, caCert, requestCount := createStaticTestWebhookServer(t)
	defer webhookServer.Close()

	// Create static manifests directory
	manifestsDir := t.TempDir()

	// Create a validating webhook configuration manifest
	webhookConfig := &admissionregistrationv1.ValidatingWebhookConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingWebhookConfiguration",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "static-webhook.static.k8s.io",
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "static-webhook.static.k8s.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &webhookServer.URL,
				CABundle: caCert,
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
				Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
			}},
			AdmissionReviewVersions: []string{"v1"},
			SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
			// The manifest loader applies defaults (selectors, failurePolicy, etc.)
			// but we set them explicitly here for test clarity.
			NamespaceSelector: &metav1.LabelSelector{},
			ObjectSelector:    &metav1.LabelSelector{},
			FailurePolicy:     new(admissionregistrationv1.Fail),
		}},
	}

	// Write the manifest file
	manifestPath := filepath.Join(manifestsDir, "webhook.yaml")
	manifestBytes, err := yaml.Marshal(webhookConfig)
	if err != nil {
		t.Fatalf("Failed to marshal webhook config: %v", err)
	}
	if err := os.WriteFile(manifestPath, manifestBytes, 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// Create admission config file pointing to the static manifests directory
	admissionConfigFile := createStaticAdmissionConfig(t, manifestsDir)

	// Start the API server with the feature gate enabled
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// Create a client
	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Test that the webhook is called when creating a configmap
	ns := "default"

	// Create a configmap - the webhook should be invoked
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: ns,
		},
		Data: map[string]string{"key": "value"},
	}

	// Wait for the webhook to be ready (static sources should be ready at startup)
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
		if lastErr != nil {
			// If we get a webhook error, the webhook is working
			if apierrors.IsInternalError(lastErr) || apierrors.IsForbidden(lastErr) {
				return true, nil
			}
			// Other errors might be transient
			return false, nil
		}
		// Success means webhook allowed it
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for webhook to be ready: %v (last error: %v)", err, lastErr)
	}

	// Verify the webhook handler was actually invoked
	if requestCount.Load() == 0 {
		t.Fatal("Webhook handler was never invoked - static manifest loading may not be working")
	}
}

// TestStaticWebhookHotReload tests that adding a new manifest file while the server is running
// causes the new webhook to be loaded and enforced without restarting the API server.
func TestStaticWebhookHotReload(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	resetInterval := source.SetReloadIntervalForTests(10 * time.Millisecond)
	defer resetInterval()

	// Create a test webhook server
	webhookServer, caCert, requestCount := createStaticTestWebhookServer(t)
	defer webhookServer.Close()

	// Start with an empty manifests directory
	manifestsDir := t.TempDir()

	// Create admission config file pointing to the static manifests directory
	admissionConfigFile := createStaticAdmissionConfig(t, manifestsDir)

	// Start the API server
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	ns := "default"

	// Verify webhook is NOT invoked initially (empty directory)
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pre-reload-configmap",
			Namespace: ns,
		},
		Data: map[string]string{"key": "value"},
	}
	if _, err := client.CoreV1().ConfigMaps(ns).Create(context.Background(), cm, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Expected configmap creation to succeed before webhook is loaded: %v", err)
	}
	_ = client.CoreV1().ConfigMaps(ns).Delete(context.Background(), cm.Name, metav1.DeleteOptions{})
	if requestCount.Load() != 0 {
		t.Fatal("Expected webhook not to be invoked before hot-reload")
	}

	// Now write a new webhook manifest to the directory (hot-reload)
	webhookConfig := &admissionregistrationv1.ValidatingWebhookConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingWebhookConfiguration",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "hot-reload-webhook.static.k8s.io",
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "hot-reload-webhook.static.k8s.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &webhookServer.URL,
				CABundle: caCert,
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
	manifestBytes, err := yaml.Marshal(webhookConfig)
	if err != nil {
		t.Fatalf("Failed to marshal webhook config: %v", err)
	}
	manifestPath := filepath.Join(manifestsDir, "hot-reload-webhook.yaml")
	if err := os.WriteFile(manifestPath, manifestBytes, 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// Wait for the hot-reload to take effect by polling until the webhook is invoked
	postReloadCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "post-reload-configmap",
			Namespace: ns,
		},
		Data: map[string]string{"key": "value"},
	}
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, postReloadCM, metav1.CreateOptions{})
		if lastErr == nil {
			// Webhook allowed it — check if it was actually invoked
			_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, postReloadCM.Name, metav1.DeleteOptions{})
			if requestCount.Load() > 0 {
				return true, nil
			}
			return false, nil
		}
		// Transient error — keep polling
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for hot-reload webhook to be invoked: %v (last error: %v)", err, lastErr)
	}
	if requestCount.Load() == 0 {
		t.Fatal("Hot-reloaded webhook was never invoked")
	}

	// Validate reload metrics after adding manifest
	addMetrics := getReloadMetrics(t, client, "ValidatingAdmissionWebhook")
	if addMetrics.reloadSuccessCount == 0 {
		t.Error("Expected reload success count > 0 after hot-reload")
	}
	if addMetrics.reloadSuccessTime == nil {
		t.Error("Expected reload success timestamp to be set after hot-reload")
	}
	if len(addMetrics.configHash) == 0 {
		t.Error("Expected config hash to be set after hot-reload")
	}
	addHash := addMetrics.configHash

	// Now delete the manifest file — webhook should stop being invoked
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("Failed to remove manifest file: %v", err)
	}

	// Wait for the deletion to take effect
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		requestCount.Store(0)
		deletionCM := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "post-deletion-configmap",
				Namespace: ns,
			},
			Data: map[string]string{"key": "value"},
		}
		_, err := client.CoreV1().ConfigMaps(ns).Create(ctx, deletionCM, metav1.CreateOptions{})
		if err != nil {
			return false, nil
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, deletionCM.Name, metav1.DeleteOptions{})
		// If webhook is no longer invoked, requestCount stays 0
		return requestCount.Load() == 0, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for webhook to stop being invoked after file deletion: %v", err)
	}

	// Validate reload metrics after deleting manifest
	deleteMetrics := getReloadMetrics(t, client, "ValidatingAdmissionWebhook")
	if deleteMetrics.reloadSuccessCount <= addMetrics.reloadSuccessCount {
		t.Errorf("Expected reload success count to increase after deletion, got %d (was %d)", deleteMetrics.reloadSuccessCount, addMetrics.reloadSuccessCount)
	}
	if deleteMetrics.configHash == addHash {
		t.Error("Expected config hash to change after manifest deletion")
	}
}

// TestStaticMutatingWebhookManifestLoading tests that mutating webhook configurations can be loaded
// from static manifest files at API server startup.
func TestStaticMutatingWebhookManifestLoading(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Create a test mutating webhook server
	webhookServer, caCert := createStaticMutatingTestWebhookServer(t)
	defer webhookServer.Close()

	// Create static manifests directory
	manifestsDir := t.TempDir()

	// Create a mutating webhook configuration manifest
	webhookConfig := &admissionregistrationv1.MutatingWebhookConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "MutatingWebhookConfiguration",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "static-mutating-webhook.static.k8s.io",
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name: "static-mutating-webhook.static.k8s.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &webhookServer.URL,
				CABundle: caCert,
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

	// Write the manifest file
	manifestPath := filepath.Join(manifestsDir, "mutating-webhook.yaml")
	manifestBytes, err := yaml.Marshal(webhookConfig)
	if err != nil {
		t.Fatalf("Failed to marshal webhook config: %v", err)
	}
	if err := os.WriteFile(manifestPath, manifestBytes, 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// Create admission config file pointing to the static manifests directory
	admissionConfigFile := createStaticMutatingAdmissionConfig(t, manifestsDir)

	// Start the API server with the feature gate enabled
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// Create a client
	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	ns := "default"

	// Create a configmap - the mutating webhook should add a label
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-mutating-configmap",
			Namespace: ns,
			Labels:    map[string]string{"app": "test"},
		},
		Data: map[string]string{"key": "value"},
	}

	// Wait for the webhook to be ready and verify the mutation is applied
	var createdCM *corev1.ConfigMap
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		createdCM, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
		if lastErr != nil {
			return false, nil
		}
		// Check if the mutation was applied
		if val, ok := createdCM.Labels["static-mutated"]; ok && val == "true" {
			return true, nil
		}
		// Webhook might not be ready yet, clean up and retry
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for mutating webhook to be ready: %v (last error: %v)", err, lastErr)
	}

	// Verify the label was added by the webhook
	if val, ok := createdCM.Labels["static-mutated"]; !ok || val != "true" {
		t.Fatalf("Expected label static-mutated=true on configmap, got labels: %v", createdCM.Labels)
	}
}

// TestStaticMutatingWebhookHotReload tests that adding a mutating webhook manifest file while the
// server is running causes the new webhook to be loaded and enforced without restarting the API server.
func TestStaticMutatingWebhookHotReload(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	resetInterval := source.SetReloadIntervalForTests(10 * time.Millisecond)
	defer resetInterval()

	// Create a test mutating webhook server
	webhookServer, caCert := createStaticMutatingTestWebhookServer(t)
	defer webhookServer.Close()

	// Start with an empty manifests directory
	manifestsDir := t.TempDir()

	// Create admission config file pointing to the static manifests directory
	admissionConfigFile := createStaticMutatingAdmissionConfig(t, manifestsDir)

	// Start the API server
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	ns := "default"

	// Verify configmaps are NOT mutated initially (empty directory)
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pre-reload-mutating-configmap",
			Namespace: ns,
			Labels:    map[string]string{"app": "test"},
		},
		Data: map[string]string{"key": "value"},
	}
	createdCM, err := client.CoreV1().ConfigMaps(ns).Create(context.Background(), cm, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected configmap creation to succeed before webhook is loaded: %v", err)
	}
	if _, ok := createdCM.Labels["static-mutated"]; ok {
		t.Fatal("Expected configmap NOT to be mutated before webhook is loaded")
	}
	_ = client.CoreV1().ConfigMaps(ns).Delete(context.Background(), cm.Name, metav1.DeleteOptions{})

	// Now write a new mutating webhook manifest to the directory (hot-reload)
	webhookConfig := &admissionregistrationv1.MutatingWebhookConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "MutatingWebhookConfiguration",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "hot-reload-mutating-webhook.static.k8s.io",
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name: "hot-reload-mutating-webhook.static.k8s.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &webhookServer.URL,
				CABundle: caCert,
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
	manifestBytes, err := yaml.Marshal(webhookConfig)
	if err != nil {
		t.Fatalf("Failed to marshal webhook config: %v", err)
	}
	manifestPath := filepath.Join(manifestsDir, "hot-reload-mutating-webhook.yaml")
	if err := os.WriteFile(manifestPath, manifestBytes, 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// Wait for the hot-reload to take effect by polling until the configmap is mutated
	postReloadCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "post-reload-mutating-configmap",
			Namespace: ns,
			Labels:    map[string]string{"app": "test"},
		},
		Data: map[string]string{"key": "value"},
	}
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		created, createErr := client.CoreV1().ConfigMaps(ns).Create(ctx, postReloadCM, metav1.CreateOptions{})
		if createErr != nil {
			lastErr = createErr
			return false, nil
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, postReloadCM.Name, metav1.DeleteOptions{})
		if val, ok := created.Labels["static-mutated"]; ok && val == "true" {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for hot-reload mutating webhook to take effect: %v (last error: %v)", err, lastErr)
	}

	// Validate reload metrics after adding manifest
	addMetrics := getReloadMetrics(t, client, "MutatingAdmissionWebhook")
	if addMetrics.reloadSuccessCount == 0 {
		t.Error("Expected reload success count > 0 after hot-reload")
	}
	if addMetrics.reloadSuccessTime == nil {
		t.Error("Expected reload success timestamp to be set after hot-reload")
	}
	if len(addMetrics.configHash) == 0 {
		t.Error("Expected config hash to be set after hot-reload")
	}
	addHash := addMetrics.configHash

	// Now delete the manifest file — webhook should stop mutating
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("Failed to remove manifest file: %v", err)
	}

	// Wait for the deletion to take effect
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		deletionCM := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "post-deletion-mutating-configmap",
				Namespace: ns,
				Labels:    map[string]string{"app": "test"},
			},
			Data: map[string]string{"key": "value"},
		}
		created, createErr := client.CoreV1().ConfigMaps(ns).Create(ctx, deletionCM, metav1.CreateOptions{})
		if createErr != nil {
			return false, nil
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, deletionCM.Name, metav1.DeleteOptions{})
		// If webhook is no longer mutating, the label should not be present
		_, ok := created.Labels["static-mutated"]
		return !ok, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for mutating webhook to stop after file deletion: %v", err)
	}

	// Validate reload metrics after deleting manifest
	deleteMetrics := getReloadMetrics(t, client, "MutatingAdmissionWebhook")
	if deleteMetrics.reloadSuccessCount <= addMetrics.reloadSuccessCount {
		t.Errorf("Expected reload success count to increase after deletion, got %d (was %d)", deleteMetrics.reloadSuccessCount, addMetrics.reloadSuccessCount)
	}
	if deleteMetrics.configHash == addHash {
		t.Error("Expected config hash to change after manifest deletion")
	}
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

// createStaticAdmissionConfig creates an admission configuration file for testing.
// It only configures the ValidatingAdmissionWebhook plugin with a static manifests
// directory since the tests use ValidatingWebhookConfiguration manifests.
func createStaticAdmissionConfig(t *testing.T, staticManifestsDir string) string {
	admissionConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ValidatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    staticManifestsDir: %q
`, staticManifestsDir)

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

// createStaticMutatingAdmissionConfig creates an admission configuration file for testing.
// It configures the MutatingAdmissionWebhook plugin with a static manifests directory.
func createStaticMutatingAdmissionConfig(t *testing.T, staticManifestsDir string) string {
	admissionConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    staticManifestsDir: %q
`, staticManifestsDir)

	configFile, err := os.CreateTemp("", "mutating-admission-config-*.yaml")
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

func readStaticBody(r *http.Request) ([]byte, error) {
	defer func() { _ = r.Body.Close() }()
	return io.ReadAll(r.Body)
}

// reloadMetrics holds parsed reload metrics from /metrics endpoint.
type reloadMetrics struct {
	reloadSuccessCount int
	reloadSuccessTime  *time.Time
	configHash         string
}

var (
	reloadCountMetric  = regexp.MustCompile(`apiserver_manifest_admission_config_controller_automatic_reloads_total\{[^}]*plugin="([^"]*)"[^}]*status="success"[^}]*\}\s+(\d+)`)
	reloadTimestampRe  = regexp.MustCompile(`apiserver_manifest_admission_config_controller_automatic_reload_last_timestamp_seconds\{[^}]*plugin="([^"]*)"[^}]*status="success"[^}]*\}\s+([0-9.e+]+)`)
	configInfoMetricRe = regexp.MustCompile(`apiserver_manifest_admission_config_controller_last_config_info\{[^}]*hash="([^"]*)"[^}]*plugin="([^"]*)"[^}]*\}\s+\d+`)
)

func getReloadMetrics(t *testing.T, client clientset.Interface, pluginName string) *reloadMetrics {
	t.Helper()
	data, err := client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(context.Background())
	if err != nil {
		t.Fatalf("Failed to fetch metrics: %v", err)
	}

	m := &reloadMetrics{}
	for line := range strings.SplitSeq(string(data), "\n") {
		if matches := reloadCountMetric.FindStringSubmatch(line); matches != nil && matches[1] == pluginName {
			m.reloadSuccessCount, _ = strconv.Atoi(matches[2])
		}
		if matches := reloadTimestampRe.FindStringSubmatch(line); matches != nil && matches[1] == pluginName {
			value, _ := strconv.ParseFloat(matches[2], 64)
			seconds := int64(value)
			nanoseconds := int64((value - float64(seconds)) * 1000000000)
			tm := time.Unix(seconds, nanoseconds)
			m.reloadSuccessTime = &tm
		}
		if matches := configInfoMetricRe.FindStringSubmatch(line); matches != nil && matches[2] == pluginName {
			m.configHash = matches[1]
		}
	}
	return m
}
