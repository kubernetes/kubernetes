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

package cel

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	policysource "k8s.io/apiserver/pkg/admission/plugin/policy/manifest/source"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"sigs.k8s.io/yaml"
)

// TestStaticVAPBlocksAPICreation tests that VAP resources with .static.k8s.io suffix
// cannot be created via the REST API when the feature gate is enabled.
func TestStaticVAPBlocksAPICreation(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Try to create a ValidatingAdmissionPolicy with .static.k8s.io suffix
	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-policy.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: new(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "true",
			}},
		},
	}

	_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(
		context.Background(), policy, metav1.CreateOptions{})

	if err == nil {
		t.Fatal("Expected error when creating VAP with .static.k8s.io suffix, got nil")
	}

	if !apierrors.IsInvalid(err) {
		t.Fatalf("Expected Invalid error, got: %v", err)
	}

	// Also test ValidatingAdmissionPolicyBinding
	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-binding.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "some-policy",
			ValidationActions: []admissionregistrationv1.ValidationAction{
				admissionregistrationv1.Deny,
			},
		},
	}

	_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(
		context.Background(), binding, metav1.CreateOptions{})

	if err == nil {
		t.Fatal("Expected error when creating VAP binding with .static.k8s.io suffix, got nil")
	}

	if !apierrors.IsInvalid(err) {
		t.Fatalf("Expected Invalid error, got: %v", err)
	}
}

// TestStaticVAPManifestLoading tests that VAP configurations can be loaded
// from static manifest files at API server startup.
func TestStaticVAPManifestLoading(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Create static manifests directory
	manifestsDir := t.TempDir()

	// Create a validating admission policy manifest that denies configmaps with specific label
	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicy",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "deny-test-label.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: new(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "!has(object.metadata.labels) || !('deny-me' in object.metadata.labels)",
				Message:    "configmaps with label 'deny-me' are not allowed",
			}},
		},
	}

	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicyBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "deny-test-label-binding.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "deny-test-label.static.k8s.io",
			ValidationActions: []admissionregistrationv1.ValidationAction{
				admissionregistrationv1.Deny,
			},
		},
	}

	// Write the manifest files
	policyBytes, err := yaml.Marshal(policy)
	if err != nil {
		t.Fatalf("Failed to marshal policy: %v", err)
	}
	bindingBytes, err := yaml.Marshal(binding)
	if err != nil {
		t.Fatalf("Failed to marshal binding: %v", err)
	}

	// Write policy and binding to same file with YAML separator
	manifestPath := filepath.Join(manifestsDir, "policy.yaml")
	content := string(policyBytes) + "\n---\n" + string(bindingBytes)
	if err := os.WriteFile(manifestPath, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// Create admission config file pointing to the static manifests directory
	admissionConfigFile := createStaticVAPAdmissionConfig(t, manifestsDir)

	// Start the API server with the feature gate enabled
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	// Create a client
	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Wait for the policy to be ready by testing that it blocks labeled configmaps
	ns := "default"

	// First verify that a configmap without the label is allowed
	allowedCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "allowed-configmap",
			Namespace: ns,
		},
		Data: map[string]string{"key": "value"},
	}

	// Wait for API server to be ready and policy to potentially sync
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, allowedCM, metav1.CreateOptions{})
		if lastErr != nil {
			// Keep trying - may be waiting for API server
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for API server to be ready: %v (last error: %v)", err, lastErr)
	}

	// Now test that the policy is enforcing - create a configmap with the deny label
	deniedCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "denied-configmap",
			Namespace: ns,
			Labels:    map[string]string{"deny-me": "true"},
		},
		Data: map[string]string{"key": "value"},
	}

	// Wait for the policy to be enforced
	attempts := 0
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		attempts++
		_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, deniedCM, metav1.CreateOptions{})
		if lastErr != nil {
			// VAP denials return StatusReasonInvalid (422), not Forbidden (403)
			if apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			// Other error - keep trying
			return false, nil
		}
		// Success means policy isn't enforcing yet - cleanup and retry
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, deniedCM.Name, metav1.DeleteOptions{})
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for VAP policy to enforce: %v (last error: %v)", err, lastErr)
	}
}

// TestStaticVAPHotReload tests that adding a new VAP manifest file while the server is running
// causes the new policy to be loaded and enforced without restarting the API server.
func TestStaticVAPHotReload(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()
	resetStaticReloadInterval := policysource.SetReloadIntervalForTests(10 * time.Millisecond)
	defer resetStaticReloadInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Start with an empty manifests directory
	manifestsDir := t.TempDir()

	// Create admission config file
	admissionConfigFile := createStaticVAPAdmissionConfig(t, manifestsDir)

	// Start the API server
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--enable-admission-plugins=ValidatingAdmissionPolicy",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatalf("Failed to start server: %v", err)
	}
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	ns := "default"

	// Confirm that a "deny-me" configmap is allowed before the policy is loaded
	preReloadCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pre-reload-denied",
			Namespace: ns,
			Labels:    map[string]string{"deny-me": "true"},
		},
	}
	if _, err := client.CoreV1().ConfigMaps(ns).Create(context.Background(), preReloadCM, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Expected creation to succeed before policy is hot-loaded: %v", err)
	}
	_ = client.CoreV1().ConfigMaps(ns).Delete(context.Background(), preReloadCM.Name, metav1.DeleteOptions{})

	// Hot-reload: write the policy and binding manifest
	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicy",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "deny-hot-reload-label.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: new(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "!has(object.metadata.labels) || !('deny-me' in object.metadata.labels)",
				Message:    "configmaps with label 'deny-me' are not allowed (hot-reload)",
			}},
		},
	}
	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicyBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "deny-hot-reload-label-binding.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        "deny-hot-reload-label.static.k8s.io",
			ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
		},
	}

	policyBytes, err := yaml.Marshal(policy)
	if err != nil {
		t.Fatalf("Failed to marshal policy: %v", err)
	}
	bindingBytes, err := yaml.Marshal(binding)
	if err != nil {
		t.Fatalf("Failed to marshal binding: %v", err)
	}
	content := string(policyBytes) + "\n---\n" + string(bindingBytes)
	manifestPath := filepath.Join(manifestsDir, "hot-reload-policy.yaml")
	if err := os.WriteFile(manifestPath, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// Wait for the hot-reload to take effect: poll until the deny-me configmap is rejected
	deniedCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "post-reload-denied",
			Namespace: ns,
			Labels:    map[string]string{"deny-me": "true"},
		},
	}
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, deniedCM, metav1.CreateOptions{})
		if lastErr != nil {
			if apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			return false, nil
		}
		// Allowed — policy not yet enforced; clean up and retry
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, deniedCM.Name, metav1.DeleteOptions{})
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for hot-reloaded VAP to enforce: %v (last error: %v)", err, lastErr)
	}

	// Validate reload metrics after adding manifest
	addMetrics := getReloadMetrics(t, client, "ValidatingAdmissionPolicy")
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

	// Now delete the manifest file — policy should stop being enforced
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("Failed to remove manifest file: %v", err)
	}

	// Wait for the deletion to take effect: poll until the deny-me configmap is allowed again
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		deletionCM := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "post-deletion-denied",
				Namespace: ns,
				Labels:    map[string]string{"deny-me": "true"},
			},
		}
		_, err := client.CoreV1().ConfigMaps(ns).Create(ctx, deletionCM, metav1.CreateOptions{})
		if err != nil {
			if apierrors.IsInvalid(err) {
				// Policy still enforced
				return false, nil
			}
			return false, nil
		}
		// Allowed — policy is no longer enforced
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, deletionCM.Name, metav1.DeleteOptions{})
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for VAP to stop being enforced after file deletion: %v", err)
	}

	// Validate reload metrics after deleting manifest
	deleteMetrics := getReloadMetrics(t, client, "ValidatingAdmissionPolicy")
	if deleteMetrics.reloadSuccessCount <= addMetrics.reloadSuccessCount {
		t.Errorf("Expected reload success count to increase after deletion, got %d (was %d)", deleteMetrics.reloadSuccessCount, addMetrics.reloadSuccessCount)
	}
	if deleteMetrics.configHash == addHash {
		t.Error("Expected config hash to change after manifest deletion")
	}
}

// createStaticVAPAdmissionConfig creates an admission configuration file for VAP testing
func createStaticVAPAdmissionConfig(t *testing.T, staticManifestsDir string) string {
	admissionConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ValidatingAdmissionPolicy
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: ValidatingAdmissionPolicyConfiguration
    staticManifestsDir: %q
`, staticManifestsDir)

	configFile, err := os.CreateTemp("", "vap-admission-config-*.yaml")
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

// TestStaticVAPProtectsAPIWebhook tests the KEP Story 2: Self-Protection scenario.
// A static ValidatingAdmissionPolicy protects API-based ValidatingWebhookConfigurations
// from deletion when they have a specific label.
func TestStaticVAPProtectsAPIWebhook(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Create static manifests directory with a policy that blocks deletion of
	// ValidatingWebhookConfigurations with label "protected=true"
	manifestsDir := t.TempDir()

	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicy",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "protect-webhooks.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: new(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Delete},
						Rule: admissionregistrationv1.Rule{
							APIGroups:   []string{"admissionregistration.k8s.io"},
							APIVersions: []string{"v1"},
							Resources:   []string{"validatingwebhookconfigurations"},
						},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "!has(oldObject.metadata.labels) || !('protected' in oldObject.metadata.labels) || oldObject.metadata.labels['protected'] != 'true'",
				Message:    "cannot delete protected ValidatingWebhookConfiguration",
			}},
		},
	}

	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicyBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "protect-webhooks-binding.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "protect-webhooks.static.k8s.io",
			ValidationActions: []admissionregistrationv1.ValidationAction{
				admissionregistrationv1.Deny,
			},
		},
	}

	policyBytes, err := yaml.Marshal(policy)
	if err != nil {
		t.Fatalf("Failed to marshal policy: %v", err)
	}
	bindingBytes, err := yaml.Marshal(binding)
	if err != nil {
		t.Fatalf("Failed to marshal binding: %v", err)
	}
	content := string(policyBytes) + "\n---\n" + string(bindingBytes)
	if err := os.WriteFile(filepath.Join(manifestsDir, "protect-webhooks.yaml"), []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	admissionConfigFile := createStaticVAPAdmissionConfig(t, manifestsDir)

	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--enable-admission-plugins=ValidatingAdmissionPolicy",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatalf("Failed to start server: %v", err)
	}
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Create an API-based ValidatingWebhookConfiguration with protected=true label
	vwc := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "api-webhook",
			Labels: map[string]string{"protected": "true"},
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "api-webhook.example.com",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL: new("https://localhost:12345/validate"),
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
				Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"secrets"}},
			}},
			AdmissionReviewVersions: []string{"v1"},
			SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
		}},
	}

	_, err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(
		context.Background(), vwc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create API-based webhook: %v", err)
	}

	// Wait for the static policy to be enforced, then try to delete the protected webhook
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		lastErr = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(
			ctx, "api-webhook", metav1.DeleteOptions{})
		if lastErr != nil {
			// VAP denials return StatusReasonInvalid (422)
			if apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			return false, nil
		}
		// Deletion succeeded — policy not enforcing yet; recreate and retry
		_, _ = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(
			ctx, vwc, metav1.CreateOptions{})
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for static VAP to block webhook deletion: %v (last error: %v)", err, lastErr)
	}

	// Verify the webhook still exists (deletion was blocked)
	_, err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Get(
		context.Background(), "api-webhook", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Protected webhook should still exist after blocked deletion: %v", err)
	}

	// Verify an unprotected webhook CAN be deleted
	unprotectedVWC := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "unprotected-webhook",
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "unprotected.example.com",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL: new("https://localhost:12345/validate"),
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
				Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"secrets"}},
			}},
			AdmissionReviewVersions: []string{"v1"},
			SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
		}},
	}
	_, err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(
		context.Background(), unprotectedVWC, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create unprotected webhook: %v", err)
	}
	err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(
		context.Background(), "unprotected-webhook", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Should be able to delete unprotected webhook: %v", err)
	}
}

// TestStaticVAPProtectsVAP tests that a static ValidatingAdmissionPolicy can protect
// other ValidatingAdmissionPolicy resources from deletion. This verifies that the
// shouldIgnoreResource exclusion is bypassed for static (manifest-based) policies,
// allowing them to intercept operations on admission registration resources.
func TestStaticVAPProtectsVAP(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Create static manifests directory with a policy that blocks deletion of
	// ValidatingAdmissionPolicies with label "protected=true"
	manifestsDir := t.TempDir()

	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicy",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "protect-vaps.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: new(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Delete},
						Rule: admissionregistrationv1.Rule{
							APIGroups:   []string{"admissionregistration.k8s.io"},
							APIVersions: []string{"v1"},
							Resources:   []string{"validatingadmissionpolicies"},
						},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "!has(oldObject.metadata.labels) || !('protected' in oldObject.metadata.labels) || oldObject.metadata.labels['protected'] != 'true'",
				Message:    "cannot delete protected ValidatingAdmissionPolicy",
			}},
		},
	}

	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "ValidatingAdmissionPolicyBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "protect-vaps-binding.static.k8s.io",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "protect-vaps.static.k8s.io",
			ValidationActions: []admissionregistrationv1.ValidationAction{
				admissionregistrationv1.Deny,
			},
		},
	}

	policyBytes, err := yaml.Marshal(policy)
	if err != nil {
		t.Fatalf("Failed to marshal policy: %v", err)
	}
	bindingBytes, err := yaml.Marshal(binding)
	if err != nil {
		t.Fatalf("Failed to marshal binding: %v", err)
	}
	content := string(policyBytes) + "\n---\n" + string(bindingBytes)
	if err := os.WriteFile(filepath.Join(manifestsDir, "protect-vaps.yaml"), []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	admissionConfigFile := createStaticVAPAdmissionConfig(t, manifestsDir)

	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--enable-admission-plugins=ValidatingAdmissionPolicy",
		"--admission-control-config-file=" + admissionConfigFile,
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatalf("Failed to start server: %v", err)
	}
	defer server.TearDownFn()

	config := rest.CopyConfig(server.ClientConfig)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Create an API-based VAP with protected=true label
	protectedVAP := &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "api-vap-protected",
			Labels: map[string]string{"protected": "true"},
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: new(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule: admissionregistrationv1.Rule{
							APIGroups:   []string{""},
							APIVersions: []string{"v1"},
							Resources:   []string{"secrets"},
						},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "true",
			}},
		},
	}

	_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(
		context.Background(), protectedVAP, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create API-based VAP: %v", err)
	}

	// Wait for the static policy to be enforced, then try to delete the protected VAP
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		lastErr = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Delete(
			ctx, "api-vap-protected", metav1.DeleteOptions{})
		if lastErr != nil {
			// VAP denials return StatusReasonInvalid (422)
			if apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			return false, nil
		}
		// Deletion succeeded — policy not enforcing yet; recreate and retry
		_, _ = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(
			ctx, protectedVAP, metav1.CreateOptions{})
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for static VAP to block VAP deletion: %v (last error: %v)", err, lastErr)
	}

	// Verify the protected VAP still exists
	_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Get(
		context.Background(), "api-vap-protected", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Protected VAP should still exist after blocked deletion: %v", err)
	}

	// Verify an unprotected VAP CAN be deleted
	unprotectedVAP := &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "api-vap-unprotected",
		},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: new(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule: admissionregistrationv1.Rule{
							APIGroups:   []string{""},
							APIVersions: []string{"v1"},
							Resources:   []string{"secrets"},
						},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "true",
			}},
		},
	}
	_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(
		context.Background(), unprotectedVAP, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create unprotected VAP: %v", err)
	}
	err = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Delete(
		context.Background(), "api-vap-unprotected", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Should be able to delete unprotected VAP: %v", err)
	}
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
