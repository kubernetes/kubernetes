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

// TestStaticMAPManifestLoading tests that MutatingAdmissionPolicy loaded from static manifests
// can mutate resources at API server startup.
func TestStaticMAPManifestLoading(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Create static manifests directory
	manifestsDir := t.TempDir()

	// Create a mutating admission policy that adds a label to configmaps
	policy := &admissionregistrationv1.MutatingAdmissionPolicy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "MutatingAdmissionPolicy",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "add-label.static.k8s.io",
		},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
			FailurePolicy: func() *admissionregistrationv1.FailurePolicyType {
				fp := admissionregistrationv1.Fail
				return &fp
			}(),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
					},
				}},
			},
			Mutations: []admissionregistrationv1.Mutation{{
				PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
				ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
					Expression: `Object{metadata: Object.metadata{labels: {"mutated-by-static-map": "true"}}}`,
				},
			}},
		},
	}

	binding := &admissionregistrationv1.MutatingAdmissionPolicyBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "MutatingAdmissionPolicyBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "add-label-binding.static.k8s.io",
		},
		Spec: admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{
			PolicyName: "add-label.static.k8s.io",
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

	manifestPath := filepath.Join(manifestsDir, "policy.yaml")
	content := string(policyBytes) + "\n---\n" + string(bindingBytes)
	if err := os.WriteFile(manifestPath, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// Create admission config file
	admissionConfigFile := createStaticMAPAdmissionConfig(t, manifestsDir)

	// Start the API server
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--enable-admission-plugins=MutatingAdmissionPolicy",
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

	// Create a configmap and verify the mutation is applied
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-map-mutation",
			Namespace: ns,
		},
		Data: map[string]string{"key": "value"},
	}

	// Wait for the policy to be active and mutating
	var lastErr error
	var createdCM *corev1.ConfigMap
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		createdCM, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
		if lastErr != nil {
			return false, nil
		}
		// Check if the mutation was applied
		if createdCM.Labels != nil && createdCM.Labels["mutated-by-static-map"] == "true" {
			return true, nil
		}
		// Mutation not yet applied, clean up and retry
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for MAP policy to mutate: %v (last error: %v)", err, lastErr)
	}

	if createdCM.Labels["mutated-by-static-map"] != "true" {
		t.Errorf("Expected label 'mutated-by-static-map=true', got labels: %v", createdCM.Labels)
	}
}

// TestStaticMAPBlocksAPICreation tests that MAP resources with .static.k8s.io suffix
// cannot be created via the REST API when the feature gate is enabled.
func TestStaticMAPBlocksAPICreation(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "MutatingAdmissionPolicy",
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

	// Try to create a MutatingAdmissionPolicy with .static.k8s.io suffix — should be rejected
	policy := &admissionregistrationv1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-policy.static.k8s.io",
		},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule: admissionregistrationv1.Rule{
							APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"},
						},
					},
				}},
			},
			Mutations: []admissionregistrationv1.Mutation{{
				PatchType:          admissionregistrationv1.PatchTypeApplyConfiguration,
				ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{}`},
			}},
		},
	}
	_, err = client.AdmissionregistrationV1().MutatingAdmissionPolicies().Create(context.Background(), policy, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("Expected error when creating MutatingAdmissionPolicy with .static.k8s.io suffix, got nil")
	}
	if !apierrors.IsInvalid(err) {
		t.Errorf("Expected Invalid error, got: %v", err)
	}

	// Try to create a MutatingAdmissionPolicyBinding with .static.k8s.io suffix — should be rejected
	binding := &admissionregistrationv1.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-binding.static.k8s.io",
		},
		Spec: admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{
			PolicyName: "some-policy",
		},
	}
	_, err = client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Create(context.Background(), binding, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("Expected error when creating MutatingAdmissionPolicyBinding with .static.k8s.io suffix, got nil")
	}
	if !apierrors.IsInvalid(err) {
		t.Errorf("Expected Invalid error, got: %v", err)
	}
}

// TestStaticMAPHotReload tests that adding a new MAP manifest file while the server is running
// causes the new policy to be loaded and applied without restarting.
func TestStaticMAPHotReload(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()
	resetStaticReloadInterval := policysource.SetReloadIntervalForTests(10 * time.Millisecond)
	defer resetStaticReloadInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	// Start with an empty manifests directory
	manifestsDir := t.TempDir()
	admissionConfigFile := createStaticMAPAdmissionConfig(t, manifestsDir)

	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--enable-admission-plugins=MutatingAdmissionPolicy",
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

	// Verify no mutation happens initially
	preCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "pre-reload", Namespace: ns},
	}
	created, err := client.CoreV1().ConfigMaps(ns).Create(context.Background(), preCM, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	if created.Labels != nil && created.Labels["hot-reload-map"] == "true" {
		t.Fatal("Label should not be present before hot-reload")
	}
	_ = client.CoreV1().ConfigMaps(ns).Delete(context.Background(), preCM.Name, metav1.DeleteOptions{})

	// Hot-reload: write MAP manifest
	policy := &admissionregistrationv1.MutatingAdmissionPolicy{
		TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "MutatingAdmissionPolicy"},
		ObjectMeta: metav1.ObjectMeta{Name: "hot-reload-map.static.k8s.io"},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
			FailurePolicy:      func() *admissionregistrationv1.FailurePolicyType { fp := admissionregistrationv1.Fail; return &fp }(),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
					},
				}},
			},
			Mutations: []admissionregistrationv1.Mutation{{
				PatchType:          admissionregistrationv1.PatchTypeApplyConfiguration,
				ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{metadata: Object.metadata{labels: {"hot-reload-map": "true"}}}`},
			}},
		},
	}
	binding := &admissionregistrationv1.MutatingAdmissionPolicyBinding{
		TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "MutatingAdmissionPolicyBinding"},
		ObjectMeta: metav1.ObjectMeta{Name: "hot-reload-map-binding.static.k8s.io"},
		Spec:       admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{PolicyName: "hot-reload-map.static.k8s.io"},
	}

	policyBytes, _ := yaml.Marshal(policy)
	bindingBytes, _ := yaml.Marshal(binding)
	content := string(policyBytes) + "\n---\n" + string(bindingBytes)
	if err := os.WriteFile(filepath.Join(manifestsDir, "hot-reload.yaml"), []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write manifest: %v", err)
	}

	// Wait for hot-reload
	postCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "post-reload", Namespace: ns},
	}
	var lastErr error
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		var cm *corev1.ConfigMap
		cm, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, postCM, metav1.CreateOptions{})
		if lastErr != nil {
			return false, nil
		}
		if cm.Labels != nil && cm.Labels["hot-reload-map"] == "true" {
			return true, nil
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, postCM.Name, metav1.DeleteOptions{})
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for hot-reloaded MAP to mutate: %v (last error: %v)", err, lastErr)
	}

	// Validate reload metrics after adding manifest
	addMetrics := getReloadMetrics(t, client, "MutatingAdmissionPolicy")
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

	// Now delete the manifest file — mutation should stop being applied
	manifestPath := filepath.Join(manifestsDir, "hot-reload.yaml")
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("Failed to remove manifest file: %v", err)
	}

	// Clean up the post-reload configmap
	_ = client.CoreV1().ConfigMaps(ns).Delete(context.Background(), postCM.Name, metav1.DeleteOptions{})

	// Wait for the deletion to take effect: poll until a new configmap is NOT mutated
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		deletionCM := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "post-deletion", Namespace: ns},
		}
		cm, err := client.CoreV1().ConfigMaps(ns).Create(ctx, deletionCM, metav1.CreateOptions{})
		if err != nil {
			return false, nil
		}
		hasLabel := cm.Labels != nil && cm.Labels["hot-reload-map"] == "true"
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, deletionCM.Name, metav1.DeleteOptions{})
		// Mutation no longer applied means the policy was removed
		return !hasLabel, nil
	})
	if err != nil {
		t.Fatalf("Timeout waiting for MAP mutation to stop after file deletion: %v", err)
	}

	// Validate reload metrics after deleting manifest
	deleteMetrics := getReloadMetrics(t, client, "MutatingAdmissionPolicy")
	if deleteMetrics.reloadSuccessCount <= addMetrics.reloadSuccessCount {
		t.Errorf("Expected reload success count to increase after deletion, got %d (was %d)", deleteMetrics.reloadSuccessCount, addMetrics.reloadSuccessCount)
	}
	if deleteMetrics.configHash == addHash {
		t.Error("Expected config hash to change after manifest deletion")
	}
}

// createStaticMAPAdmissionConfig creates an admission configuration file for MAP testing.
func createStaticMAPAdmissionConfig(t *testing.T, staticManifestsDir string) string {
	admissionConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: MutatingAdmissionPolicy
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: MutatingAdmissionPolicyConfiguration
    staticManifestsDir: %q
`, staticManifestsDir)

	configFile, err := os.CreateTemp("", "map-admission-config-*.yaml")
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
