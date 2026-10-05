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
	authorizationv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission/plugin/manifest/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	policysource "k8s.io/apiserver/pkg/admission/plugin/policy/manifest/source"
	"k8s.io/apiserver/pkg/features"
	genericapiserver "k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"sigs.k8s.io/yaml"
)

// TestStaticPolicyBlocksAPICreation tests that all policy resources with .static.k8s.io
// suffix cannot be created via the REST API when the feature gate is enabled. Covers
// ValidatingAdmissionPolicy, ValidatingAdmissionPolicyBinding, MutatingAdmissionPolicy,
// and MutatingAdmissionPolicyBinding in a single server instance.
func TestStaticPolicyBlocksAPICreation(t *testing.T) {
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	ctx := t.Context()

	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy,MutatingAdmissionPolicy",
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

	// ValidatingAdmissionPolicy
	_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(ctx,
		&admissionregistrationv1.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{Name: "test-policy.static.k8s.io"},
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
				Validations: []admissionregistrationv1.Validation{{Expression: "true"}},
			},
		}, metav1.CreateOptions{})
	expectStaticSuffixRejection("ValidatingAdmissionPolicy", err)

	// ValidatingAdmissionPolicyBinding
	_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(ctx,
		&admissionregistrationv1.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "test-binding.static.k8s.io"},
			Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        "some-policy",
				ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
			},
		}, metav1.CreateOptions{})
	expectStaticSuffixRejection("ValidatingAdmissionPolicyBinding", err)

	// MutatingAdmissionPolicy
	_, err = client.AdmissionregistrationV1().MutatingAdmissionPolicies().Create(ctx,
		&admissionregistrationv1.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{Name: "test-policy.static.k8s.io"},
			Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
				ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
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
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{}`},
				}},
			},
		}, metav1.CreateOptions{})
	expectStaticSuffixRejection("MutatingAdmissionPolicy", err)

	// MutatingAdmissionPolicyBinding
	_, err = client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Create(ctx,
		&admissionregistrationv1.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "test-binding.static.k8s.io"},
			Spec:       admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{PolicyName: "some-policy"},
		}, metav1.CreateOptions{})
	expectStaticSuffixRejection("MutatingAdmissionPolicyBinding", err)
}

func createCombinedPolicyAdmissionConfig(t *testing.T, vapManifestsDir, mapManifestsDir string) string {
	t.Helper()
	admissionConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ValidatingAdmissionPolicy
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: ValidatingAdmissionPolicyConfiguration
    staticManifestsDir: %q
- name: MutatingAdmissionPolicy
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: MutatingAdmissionPolicyConfiguration
    staticManifestsDir: %q
`, vapManifestsDir, mapManifestsDir)
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

// TestStaticPolicyComprehensive exercises static policy manifest loading using a
// single shared API server with combined ValidatingAdmissionPolicy and
// MutatingAdmissionPolicy admission config.
//
// Scenarios tested (in order):
//  1. Startup loading — VAP+VAPB manifest pre-populated before server start, verified active immediately
//  2. Hot reload — MAP+MAPB manifest added after server start, verified via polling
//  3. Both policies active simultaneously — configmap triggers both validation and mutation
//  4. v1.List format manifest — VAP+VAPB loaded from a v1.List YAML file
//  5. Static policy intercepts admission config resources — VAP creation denied by static policy
//  6. Excluded auth resources skipped — SubjectAccessReview not intercepted
//  7. Coexistence with API-based policy — static and API-defined policies enforce side by side
//  8. Hot reload removal — VAP manifest deleted, policy stops enforcing
//
// Reload metrics (count, hash, timestamp, apiserver_id_hash) are validated after every
// subtest that triggers a manifest reload to confirm the reload was successful.
func TestStaticPolicyComprehensive(t *testing.T) {
	genericapiserver.SetHostnameFuncForTests("testAPIServerID")
	metrics.ResetMetricsForTest()
	resetPolicyRefreshInterval := generic.SetPolicyRefreshIntervalForTests(policyRefreshInterval)
	defer resetPolicyRefreshInterval()
	resetStaticReloadInterval := policysource.SetReloadIntervalForTests(10 * time.Millisecond)
	defer resetStaticReloadInterval()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	ctx := t.Context()

	// Separate directories for VAP and MAP manifests (each loader only accepts its own resource types)
	vapManifestsDir := t.TempDir()
	mapManifestsDir := t.TempDir()

	// Pre-populate VAP + VAPB manifest before server start (tests startup loading path)
	vapPolicy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "ValidatingAdmissionPolicy"},
		ObjectMeta: metav1.ObjectMeta{Name: "deny-labels.static.k8s.io"},
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
	vapBinding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "ValidatingAdmissionPolicyBinding"},
		ObjectMeta: metav1.ObjectMeta{Name: "deny-labels-binding.static.k8s.io"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        "deny-labels.static.k8s.io",
			ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
		},
	}
	writeManifestToFile(t, vapManifestsDir, "deny-labels.yaml", vapPolicy, vapBinding)

	// Combined admission config for BOTH ValidatingAdmissionPolicy AND MutatingAdmissionPolicy
	admissionConfigFile := createCombinedPolicyAdmissionConfig(t, vapManifestsDir, mapManifestsDir)

	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
		"--enable-admission-plugins=ValidatingAdmissionPolicy,MutatingAdmissionPolicy",
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
	ns := metav1.NamespaceDefault

	// Track reload metrics across subtests for both plugins
	var lastVAPMetrics, lastMAPMetrics *reloadMetrics
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

	t.Run("validating policy active at startup", func(t *testing.T) {
		// VAP+VAPB was pre-populated — should be active from the first request
		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "startup-vap-probe", Namespace: ns, Labels: map[string]string{"deny-me": "true"}},
			}, metav1.CreateOptions{})
			if lastErr != nil && apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			if lastErr == nil {
				_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, "startup-vap-probe", metav1.DeleteOptions{})
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for VAP to enforce at startup: %v (last error: %v)", err, lastErr)
		}

		expectReloadCountIncrease("ValidatingAdmissionPolicy", &lastVAPMetrics, true)
	})

	t.Run("hot reload mutating policy", func(t *testing.T) {
		// Write MAP + MAPB manifest in a SEPARATE file (tests multiple manifest files in same dir)
		mapPolicy := &admissionregistrationv1.MutatingAdmissionPolicy{
			TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "MutatingAdmissionPolicy"},
			ObjectMeta: metav1.ObjectMeta{Name: "add-label.static.k8s.io"},
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
					PatchType:          admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{metadata: Object.metadata{labels: {"static-mutated": "true"}}}`},
				}},
			},
		}
		mapBinding := &admissionregistrationv1.MutatingAdmissionPolicyBinding{
			TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "MutatingAdmissionPolicyBinding"},
			ObjectMeta: metav1.ObjectMeta{Name: "add-label-binding.static.k8s.io"},
			Spec:       admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{PolicyName: "add-label.static.k8s.io"},
		}
		writeManifestToFile(t, mapManifestsDir, "add-label.yaml", mapPolicy, mapBinding)

		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			cm, createErr := client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "hot-reload-map-probe", Namespace: ns},
			}, metav1.CreateOptions{})
			if createErr != nil {
				lastErr = createErr
				return false, nil
			}
			if cm.Labels != nil && cm.Labels["static-mutated"] == "true" {
				_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
				return true, nil
			}
			_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, "hot-reload-map-probe", metav1.DeleteOptions{})
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for hot-reloaded MAP to mutate: %v (last error: %v)", err, lastErr)
		}

		expectReloadCountIncrease("MutatingAdmissionPolicy", &lastMAPMetrics, true)
	})

	t.Run("both policies active simultaneously", func(t *testing.T) {
		// Create configmap without deny-me label — should be mutated but not denied
		cm, err := client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "both-active-test", Namespace: ns},
			Data:       map[string]string{"key": "value"},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Expected configmap without deny-me label to be allowed: %v", err)
		}
		if cm.Labels == nil || cm.Labels["static-mutated"] != "true" {
			t.Errorf("Expected label 'static-mutated=true', got labels: %v", cm.Labels)
		}
		_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
	})

	t.Run("v1.List format manifest", func(t *testing.T) {
		// Write a third manifest file using v1.List format containing a VAP+VAPB
		// that denies configmaps with annotation blocked=true
		listYAML := `apiVersion: v1
kind: List
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingAdmissionPolicy
  metadata:
    name: deny-annotations.static.k8s.io
  spec:
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - operations: ["CREATE"]
        apiGroups: [""]
        apiVersions: ["v1"]
        resources: ["configmaps"]
    validations:
    - expression: "!has(object.metadata.annotations) || !('blocked' in object.metadata.annotations) || object.metadata.annotations['blocked'] != 'true'"
      message: "configmaps with annotation 'blocked=true' are not allowed"
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingAdmissionPolicyBinding
  metadata:
    name: deny-annotations-binding.static.k8s.io
  spec:
    policyName: deny-annotations.static.k8s.io
    validationActions: ["Deny"]
`
		listPath := filepath.Join(vapManifestsDir, "deny-annotations.yaml")
		tmp := listPath + ".tmp"
		if err := os.WriteFile(tmp, []byte(listYAML), 0644); err != nil {
			t.Fatalf("Failed to write v1.List manifest: %v", err)
		}
		if err := os.Rename(tmp, listPath); err != nil {
			t.Fatalf("Failed to rename v1.List manifest: %v", err)
		}

		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "annotation-probe", Namespace: ns, Annotations: map[string]string{"blocked": "true"}},
			}, metav1.CreateOptions{})
			if lastErr != nil && apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			if lastErr == nil {
				_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, "annotation-probe", metav1.DeleteOptions{})
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for v1.List VAP to enforce: %v (last error: %v)", err, lastErr)
		}

		expectReloadCountIncrease("ValidatingAdmissionPolicy", &lastVAPMetrics, true)
	})

	t.Run("static policy protects admission config resources", func(t *testing.T) {
		// Write a new manifest that denies VAP creation
		protectPolicy := &admissionregistrationv1.ValidatingAdmissionPolicy{
			TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "ValidatingAdmissionPolicy"},
			ObjectMeta: metav1.ObjectMeta{Name: "protect-vaps.static.k8s.io"},
			Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
				FailurePolicy: new(admissionregistrationv1.Fail),
				MatchConstraints: &admissionregistrationv1.MatchResources{
					ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
						RuleWithOperations: admissionregistrationv1.RuleWithOperations{
							Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
							Rule: admissionregistrationv1.Rule{
								APIGroups:   []string{"admissionregistration.k8s.io"},
								APIVersions: []string{"v1"},
								Resources:   []string{"validatingadmissionpolicies"},
							},
						},
					}},
				},
				Validations: []admissionregistrationv1.Validation{{
					Expression: "false",
					Message:    "VAP creation denied by static policy",
				}},
			},
		}
		protectBinding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
			TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "ValidatingAdmissionPolicyBinding"},
			ObjectMeta: metav1.ObjectMeta{Name: "protect-vaps-binding.static.k8s.io"},
			Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        "protect-vaps.static.k8s.io",
				ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
			},
		}
		writeManifestToFile(t, vapManifestsDir, "protect-vaps.yaml", protectPolicy, protectBinding)

		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(ctx,
				&admissionregistrationv1.ValidatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "test-api-vap"},
					Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
						FailurePolicy: new(admissionregistrationv1.Fail),
						MatchConstraints: &admissionregistrationv1.MatchResources{
							ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
								RuleWithOperations: admissionregistrationv1.RuleWithOperations{
									Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
									Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"secrets"}},
								},
							}},
						},
						Validations: []admissionregistrationv1.Validation{{Expression: "true"}},
					},
				}, metav1.CreateOptions{})
			if lastErr == nil {
				_ = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Delete(ctx, "test-api-vap", metav1.DeleteOptions{})
				return false, nil
			}
			if apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for static policy to protect VAP creation: %v (last error: %v)", err, lastErr)
		}

		expectReloadCountIncrease("ValidatingAdmissionPolicy", &lastVAPMetrics, true)
	})

	t.Run("static policy skips excluded auth resources", func(t *testing.T) {
		// SubjectAccessReview should NOT be intercepted even with broad static policies
		_, err := client.AuthorizationV1().SubjectAccessReviews().Create(ctx,
			&authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					User: "test-user",
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:     "get",
						Resource: "pods",
					},
				},
			}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("SubjectAccessReview should succeed (not intercepted by static policy): %v", err)
		}
	})

	t.Run("coexists with API-based policy", func(t *testing.T) {
		// Remove protect-vaps.yaml so we can create the API-based VAP
		if err := os.Remove(filepath.Join(vapManifestsDir, "protect-vaps.yaml")); err != nil {
			t.Fatalf("Failed to remove protect-vaps.yaml: %v", err)
		}

		expectReloadCountIncrease("ValidatingAdmissionPolicy", &lastVAPMetrics, true)

		// Wait for protect-vaps policy to be unloaded, then create API-based VAP
		var lastErr error
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(ctx,
				&admissionregistrationv1.ValidatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "deny-api-label"},
					Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
						FailurePolicy: new(admissionregistrationv1.Fail),
						MatchConstraints: &admissionregistrationv1.MatchResources{
							ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
								RuleWithOperations: admissionregistrationv1.RuleWithOperations{
									Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
									Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"secrets"}},
								},
							}},
						},
						Validations: []admissionregistrationv1.Validation{{
							Expression: "!has(object.metadata.labels) || !('deny-api' in object.metadata.labels) || object.metadata.labels['deny-api'] != 'true'",
							Message:    "secrets with label 'deny-api=true' are not allowed",
						}},
					},
				}, metav1.CreateOptions{})
			return lastErr == nil, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting to create API-based VAP: %v (last error: %v)", err, lastErr)
		}

		_, err = client.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(ctx,
			&admissionregistrationv1.ValidatingAdmissionPolicyBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "deny-api-label-binding"},
				Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
					PolicyName:        "deny-api-label",
					ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
				},
			}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create API-based VAP binding: %v", err)
		}

		// Wait for API policy to be active
		err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, lastErr = client.CoreV1().Secrets(ns).Create(ctx, &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "api-vap-probe", Namespace: ns, Labels: map[string]string{"deny-api": "true"}},
			}, metav1.CreateOptions{})
			if lastErr != nil && apierrors.IsInvalid(lastErr) {
				return true, nil
			}
			if lastErr == nil {
				_ = client.CoreV1().Secrets(ns).Delete(ctx, "api-vap-probe", metav1.DeleteOptions{})
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for API VAP to enforce: %v (last error: %v)", err, lastErr)
		}

		// Verify static policy still works (deny configmap with deny-me)
		_, err = client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "coexist-static-check", Namespace: ns, Labels: map[string]string{"deny-me": "true"}},
		}, metav1.CreateOptions{})
		if err == nil {
			t.Fatal("Expected configmap with deny-me=true to be denied by static VAP")
		}
		if !apierrors.IsInvalid(err) {
			t.Fatalf("Expected Invalid error for deny-me, got: %v", err)
		}
	})

	t.Run("hot reload removal", func(t *testing.T) {
		// Remove deny-labels.yaml
		if err := os.Remove(filepath.Join(vapManifestsDir, "deny-labels.yaml")); err != nil {
			t.Fatalf("Failed to remove deny-labels.yaml: %v", err)
		}

		// Poll: configmap with deny-me=true now allowed
		err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			cm, createErr := client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "post-removal-cm", Namespace: ns, Labels: map[string]string{"deny-me": "true"}},
			}, metav1.CreateOptions{})
			if createErr != nil {
				if apierrors.IsInvalid(createErr) {
					return false, nil
				}
				return false, nil
			}
			_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, cm.Name, metav1.DeleteOptions{})
			return true, nil
		})
		if err != nil {
			t.Fatalf("Timeout waiting for VAP removal to take effect: %v", err)
		}

		// Other static policies still active (annotations via v1.List, mutation via MAP)
		_, err = client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "post-removal-annotation", Namespace: ns, Annotations: map[string]string{"blocked": "true"}},
		}, metav1.CreateOptions{})
		if err == nil {
			t.Fatal("Expected configmap with annotation blocked=true to still be denied after removing deny-labels.yaml")
		}
		if !apierrors.IsInvalid(err) {
			t.Fatalf("Expected Invalid error for blocked annotation, got: %v", err)
		}

		expectReloadCountIncrease("ValidatingAdmissionPolicy", &lastVAPMetrics, true)
	})
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
