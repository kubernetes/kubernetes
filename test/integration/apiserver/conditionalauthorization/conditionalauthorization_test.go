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

package conditionalauthorization

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/google/cel-go/cel"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestConditionalAuthorizationEnabled tests the conditional authorization flow
// end-to-end with the feature gate enabled and the AuthorizationConditionsEnforcer
// admission plugin active. The webhook authorizer returns conditional decisions
// (with conditions) for SubjectAccessReview requests, and then evaluates those
// conditions via AuthorizationConditionsReview during admission.
func TestConditionalAuthorizationEnabled(t *testing.T) {
	runConditionalAuthorizationTests(t, true)
}

// TestConditionalAuthorizationDisabled tests that when the ConditionalAuthorization
// feature gate is disabled, conditional decisions from webhooks are treated as
// NoOpinion (falling through to RBAC).
func TestConditionalAuthorizationDisabled(t *testing.T) {
	runConditionalAuthorizationTests(t, false)
}

func runConditionalAuthorizationTests(t *testing.T, featureEnabled bool) {
	dir := t.TempDir()

	// Start a webhook server that handles both SubjectAccessReview (authorization)
	// and AuthorizationConditionsReview (conditions evaluation) on the same endpoint.
	webhookServer := newWebhookServer(t)
	defer webhookServer.server.Close()

	// Write a kubeconfig for the webhook server with two contexts:
	// - "default" context for SAR on /authorize
	// - "conditions" context for ACR on /conditionsreview
	kubeconfigPath := filepath.Join(dir, "webhook-kubeconfig.yaml")
	if err := os.WriteFile(kubeconfigPath, fmt.Appendf(nil, `
apiVersion: v1
kind: Config
clusters:
- name: authorize
  cluster:
    server: %q
    insecure-skip-tls-verify: true
- name: conditions
  cluster:
    server: %q
    insecure-skip-tls-verify: true
contexts:
- name: default
  context:
    cluster: authorize
    user: test
- name: conditions
  context:
    cluster: conditions
    user: test
current-context: default
users:
- name: test
`, webhookServer.server.URL+"/authorize", webhookServer.server.URL+"/conditionsreview"), 0644); err != nil {
		t.Fatal(err)
	}

	// Write an AuthorizationConfiguration file
	authzConfigPath := filepath.Join(dir, "authz-config.yaml")
	conditionsReviewSection := ""
	if featureEnabled {
		conditionsReviewSection = `
    conditionsReview:
      kubeConfigContextName: conditions
      version: v1alpha1`
	}
	if err := os.WriteFile(authzConfigPath, fmt.Appendf(nil, `
apiVersion: apiserver.config.k8s.io/v1beta1
kind: AuthorizationConfiguration
authorizers:
- type: Webhook
  name: conditional-webhook
  webhook:
    timeout: 10s
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    failurePolicy: NoOpinion
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: %q%s
- type: RBAC
  name: rbac
`, kubeconfigPath, conditionsReviewSection), 0644); err != nil {
		t.Fatal(err)
	}

	// Start the test API server with the AuthorizationConfiguration, feature gate,
	// and the AuthorizationConditionsEnforcer admission plugin
	flags := []string{
		fmt.Sprintf("--feature-gates=ConditionalAuthorization=%v", featureEnabled),
		"--authorization-config=" + authzConfigPath,
		"--enable-admission-plugins=AuthorizationConditionsEnforcer",
	}
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	// Create the "test-ns" namespace for tests, with labels for namespaceObject tests
	_, err := adminClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-ns",
			Labels: map[string]string{"env": "production", "team": "platform"},
		},
	}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatal(err)
	}

	// Create a CRD with two versions for multi-version conditional authorization tests.
	// v1 (storage version) has spec.replicas as an integer.
	// v2 has spec.replicas as an object with a "max" integer field.
	apiExtClient, err := apiextensionsclientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	crdDef := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "scalablewidgets.example.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "example.com",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "scalablewidgets",
				Singular: "scalablewidget",
				Kind:     "ScalableWidget",
				ListKind: "ScalableWidgetList",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"spec": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {Type: "integer"},
									},
								},
							},
						},
					},
				},
				{
					Name:    "v2",
					Served:  true,
					Storage: false,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"spec": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {
											Type: "object",
											Properties: map[string]apiextensionsv1.JSONSchemaProps{
												"max": {Type: "integer"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if _, err := apiExtClient.ApiextensionsV1().CustomResourceDefinitions().Create(context.TODO(), crdDef, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := wait.PollUntilContextTimeout(context.TODO(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		crd, err := apiExtClient.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, crdDef.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		for _, cond := range crd.Status.Conditions {
			if cond.Type == apiextensionsv1.Established && cond.Status == apiextensionsv1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	}); err != nil {
		t.Fatalf("timed out waiting for CRD %s to be established: %v", crdDef.Name, err)
	}

	testCases := []struct {
		name string
		// user is the username that will be impersonated
		user string
		// webhookBehaviors configures the webhook for this test case.
		// It is called before makeRequest to set the desired behavior.
		// Multiple webhook behaviors can be specified to assert the same
		// result for various webhook configurations (e.g. out-of-tree
		// webhook evaluation vs in-tree CEL evaluation).
		webhookBehaviors map[string]func(ws *webhookServerHandler)
		// makeRequest creates a client with the given user and performs an API request.
		// Returns an error if the request fails. The suffix parameter is derived from
		// the webhook behavior name and must be used in resource names to avoid
		// conflicts between subtests that share the same API server.
		makeRequest func(t *testing.T, client *clientset.Clientset, suffix string) error
		// expectAllowed is true if the request should be allowed
		expectAllowed bool
		// expectAllowedWhenDisabled overrides expectAllowed when the feature is disabled.
		// If nil, uses expectAllowed.
		expectAllowedWhenDisabled *bool
	}{
		// Unconditional decisions: the webhook returns a concrete Allow/Deny/NoOpinion.
		{
			name: "unconditional allow from webhook",
			user: "allow-user",
			webhookBehaviors: map[string]func(ws *webhookServerHandler){
				"": func(ws *webhookServerHandler) {
					ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
						sar.Status.Allowed = true
						sar.Status.Reason = "unconditionally allowed"
					}
				},
			},
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "test-allowed" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: true,
		},
		{
			name: "unconditional deny from webhook",
			user: "deny-user",
			webhookBehaviors: map[string]func(ws *webhookServerHandler){
				"": func(ws *webhookServerHandler) {
					ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
						sar.Status.Allowed = false
						sar.Status.Denied = true
						sar.Status.Reason = "unconditionally denied"
					}
				},
			},
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "test-denied" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name: "webhook no-opinion falls through to RBAC allow",
			user: "webhook-noop-rbac-user",
			webhookBehaviors: map[string]func(ws *webhookServerHandler){
				"": func(ws *webhookServerHandler) {
					ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
						// NoOpinion: neither allowed nor denied
						sar.Status.Allowed = false
						sar.Status.Denied = false
					}
				},
			},
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "foo" + suffix,
						Namespace: "test-ns",
					},
					Data: map[string]string{
						"foo": "bar",
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: true,
		},

		// Conditional decisions: the webhook returns conditions that must be
		// evaluated. Each test runs with multiple webhook behaviors that produce
		// the same logical outcome: out-of-tree webhook evaluation, in-tree CEL
		// evaluation, and in-tree failure falling back to the webhook.
		{
			name: "conditional allow - condition evaluates to allow",
			user: "conditional-allow-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "always-allow",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   "true",
								Type:        conditionsType,
								Description: "always allow condition",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "test-conditional-allow" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: true,
			// When disabled, the conditional decision is treated as NoOpinion,
			// falling through to RBAC which denies (no RBAC rules for this user).
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name: "conditional deny - condition evaluates to deny",
			user: "conditional-deny-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "always-allow",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   "true",
								Type:        conditionsType,
								Description: "base allow condition",
							},
							{
								ID:          "always-deny",
								Effect:      authorizationv1.ConditionEffectDeny,
								Condition:   "true",
								Type:        conditionsType,
								Description: "always deny condition",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "test-conditional-deny" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name: "conditional no-opinion falls through to RBAC allow",
			user: "conditional-noop-rbac-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "no-opinion",
								Effect:      authorizationv1.ConditionEffectNoOpinion,
								Condition:   "true",
								Type:        conditionsType,
								Description: "no opinion condition",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "foo" + suffix,
						Namespace: "test-ns",
					},
					Data: map[string]string{
						"foo": "bar",
					},
				}, metav1.CreateOptions{})
				return err
			},
			// The conditional response is NoOpinion, so both when enabled and disabled does this fall through to RBAC which allows.
			expectAllowed: true,
		},

		// CEL-based conditional authorization tests.
		// These test that CEL expressions flow through the SAR -> Decision -> conditions
		// evaluation pipeline. The assertions are the same whether conditions are evaluated
		// in-tree (k8s.io/authorization-cel) or out-of-tree (opaque type via webhook).
		{
			name: "cel allow by name pattern",
			user: "cel-name-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "allow-safe-prefix",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   `object.metadata.name.startsWith("safe-")`,
								Type:        conditionsType,
								Description: "only allow configmaps with safe- prefix",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "safe-configmap" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name: "cel deny by name pattern mismatch",
			user: "cel-name-deny-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "allow-safe-prefix",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   `object.metadata.name.startsWith("safe-")`,
								Type:        conditionsType,
								Description: "only allow configmaps with safe- prefix",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "unsafe-configmap" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name: "cel deny by label overrides allow",
			user: "cel-label-deny-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "allow-all",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   "true",
								Type:        conditionsType,
								Description: "base allow",
							},
							{
								ID:     "deny-restricted-label",
								Effect: authorizationv1.ConditionEffectDeny,
								Condition: `has(object.metadata.labels) && ` +
									`has(object.metadata.labels.restricted) && ` +
									`object.metadata.labels.restricted == "true"`,
								Type:        conditionsType,
								Description: "deny restricted labels",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "cel-restricted-cm" + suffix,
						Labels: map[string]string{"restricted": "true"},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name: "cel allow by data content",
			user: "cel-data-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:     "allow-approved-data",
								Effect: authorizationv1.ConditionEffectAllow,
								Condition: `has(object.data) && ` +
									`has(object.data.approved) && ` +
									`object.data.approved == "yes"`,
								Type:        conditionsType,
								Description: "only allow configmaps with approved=yes in data",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "cel-approved-cm" + suffix},
					Data:       map[string]string{"approved": "yes"},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name: "cel deny by data content missing",
			user: "cel-data-deny-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:     "allow-approved-data",
								Effect: authorizationv1.ConditionEffectAllow,
								Condition: `has(object.data) && ` +
									`has(object.data.approved) && ` +
									`object.data.approved == "yes"`,
								Type:        conditionsType,
								Description: "only allow configmaps with approved=yes in data",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "cel-unapproved-cm" + suffix},
					Data:       map[string]string{"approved": "no"},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name: "cel operation-aware deny update",
			user: "cel-op-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "allow-creates",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   `request.operation == "CREATE"`,
								Type:        conditionsType,
								Description: "allow create operations",
							},
							{
								ID:          "deny-updates",
								Effect:      authorizationv1.ConditionEffectDeny,
								Condition:   `request.operation == "UPDATE"`,
								Type:        conditionsType,
								Description: "deny update operations",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create should succeed (CEL allows CREATE)
				cm, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "cel-op-cm" + suffix},
					Data:       map[string]string{"key": "value"},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("create should have succeeded: %w", err)
				}
				// Update should be denied (CEL denies UPDATE)
				cm.Data["key"] = "new-value"
				_, err = client.CoreV1().ConfigMaps("test-ns").Update(context.TODO(), cm, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             false,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name: "cel deny overrides allow and noopinion",
			user: "cel-priority-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "allow-all",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   "true",
								Type:        conditionsType,
								Description: "allow everything",
							},
							{
								ID:          "noop-all",
								Effect:      authorizationv1.ConditionEffectNoOpinion,
								Condition:   "true",
								Type:        conditionsType,
								Description: "no opinion on everything",
							},
							{
								ID:          "deny-all",
								Effect:      authorizationv1.ConditionEffectDeny,
								Condition:   "true",
								Type:        conditionsType,
								Description: "deny everything",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "cel-priority-cm" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name: "cel noopinion overrides allow",
			user: "cel-noop-vs-allow-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "allow-all",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   "true",
								Type:        conditionsType,
								Description: "allow everything",
							},
							{
								ID:     "noop-on-pending-review",
								Effect: authorizationv1.ConditionEffectNoOpinion,
								Condition: `has(object.metadata.labels) && ` +
									`has(object.metadata.labels.review) && ` +
									`object.metadata.labels.review == "pending"`,
								Type:        conditionsType,
								Description: "no opinion when review=pending label is present",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "cel-noop-cm" + suffix,
						Labels: map[string]string{"review": "pending"},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},

		// TODO(luxas): Reactivate this when we add support for update/patch -> create conditions.
		/*
			// Update-to-create tests: When a PUT (update) request targets a non-existent
			// resource with AllowCreateOnUpdate=true, the update handler authorizes a
			// "create" verb. These tests verify that conditional authorization works
			// correctly in this flow. Leases support AllowCreateOnUpdate.
			// TODO: Verify the same behavior for patch
			{
				name: "update-to-create conditional allow by label",
				user: "update-create-allow-user",
				webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
					if sar.Spec.ResourceAttributes == nil {
						return
					}
					switch sar.Spec.ResourceAttributes.Verb {
					case "update", "patch":
						// Unconditionally allow updates
						sar.Status.Allowed = true
						sar.Status.Reason = "updates always allowed"
					case "create":
						// Conditionally allow creates: only when creator=update-create-allow-user
						sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								Conditions: []authorizationv1.Condition{
									{
										ID:     "require-owner-label",
										Effect: authorizationv1.ConditionEffectAllow,
										Condition: `has(object.metadata.labels) && ` +
											`has(object.metadata.labels.creator) && ` +
											`object.metadata.labels.creator == "update-create-allow-user"`,
										Type:        conditionsType,
										Description: "only allow creates when creator=update-create-allow-user",
									},
								},
							},
						}
					}
				}),
				makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
					// PUT a non-existent lease with creator=update-create-allow-user.
					// Since Leases support AllowCreateOnUpdate, this becomes a create.
					// The create authorization should succeed because the condition is met.
					_, err := client.CoordinationV1().Leases("test-ns").Update(context.TODO(), &coordinationv1.Lease{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "update-create-allowed" + suffix,
							Labels: map[string]string{"creator": "update-create-allow-user"},
						},
					}, metav1.UpdateOptions{})
					_, err := client.CoordinationV1().Leases("test-ns").Apply(context.TODO(),
					applyconfigurationscoordinationv1.
						Lease("update-create-denied", "test-ns").
						WithLabels(map[string]string{"creator": "update-create-allow-user"}),
					metav1.ApplyOptions{
						FieldManager: "foo",
					})
					return err
				},
				expectAllowed: true,
				// When disabled, the conditional create decision is treated as NoOpinion,
				// falls through to RBAC which denies (no RBAC rules for this user).
				expectAllowedWhenDisabled: boolPtr(false),
			},
			{
				name: "update-to-create conditional deny by label",
				user: "update-create-deny-user",
				webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
					if sar.Spec.ResourceAttributes == nil {
						return
					}
					switch sar.Spec.ResourceAttributes.Verb {
					case "update", "patch":
						// Unconditionally allow updates
						sar.Status.Allowed = true
						sar.Status.Reason = "updates always allowed"
					case "create":
						// Conditionally allow creates: only when creator=update-create-deny-user
						sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								Conditions: []authorizationv1.Condition{
									{
										ID:     "require-owner-label",
										Effect: authorizationv1.ConditionEffectAllow,
										Condition: `has(object.metadata.labels) && ` +
											`has(object.metadata.labels.creator) && ` +
											`object.metadata.labels.creator == "update-create-deny-user"`,
										Type:        conditionsType,
										Description: "only allow creates when creator=update-create-deny-user",
									},
								},
							},
						}
					}
				}),
				makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
					// PUT a non-existent lease with creator=not-authorized-user.
					// The create authorization should fail because the condition is not met.
					_, err := client.CoordinationV1().Leases("test-ns").Update(context.TODO(), &coordinationv1.Lease{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "update-create-denied" + suffix,
							Labels: map[string]string{"creator": "not-authorized-user"},
						},
					}, metav1.UpdateOptions{})
					_, err := client.CoordinationV1().Leases("test-ns").Apply(context.TODO(),
					applyconfigurationscoordinationv1.
						Lease("update-create-denied", "test-ns").
						WithLabels(map[string]string{"creator": "not-authorized-user"}),
					metav1.ApplyOptions{
						FieldManager: "foo",
					})
					return err
				},
				expectAllowed: false,
			},
			{
				name: "update-to-create, both update and create conditions must be satisfied",
				user: "update-create-deny-user",
				webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
					if sar.Spec.ResourceAttributes == nil {
						return
					}
					switch sar.Spec.ResourceAttributes.Verb {
					case "update", "patch":
						// Conditionally allow updates: only when classified=false
						sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								Conditions: []authorizationv1.Condition{
									{
										ID:     "allow-unclassified",
										Effect: authorizationv1.ConditionEffectAllow,
										Condition: `has(object.metadata.labels) && ` +
											`has(object.metadata.labels.classified) && ` +
											`object.metadata.labels.classified == "false"`,
										Type:        conditionsType,
										Description: "only allow creates when classified=false",
									},
								},
							},
						}
					case "create":
						// Conditionally allow creates: only when creator=update-create-deny-user
						sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								Conditions: []authorizationv1.Condition{
									{
										ID:     "require-owner-label",
										Effect: authorizationv1.ConditionEffectAllow,
										Condition: `has(object.metadata.labels) && ` +
											`has(object.metadata.labels.creator) && ` +
											`object.metadata.labels.creator == "update-create-deny-user"`,
										Type:        conditionsType,
										Description: "only allow creates when creator=update-create-deny-user",
									},
								},
							},
						}
					}
				}),
				makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
					// PUT a non-existent lease with creator=update-create-deny-user but no classified label.
					// The create authorization should fail because the update condition (classified=false) is not met.
					_, err := client.CoordinationV1().Leases("test-ns").Update(context.TODO(), &coordinationv1.Lease{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "update-create-denied-by-update-condition" + suffix,
							Labels: map[string]string{"creator": "update-create-deny-user"},
						},
					}, metav1.UpdateOptions{})
					_, err := client.CoordinationV1().Leases("test-ns").Apply(context.TODO(),
					applyconfigurationscoordinationv1.
						Lease("update-create-denied-by-update-condition", "test-ns").
						// Satisfies the create condition, but not the update one
						WithLabels(map[string]string{"creator": "update-create-deny-user"}),
					metav1.ApplyOptions{
						FieldManager: "foo",
					})
					return err
				},
				expectAllowed: false,
			},
		*/
		// TODO(luxas): Reactivate this when we support in-tree evaluation.
		/*
			// Tests for the builtin authorizer function in k8s authorization CEL.
			// The "compound authorization" pattern: creating an object with the "protected-label"
			// label requires the additional permission "verb=use apigroup=example.com
			// resource=protectedlabels name=protected-label", which is checked via the
			// authorizer CEL function.
			{
				name: "cel authorizer compound authorization - allow",
				user: "compound-authz-allow-user",
				webhookBehaviors: map[string]func(ws *webhookServerHandler){
					"in-process-eval-only": func(ws *webhookServerHandler) {
						ws.sarHandler = compoundAuthzSARHandler("configmaps")
						ws.acrHandler = acrEvaluateCEL(ws.t, "nonexistent-panic-on-ACR-webhook")
					},
				},
				makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
					_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "protected-cm-allow" + suffix,
							Labels: map[string]string{"protected-label": "yes"},
						},
					}, metav1.CreateOptions{})
					return err
				},
				expectAllowed:             true,
				expectAllowedWhenDisabled: boolPtr(false),
			},
			{
				name: "cel authorizer compound authorization - deny",
				user: "compound-authz-deny-user",
				webhookBehaviors: map[string]func(ws *webhookServerHandler){
					"in-process-eval-only": func(ws *webhookServerHandler) {
						ws.sarHandler = compoundAuthzSARHandler("configmaps")
						ws.acrHandler = acrEvaluateCEL(ws.t, "nonexistent-panic-on-ACR-webhook")
					},
				},
				makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
					_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "protected-cm-deny" + suffix,
							Labels: map[string]string{"protected-label": "yes"},
						},
					}, metav1.CreateOptions{})
					return err
				},
				expectAllowed: false,
			},

			// Tests for the namespaceObject variable in k8s authorization CEL.
			// The "test-ns" namespace is created with labels {env: production, team: platform}.
			{
				name: "cel namespaceObject - positive match",
				user: "ns-label-match-user",
				webhookBehaviors: map[string]func(ws *webhookServerHandler){
					"in-process-eval-only": func(ws *webhookServerHandler) {
						ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
							if sar.Spec.ResourceAttributes == nil || sar.Spec.ResourceAttributes.Resource != "configmaps" {
								return
							}
							sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
								Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
								ConditionsMap: &authorizationv1.ConditionsMap{
									Conditions: []authorizationv1.Condition{
										{
											ID:     "require-production-namespace",
											Effect: authorizationv1.ConditionEffectAllow,
											Condition: `namespaceObject != null && ` +
												`has(namespaceObject.metadata) && ` +
												`has(namespaceObject.metadata.labels) && ` +
												`'env' in namespaceObject.metadata.labels && ` +
												`namespaceObject.metadata.labels['env'] == 'production'`,
											Type:        "k8s.io/authorization-cel",
											Description: "only allow in production namespaces",
										},
									},
								},
							}
						}
						ws.acrHandler = acrEvaluateCEL(ws.t, "nonexistent-panic-on-ACR-webhook")
					},
				},
				makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
					_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
						ObjectMeta: metav1.ObjectMeta{Name: "ns-match-cm" + suffix},
					}, metav1.CreateOptions{})
					return err
				},
				expectAllowed:             true,
				expectAllowedWhenDisabled: boolPtr(false),
			},
			{
				name: "cel namespaceObject - negative match",
				user: "ns-label-nomatch-user",
				webhookBehaviors: map[string]func(ws *webhookServerHandler){
					"in-process-eval-only": func(ws *webhookServerHandler) {
						ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
							if sar.Spec.ResourceAttributes == nil || sar.Spec.ResourceAttributes.Resource != "configmaps" {
								return
							}
							sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
								Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
								ConditionsMap: &authorizationv1.ConditionsMap{
									Conditions: []authorizationv1.Condition{
										{
											ID:     "require-staging-namespace",
											Effect: authorizationv1.ConditionEffectAllow,
											Condition: `namespaceObject != null && ` +
												`has(namespaceObject.metadata) && ` +
												`has(namespaceObject.metadata.labels) && ` +
												`'env' in namespaceObject.metadata.labels && ` +
												`namespaceObject.metadata.labels['env'] == 'staging'`,
											Type:        "k8s.io/authorization-cel",
											Description: "only allow in staging namespaces",
										},
									},
								},
							}
						}
						ws.acrHandler = acrEvaluateCEL(ws.t, "nonexistent-panic-on-ACR-webhook")
					},
				},
				makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
					// test-ns has env=production, not staging, so this should be denied
					_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
						ObjectMeta: metav1.ObjectMeta{Name: "ns-nomatch-cm" + suffix},
					}, metav1.CreateOptions{})
					return err
				},
				expectAllowed: false,
			},*/

		// Tests for HPA v1 and v2 with CPU utilization conditions.
		// The authorizer returns version-specific CEL conditions that require
		// the target CPU utilization to be at most 80%.
		{
			name:             "hpa v1 cpu utilization - allow",
			user:             "hpa-cpu-allow-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-allow" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: int32Ptr(80),
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name:             "hpa v1 cpu utilization - deny",
			user:             "hpa-cpu-deny-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-deny" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: int32Ptr(90),
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:             "hpa v2 cpu utilization - allow",
			user:             "hpa-cpu-allow-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-allow" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: int32Ptr(80),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name:             "hpa v2 cpu utilization - deny",
			user:             "hpa-cpu-deny-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-deny" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: int32Ptr(90),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:             "hpa v1 cpu utilization - update allowed",
			user:             "hpa-cpu-allow-update-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-update-allow" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: int32Ptr(70),
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 80% (allowed: new=80<=80 && old=70<=80)
				created.Spec.TargetCPUUtilizationPercentage = int32Ptr(80)
				_, err = client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name:             "hpa v1 cpu utilization - update denied",
			user:             "hpa-cpu-deny-update-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-update-deny" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: int32Ptr(70),
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 90% (denied: new=90>80)
				created.Spec.TargetCPUUtilizationPercentage = int32Ptr(90)
				_, err = client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:             "hpa v2 cpu utilization - update allowed",
			user:             "hpa-cpu-allow-update-v2-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-update-allow" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: int32Ptr(70),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 80% (allowed: new=80<=80 && old=70<=80)
				created.Spec.Metrics[0].Resource.Target.AverageUtilization = int32Ptr(80)
				_, err = client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name:             "hpa v2 cpu utilization - update denied",
			user:             "hpa-cpu-deny-update-v2-user",
			webhookBehaviors: celConditionalTestCases(hpaCPUUtilizationSARHandler),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-update-deny" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: int32Ptr(70),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 90% (denied: new=90>80)
				created.Spec.Metrics[0].Resource.Target.AverageUtilization = int32Ptr(90)
				_, err = client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},

		// Tests for a multi-version CRD (ScalableWidget) with version-specific schemas.
		// v1 has spec.replicas (integer), v2 has spec.replicas.max (integer in object).
		// The authorizer returns version-specific CEL conditions requiring replicas <= 10.
		// For updates, both the old and new objects must satisfy the condition.
		{
			name:             "crd v1 replicas - create allowed",
			user:             "alice-crd-v1-create-allow",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-create-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-create-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(5),
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name:             "crd v1 replicas - create denied",
			user:             "alice-crd-v1-create-deny",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-create-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-create-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(15),
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:             "crd v2 replicas.max - create allowed",
			user:             "alice-crd-v2-create-allow",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-create-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-create-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(5),
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name:             "crd v2 replicas.max - create denied",
			user:             "alice-crd-v2-create-deny",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-create-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-create-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(15),
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:             "crd v1 replicas - update allowed",
			user:             "alice-crd-v1-update-allow",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-update-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				// Create with replicas=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-update-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(5),
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas=8 (allowed: new=8<=10 && old=5<=10)
				created.Object["spec"] = map[string]interface{}{"replicas": int64(8)}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},
		{
			name:             "crd v1 replicas - update denied",
			user:             "alice-crd-v1-update-deny",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-update-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				// Create with replicas=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-update-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(5),
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas=15 (denied: new=15>10)
				created.Object["spec"] = map[string]interface{}{"replicas": int64(15)}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},
		// This fails with an error, as no "real" CRD conversion happens, and the old object comes in with spec={"replicas": {}}, as
		// the v1 data was spec={"replicas": 5}, and during "conversion", the replicas field was just cast into an object, which means
		// that it's not even possible to apply the v1 condition (that targeted spec.replicas as an int).
		// TODO(luxas): See if there's anything we can do about this.
		/*{
			name:             "crd v2 replicas.max - update allowed",
			user:             "alice-crd-v2-update-allow",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-update-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				// Create with replicas.max=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-update-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(5),
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas.max=8 (allowed: new=8<=10 && old=5<=10)
				created.Object["spec"] = map[string]interface{}{
					"replicas": map[string]interface{}{
						"max": int64(8),
					},
				}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: boolPtr(false),
		},*/
		{
			name:             "crd v2 replicas.max - update denied",
			user:             "alice-crd-v2-update-deny",
			webhookBehaviors: celConditionalTestCases(crdReplicasSARHandler),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-update-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				// Create with replicas.max=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-update-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(5),
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas.max=15 (denied: new=15>10)
				created.Object["spec"] = map[string]interface{}{
					"replicas": map[string]interface{}{
						"max": int64(15),
					},
				}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Support multiple webhook behaviors with the same assertions
			for webhookBehaviorName, webhookBehavior := range tc.webhookBehaviors {
				t.Run(webhookBehaviorName, func(t *testing.T) {
					// Configure the webhook behavior for this test case
					webhookBehavior(webhookServer.handler)

					// Compute the user name and resource suffix. Append the webhook
					// behavior name so the webhook response cache (keyed on the SAR
					// spec including user) doesn't return stale entries from a
					// sibling subtest with a different webhook configuration.
					userName := tc.user
					suffix := webhookBehaviorName
					if suffix != "" {
						userName += "-" + suffix
						suffix = "-" + suffix
					}

					// For tests that need RBAC fallthrough, grant RBAC access
					if tc.user == "conditional-noop-rbac-user" || tc.user == "webhook-noop-rbac-user" {
						authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
							rbacv1.PolicyRule{
								Verbs:     []string{"create"},
								APIGroups: []string{""},
								Resources: []string{"configmaps"},
							},
						)
					}
					// For compound authorization tests: grant the "use" permission on protectedlabels
					if tc.user == "compound-authz-allow-user" {
						authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
							rbacv1.PolicyRule{
								Verbs:         []string{"use"},
								APIGroups:     []string{"example.com"},
								Resources:     []string{"protectedlabels"},
								ResourceNames: []string{"protected-label"},
							},
						)
					}
					impersonationConfig := rest.CopyConfig(server.ClientConfig)
					impersonationConfig.Impersonate.UserName = userName
					userClient := clientset.NewForConfigOrDie(impersonationConfig)
					err := tc.makeRequest(t, userClient, suffix)

					expected := tc.expectAllowed
					if !featureEnabled && tc.expectAllowedWhenDisabled != nil {
						expected = *tc.expectAllowedWhenDisabled
					}

					if expected {
						if err != nil {
							t.Fatalf("expected request to be allowed, got error: %v", err)
						}
					} else {
						if err == nil {
							t.Fatalf("expected request to be denied, got success")
						}
						if !apierrors.IsForbidden(err) && !apierrors.IsUnauthorized(err) {
							t.Fatalf("expected Forbidden or Unauthorized error, got: %v", err)
						}
					}
				})
			}
		})
	}

	if featureEnabled {
		// The conditional decision the webhook returns for configmap SARs.
		conditionalDecision := &authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationv1.ConditionsMap{
				Conditions: []authorizationv1.Condition{
					{
						ID:          "allow-safe-prefix",
						Effect:      authorizationv1.ConditionEffectAllow,
						Condition:   `object.metadata.name.startsWith("safe-")`,
						Type:        "opaque-cel-condition-type",
						Description: "only allow configmaps with safe- prefix",
					},
				},
			},
		}

		expectedConditionalDecision := &authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
			Union: []authorizationv1.ConditionsAwareDecision{
				{Type: authorizationv1.ConditionsAwareDecisionTypeNoOpinion}, // system privileged authorizer
				*conditionalDecision, // Conditional webhook
				{Type: authorizationv1.ConditionsAwareDecisionTypeNoOpinion}, // RBAC authorizer
			},
		}

		// Configure the webhook: return a conditional decision for configmap SARs,
		// NoOpinion for everything else (falls through to RBAC).
		webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
			if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Resource == "configmaps" {
				sar.Status.ConditionalDecision = conditionalDecision
			}
		}
		webhookServer.handler.acrHandler = acrEvaluateCEL(t, "opaque-cel-condition-type")

		t.Run("SubjectAccessReview with conditional authorization requested", func(t *testing.T) {
			sar := &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					User: "sar-test-user",
					ConditionalAuthorization: &authorizationv1.ConditionalAuthorizationOptions{
						Enabled: true,
					},
				},
			}
			response, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SubjectAccessReview: %v", err)
			}
			// The decision is conditional, so Allowed and Denied must both be false.
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false for conditional decision, got true")
			}
			if response.Status.Denied {
				t.Errorf("expected Denied=false for conditional decision, got true")
			}
			if response.Status.ConditionalDecision == nil {
				t.Fatalf("expected ConditionalDecision to be set, got nil")
			}
			if !reflect.DeepEqual(response.Status.ConditionalDecision, expectedConditionalDecision) {
				t.Errorf("unexpected ConditionalDecision:\ngot:  %+v\nwant: %+v", response.Status.ConditionalDecision, conditionalDecision)
			}
		})

		t.Run("SubjectAccessReview without conditional authorization requested", func(t *testing.T) {
			sar := &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					User: "sar-test-user-no-cond",
					// ConditionalAuthorization not set
				},
			}
			response, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SubjectAccessReview: %v", err)
			}
			// Without conditional authorization requested, the conditional decision
			// from the webhook is treated as NoOpinion, falling through to RBAC
			// which does not have a rule for this user.
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false, got true")
			}
			if response.Status.Denied {
				t.Errorf("expected Denied=false, got true")
			}
			if response.Status.ConditionalDecision != nil {
				t.Errorf("expected ConditionalDecision to be nil when not requested, got: %+v", response.Status.ConditionalDecision)
			}
		})

		t.Run("SelfSubjectAccessReview with conditional authorization requested", func(t *testing.T) {
			// Use an impersonated client so the "self" user is known.
			impersonationConfig := rest.CopyConfig(server.ClientConfig)
			impersonationConfig.Impersonate.UserName = "selfsar-test-user"
			userClient := clientset.NewForConfigOrDie(impersonationConfig)

			ssar := &authorizationv1.SelfSubjectAccessReview{
				Spec: authorizationv1.SelfSubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					ConditionalAuthorization: &authorizationv1.ConditionalAuthorizationOptions{
						Enabled: true,
					},
				},
			}
			response, err := userClient.AuthorizationV1().SelfSubjectAccessReviews().Create(context.TODO(), ssar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SelfSubjectAccessReview: %v", err)
			}
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false for conditional decision, got true")
			}
			if response.Status.Denied {
				t.Errorf("expected Denied=false for conditional decision, got true")
			}
			if response.Status.ConditionalDecision == nil {
				t.Fatalf("expected ConditionalDecision to be set, got nil")
			}
			if !reflect.DeepEqual(response.Status.ConditionalDecision, expectedConditionalDecision) {
				t.Errorf("unexpected ConditionalDecision:\ngot:  %+v\nwant: %+v", response.Status.ConditionalDecision, conditionalDecision)
			}
		})

		t.Run("SelfSubjectAccessReview without conditional authorization requested", func(t *testing.T) {
			impersonationConfig := rest.CopyConfig(server.ClientConfig)
			impersonationConfig.Impersonate.UserName = "selfsar-test-user-no-cond"
			userClient := clientset.NewForConfigOrDie(impersonationConfig)

			ssar := &authorizationv1.SelfSubjectAccessReview{
				Spec: authorizationv1.SelfSubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					// ConditionalAuthorization not set
				},
			}
			response, err := userClient.AuthorizationV1().SelfSubjectAccessReviews().Create(context.TODO(), ssar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SelfSubjectAccessReview: %v", err)
			}
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false, got true")
			}
			if response.Status.Allowed {
				t.Errorf("expected Denied=false, got true")
			}
			if response.Status.ConditionalDecision != nil {
				t.Errorf("expected ConditionalDecision to be nil when not requested, got: %+v", response.Status.ConditionalDecision)
			}
		})

		t.Run("LocalSubjectAccessReview with conditional authorization requested", func(t *testing.T) {
			lsar := &authorizationv1.LocalSubjectAccessReview{
				ObjectMeta: metav1.ObjectMeta{Namespace: "test-ns"},
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					User: "local-sar-test-user",
					ConditionalAuthorization: &authorizationv1.ConditionalAuthorizationOptions{
						Enabled: true,
					},
				},
			}
			response, err := adminClient.AuthorizationV1().LocalSubjectAccessReviews("test-ns").Create(context.TODO(), lsar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create LocalSubjectAccessReview: %v", err)
			}
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false for conditional decision, got true")
			}
			if response.Status.Denied {
				t.Errorf("expected Denied=false for conditional decision, got true")
			}
			if response.Status.ConditionalDecision == nil {
				t.Fatalf("expected ConditionalDecision to be set, got nil")
			}
			if !reflect.DeepEqual(response.Status.ConditionalDecision, expectedConditionalDecision) {
				t.Errorf("unexpected ConditionalDecision:\ngot:  %+v\nwant: %+v", response.Status.ConditionalDecision, conditionalDecision)
			}
		})

		t.Run("LocalSubjectAccessReview without conditional authorization requested", func(t *testing.T) {
			lsar := &authorizationv1.LocalSubjectAccessReview{
				ObjectMeta: metav1.ObjectMeta{Namespace: "test-ns"},
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					User: "local-sar-test-user-no-cond",
					// ConditionalAuthorization not set
				},
			}
			response, err := adminClient.AuthorizationV1().LocalSubjectAccessReviews("test-ns").Create(context.TODO(), lsar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create LocalSubjectAccessReview: %v", err)
			}
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false, got true")
			}
			if response.Status.Allowed {
				t.Errorf("expected Denied=false, got true")
			}
			if response.Status.ConditionalDecision != nil {
				t.Errorf("expected ConditionalDecision to be nil when not requested, got: %+v", response.Status.ConditionalDecision)
			}
		})

		// Test that an unconditional allow from the webhook is properly returned
		// even when conditional authorization is requested.
		t.Run("SubjectAccessReview unconditional allow with conditional authorization requested", func(t *testing.T) {
			webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
				if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Resource == "configmaps" {
					sar.Status.Allowed = true
					sar.Status.Reason = "unconditionally allowed"
				}
			}
			sar := &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					User: "sar-unconditional-allow-user",
					ConditionalAuthorization: &authorizationv1.ConditionalAuthorizationOptions{
						Enabled: true,
					},
				},
			}
			response, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SubjectAccessReview: %v", err)
			}
			if !response.Status.Allowed {
				t.Errorf("expected Allowed=true for unconditional allow, got false")
			}
			if response.Status.Denied {
				t.Errorf("expected Denied=false for unconditional allow, got true")
			}
			if response.Status.ConditionalDecision != nil {
				t.Errorf("expected ConditionalDecision to be nil for unconditional allow, got: %+v", response.Status.ConditionalDecision)
			}
		})

		// Test that an unconditional deny from the webhook is properly returned
		// even when conditional authorization is requested.
		t.Run("SubjectAccessReview unconditional deny with conditional authorization requested", func(t *testing.T) {
			webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
				if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Resource == "configmaps" {
					sar.Status.Allowed = false
					sar.Status.Denied = true
					sar.Status.Reason = "unconditionally denied"
				}
			}
			sar := &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					User: "sar-unconditional-deny-user",
					ConditionalAuthorization: &authorizationv1.ConditionalAuthorizationOptions{
						Enabled: true,
					},
				},
			}
			response, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SubjectAccessReview: %v", err)
			}
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false for unconditional deny, got true")
			}
			if !response.Status.Denied {
				t.Errorf("expected Denied=true for unconditional deny, got false")
			}
			if response.Status.ConditionalDecision != nil {
				t.Errorf("expected ConditionalDecision to be nil for unconditional deny, got: %+v", response.Status.ConditionalDecision)
			}
		})
		// Test that an unconditional deny from the webhook is properly returned
		// even when conditional authorization is requested.
		t.Run("SubjectAccessReview conditional deny fails closed when client is not conditions-aware", func(t *testing.T) {
			webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
				if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Resource == "configmaps" {
					sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
						Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
						ConditionsMap: &authorizationv1.ConditionsMap{
							Conditions: []authorizationv1.Condition{
								{
									ID:        "deny-sensitive-label",
									Effect:    authorizationv1.ConditionEffectDeny,
									Condition: `has(object.metadata.labels) && has(object.metadata.labels.sensitive)`,
									Type:      "opaque-cel-condition-type",
								},
							},
						},
					}
				}
			}
			sar := &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Verb:      "create",
						Group:     "",
						Version:   "v1",
						Resource:  "configmaps",
						Namespace: "test-ns",
					},
					User: "sar-unconditional-deny-user",
					// Conditions unaware client
				},
			}
			response, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SubjectAccessReview: %v", err)
			}
			if response.Status.Allowed {
				t.Errorf("expected Allowed=false for unconditional deny, got true")
			}
			if !response.Status.Denied {
				t.Errorf("expected Denied=true for unconditional deny, got false")
			}
			if response.Status.ConditionalDecision != nil {
				t.Errorf("expected ConditionalDecision to be nil for unconditional deny, got: %+v", response.Status.ConditionalDecision)
			}
		})

		// Tests for conditional decision fold-down behavior.
		//
		// When the feature gate is enabled but a request is not subject to conditional
		// authorization (either because the verb is not an admission verb, or because
		// the resource is on the admission exclusion list), the conditional decision
		// returned by the webhook is folded down by the regular Authorize() path:
		//   - Any Deny condition in the decision → DecisionDeny → HTTP 403
		//   - No Deny condition → DecisionNoOpinion → passes to next authorizer in chain
		//
		// This means that:
		//  a) Requests not supported by conditional authorization (e.g. GET/LIST):
		//     the client gets 403 if the fold-down is Deny, or 200 if fold-down is NoOpinion
		//     and a subsequent authorizer (RBAC) allows it.
		//  b) Requests for resources excluded from admission (e.g. create SubjectAccessReview):
		//     same fold-down logic, covering all four combinations of (Deny/NoOpinion) × (no RBAC/RBAC allows).

		// Case (a): GET request (verb not in admissionVerbs, classifier returns false).
		// The webhook returns a conditional decision, but Authorize() is called, not
		// ConditionsAwareAuthorize(), so the fold-down logic in Authorize() applies.
		t.Run("fold down for non-admission verb (GET configmap)", func(t *testing.T) {
			// Pre-create a ConfigMap as admin for the GET tests.
			if _, err := adminClient.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "fold-test-cm"},
			}, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
				t.Fatal(err)
			}

			// Conditional with Deny condition → fold to Deny → 403, even without RBAC.
			t.Run("conditional with deny condition folds to deny", func(t *testing.T) {
				webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
					if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Resource == "configmaps" {
						sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								Conditions: []authorizationv1.Condition{{
									ID: "deny-all", Effect: authorizationv1.ConditionEffectDeny,
									Condition: "true", Type: "opaque",
								}},
							},
						}
					}
				}
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = "fold-get-deny-user"
				_, err := clientset.NewForConfigOrDie(userCfg).CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "fold-test-cm", metav1.GetOptions{})
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !apierrors.IsForbidden(err) && !apierrors.IsUnauthorized(err) {
					t.Fatalf("expected Forbidden or Unauthorized, got: %v", err)
				}
			})

			// Conditional without Deny → fold to NoOpinion → RBAC allows → 200.
			t.Run("conditional without deny folds to no-opinion, RBAC allows", func(t *testing.T) {
				// Reset sarHandler to nil first so that the SAR polling inside
				// GrantUserAuthorization (which checks "get configmaps") is not
				// intercepted by the previous sub-test's conditional-Deny handler.
				webhookServer.handler.sarHandler = nil
				userName := "fold-get-noop-rbac-user"
				authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
					rbacv1.PolicyRule{
						Verbs:     []string{"get"},
						APIGroups: []string{""},
						Resources: []string{"configmaps"},
					},
				)
				webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
					if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Resource == "configmaps" {
						sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								Conditions: []authorizationv1.Condition{{
									ID: "allow-all", Effect: authorizationv1.ConditionEffectAllow,
									Condition: "true", Type: "opaque",
								}},
							},
						}
					}
				}
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = userName
				_, err := clientset.NewForConfigOrDie(userCfg).CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "fold-test-cm", metav1.GetOptions{})
				if err != nil {
					t.Fatalf("expected request to succeed, got: %v", err)
				}
			})
		})

		// Case (b): SubjectAccessReview create (resource is on the admission exclusion list,
		// so conditionalRequestClassifier returns false → regular Authorize() path).
		// Tests all four combinations of fold outcome × RBAC state.
		t.Run("fold down for SubjectAccessReview create (excluded from admission)", func(t *testing.T) {
			// newSAR returns a minimal SAR payload to use in each sub-test.
			newSAR := func() *authorizationv1.SubjectAccessReview {
				return &authorizationv1.SubjectAccessReview{
					Spec: authorizationv1.SubjectAccessReviewSpec{
						ResourceAttributes: &authorizationv1.ResourceAttributes{
							Verb: "get", Group: "", Version: "v1",
							Resource: "configmaps", Namespace: "test-ns",
						},
						User: "some-subject",
					},
				}
			}

			// setConditionalForSARCreate installs a sarHandler that returns the given
			// conditional decision when the webhook is asked to authorize a SAR create,
			// and NoOpinion for all other resources.
			setConditionalForSARCreate := func(decision *authorizationv1.ConditionsAwareDecision) {
				webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
					if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Resource == "subjectaccessreviews" {
						sar.Status.ConditionalDecision = decision
					}
				}
			}

			conditionalWithDeny := &authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorizationv1.ConditionsMap{
					Conditions: []authorizationv1.Condition{{
						ID: "deny-all", Effect: authorizationv1.ConditionEffectDeny,
						Condition: "true", Type: "opaque",
					}},
				},
			}
			conditionalWithoutDeny := &authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorizationv1.ConditionsMap{
					Conditions: []authorizationv1.Condition{{
						ID: "allow-all", Effect: authorizationv1.ConditionEffectAllow,
						Condition: "true", Type: "opaque",
					}},
				},
			}

			// Case 1: fold to Deny (has Deny condition), no RBAC → 403.
			t.Run("fold to deny, no RBAC", func(t *testing.T) {
				setConditionalForSARCreate(conditionalWithDeny)
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = "fold-sar-deny-no-rbac-user"
				_, err := clientset.NewForConfigOrDie(userCfg).AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), newSAR(), metav1.CreateOptions{})
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !apierrors.IsForbidden(err) && !apierrors.IsUnauthorized(err) {
					t.Fatalf("expected Forbidden or Unauthorized, got: %v", err)
				}
			})

			// Case 2: fold to Deny (has Deny condition), RBAC allows SAR creates → 403.
			// Deny short-circuits the union authorizer chain, so RBAC is never consulted.
			t.Run("fold to deny, RBAC allows SAR creates", func(t *testing.T) {
				userName := "fold-sar-deny-with-rbac-user"

				// Reset sarHandler to nil first so that the SAR polling inside
				// GrantUserAuthorization (which checks "create subjectaccessreviews") is not
				// intercepted by the previous sub-test's webhook authorizer handler.
				webhookServer.handler.sarHandler = nil
				// When this function returns, userName can successfully create SARs
				authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
					rbacv1.PolicyRule{
						Verbs:     []string{"create"},
						APIGroups: []string{"authorization.k8s.io"},
						Resources: []string{"subjectaccessreviews"},
					},
				)
				setConditionalForSARCreate(conditionalWithDeny)
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = userName
				_, err = clientset.NewForConfigOrDie(userCfg).AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), newSAR(), metav1.CreateOptions{})
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !apierrors.IsForbidden(err) && !apierrors.IsUnauthorized(err) {
					t.Fatalf("expected Forbidden or Unauthorized, got: %v", err)
				}
			})

			// Case 3: fold to NoOpinion (no Deny condition), no RBAC → 403.
			t.Run("fold to no-opinion, no RBAC", func(t *testing.T) {
				setConditionalForSARCreate(conditionalWithoutDeny)
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = "fold-sar-noop-no-rbac-user"
				_, err := clientset.NewForConfigOrDie(userCfg).AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), newSAR(), metav1.CreateOptions{})
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !apierrors.IsForbidden(err) && !apierrors.IsUnauthorized(err) {
					t.Fatalf("expected Forbidden or Unauthorized, got: %v", err)
				}
			})

			// Case 4: fold to NoOpinion (no Deny condition), RBAC allows SAR creates → 200.
			// NoOpinion passes to the next authorizer in the chain, where RBAC allows.
			t.Run("fold to no-opinion, RBAC allows SAR creates", func(t *testing.T) {
				// Reset the sarHandler to no-op so that the SAR polling inside
				// GrantUserAuthorization is not affected by the previous test's handler.
				// Then install the conditional handler only after RBAC is ready.
				webhookServer.handler.sarHandler = nil
				userName := "fold-sar-noop-with-rbac-user"
				authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
					rbacv1.PolicyRule{
						Verbs:     []string{"create"},
						APIGroups: []string{"authorization.k8s.io"},
						Resources: []string{"subjectaccessreviews"},
					},
				)
				setConditionalForSARCreate(conditionalWithoutDeny)
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = userName
				_, err := clientset.NewForConfigOrDie(userCfg).AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), newSAR(), metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("expected request to succeed, got: %v", err)
				}
			})
		})
	}
}

func boolPtr(b bool) *bool {
	return &b
}

func int32Ptr(i int32) *int32 {
	return &i
}

// TODO(luxas): Reactivate this when we add support for in-tree evaluation.
/*
// compoundAuthzSARHandler returns a SAR handler that implements compound
// authorization: creating a resource with the "protected-label" label
// requires the additional "use protectedlabels" permission, checked via the
// authorizer CEL function. The handler returns NoOpinion for non-matching SARs
// (e.g. the authorizer function's internal permission check).
func compoundAuthzSARHandler(matchResource string) func(sar *authorizationv1.SubjectAccessReview) {
	return func(sar *authorizationv1.SubjectAccessReview) {
		if sar.Spec.ResourceAttributes == nil || sar.Spec.ResourceAttributes.Resource != matchResource {
			return // NoOpinion for non-matching SARs (e.g. the authorizer function's internal check)
		}
		sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationv1.ConditionsMap{
				Conditions: []authorizationv1.Condition{
					{
						ID:     "require-protectedlabel-permission",
						Effect: authorizationv1.ConditionEffectAllow,
						Condition: `!has(object.metadata.labels) || !('protected-label' in object.metadata.labels) || ` +
							`authorizer.group('example.com').resource('protectedlabels').name('protected-label').check('use').allowed()`,
						Type:        "k8s.io/authorization-cel",
						Description: "compound authorization: require 'use' permission on protectedlabels resource when protected-label is set",
					},
				},
			},
		}
	}
}
*/

// hpaCPUUtilizationSARHandler returns a processSAR function for use with
// celConditionalTestCases. It sets CEL conditions on HPA SARs that require
// CPU utilization to be at most 80%. The CEL expression is version-specific:
// for v1 it checks spec.targetCPUUtilizationPercentage, for v2 it iterates
// spec.metrics to find the CPU resource metric and checks averageUtilization.
// For updates, both the old and new objects must satisfy the condition.
func hpaCPUUtilizationSARHandler(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
	if sar.Spec.ResourceAttributes == nil || sar.Spec.ResourceAttributes.Resource != "horizontalpodautoscalers" {
		return
	}

	var objectCondition, oldObjectCondition string
	switch sar.Spec.ResourceAttributes.Version {
	case "v1":
		objectCondition = `has(object.spec.targetCPUUtilizationPercentage) && object.spec.targetCPUUtilizationPercentage <= 80`
		oldObjectCondition = `has(oldObject.spec.targetCPUUtilizationPercentage) && oldObject.spec.targetCPUUtilizationPercentage <= 80`
	default: // v2, v2beta2, etc.
		objectCondition = `has(object.spec.metrics) && object.spec.metrics.exists(m, ` +
			`m.type == "Resource" && ` +
			`has(m.resource) && ` +
			`m.resource.name == "cpu" && ` +
			`has(m.resource.target) && ` +
			`m.resource.target.type == "Utilization" && ` +
			`has(m.resource.target.averageUtilization) && ` +
			`m.resource.target.averageUtilization <= 80)`
		oldObjectCondition = `has(oldObject.spec.metrics) && oldObject.spec.metrics.exists(m, ` +
			`m.type == "Resource" && ` +
			`has(m.resource) && ` +
			`m.resource.name == "cpu" && ` +
			`has(m.resource.target) && ` +
			`m.resource.target.type == "Utilization" && ` +
			`has(m.resource.target.averageUtilization) && ` +
			`m.resource.target.averageUtilization <= 80)`
	}

	var condition string
	switch sar.Spec.ResourceAttributes.Verb {
	case "create":
		condition = objectCondition
	case "update":
		condition = objectCondition + " && " + oldObjectCondition
	default:
		return
	}

	sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorizationv1.ConditionsMap{
			Conditions: []authorizationv1.Condition{
				{
					ID:          "limit-cpu-utilization",
					Effect:      authorizationv1.ConditionEffectAllow,
					Condition:   condition,
					Type:        conditionsType,
					Description: "only allow HPAs with CPU utilization at most 80%",
				},
			},
		},
	}
}

// crdReplicasSARHandler returns a processSAR function for use with
// celConditionalTestCases. It sets CEL conditions on ScalableWidget SARs that
// require replicas to be at most 10. The CEL expression is version-specific:
// for v1 it checks spec.replicas (integer), for v2 it checks spec.replicas.max
// (integer nested in object). For updates, both the old and new objects must
// satisfy the condition.
func crdReplicasSARHandler(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
	if sar.Spec.ResourceAttributes == nil || sar.Spec.ResourceAttributes.Resource != "scalablewidgets" {
		return
	}

	var objectCondition, oldObjectCondition string
	switch sar.Spec.ResourceAttributes.Version {
	case "v1":
		objectCondition = `has(object.spec.replicas) && object.spec.replicas <= 10`
		oldObjectCondition = `has(oldObject.spec.replicas) && oldObject.spec.replicas <= 10`
	case "v2":
		objectCondition = `has(object.spec.replicas) && has(object.spec.replicas.max) && object.spec.replicas.max <= 10`
		oldObjectCondition = `has(oldObject.spec.replicas) && has(oldObject.spec.replicas.max) && oldObject.spec.replicas.max <= 10`
	default:
		return
	}

	var condition string
	switch sar.Spec.ResourceAttributes.Verb {
	case "create":
		condition = objectCondition
	case "update":
		condition = objectCondition + " && " + oldObjectCondition
	default:
		return
	}

	sar.Status.ConditionalDecision = &authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorizationv1.ConditionsMap{
			Conditions: []authorizationv1.Condition{
				{
					ID:          "limit-replicas",
					Effect:      authorizationv1.ConditionEffectAllow,
					Condition:   condition,
					Type:        conditionsType,
					Description: "only allow if replicas <= 10",
				},
			},
		},
	}
}

// acrEvaluateCEL returns an ACR handler that reads conditions from the ACR
// request, verifies the conditions type, evaluates CEL expressions against the
// write request objects, and sets the response.
func acrEvaluateCEL(t *testing.T, expectedConditionsType string) func(acr *authorizationv1alpha1.AuthorizationConditionsReview) {
	return func(acr *authorizationv1alpha1.AuthorizationConditionsReview) {
		if acr.Request.Decision.Type != authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap {
			t.Fatalf("expected ConditionsMap decision to evaluate, got %q", acr.Request.Decision.Type)
		}
		conditionsMap := acr.Request.Decision.ConditionsMap
		if conditionsMap == nil {
			t.Fatalf("expected ConditionsMap in ACR to be non-nil")
		}
		for _, cond := range conditionsMap.Conditions {
			if cond.Type != expectedConditionsType {
				t.Fatalf("expected condition type %q, got %q for condition %q", expectedConditionsType, cond.Type, cond.ID)
			}
		}
		decisionType := celEvaluateConditions(t, acr.Request.AdmissionControlData, conditionsMap)
		acr.Response = &authorizationv1alpha1.AuthorizationConditionsResponse{
			Decision: authorizationv1alpha1.ConditionsAwareDecision{
				Type: decisionType,
			},
		}
	}
}

// celConditionalTestCases creates three webhook behavior variants for the same
// conditional authorization test, asserting the same outcome regardless of
// whether conditions are evaluated out-of-tree (via the webhook) or in-tree
// (via the built-in CEL evaluator):
//
//   - "using-webhook-only": conditions use an opaque type that only the webhook
//     can evaluate. Verifies the out-of-tree evaluation path.
//   - "in-process-eval-only": conditions use "k8s.io/authorization-cel" so the
//     built-in evaluator handles them. The ACR handler is set to panic if called,
//     verifying that the webhook is NOT consulted.
//   - "if-in-process-fails-call-webhook": conditions use "k8s.io/authorization-cel"
//     but are prefixed with invalid syntax so in-tree evaluation fails. Verifies
//     that the kube-apiserver falls back to the webhook, which strips the prefix
//     and evaluates successfully.
func celConditionalTestCases(processSAR func(sar *authorizationv1.SubjectAccessReview, conditionsType string)) map[string]func(*webhookServerHandler) {
	return map[string]func(*webhookServerHandler){
		// When the condition type is opaque, the webhook should be called to resolve the condition.
		"using-webhook-only": func(ws *webhookServerHandler) {
			ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
				processSAR(sar, "opaque-cel-condition-type")
			}
			ws.acrHandler = acrEvaluateCEL(ws.t, "opaque-cel-condition-type")
		},
		// TODO(luxas): Reactivate this when we support in-tree evaluation.
		// When the condition type is k8s.io/authorization-cel, in-tree evaluation handles it.
		/*"in-process-eval-only": func(ws *webhookServerHandler) {
			ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
				processSAR(sar, "k8s.io/authorization-cel")
			}
			// Ensure no calls to ACR are made, as the above should be handled in-tree only.
			// If Kubernetes did webhook out to us, the test panics as the conditions type does not match.
			ws.acrHandler = acrEvaluateCEL(ws.t, "nonexistent-panic-on-ACR-webhook")
		},
		// When in-tree evaluation fails (e.g. newer CEL syntax), it falls back to the webhook.
		"if-in-process-fails-call-webhook": func(ws *webhookServerHandler) {
			ws.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
				processSAR(sar, "k8s.io/authorization-cel")

				if d := sar.Status.ConditionalDecision; d != nil {
					if d.ConditionsMap == nil {
						ws.t.Fatal("current test expects a ConditionsMap")
					}
					// Corrupt each condition with an invalid prefix so in-tree evaluation fails.
					for j := range d.ConditionsMap.Conditions {
						d.ConditionsMap.Conditions[j].Condition = "unsupported_in_tree && " + d.ConditionsMap.Conditions[j].Condition
					}
				}
			}
			// The webhook should be called as fallback. Strip the prefix and evaluate.
			ws.acrHandler = func(acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				decision := &acr.Request.Decision
				if decision.ConditionsMap == nil {
					ws.t.Fatal("current test expects a conditionset")
				}
				for i := range decision.ConditionsMap.Conditions {
					decision.ConditionsMap.Conditions[i].Condition = strings.TrimPrefix(decision.ConditionsMap.Conditions[i].Condition, "unsupported_in_tree && ")
				}
				acrEvaluateCEL(ws.t, "k8s.io/authorization-cel")(acr)
			}
		},*/
	}
}

// webhookServer wraps an httptest.Server serving both SubjectAccessReview and
// AuthorizationConditionsReview on its /authorize endpoint.
type webhookServer struct {
	server  *httptest.Server
	handler *webhookServerHandler
}

type webhookServerHandler struct {
	t          *testing.T
	sarHandler func(sar *authorizationv1.SubjectAccessReview)
	acrHandler func(acr *authorizationv1alpha1.AuthorizationConditionsReview)
}

func newWebhookServer(t *testing.T) *webhookServer {
	handler := &webhookServerHandler{t: t}
	mux := http.NewServeMux()
	mux.HandleFunc("/authorize", handler.serveSAR)
	mux.HandleFunc("/conditionsreview", handler.serveACR)
	server := httptest.NewTLSServer(mux)
	return &webhookServer{
		server:  server,
		handler: handler,
	}
}

func (h *webhookServerHandler) serveSAR(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "only POST is supported", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		h.t.Errorf("failed to read request body: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer func() { _ = req.Body.Close() }()

	h.handleSAR(w, body)
}

func (h *webhookServerHandler) serveACR(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "only POST is supported", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		h.t.Errorf("failed to read request body: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer func() { _ = req.Body.Close() }()

	h.handleACR(w, body)
}

func (h *webhookServerHandler) handleSAR(w http.ResponseWriter, body []byte) {
	sar := &authorizationv1.SubjectAccessReview{}
	if err := json.Unmarshal(body, sar); err != nil {
		h.t.Errorf("failed to unmarshal SubjectAccessReview: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	h.t.Logf("SAR request for user=%q resource=%v verb=%v ns=%v",
		sar.Spec.User,
		safeResourceAttr(sar, func(ra *authorizationv1.ResourceAttributes) string { return ra.Resource }),
		safeResourceAttr(sar, func(ra *authorizationv1.ResourceAttributes) string { return ra.Verb }),
		safeResourceAttr(sar, func(ra *authorizationv1.ResourceAttributes) string { return ra.Namespace }),
	)

	if h.sarHandler != nil {
		h.sarHandler(sar)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(sar); err != nil {
		h.t.Errorf("failed to encode SAR response: %v", err)
	}
}

func (h *webhookServerHandler) handleACR(w http.ResponseWriter, body []byte) {
	acr := &authorizationv1alpha1.AuthorizationConditionsReview{}
	if err := json.Unmarshal(body, acr); err != nil {
		h.t.Errorf("failed to unmarshal AuthorizationConditionsReview: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// h.t.Logf("ACR request: decision conditions count=%d", len(acr.Request.Decision.Conditions))

	if h.acrHandler != nil {
		h.acrHandler(acr)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(acr); err != nil {
		h.t.Errorf("failed to encode ACR response: %v", err)
	}
}

func safeResourceAttr(sar *authorizationv1.SubjectAccessReview, fn func(*authorizationv1.ResourceAttributes) string) string {
	if sar.Spec.ResourceAttributes != nil {
		return fn(sar.Spec.ResourceAttributes)
	}
	return "<non-resource>"
}

// celEvaluateConditions evaluates CEL conditions from a serialized decision
// against the objects in the write request. It follows the condition precedence:
// Deny > NoOpinion > Allow (matching EvaluateConditionSet semantics).
// Returns (allowed, denied).
func celEvaluateConditions(t *testing.T, wr *authorizationv1alpha1.AuthorizationConditionsTargetAdmissionControl, conditionsMap *authorizationv1alpha1.ConditionsMap) authorizationv1alpha1.ConditionsAwareDecisionType {
	t.Helper()

	if conditionsMap == nil || len(conditionsMap.Conditions) == 0 {
		t.Fatal("expected a non-empty ConditionsMap in celEvaluateConditions")
	}

	env, err := cel.NewEnv(
		cel.Variable("object", cel.DynType),
		cel.Variable("oldObject", cel.DynType),
		cel.Variable("request", cel.DynType),
	)
	if err != nil {
		t.Fatalf("failed to create CEL env: %v", err)
	}

	// Deserialize object and oldObject from RawExtension JSON
	var objectMap map[string]any
	if len(wr.Object.Raw) > 0 {
		if err := json.Unmarshal(wr.Object.Raw, &objectMap); err != nil {
			t.Fatalf("failed to unmarshal object: %v", err)
		}
	}

	var oldObjectMap map[string]any
	if len(wr.OldObject.Raw) > 0 {
		if err := json.Unmarshal(wr.OldObject.Raw, &oldObjectMap); err != nil {
			t.Fatalf("failed to unmarshal oldObject: %v", err)
		}
	}

	requestMap := map[string]any{
		// Expose only previously-unseen data for now.
		"operation": string(wr.Operation),
	}

	vars := map[string]any{
		"object":    objectMap,
		"oldObject": oldObjectMap,
		"request":   requestMap,
	}

	// Phase 1: Deny conditions
	for _, cond := range conditionsMap.Conditions {
		if cond.Effect != authorizationv1alpha1.ConditionEffectDeny {
			continue
		}
		if evalCEL(t, env, cond.Condition, vars) {
			return authorizationv1alpha1.ConditionsAwareDecisionTypeDeny
		}
	}

	// Phase 2: NoOpinion conditions
	for _, cond := range conditionsMap.Conditions {
		if cond.Effect != authorizationv1alpha1.ConditionEffectNoOpinion {
			continue
		}
		if evalCEL(t, env, cond.Condition, vars) {
			return authorizationv1alpha1.ConditionsAwareDecisionTypeNoOpinion
		}
	}

	// Phase 3: Allow conditions
	for _, cond := range conditionsMap.Conditions {
		if cond.Effect != authorizationv1alpha1.ConditionEffectAllow {
			continue
		}
		if evalCEL(t, env, cond.Condition, vars) {
			return authorizationv1alpha1.ConditionsAwareDecisionTypeAllow
		}
	}

	// Default: NoOpinion
	return authorizationv1alpha1.ConditionsAwareDecisionTypeNoOpinion
}

// evalCEL compiles and evaluates a single CEL expression, returning true/false.
func evalCEL(t *testing.T, env *cel.Env, expr string, vars map[string]any) bool {
	t.Helper()
	ast, issues := env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		t.Fatalf("CEL compile error for %q: %v", expr, issues.Err())
	}
	prg, err := env.Program(ast)
	if err != nil {
		t.Fatalf("CEL program error for %q: %v", expr, err)
	}
	out, _, err := prg.Eval(vars)
	if err != nil {
		t.Fatalf("CEL eval error for %q: %v", expr, err)
	}
	result, ok := out.Value().(bool)
	if !ok {
		t.Fatalf("CEL expression %q did not return bool, got %T", expr, out.Value())
	}
	return result
}
