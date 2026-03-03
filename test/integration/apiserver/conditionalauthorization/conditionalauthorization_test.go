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
	"strings"
	"testing"

	"github.com/google/cel-go/cel"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	if err := os.WriteFile(kubeconfigPath, []byte(fmt.Sprintf(`
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
`, webhookServer.server.URL+"/authorize", webhookServer.server.URL+"/conditionsreview")), 0644); err != nil {
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
	if err := os.WriteFile(authzConfigPath, []byte(fmt.Sprintf(`
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
`, kubeconfigPath, conditionsReviewSection)), 0644); err != nil {
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
				_, err := client.CoreV1().ConfigMaps("test-ns").List(context.TODO(), metav1.ListOptions{})
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "always-allow",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   "true",
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "always-allow",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   "true",
								Description: "base allow condition",
							},
							{
								ID:          "always-deny",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectDeny,
								Condition:   "true",
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "no-opinion",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectNoOpinion,
								Condition:   "true",
								Description: "no opinion condition",
							},
						},
					},
				}
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").List(context.TODO(), metav1.ListOptions{})
				return err
			},
			// With feature enabled: conditional => NoOpinion from conditions evaluation.
			// The original Authorize() in the chain returns Conditional (which CanBecomeAllowed),
			// so RBAC is never consulted. The conditions evaluator returns NoOpinion => denied.
			expectAllowed: false,
			// When disabled: conditional decision is NoOpinion, RBAC is consulted and allows.
			expectAllowedWhenDisabled: boolPtr(true),
		},

		// CEL-based conditional authorization tests.
		// These test that CEL expressions flow through the SAR -> Decision -> conditions
		// evaluation pipeline. The assertions are the same whether conditions are evaluated
		// in-tree (k8s.io/authorization-cel) or out-of-tree (opaque type via webhook).
		{
			name: "cel allow by name pattern",
			user: "cel-name-user",
			webhookBehaviors: celConditionalTestCases(func(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "allow-safe-prefix",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   `object.metadata.name.startsWith("safe-")`,
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "allow-safe-prefix",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   `object.metadata.name.startsWith("safe-")`,
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "allow-all",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   "true",
								Description: "base allow",
							},
							{
								ID:     "deny-restricted-label",
								Effect: authorizationv1.SubjectAccessReviewConditionEffectDeny,
								Condition: `has(object.metadata.labels) && ` +
									`has(object.metadata.labels.restricted) && ` +
									`object.metadata.labels.restricted == "true"`,
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:     "allow-approved-data",
								Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition: `has(object.data) && ` +
									`has(object.data.approved) && ` +
									`object.data.approved == "yes"`,
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:     "allow-approved-data",
								Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition: `has(object.data) && ` +
									`has(object.data.approved) && ` +
									`object.data.approved == "yes"`,
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "allow-creates",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   `request.operation == "CREATE"`,
								Description: "allow create operations",
							},
							{
								ID:          "deny-updates",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectDeny,
								Condition:   `request.operation == "UPDATE"`,
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "allow-all",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   "true",
								Description: "allow everything",
							},
							{
								ID:          "noop-all",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectNoOpinion,
								Condition:   "true",
								Description: "no opinion on everything",
							},
							{
								ID:          "deny-all",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectDeny,
								Condition:   "true",
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
				sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
					{
						ConditionsType: conditionsType,
						Conditions: []authorizationv1.SubjectAccessReviewCondition{
							{
								ID:          "allow-all",
								Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
								Condition:   "true",
								Description: "allow everything",
							},
							{
								ID:     "noop-on-pending-review",
								Effect: authorizationv1.SubjectAccessReviewConditionEffectNoOpinion,
								Condition: `has(object.metadata.labels) && ` +
									`has(object.metadata.labels.review) && ` +
									`object.metadata.labels.review == "pending"`,
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
					sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
						{
							ConditionsType: conditionsType,
							Conditions: []authorizationv1.SubjectAccessReviewCondition{
								{
									ID:     "require-owner-label",
									Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
									Condition: `has(object.metadata.labels) && ` +
										`has(object.metadata.labels.creator) && ` +
										`object.metadata.labels.creator == "update-create-allow-user"`,
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
				/*_, err := client.CoordinationV1().Leases("test-ns").Apply(context.TODO(),
				applyconfigurationscoordinationv1.
					Lease("update-create-denied", "test-ns").
					WithLabels(map[string]string{"creator": "update-create-allow-user"}),
				metav1.ApplyOptions{
					FieldManager: "foo",
				})*/
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
					sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
						{
							ConditionsType: conditionsType,
							Conditions: []authorizationv1.SubjectAccessReviewCondition{
								{
									ID:     "require-owner-label",
									Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
									Condition: `has(object.metadata.labels) && ` +
										`has(object.metadata.labels.creator) && ` +
										`object.metadata.labels.creator == "update-create-deny-user"`,
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
				/*_, err := client.CoordinationV1().Leases("test-ns").Apply(context.TODO(),
				applyconfigurationscoordinationv1.
					Lease("update-create-denied", "test-ns").
					WithLabels(map[string]string{"creator": "not-authorized-user"}),
				metav1.ApplyOptions{
					FieldManager: "foo",
				})*/
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
					sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
						{
							ConditionsType: conditionsType,
							Conditions: []authorizationv1.SubjectAccessReviewCondition{
								{
									ID:     "allow-unclassified",
									Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
									Condition: `has(object.metadata.labels) && ` +
										`has(object.metadata.labels.classified) && ` +
										`object.metadata.labels.classified == "false"`,
									Description: "only allow creates when classified=false",
								},
							},
						},
					}
				case "create":
					// Conditionally allow creates: only when creator=update-create-deny-user
					sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
						{
							ConditionsType: conditionsType,
							Conditions: []authorizationv1.SubjectAccessReviewCondition{
								{
									ID:     "require-owner-label",
									Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
									Condition: `has(object.metadata.labels) && ` +
										`has(object.metadata.labels.creator) && ` +
										`object.metadata.labels.creator == "update-create-deny-user"`,
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
				/*_, err := client.CoordinationV1().Leases("test-ns").Apply(context.TODO(),
				applyconfigurationscoordinationv1.
					Lease("update-create-denied-by-update-condition", "test-ns").
					// Satisfies the create condition, but not the update one
					WithLabels(map[string]string{"creator": "update-create-deny-user"}),
				metav1.ApplyOptions{
					FieldManager: "foo",
				})*/
				return err
			},
			expectAllowed: false,
		},
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
						sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
							{
								ConditionsType: "k8s.io/authorization-cel",
								Conditions: []authorizationv1.SubjectAccessReviewCondition{
									{
										ID:     "require-production-namespace",
										Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
										Condition: `namespaceObject != null && ` +
											`has(namespaceObject.metadata) && ` +
											`has(namespaceObject.metadata.labels) && ` +
											`'env' in namespaceObject.metadata.labels && ` +
											`namespaceObject.metadata.labels['env'] == 'production'`,
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
						sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
							{
								ConditionsType: "k8s.io/authorization-cel",
								Conditions: []authorizationv1.SubjectAccessReviewCondition{
									{
										ID:     "require-staging-namespace",
										Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
										Condition: `namespaceObject != null && ` +
											`has(namespaceObject.metadata) && ` +
											`has(namespaceObject.metadata.labels) && ` +
											`'env' in namespaceObject.metadata.labels && ` +
											`namespaceObject.metadata.labels['env'] == 'staging'`,
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
		},

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
								Verbs:     []string{"list", "get"},
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
}

func boolPtr(b bool) *bool {
	return &b
}

func int32Ptr(i int32) *int32 {
	return &i
}

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
		sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
			{
				ConditionsType: "k8s.io/authorization-cel",
				Conditions: []authorizationv1.SubjectAccessReviewCondition{
					{
						ID:     "require-protectedlabel-permission",
						Effect: authorizationv1.SubjectAccessReviewConditionEffectAllow,
						Condition: `!has(object.metadata.labels) || !('protected-label' in object.metadata.labels) || ` +
							`authorizer.group('example.com').resource('protectedlabels').name('protected-label').check('use').allowed()`,
						Description: "compound authorization: require 'use' permission on protectedlabels resource when protected-label is set",
					},
				},
			},
		}
	}
}

// hpaCPUUtilizationSARHandler returns a processSAR function for use with
// celConditionalTestCases. It sets CEL conditions on HPA SARs that require
// CPU utilization to be at most 80%. The CEL expression is version-specific:
// for v1 it checks spec.targetCPUUtilizationPercentage, for v2 it iterates
// spec.metrics to find the CPU resource metric and checks averageUtilization.
func hpaCPUUtilizationSARHandler(sar *authorizationv1.SubjectAccessReview, conditionsType string) {
	if sar.Spec.ResourceAttributes == nil || sar.Spec.ResourceAttributes.Resource != "horizontalpodautoscalers" {
		return
	}

	var condition string
	switch sar.Spec.ResourceAttributes.Version {
	case "v1":
		condition = `has(object.spec.targetCPUUtilizationPercentage) && object.spec.targetCPUUtilizationPercentage <= 80`
	default: // v2, v2beta2, etc.
		condition = `has(object.spec.metrics) && object.spec.metrics.exists(m, ` +
			`m.type == "Resource" && ` +
			`has(m.resource) && ` +
			`m.resource.name == "cpu" && ` +
			`has(m.resource.target) && ` +
			`m.resource.target.type == "Utilization" && ` +
			`has(m.resource.target.averageUtilization) && ` +
			`m.resource.target.averageUtilization <= 80)`
	}

	sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
		{
			ConditionsType: conditionsType,
			Conditions: []authorizationv1.SubjectAccessReviewCondition{
				{
					ID:          "limit-cpu-utilization",
					Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
					Condition:   condition,
					Description: "only allow HPAs with CPU utilization at most 80%",
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
		decision := acr.Request.Decision
		// The webhook serializes conditional decisions as a chain; unwrap.
		if len(decision.ConditionalDecisionChain) > 0 {
			if len(decision.ConditionalDecisionChain) != 1 {
				t.Fatalf("expected exactly one ConditionSet in chain, got %d", len(decision.ConditionalDecisionChain))
			}
			decision = decision.ConditionalDecisionChain[0]
		}
		if decision.ConditionsType != expectedConditionsType {
			t.Fatalf("expected conditions type %q, got %q", expectedConditionsType, decision.ConditionsType)
		}
		allowed, denied := celEvaluateConditions(t, acr.Request.WriteRequest, &decision)
		acr.Response = &authorizationv1alpha1.AuthorizationConditionsResponse{
			SubjectAccessReviewAuthorizationDecision: authorizationv1alpha1.SubjectAccessReviewAuthorizationDecision{
				Allowed: allowed,
				Denied:  denied,
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
		// When the condition type is k8s.io/authorization-cel, in-tree evaluation handles it.
		"in-process-eval-only": func(ws *webhookServerHandler) {
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

				if len(sar.Status.ConditionalDecisionChain) != 0 {
					if len(sar.Status.ConditionalDecisionChain) != 1 {
						ws.t.Fatal("current test expects exactly 0 or 1 conditional decision")
					}
					d := &sar.Status.ConditionalDecisionChain[0]
					if len(d.Conditions) == 0 {
						ws.t.Fatal("current test expects a conditionset in the first conditional decision")
					}
					// Corrupt each condition with an invalid prefix so in-tree evaluation fails.
					for j := range d.Conditions {
						d.Conditions[j].Condition = "unsupported_in_tree && " + d.Conditions[j].Condition
					}
				}
			}
			// The webhook should be called as fallback. Strip the prefix and evaluate.
			ws.acrHandler = func(acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				decision := &acr.Request.Decision
				if len(decision.Conditions) == 0 {
					ws.t.Fatal("current test expects a conditionset")
				}
				for i := range decision.Conditions {
					decision.Conditions[i].Condition = strings.TrimPrefix(decision.Conditions[i].Condition, "unsupported_in_tree && ")
				}
				acrEvaluateCEL(ws.t, "k8s.io/authorization-cel")(acr)
			}
		},
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
	defer req.Body.Close()

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
	defer req.Body.Close()

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

	h.t.Logf("ACR request: decision conditions count=%d", len(acr.Request.Decision.Conditions))

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
func celEvaluateConditions(t *testing.T, wr *authorizationv1alpha1.AuthorizationConditionsWriteRequest, serializedDecision *authorizationv1alpha1.SubjectAccessReviewAuthorizationDecision) (bool, bool) {
	t.Helper()

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
		"operation": string(wr.Operation),
		"namespace": wr.Namespace,
		"name":      wr.Name,
	}

	vars := map[string]any{
		"object":    objectMap,
		"oldObject": oldObjectMap,
		"request":   requestMap,
	}

	if len(serializedDecision.Conditions) == 0 || len(serializedDecision.ConditionsType) == 0 {
		// TODO(luxas): This could be extended to cover the recursive case.
		t.Fatal("expected a ConditionSet in celEvaluateConditions")
	}

	// Phase 1: Deny conditions
	for _, cond := range serializedDecision.Conditions {
		if cond.Effect != authorizationv1alpha1.SubjectAccessReviewConditionEffectDeny {
			continue
		}
		if evalCEL(t, env, cond.Condition, vars) {
			return false, true
		}
	}

	// Phase 2: NoOpinion conditions
	for _, cond := range serializedDecision.Conditions {
		if cond.Effect != authorizationv1alpha1.SubjectAccessReviewConditionEffectNoOpinion {
			continue
		}
		if evalCEL(t, env, cond.Condition, vars) {
			return false, false
		}
	}

	// Phase 3: Allow conditions
	for _, cond := range serializedDecision.Conditions {
		if cond.Effect != authorizationv1alpha1.SubjectAccessReviewConditionEffectAllow {
			continue
		}
		if evalCEL(t, env, cond.Condition, vars) {
			return true, false
		}
	}

	// Default: NoOpinion
	return false, false
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
