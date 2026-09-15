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
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/go-cmp/cmp"

	admissionv1 "k8s.io/api/admission/v1"
	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"

	crdconversiontesting "k8s.io/apiextensions-apiserver/test/integration/conversion"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	authorizationv1apiserver "k8s.io/apiserver/pkg/apis/authorization/v1"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	authorizationinternalv1 "k8s.io/kubernetes/pkg/apis/authorization/v1"
	authorizationutil "k8s.io/kubernetes/pkg/registry/authorization/util"
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

type conditionalAuthzTestCase struct {
	name string
	// user is the username that will be impersonated
	user string
	// authorizers configures the per-variant webhook authorizer for this
	// test case. Each entry becomes a subtest under tc.name; the map key
	// is the variant name (used as a suffix on impersonated users and
	// resource names to avoid cross-subtest cache/collision).
	// Multiple entries can assert the same outcome under different
	// webhook configurations (e.g. out-of-tree webhook evaluation vs
	// in-tree CEL evaluation) — today only "using-webhook-only" is
	// active; more variants will land when in-tree evaluation is
	// re-enabled.
	authorizers map[string]authorizer.Authorizer
	// makeRequest creates a client with the given user and performs an API request.
	// Returns an error if the request fails. The suffix parameter is derived from
	// the variant name and must be used in resource names to avoid
	// conflicts between subtests that share the same API server.
	makeRequest func(t *testing.T, client *clientset.Clientset, suffix string) error
	// expectAllowed is true if the request should be allowed
	expectAllowed bool
	// expectAllowedWhenDisabled overrides expectAllowed when the feature is disabled.
	// If nil, uses expectAllowed.
	expectAllowedWhenDisabled *bool
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
apiVersion: apiserver.config.k8s.io/v1
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

	closeConversion, conversionClientConfig, err := crdconversiontesting.StartConversionWebhookServer(
		crdconversiontesting.NewObjectConverterWebhookHandler(t, convertScalableWidget),
	)
	if err != nil {
		t.Fatalf("failed to start conversion webhook: %v", err)
	}
	t.Cleanup(closeConversion)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		fmt.Sprintf("--feature-gates=ConditionalAuthorization=%v", featureEnabled),
		"--authorization-config=" + authzConfigPath,
	}, framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	// Create the "test-ns" namespace for tests, with labels for namespaceObject tests
	_, err = adminClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-ns",
			Labels: map[string]string{"env": "production", "team": "platform"},
		},
	}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatal(err)
	}

	createAndWaitForCRD(t, server, conversionClientConfig)

	testCases := []conditionalAuthzTestCase{
		// Unconditional decisions: the webhook returns a concrete Allow/Deny/NoOpinion.
		{
			name: "unconditional allow from webhook",
			user: "allow-user",
			authorizers: map[string]authorizer.Authorizer{
				"": testAuthorizer(func(_ context.Context, _ authorizer.Attributes) authorizer.ConditionsAwareDecision {
					return authorizer.ConditionsAwareDecisionAllow("unconditionally allowed", nil)
				}),
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
			authorizers: map[string]authorizer.Authorizer{
				"": testAuthorizer(func(_ context.Context, _ authorizer.Attributes) authorizer.ConditionsAwareDecision {
					return authorizer.ConditionsAwareDecisionDeny("unconditionally denied", nil)
				}),
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
			authorizers: map[string]authorizer.Authorizer{
				"": testAuthorizer(func(_ context.Context, _ authorizer.Attributes) authorizer.ConditionsAwareDecision {
					return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
				}),
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
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/always-allow",
							Condition:   "true",
							Type:        conditionsType,
							Description: "always allow condition",
						},
					},
				)
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
			expectAllowedWhenDisabled: new(false),
		},
		{
			name: "conditional deny - condition evaluates to deny",
			user: "conditional-deny-user",
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/always-deny",
							Condition:   "true",
							Type:        conditionsType,
							Description: "always deny condition",
						},
					},
					nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/always-allow",
							Condition:   "true",
							Type:        conditionsType,
							Description: "base allow condition",
						},
					},
				)
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
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/no-opinion",
							Condition:   "true",
							Type:        conditionsType,
							Description: "no opinion condition",
						},
					},
					nil,
				)
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
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/allow-safe-prefix",
							Condition:   `object.metadata.name.startsWith("safe-")`,
							Type:        conditionsType,
							Description: "only allow configmaps with safe- prefix",
						},
					},
				)
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "safe-configmap" + suffix},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name: "cel deny by name pattern mismatch",
			user: "cel-name-deny-user",
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/allow-safe-prefix",
							Condition:   `object.metadata.name.startsWith("safe-")`,
							Type:        conditionsType,
							Description: "only allow configmaps with safe- prefix",
						},
					},
				)
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
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID: "example.com/deny-restricted-label",
							Condition: `has(object.metadata.labels) && ` +
								`has(object.metadata.labels.restricted) && ` +
								`object.metadata.labels.restricted == "true"`,
							Type:        conditionsType,
							Description: "deny restricted labels",
						},
					},
					nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/allow-all",
							Condition:   "true",
							Type:        conditionsType,
							Description: "base allow",
						},
					},
				)
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
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID: "example.com/allow-approved-data",
							Condition: `has(object.data) && ` +
								`has(object.data.approved) && ` +
								`object.data.approved == "yes"`,
							Type:        conditionsType,
							Description: "only allow configmaps with approved=yes in data",
						},
					},
				)
			}),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "cel-approved-cm" + suffix},
					Data:       map[string]string{"approved": "yes"},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name: "cel deny by data content missing",
			user: "cel-data-deny-user",
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID: "example.com/allow-approved-data",
							Condition: `has(object.data) && ` +
								`has(object.data.approved) && ` +
								`object.data.approved == "yes"`,
							Type:        conditionsType,
							Description: "only allow configmaps with approved=yes in data",
						},
					},
				)
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
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/deny-updates",
							Condition:   `request.operation == "UPDATE"`,
							Type:        conditionsType,
							Description: "deny update operations",
						},
					},
					nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/allow-creates",
							Condition:   `request.operation == "CREATE"`,
							Type:        conditionsType,
							Description: "allow create operations",
						},
					},
				)
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
			expectAllowedWhenDisabled: new(false),
		},
		{
			name: "cel deny overrides allow and noopinion",
			user: "cel-priority-user",
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/deny-all",
							Condition:   "true",
							Type:        conditionsType,
							Description: "deny everything",
						},
					},
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/noop-all",
							Condition:   "true",
							Type:        conditionsType,
							Description: "no opinion on everything",
						},
					},
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/allow-all",
							Condition:   "true",
							Type:        conditionsType,
							Description: "allow everything",
						},
					},
				)
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
			authorizers: celConditionalAuthorizerVariants(func(_ authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionConditionsMap(
					nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID: "example.com/noop-on-pending-review",
							Condition: `has(object.metadata.labels) && ` +
								`has(object.metadata.labels.review) && ` +
								`object.metadata.labels.review == "pending"`,
							Type:        conditionsType,
							Description: "no opinion when review=pending label is present",
						},
					},
					[]authorizer.Condition{
						authorizer.GenericCondition{
							ID:          "example.com/allow-all",
							Condition:   "true",
							Type:        conditionsType,
							Description: "allow everything",
						},
					},
				)
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
								AllowConditions: []authorizationv1.Condition{
									{
										ID: "example.com/require-owner-label",
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
				expectAllowedWhenDisabled: new(false),
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
								AllowConditions: []authorizationv1.Condition{
									{
										ID: "example.com/require-owner-label",
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
								AllowConditions: []authorizationv1.Condition{
									{
										ID: "example.com/allow-unclassified",
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
								AllowConditions: []authorizationv1.Condition{
									{
										ID: "example.com/require-owner-label",
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
				expectAllowedWhenDisabled: new(false),
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
									AllowConditions: []authorizationv1.Condition{
										{
											ID: "example.com/require-production-namespace",
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
				expectAllowedWhenDisabled: new(false),
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
									AllowConditions: []authorizationv1.Condition{
										{
											ID: "example.com/require-staging-namespace",
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
	}

	testCases = append(testCases, crdTestCases(server)...)
	testCases = append(testCases, hpaTestCases()...)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Support multiple variant authorizers with the same assertions
			for variantName, authz := range tc.authorizers {
				t.Run(variantName, func(t *testing.T) {
					// Install the authorizer for this variant. setAuthorizer
					// registers a t.Cleanup that restores the previous state.
					webhookServer.handler.setAuthorizer(t, authz)

					// Compute the user name and resource suffix. Append the variant
					// name so the webhook response cache (keyed on the SAR
					// spec including user) doesn't return stale entries from a
					// sibling subtest with a different webhook configuration.
					userName := tc.user
					suffix := variantName
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
						if !apierrors.IsForbidden(err) {
							t.Fatalf("expected Forbidden error, got: %v", err)
						}
					}
				})
			}
		})
	}

	if featureEnabled {
		// The internal conditional-allow decision that the test authorizer
		// returns for configmap SARs. The webhook mock serializes this via
		// SerializeConditionsAwareDecision on the way out.
		internalConditionalAllow := authorizer.ConditionsAwareDecisionConditionsMap(
			nil, nil,
			[]authorizer.Condition{
				authorizer.GenericCondition{
					ID:          "example.com/allow-safe-prefix",
					Condition:   `object.metadata.name.startsWith("safe-")`,
					Type:        opaqueCELConditionType,
					Description: "only allow configmaps with safe- prefix",
				},
			},
		)
		// The expected v1 form after a JSON round-trip through the API server.
		// Empty condition slices come back as nil (not empty slices), so we
		// leave DenyConditions/NoOpinionConditions unset here — matching the
		// wire shape.
		conditionalAllowDecision := &authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationv1.ConditionsMap{
				AllowConditions: []authorizationv1.Condition{
					{
						ID:          "example.com/allow-safe-prefix",
						Condition:   `object.metadata.name.startsWith("safe-")`,
						Type:        opaqueCELConditionType,
						Description: "only allow configmaps with safe- prefix",
					},
				},
			},
		}

		// The internal conditional-deny decision the "conditional deny" test
		// row's authorizer returns.
		internalConditionalDeny := authorizer.ConditionsAwareDecisionConditionsMap(
			[]authorizer.Condition{
				authorizer.GenericCondition{
					ID:        "example.com/deny-sensitive-label",
					Condition: `has(object.metadata.labels) && has(object.metadata.labels.sensitive)`,
					Type:      opaqueCELConditionType,
				},
			},
			nil, nil,
		)
		conditionalDenyDecision := &authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationv1.ConditionsMap{
				DenyConditions: []authorizationv1.Condition{
					{
						ID:        "example.com/deny-sensitive-label",
						Condition: `has(object.metadata.labels) && has(object.metadata.labels.sensitive)`,
						Type:      opaqueCELConditionType,
					},
				},
			},
		}

		// expectedUnion wraps the conditional-webhook's decision in the Union structure
		// the apiserver returns to conditions-aware clients: system-privileged NoOpinion,
		// then the webhook's decision, then rbac NoOpinion.
		expectedUnion := func(webhookDecision authorizationv1.ConditionsAwareDecision) *authorizationv1.ConditionsAwareDecision {
			return &authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
				Union: []authorizationv1.NamedConditionsAwareDecision{
					{
						AuthorizerName: "system-privileged-group.authorizer.kubernetes.io",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
							NoOpinion: &authorizationv1.UnconditionalDecision{},
						},
					},
					{
						AuthorizerName: "conditional-webhook",
						Decision:       webhookDecision,
					},
					{
						AuthorizerName: "rbac",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
							NoOpinion: &authorizationv1.UnconditionalDecision{},
						},
					},
				},
			}
		}

		expectedConditionalAllowDecision := expectedUnion(*conditionalAllowDecision)
		expectedConditionalDenyDecision := expectedUnion(*conditionalDenyDecision)

		handledAll := []authorizationv1.ConditionsAwareDecisionType{
			authorizationv1.ConditionsAwareDecisionTypeAllow,
			authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			authorizationv1.ConditionsAwareDecisionTypeDeny,
			authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
			authorizationv1.ConditionsAwareDecisionTypeUnion,
		}

		resourceAttrs := &authorizationv1.ResourceAttributes{
			Verb:      "create",
			Group:     "",
			Version:   "v1",
			Resource:  "configmaps",
			Namespace: "test-ns",
		}

		type sendFn func(t *testing.T, user string, authOpts *authorizationv1.AuthorizationOptions) authorizationv1.SubjectAccessReviewStatus

		sendSAR := func(t *testing.T, user string, authOpts *authorizationv1.AuthorizationOptions) authorizationv1.SubjectAccessReviewStatus {
			t.Helper()
			sar := &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes:   resourceAttrs,
					User:                 user,
					AuthorizationOptions: authOpts,
				},
			}
			resp, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SubjectAccessReview: %v", err)
			}
			return resp.Status
		}

		sendSelfSAR := func(t *testing.T, user string, authOpts *authorizationv1.AuthorizationOptions) authorizationv1.SubjectAccessReviewStatus {
			t.Helper()
			impersonationConfig := rest.CopyConfig(server.ClientConfig)
			impersonationConfig.Impersonate.UserName = user
			userClient := clientset.NewForConfigOrDie(impersonationConfig)
			ssar := &authorizationv1.SelfSubjectAccessReview{
				Spec: authorizationv1.SelfSubjectAccessReviewSpec{
					ResourceAttributes:   resourceAttrs,
					AuthorizationOptions: authOpts,
				},
			}
			resp, err := userClient.AuthorizationV1().SelfSubjectAccessReviews().Create(context.TODO(), ssar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create SelfSubjectAccessReview: %v", err)
			}
			return resp.Status
		}

		sendLocalSAR := func(t *testing.T, user string, authOpts *authorizationv1.AuthorizationOptions) authorizationv1.SubjectAccessReviewStatus {
			t.Helper()
			lsar := &authorizationv1.LocalSubjectAccessReview{
				ObjectMeta: metav1.ObjectMeta{Namespace: "test-ns"},
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes:   resourceAttrs,
					User:                 user,
					AuthorizationOptions: authOpts,
				},
			}
			resp, err := adminClient.AuthorizationV1().LocalSubjectAccessReviews("test-ns").Create(context.TODO(), lsar, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create LocalSubjectAccessReview: %v", err)
			}
			return resp.Status
		}

		endpoints := []struct {
			name string
			send sendFn
		}{
			{"SubjectAccessReview", sendSAR},
			{"SelfSubjectAccessReview", sendSelfSAR},
			{"LocalSubjectAccessReview", sendLocalSAR},
		}

		// configmapAuthorizer wraps a per-case decision producer so it only fires
		// for the "configmaps" resource; other resources (e.g. the SAR polls made
		// by test setup) fall through to NoOpinion.
		configmapAuthorizer := func(produce func() authorizer.ConditionsAwareDecision) authorizer.Authorizer {
			return testAuthorizer(func(_ context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
				if a.GetResource() != "configmaps" {
					return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
				}
				return produce()
			})
		}

		cases := []struct {
			name string
			// user is to distinguish the cases from each other, otherwise cached responses are returned as the spec is otherwise shared between all cases
			user                    string
			authz                   authorizer.Authorizer
			wantConditionalStatus   authorizationv1.SubjectAccessReviewStatus
			wantUnconditionalStatus authorizationv1.SubjectAccessReviewStatus
		}{
			{
				name: "conditional allow",
				user: "conditional-allow-user",
				authz: configmapAuthorizer(func() authorizer.ConditionsAwareDecision {
					return internalConditionalAllow
				}),
				wantConditionalStatus: authorizationv1.SubjectAccessReviewStatus{ConditionalDecision: expectedConditionalAllowDecision},
				wantUnconditionalStatus: authorizationv1.SubjectAccessReviewStatus{
					Reason: "conditional-webhook: failed closed",
				},
			},
			{
				name: "unconditional allow",
				user: "unconditional-allow-user",
				authz: configmapAuthorizer(func() authorizer.ConditionsAwareDecision {
					return authorizer.ConditionsAwareDecisionAllow("unconditionally allowed", nil)
				}),
				wantConditionalStatus: authorizationv1.SubjectAccessReviewStatus{
					Allowed: true,
					Reason:  "conditional-webhook: {unconditionally allowed}",
				},
				wantUnconditionalStatus: authorizationv1.SubjectAccessReviewStatus{
					Allowed: true,
					Reason:  "unconditionally allowed",
				},
			},
			{
				name: "unconditional deny",
				user: "unconditional-deny-user",
				authz: configmapAuthorizer(func() authorizer.ConditionsAwareDecision {
					return authorizer.ConditionsAwareDecisionDeny("unconditionally denied", nil)
				}),
				wantConditionalStatus: authorizationv1.SubjectAccessReviewStatus{
					Denied: true,
					Reason: "conditional-webhook: {unconditionally denied}",
				},
				wantUnconditionalStatus: authorizationv1.SubjectAccessReviewStatus{
					Denied: true,
					Reason: "unconditionally denied",
				},
			},
			{
				name: "conditional deny",
				user: "conditional-deny-user",
				authz: configmapAuthorizer(func() authorizer.ConditionsAwareDecision {
					return internalConditionalDeny
				}),
				wantConditionalStatus: authorizationv1.SubjectAccessReviewStatus{
					ConditionalDecision: expectedConditionalDenyDecision,
				},
				wantUnconditionalStatus: authorizationv1.SubjectAccessReviewStatus{
					Denied: true,
					Reason: "failed closed",
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if tc.authz == nil {
					t.Fatal("authz is required")
				}

				webhookServer.handler.setAuthorizer(t, tc.authz)

				variants := []struct {
					name       string
					authOpts   *authorizationv1.AuthorizationOptions
					wantStatus authorizationv1.SubjectAccessReviewStatus
				}{
					{
						name:       "conditional-client",
						authOpts:   &authorizationv1.AuthorizationOptions{HandledDecisionTypes: handledAll},
						wantStatus: tc.wantConditionalStatus,
					},
					{
						name:       "unconditional-client",
						authOpts:   nil,
						wantStatus: tc.wantUnconditionalStatus,
					},
				}

				for _, endpoint := range endpoints {
					t.Run(endpoint.name, func(t *testing.T) {
						for _, variant := range variants {
							t.Run(variant.name, func(t *testing.T) {

								gotStatus := endpoint.send(t, tc.user, variant.authOpts)
								if diff := cmp.Diff(variant.wantStatus, gotStatus); diff != "" {
									t.Errorf("unexpected Status (-want +got):\n%s", diff)
								}
							})
						}
					})
				}
			})
		}

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
			// Install a NoOpinion default so SAR polls done by helpers like
			// GrantUserAuthorization pass through to RBAC. Inner subtests
			// override this and restore it via setAuthorizer's stacked cleanup.
			webhookServer.handler.setAuthorizer(t, noOpinionAuthorizer)

			// Pre-create a ConfigMap as admin for the GET tests.
			if _, err := adminClient.CoreV1().ConfigMaps("test-ns").Create(context.TODO(), &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "fold-test-cm"},
			}, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
				t.Fatal(err)
			}

			// Conditional with Deny condition → fold to Deny → 403, even without RBAC.
			t.Run("conditional with deny condition folds to deny", func(t *testing.T) {
				webhookServer.handler.setAuthorizer(t, testAuthorizer(func(_ context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
					if a.GetResource() != "configmaps" {
						return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
					}
					return authorizer.ConditionsAwareDecisionConditionsMap(
						[]authorizer.Condition{
							authorizer.GenericCondition{
								ID: "example.com/deny-all", Condition: "true", Type: "example.com/opaque",
							},
						},
						nil, nil,
					)
				}))
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = "fold-get-deny-user"
				_, err := clientset.NewForConfigOrDie(userCfg).CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "fold-test-cm", metav1.GetOptions{})
				if !apierrors.IsForbidden(err) {
					t.Fatalf("expected Forbidden, got: %v", err)
				}
			})

			// Conditional without Deny → fold to NoOpinion → RBAC allows → 200.
			t.Run("conditional without deny folds to no-opinion, RBAC allows", func(t *testing.T) {
				// The outer NoOpinion default is still in effect here, so the
				// SAR polling inside GrantUserAuthorization ("get configmaps")
				// passes through to RBAC as intended.
				userName := "fold-get-noop-rbac-user"
				authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
					rbacv1.PolicyRule{
						Verbs:     []string{"get"},
						APIGroups: []string{""},
						Resources: []string{"configmaps"},
					},
				)
				webhookServer.handler.setAuthorizer(t, testAuthorizer(func(_ context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
					if a.GetResource() != "configmaps" {
						return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
					}
					return authorizer.ConditionsAwareDecisionConditionsMap(
						nil, nil,
						[]authorizer.Condition{
							authorizer.GenericCondition{
								ID: "example.com/allow-all", Condition: "true", Type: "example.com/opaque",
							},
						},
					)
				}))
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
			// Install a NoOpinion default so SAR polls done by helpers like
			// GrantUserAuthorization pass through to RBAC. Inner subtests
			// override this and restore it via setAuthorizer's stacked cleanup.
			webhookServer.handler.setAuthorizer(t, noOpinionAuthorizer)

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

			// setConditionalForSARCreate installs an authorizer that returns the
			// given conditional decision when the webhook is asked to authorize
			// a SAR create, and NoOpinion for all other resources.
			setConditionalForSARCreate := func(t *testing.T, decision authorizer.ConditionsAwareDecision) {
				t.Helper()
				webhookServer.handler.setAuthorizer(t, testAuthorizer(func(_ context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
					if a.GetResource() != "subjectaccessreviews" {
						return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
					}
					return decision
				}))
			}

			conditionalWithDeny := authorizer.ConditionsAwareDecisionConditionsMap(
				[]authorizer.Condition{
					authorizer.GenericCondition{
						ID: "example.com/deny-all", Condition: "true", Type: "example.com/opaque",
					},
				},
				nil, nil,
			)
			conditionalWithoutDeny := authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{
					authorizer.GenericCondition{
						ID: "example.com/allow-all", Condition: "true", Type: "example.com/opaque",
					},
				},
			)

			// Case 1: fold to Deny (has Deny condition), no RBAC → 403.
			t.Run("fold to deny, no RBAC", func(t *testing.T) {
				setConditionalForSARCreate(t, conditionalWithDeny)
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = "fold-sar-deny-no-rbac-user"
				_, err := clientset.NewForConfigOrDie(userCfg).AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), newSAR(), metav1.CreateOptions{})
				if !apierrors.IsForbidden(err) {
					t.Fatalf("expected Forbidden, got: %v", err)
				}
			})

			// Case 2: fold to Deny (has Deny condition), RBAC allows SAR creates → 403.
			// Deny short-circuits the union authorizer chain, so RBAC is never consulted.
			t.Run("fold to deny, RBAC allows SAR creates", func(t *testing.T) {
				userName := "fold-sar-deny-with-rbac-user"

				// The outer NoOpinion default is still in effect during the SAR
				// polling inside GrantUserAuthorization.
				authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
					rbacv1.PolicyRule{
						Verbs:     []string{"create"},
						APIGroups: []string{"authorization.k8s.io"},
						Resources: []string{"subjectaccessreviews"},
					},
				)
				setConditionalForSARCreate(t, conditionalWithDeny)
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = userName
				_, err = clientset.NewForConfigOrDie(userCfg).AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), newSAR(), metav1.CreateOptions{})
				if !apierrors.IsForbidden(err) {
					t.Fatalf("expected Forbidden, got: %v", err)
				}
			})

			// Case 3: fold to NoOpinion (no Deny condition), no RBAC → 403.
			t.Run("fold to no-opinion, no RBAC", func(t *testing.T) {
				setConditionalForSARCreate(t, conditionalWithoutDeny)
				userCfg := rest.CopyConfig(server.ClientConfig)
				userCfg.Impersonate.UserName = "fold-sar-noop-no-rbac-user"
				_, err := clientset.NewForConfigOrDie(userCfg).AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), newSAR(), metav1.CreateOptions{})
				if !apierrors.IsForbidden(err) {
					t.Fatalf("expected Forbidden, got: %v", err)
				}
			})

			// Case 4: fold to NoOpinion (no Deny condition), RBAC allows SAR creates → 200.
			// NoOpinion passes to the next authorizer in the chain, where RBAC allows.
			t.Run("fold to no-opinion, RBAC allows SAR creates", func(t *testing.T) {
				userName := "fold-sar-noop-with-rbac-user"
				authutil.GrantUserAuthorization(t, t.Context(), adminClient, userName,
					rbacv1.PolicyRule{
						Verbs:     []string{"create"},
						APIGroups: []string{"authorization.k8s.io"},
						Resources: []string{"subjectaccessreviews"},
					},
				)
				setConditionalForSARCreate(t, conditionalWithoutDeny)
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
						ID: "example.com/require-protectedlabel-permission",
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

// The condition Type installed on all in-tree "using-webhook-only" variants.
// This tells the apiserver to consult the webhook for evaluation, as the
// built-in CEL evaluator does not know how to handle this opaque type.
const opaqueCELConditionType = "example.com/opaque-cel-condition-type"

// decisionFunc returns an internal ConditionsAwareDecision for the given
// authorizer.Attributes and a caller-provided conditions type. It is the
// building block for constructing test authorizers.
type decisionFunc func(a authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision

// celConditionalAuthorizerVariants returns a map of variant-name -> Authorizer
// for a conditional-authorization test case. Today only the out-of-tree
// ("using-webhook-only") variant is active; when in-tree CEL evaluation lands
// upstream, additional variants (in-process-eval-only,
// if-in-process-fails-call-webhook) can be added here.
func celConditionalAuthorizerVariants(fn decisionFunc) map[string]authorizer.Authorizer {
	return map[string]authorizer.Authorizer{
		// When the condition type is opaque, the webhook must be called to
		// resolve the condition — the apiserver has no built-in evaluator
		// for it and delegates via AuthorizationConditionsReview.
		"using-webhook-only": testAuthorizer(func(_ context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
			return fn(a, opaqueCELConditionType)
		}),
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

// webhookServerHandler serves SubjectAccessReview and
// AuthorizationConditionsReview requests. Both endpoints route through the
// currently installed authorizer.Authorizer: the SAR path calls
// ConditionsAwareAuthorize and serializes the resulting ConditionsAwareDecision
// into the SAR status; the ACR path calls EvaluateConditions on the same
// authorizer to fold a previously-issued conditional decision against the
// newly-available admission data.
type webhookServerHandler struct {
	t *testing.T

	mu    sync.Mutex
	authz authorizer.Authorizer
}

// setAuthorizer swaps in the authorizer to be used by the SAR and ACR handlers
// and registers a t.Cleanup to restore the previous authorizer when the test
// (or subtest) that installed it exits. This makes overlapping subtests
// naturally stack: an outer setAuthorizer establishes the default, an inner
// one temporarily overrides it, and the inner cleanup restores the outer.
//
// Passing nil is allowed and useful when the test needs to explicitly clear
// the authorizer. Any SAR/ACR request that arrives while the authorizer is
// nil is served by forgotToSetAuthorizer, which fails closed with a Deny +
// evaluation error so a forgotten test setup is loud rather than silent.
func (h *webhookServerHandler) setAuthorizer(t *testing.T, a authorizer.Authorizer) {
	t.Helper()
	h.mu.Lock()
	prev := h.authz
	h.authz = a
	h.mu.Unlock()
	t.Cleanup(func() {
		h.mu.Lock()
		h.authz = prev
		h.mu.Unlock()
	})
}

// getAuthorizer returns the currently installed authorizer, or the
// forgotToSetAuthorizer if none is set — never nil.
func (h *webhookServerHandler) getAuthorizer() authorizer.Authorizer {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.authz == nil {
		return forgotToSetAuthorizer
	}
	return h.authz
}

// forgotToSetAuthorizer is the fail-closed default returned by getAuthorizer
// when no per-test authorizer has been installed. It surfaces the mistake as
// a Deny with an obvious evaluation error rather than silently passing SAR
// polls with NoOpinion.
var forgotToSetAuthorizer authorizer.Authorizer = forgotAuthorizer{}

type forgotAuthorizer struct{}

func (forgotAuthorizer) errMsg() error {
	return errors.New("forgot to set authorizer in integration test")
}

func (a forgotAuthorizer) Authorize(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", a.errMsg()
}

func (a forgotAuthorizer) ConditionsAwareAuthorize(_ context.Context, _ authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionDeny("", a.errMsg())
}

func (a forgotAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", a.errMsg()
}

// noOpinionAuthorizer is a NoOpinion-for-everything authorizer. It's used in
// setup phases (e.g. around SAR polls done by GrantUserAuthorization) where a
// test needs the webhook to stay out of the way while still having a
// non-forgotAuthorizer default installed.
var noOpinionAuthorizer authorizer.Authorizer = testAuthorizer(func(_ context.Context, _ authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
})

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

	h.handleSAR(req.Context(), w, body)
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

	h.handleACR(req.Context(), w, body)
}

func (h *webhookServerHandler) handleSAR(ctx context.Context, w http.ResponseWriter, body []byte) {
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

	attrs, err := sarSpecToAttributes(sar.Spec)
	if err != nil {
		h.t.Errorf("failed to convert SAR spec to Attributes: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var decision authorizer.ConditionsAwareDecision
	if sar.Spec.AuthorizationOptions.SupportsConditionalAuthorization() {
		decision = h.getAuthorizer().ConditionsAwareAuthorize(ctx, attrs)
	} else {
		decision = authorizer.ConditionsAwareDecisionFromParts(h.getAuthorizer().Authorize(ctx, attrs))
	}

	applyDecisionToSAR(sar, decision)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(sar); err != nil {
		h.t.Errorf("failed to encode SAR response: %v", err)
	}
}

func (h *webhookServerHandler) handleACR(ctx context.Context, w http.ResponseWriter, body []byte) {
	acr := &authorizationv1alpha1.AuthorizationConditionsReview{}
	if err := json.Unmarshal(body, acr); err != nil {
		h.t.Errorf("failed to unmarshal AuthorizationConditionsReview: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if acr.Request == nil {
		h.t.Errorf("ACR request payload has nil Request")
		http.Error(w, "nil ACR request", http.StatusBadRequest)
		return
	}

	// Deserialize the wire decision into the internal ConditionsAwareDecision
	// and delegate evaluation to the same authorizer that issued it. The
	// authorizer's EvaluateConditions is the canonical entry point: it can
	// e.g. dispatch on condition Type, and for the testAuthorizer used here
	// it routes into ConditionsMap.Evaluate with a CEL-based evaluator.
	internalDecision := authorizationv1apiserver.DeserializeConditionsAwareDecision(
		acr.Request.Decision,
		func(err error) authorizer.ConditionsAwareDecision {
			return authorizer.ConditionsAwareDecisionDeny("failed closed", err)
		},
	)

	data, err := admissionRequestToAttributes(acr.Request.AdmissionRequest)
	if err != nil {
		h.t.Errorf("failed to build admission attributes: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	decision, reason, err := h.getAuthorizer().EvaluateConditions(ctx, internalDecision, data)

	evaluated := authorizer.ConditionsAwareDecisionFromParts(decision, reason, err)

	var uid types.UID
	if acr.Request.AdmissionRequest != nil {
		uid = acr.Request.AdmissionRequest.UID
	}
	acr.Response = &authorizationv1alpha1.AuthorizationConditionsResponse{
		UID:      uid,
		Decision: authorizationv1apiserver.SerializeConditionsAwareDecision(evaluated),
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

// sarSpecToAttributes converts a v1 SubjectAccessReviewSpec into
// authorizer.Attributes by round-tripping through the internal SAR spec and
// invoking the shared registry helper. This keeps the mapping in sync with
// how the real apiserver constructs Attributes for the SAR endpoints.
func sarSpecToAttributes(spec authorizationv1.SubjectAccessReviewSpec) (authorizer.AttributesRecord, error) {
	var internal authorizationapi.SubjectAccessReviewSpec
	if err := authorizationinternalv1.Convert_v1_SubjectAccessReviewSpec_To_authorization_SubjectAccessReviewSpec(&spec, &internal, nil); err != nil {
		return authorizer.AttributesRecord{}, fmt.Errorf("convert SAR spec: %w", err)
	}
	return authorizationutil.AuthorizationAttributesFrom(internal), nil
}

// applyDecisionToSAR writes an internal ConditionsAwareDecision to a v1 SAR's
// Status. Unconditional decisions go to Allowed/Denied/Reason/EvaluationError;
// conditional decisions go to ConditionalDecision.
func applyDecisionToSAR(sar *authorizationv1.SubjectAccessReview, decision authorizer.ConditionsAwareDecision) {
	var errString string
	if err := decision.Error(); err != nil {
		errString = err.Error()
	}
	switch {
	case decision.IsAllow():
		sar.Status.Allowed = true
		sar.Status.Denied = false
		sar.Status.Reason = decision.Reason()
		sar.Status.EvaluationError = errString
	case decision.IsDeny():
		sar.Status.Allowed = false
		sar.Status.Denied = true
		sar.Status.Reason = decision.Reason()
		sar.Status.EvaluationError = errString
	case decision.IsNoOpinion():
		sar.Status.Allowed = false
		sar.Status.Denied = false
		sar.Status.Reason = decision.Reason()
		sar.Status.EvaluationError = errString
	default: // ConditionsMap / Union
		sar.Status.ConditionalDecision = new(authorizationv1apiserver.SerializeConditionsAwareDecision(decision))
	}
}

// testAuthorizer implements authorizer.Authorizer by delegating
// ConditionsAwareAuthorize to the wrapped func. EvaluateConditions folds a
// ConditionsMap decision by running each condition through the CEL evaluator
// — matching the semantics of the opaque-condition-type test variant.
type testAuthorizer func(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision

var _ authorizer.Authorizer = testAuthorizer(nil)

func (f testAuthorizer) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return f(ctx, a)
}

func (f testAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	d := f(ctx, a)
	switch {
	case d.IsAllow():
		return authorizer.DecisionAllow, d.Reason(), d.Error()
	case d.IsDeny():
		return authorizer.DecisionDeny, d.Reason(), d.Error()
	case d.IsNoOpinion():
		return authorizer.DecisionNoOpinion, d.Reason(), d.Error()
	default:
		return d.FailureDecision(), "failed closed", nil
	}
}

func (f testAuthorizer) EvaluateConditions(ctx context.Context, decision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	if !decision.IsConditionsMap() {
		return decision.FailureDecision(), "", fmt.Errorf("testAuthorizer.EvaluateConditions: expected ConditionsMap decision, got %s", decision.String())
	}
	return decision.ConditionsMap().Evaluate(ctx, data, celEvaluateCondition)
}

// admissionRequestToAttributes builds an admission.Attributes value (which
// also satisfies authorizer.ConditionsData) from the admissionv1 request
// embedded in the AuthorizationConditionsReview. Object and OldObject are
// decoded from their RawExtension bytes into *unstructured.Unstructured so
// the CEL evaluator can walk them as maps.
func admissionRequestToAttributes(req *admissionv1.AdmissionRequest) (admission.Attributes, error) {
	if req == nil {
		return admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", "", nil, false, nil), nil
	}
	obj, err := unmarshalUnstructured(req.Object.Raw)
	if err != nil {
		return nil, fmt.Errorf("unmarshal Object: %w", err)
	}
	oldObj, err := unmarshalUnstructured(req.OldObject.Raw)
	if err != nil {
		return nil, fmt.Errorf("unmarshal OldObject: %w", err)
	}
	kind := schema.GroupVersionKind{Group: req.Kind.Group, Version: req.Kind.Version, Kind: req.Kind.Kind}
	resource := schema.GroupVersionResource{Group: req.Resource.Group, Version: req.Resource.Version, Resource: req.Resource.Resource}
	return admission.NewAttributesRecord(
		obj, oldObj,
		kind,
		req.Namespace, req.Name,
		resource, req.SubResource,
		admission.Operation(req.Operation),
		nil,   // operationOptions — not exposed to CEL
		false, // dryRun
		nil,   // userInfo — not exposed to CEL
	), nil
}

// unmarshalUnstructured decodes a RawExtension payload into an
// *unstructured.Unstructured, returning nil (not an error) for empty input.
func unmarshalUnstructured(raw []byte) (runtime.Object, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	u := &unstructured.Unstructured{}
	if err := json.Unmarshal(raw, &u.Object); err != nil {
		return nil, err
	}
	return u, nil
}

// celEnvOnce lazily builds a package-scoped CEL environment matching the one
// used previously by celEvaluateConditions.
var (
	celEnvOnce sync.Once
	celEnv     *cel.Env
	celEnvErr  error
)

func getCELEnv() (*cel.Env, error) {
	celEnvOnce.Do(func() {
		celEnv, celEnvErr = cel.NewEnv(
			cel.Variable("object", cel.DynType),
			cel.Variable("oldObject", cel.DynType),
			cel.Variable("request", cel.DynType),
		)
	})
	return celEnv, celEnvErr
}

// celEvaluateCondition is the EvaluateConditionFunc passed to
// authorizer.ConditionsMap.Evaluate. It compiles and evaluates a single CEL
// condition against the object/oldObject/request variables derived from the
// admission request.
func celEvaluateCondition(_ context.Context, cond authorizer.Condition, data authorizer.ConditionsData) (bool, error) {
	env, err := getCELEnv()
	if err != nil {
		return false, fmt.Errorf("build CEL env: %w", err)
	}

	vars := map[string]any{
		"object":    unstructuredContent(data.GetObject()),
		"oldObject": unstructuredContent(data.GetOldObject()),
		"request": map[string]any{
			"operation": string(data.GetOperation()),
		},
	}
	return evalCEL(env, cond.GetCondition(), vars)
}

// unstructuredContent extracts the map form of an object built by
// admissionRequestToAttributes. Returns nil when the object is absent.
func unstructuredContent(obj runtime.Object) map[string]any {
	if obj == nil {
		return nil
	}
	u, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil
	}
	return u.Object
}

// evalCEL compiles and evaluates a single CEL expression, returning its
// boolean result. Any compile/eval/type error is returned so callers of
// ConditionsMap.Evaluate can surface it via the decision.
func evalCEL(env *cel.Env, expr string, vars map[string]any) (bool, error) {
	ast, issues := env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		return false, fmt.Errorf("CEL compile error for %q: %w", expr, issues.Err())
	}
	prg, err := env.Program(ast)
	if err != nil {
		return false, fmt.Errorf("CEL program error for %q: %w", expr, err)
	}
	out, _, err := prg.Eval(vars)
	if err != nil {
		return false, fmt.Errorf("CEL eval error for %q: %w", expr, err)
	}
	result, ok := out.Value().(bool)
	if !ok {
		return false, fmt.Errorf("CEL expression %q did not return bool, got %T", expr, out.Value())
	}
	return result, nil
}
