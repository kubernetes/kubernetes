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
	"strings"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
)

const (
	vapDeniedPrefix = "vap-denied-"
	vapMessage      = "rejected by the vap-ordering ValidatingAdmissionPolicy"
)

// testValidatingAdmissionPolicyRunsBeforeConditionsEnforcer verifies that the
// ValidatingAdmissionPolicy admission plugin runs before the AuthorizationConditionsEnforcer,
// such that a request a policy rejects never has its conditions sent to the webhook.
//
// The conditions only allow the objects the policy admits, so the error tells which plugin
// rejected the request: the policy message if VAP ran first, or an authorization Forbidden if
// the conditions were evaluated first.
func testValidatingAdmissionPolicyRunsBeforeConditionsEnforcer(t *testing.T, adminClient *clientset.Clientset, clientConfig *rest.Config, webhookServer *webhookServer) {
	// Let the admin's requests and the SAR polls done by setup pass through.
	webhookServer.handler.setAuthorizer(t, noOpinionAuthorizer)

	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "vap-ordering"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule: admissionregistrationv1.Rule{
							APIGroups:   []string{""},
							APIVersions: []string{"v1"},
							Resources:   []string{"configmaps"},
						},
					},
				}},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: `!object.metadata.name.startsWith("` + vapDeniedPrefix + `")`,
				Message:    vapMessage,
			}},
		},
	}
	if _, err := adminClient.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(t.Context(), policy, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create ValidatingAdmissionPolicy: %v", err)
	}
	t.Cleanup(func() {
		_ = adminClient.AdmissionregistrationV1().ValidatingAdmissionPolicies().Delete(context.Background(), policy.Name, metav1.DeleteOptions{})
	})
	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "vap-ordering"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        policy.Name,
			ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
		},
	}
	if _, err := adminClient.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(t.Context(), binding, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create ValidatingAdmissionPolicyBinding: %v", err)
	}
	t.Cleanup(func() {
		_ = adminClient.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Delete(context.Background(), binding.Name, metav1.DeleteOptions{})
	})

	// Wait for the policy to take effect. The admin is unconditionally allowed, so the probes never
	// reach the AuthorizationConditionsEnforcer.
	if err := wait.PollUntilContextTimeout(t.Context(), 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := adminClient.CoreV1().ConfigMaps("test-ns").Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{GenerateName: vapDeniedPrefix + "probe-"},
		}, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
		return err != nil && strings.Contains(err.Error(), vapMessage), nil
	}); err != nil {
		t.Fatalf("ValidatingAdmissionPolicy did not take effect: %v", err)
	}

	webhookServer.handler.setAuthorizer(t, testAuthorizer(func(_ context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
		if a.GetResource() != "configmaps" {
			return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		}
		return authorizer.ConditionsAwareDecisionConditionsMap(
			nil, nil,
			[]authorizer.Condition{
				authorizer.GenericCondition{
					ID:        "example.com/allow-without-vap-denied-prefix",
					Condition: `!object.metadata.name.startsWith("` + vapDeniedPrefix + `")`,
					Type:      opaqueCELConditionType,
				},
			},
		)
	}))
	userConfig := rest.CopyConfig(clientConfig)
	userConfig.Impersonate.UserName = "vap-ordering-user"
	userClient := clientset.NewForConfigOrDie(userConfig)

	t.Run("policy rejects before conditions are evaluated", func(t *testing.T) {
		acrsBefore := webhookServer.handler.acrCount.Load()
		_, err := userClient.CoreV1().ConfigMaps("test-ns").Create(t.Context(), &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: vapDeniedPrefix + "cm"},
		}, metav1.CreateOptions{})
		if err == nil || !strings.Contains(err.Error(), vapMessage) {
			t.Fatalf("expected the request to be rejected by the ValidatingAdmissionPolicy, got: %v", err)
		}
		if acrs := webhookServer.handler.acrCount.Load() - acrsBefore; acrs != 0 {
			t.Errorf("expected no AuthorizationConditionsReviews for a request rejected by the ValidatingAdmissionPolicy, got %d", acrs)
		}
	})

	// Control case: once the policy admits the request, its conditions are evaluated.
	t.Run("conditions are evaluated after the policy admits", func(t *testing.T) {
		acrsBefore := webhookServer.handler.acrCount.Load()
		if _, err := userClient.CoreV1().ConfigMaps("test-ns").Create(t.Context(), &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "vap-allowed-cm"},
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("expected the request to be allowed, got: %v", err)
		}
		if acrs := webhookServer.handler.acrCount.Load() - acrsBefore; acrs == 0 {
			t.Errorf("expected the conditions to be sent to the webhook for evaluation, got no AuthorizationConditionsReviews")
		}
	})
}
