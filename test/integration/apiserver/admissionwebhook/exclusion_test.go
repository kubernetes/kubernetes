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
	"fmt"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/exclusion"
	"k8s.io/kubernetes/test/integration/framework"
)

// excludedResourceCreators returns, for every virtual resource that admission webhooks exclude, a
// function that issues a request for that resource. The keys must match exclusion.Excluded() exactly
// (asserted by the test) so the set stays in parity with the ValidatingAdmissionPolicy /
// MutatingAdmissionPolicy exclusion list.
func excludedResourceCreators(ctx context.Context, client kubernetes.Interface) map[schema.GroupResource]func() error {
	resourceAttributes := &authorizationv1.ResourceAttributes{Namespace: "default", Verb: "get", Resource: "pods"}
	return map[schema.GroupResource]func() error{
		{Group: "authentication.k8s.io", Resource: "tokenreviews"}: func() error {
			_, err := client.AuthenticationV1().TokenReviews().Create(ctx, &authenticationv1.TokenReview{Spec: authenticationv1.TokenReviewSpec{Token: "any-token"}}, metav1.CreateOptions{})
			return err
		},
		{Group: "authentication.k8s.io", Resource: "selfsubjectreviews"}: func() error {
			_, err := client.AuthenticationV1().SelfSubjectReviews().Create(ctx, &authenticationv1.SelfSubjectReview{}, metav1.CreateOptions{})
			return err
		},
		{Group: "authorization.k8s.io", Resource: "subjectaccessreviews"}: func() error {
			_, err := client.AuthorizationV1().SubjectAccessReviews().Create(ctx, &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{ResourceAttributes: resourceAttributes, User: "alice"}}, metav1.CreateOptions{})
			return err
		},
		{Group: "authorization.k8s.io", Resource: "selfsubjectaccessreviews"}: func() error {
			_, err := client.AuthorizationV1().SelfSubjectAccessReviews().Create(ctx, &authorizationv1.SelfSubjectAccessReview{Spec: authorizationv1.SelfSubjectAccessReviewSpec{ResourceAttributes: resourceAttributes}}, metav1.CreateOptions{})
			return err
		},
		{Group: "authorization.k8s.io", Resource: "selfsubjectrulesreviews"}: func() error {
			_, err := client.AuthorizationV1().SelfSubjectRulesReviews().Create(ctx, &authorizationv1.SelfSubjectRulesReview{Spec: authorizationv1.SelfSubjectRulesReviewSpec{Namespace: "default"}}, metav1.CreateOptions{})
			return err
		},
		{Group: "authorization.k8s.io", Resource: "localsubjectaccessreviews"}: func() error {
			_, err := client.AuthorizationV1().LocalSubjectAccessReviews("default").Create(ctx, &authorizationv1.LocalSubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{ResourceAttributes: resourceAttributes, User: "alice"}}, metav1.CreateOptions{})
			return err
		},
	}
}

// TestWebhookVirtualResourceExclusion verifies that, when ExcludeAdmissionWebhookVirtualResources is
// enabled, admission webhooks are not dispatched for the excluded virtual resources (so the requests
// succeed despite a broken webhook), while a non-excluded resource is still intercepted. With the
// gate disabled, the webhook intercepts everything. It also asserts the excluded set is exactly
// exclusion.Excluded() (parity with the VAP/MAP exclusion list).
func TestWebhookVirtualResourceExclusion(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("gate=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExcludeAdmissionWebhookVirtualResources, enabled)

			server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
			defer server.TearDownFn()

			client, err := kubernetes.NewForConfig(server.ClientConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			ctx := context.Background()

			// Register a broken webhook (failurePolicy=Fail, unreachable service) that matches all
			// resources, so anything it is dispatched for fails.
			webhookName := fmt.Sprintf("integration-exclusion-test-webhook-%t", enabled)
			if _, err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(ctx, brokenWebhookConfig(webhookName), metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to register broken webhook: %v", err)
			}

			// There is no API to learn when a webhook configuration becomes effective, so poll until
			// a non-excluded resource (Deployment) starts failing. This confirms the webhook is being
			// honored regardless of the gate (Deployments are never excluded).
			attempt := 0
			if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
				attempt++
				_, createErr := client.AppsV1().Deployments("default").Create(ctx, exampleDeployment(fmt.Sprintf("test-deploy-%t-%d", enabled, attempt)), metav1.CreateOptions{})
				return createErr != nil, nil
			}); err != nil {
				t.Fatalf("timed out waiting for the broken webhook to be honored: %v", err)
			}

			creators := excludedResourceCreators(ctx, client)

			// Parity guard: the resources exercised here must be exactly exclusion.Excluded(), which is
			// the same set ValidatingAdmissionPolicy / MutatingAdmissionPolicy exclude.
			if got, want := sets.KeySet(creators), sets.New(exclusion.Excluded()...); !got.Equal(want) {
				t.Fatalf("excluded resource coverage does not match exclusion.Excluded(); update excludedResourceCreators (missing=%v extra=%v)",
					want.Difference(got).UnsortedList(), got.Difference(want).UnsortedList())
			}

			for gr, create := range creators {
				err := create()
				if enabled && err != nil {
					t.Errorf("gate enabled: %s should be excluded from webhooks (skipped), but the request failed: %v", gr, err)
				}
				if !enabled && err == nil {
					t.Errorf("gate disabled: %s should be intercepted by the broken webhook, but the request succeeded", gr)
				}
			}
		})
	}
}

// TestWebhookVirtualResourceExclusionWarnings verifies a deprecation warning is returned when a
// webhook configuration explicitly names an excluded virtual resource in its rules.
func TestWebhookVirtualResourceExclusionWarnings(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExcludeAdmissionWebhookVirtualResources, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	warnings := &warningHandler{warnings: map[string]bool{}}
	config := rest.CopyConfig(server.ClientConfig)
	config.WarningHandler = warnings

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var path string
	webhookCfg := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "integration-exclusion-warnings-webhook"},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "warnings-webhook.k8s.io",
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
				Rule: admissionregistrationv1.Rule{
					APIGroups:   []string{"authentication.k8s.io"},
					APIVersions: []string{"v1"},
					Resources:   []string{"tokenreviews"}, // explicitly named excluded resource
				},
			}},
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				Service: &admissionregistrationv1.ServiceReference{Namespace: "default", Name: "some-service", Path: &path},
			},
			SideEffects:             &noSideEffects,
			AdmissionReviewVersions: []string{"v1"},
		}},
	}

	if _, err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(context.TODO(), webhookCfg, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error creating webhook configuration: %v", err)
	}

	expectedWarning := `webhooks[0].rules[0]: tokenreviews.authentication.k8s.io is excluded from admission webhooks; this rule will have no effect`
	if !warnings.hasWarning(expectedWarning) {
		t.Fatalf("expected warning %q, but it was not returned", expectedWarning)
	}
}
