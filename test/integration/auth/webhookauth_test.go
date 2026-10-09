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

package auth

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	admissionauth "k8s.io/component-helpers/auth/admission"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestWebhookAuthIssuedToken(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerWebhookAuthenticationToken, true)

	ctx := ktesting.Init(t)
	var issuer string
	kubeClient, kubeConfig, tearDown := framework.StartTestServer(ctx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Unlike issuance-only tests, discovery and the local JWKS must be reachable by the consumer.
			issuer = "https://" + opts.SecureServing.Listener.Addr().String()
			opts.Authentication.ServiceAccounts.Issuers = []string{issuer}
			opts.Authentication.ServiceAccounts.JWKSURI = issuer + "/openid/v1/jwks"
			opts.Authentication.APIAudiences = authenticator.Audiences{"api"}
			opts.Authorization.Modes = []string{"RBAC"}
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
		},
	})
	defer tearDown()

	cs, err := clientset.NewForConfig(kubeConfig)
	if err != nil {
		t.Fatalf("create Kubernetes client: %v", err)
	}
	// The default issuer-discovery RBAC role permits service accounts, not
	// anonymous callers. This test's issuer and JWKS are on the same apiserver.
	client, err := rest.HTTPClientFor(kubeConfig)
	if err != nil {
		t.Fatalf("create client for issuer discovery: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(kubeClient, "webhook-auth", t)
	defer framework.DeleteNamespaceOrDie(kubeClient, ns, t)
	sa, deleteSA := createDeleteSvcAcct(t, cs, &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "admission-verifier", Namespace: ns.Name}})
	defer deleteSA()

	rules := []admissionregistrationv1.RuleWithOperations{
		{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
			Rule:       admissionregistrationv1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments"}},
		},
		{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Update},
			Rule:       admissionregistrationv1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments/scale"}},
		},
	}
	validatingServiceWebhook := func(name string, service admissionregistrationv1.ServiceReference) admissionregistrationv1.ValidatingWebhook {
		return admissionregistrationv1.ValidatingWebhook{
			Name:                    name,
			ClientConfig:            admissionregistrationv1.WebhookClientConfig{Service: &service},
			Rules:                   rules,
			SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
			AdmissionReviewVersions: []string{"v1"},
		}
	}
	validating, deleteValidating := createDeleteValidating(t, cs, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "test-authn-validating"},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name:                    "validate.url.webhook.test",
				ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: new("https://validate.test")},
				Rules:                   rules,
				SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
				AdmissionReviewVersions: []string{"v1"},
			},
			validatingServiceWebhook("validate.service.webhook.test", admissionregistrationv1.ServiceReference{
				Namespace: "webhook-ns", Name: "webhook-svc", Port: new(int32(8443)), Path: new("/validate"),
			}),
			validatingServiceWebhook("validate.service.default.webhook.test", admissionregistrationv1.ServiceReference{
				Namespace: "webhook-ns", Name: "default-svc",
			}),
			validatingServiceWebhook("validate.service.empty.webhook.test", admissionregistrationv1.ServiceReference{
				Namespace: "webhook-ns", Name: "empty-svc", Path: new(""),
			}),
			validatingServiceWebhook("validate.service.root.webhook.test", admissionregistrationv1.ServiceReference{
				Namespace: "webhook-ns", Name: "root-svc", Path: new("/"),
			}),
			validatingServiceWebhook("validate.service.trailing.webhook.test", admissionregistrationv1.ServiceReference{
				Namespace: "webhook-ns", Name: "trailing-svc", Port: new(int32(8443)), Path: new("/validate/"),
			}),
		},
	})
	defer deleteValidating()
	mutating, deleteMutating := createDeleteMutating(t, cs, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "test-authn-mutating"},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			{
				Name:                    "mutate.url.webhook.test",
				ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: new("https://mutate.test")},
				Rules:                   rules,
				SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
				AdmissionReviewVersions: []string{"v1"},
			},
			{
				Name: "mutate.service.webhook.test",
				ClientConfig: admissionregistrationv1.WebhookClientConfig{Service: &admissionregistrationv1.ServiceReference{
					Namespace: "webhook-ns", Name: "mutating-svc", Port: new(int32(9443)),
				}},
				Rules:                   rules,
				SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
				AdmissionReviewVersions: []string{"v1"},
			},
		},
	})
	defer deleteMutating()

	authutil.GrantServiceAccountAuthorization(t, ctx, cs, sa.Name, ns.Name, rbacv1.PolicyRule{
		Verbs: []string{"attest"}, APIGroups: []string{"authentication.k8s.io"}, Resources: []string{"admissionReviewAPIGroups"}, ResourceNames: []string{"apps"},
	})
	authutil.GrantServiceAccountAuthorization(t, ctx, cs, sa.Name, ns.Name, rbacv1.PolicyRule{
		Verbs: []string{"create"}, APIGroups: []string{""}, Resources: []string{"serviceaccounts/token"},
	})

	cases := []struct {
		name, kind, webhook, audience string
	}{
		{"validating URL", "ValidatingWebhookConfiguration", validating.Name, "https://validate.test"},
		{"validating Service", "ValidatingWebhookConfiguration", validating.Name, "https://webhook-svc.webhook-ns.svc:8443/validate"},
		{"validating Service default path and port", "ValidatingWebhookConfiguration", validating.Name, "https://default-svc.webhook-ns.svc:443/"},
		{"validating Service empty path", "ValidatingWebhookConfiguration", validating.Name, "https://empty-svc.webhook-ns.svc:443/"},
		{"validating Service root path", "ValidatingWebhookConfiguration", validating.Name, "https://root-svc.webhook-ns.svc:443/"},
		{"validating Service trailing slash", "ValidatingWebhookConfiguration", validating.Name, "https://trailing-svc.webhook-ns.svc:8443/validate/"},
		{"mutating URL", "MutatingWebhookConfiguration", mutating.Name, "https://mutate.test"},
		{"mutating Service", "MutatingWebhookConfiguration", mutating.Name, "https://mutating-svc.webhook-ns.svc:9443/"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			token, err := cs.CoreV1().ServiceAccounts(ns.Name).CreateToken(ctx, sa.Name, &authenticationv1.TokenRequest{
				Spec: authenticationv1.TokenRequestSpec{
					Audiences: []string{tc.audience},
					BoundObjectRef: &authenticationv1.BoundObjectReference{
						Kind: tc.kind, APIVersion: "admissionregistration.k8s.io/v1", Name: tc.webhook,
					},
					Attestations: map[string]authenticationv1.AttestationValue{"admissionReviewAPIGroups": {"apps"}},
				},
			}, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("issue %s token with audience %q: %v", tc.kind, tc.audience, err)
			}

			verifier, err := admissionauth.NewVerifier(ctx, admissionauth.WithRemoteIssuer(issuer, tc.audience, client))
			if err != nil {
				t.Fatalf("construct consumer verifier: %v", err)
			}
			request := &admissionv1.AdmissionRequest{
				Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
				Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
				Operation: admissionv1.Create,
				Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment"}`)},
			}
			if err := verifier.Verify(ctx, request, token.Status.Token); err != nil {
				t.Fatalf("verify issued %s token with audience %q: %v", tc.kind, tc.audience, err)
			}
			// The signed attestation covers the resource group, while a /scale
			// subresource legitimately carries an autoscaling Scale object.
			scale := &admissionv1.AdmissionRequest{
				Kind:        metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"},
				Resource:    request.Resource,
				SubResource: "scale",
				Operation:   admissionv1.Update,
				Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"autoscaling/v1","kind":"Scale"}`)},
			}
			if err := verifier.Verify(ctx, scale, token.Status.Token); err != nil {
				t.Fatalf("verify issued token for legitimate /scale subresource: %v", err)
			}

			wrongGroup := &admissionv1.AdmissionRequest{
				Kind:     metav1.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
				Resource: metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
				Object:   runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)},
			}
			if err := verifier.Verify(ctx, wrongGroup, token.Status.Token); err == nil || err.Error() != "webhook token verification failed" {
				t.Fatalf("verify token for unauthorized group: got %v, want generic denial", err)
			}
			for _, mismatch := range []struct {
				name string
				req  *admissionv1.AdmissionRequest
			}{
				{
					name: "object group",
					req: &admissionv1.AdmissionRequest{
						Kind:     request.Kind,
						Resource: request.Resource,
						Object:   runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)},
					},
				},
				{
					name: "old object group",
					req: &admissionv1.AdmissionRequest{
						Kind:      request.Kind,
						Resource:  request.Resource,
						Object:    request.Object,
						OldObject: runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)},
					},
				},
			} {
				if err := verifier.Verify(ctx, mismatch.req, token.Status.Token); err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verify token for mismatched %s: got %v, want generic denial", mismatch.name, err)
				}
			}
			wrongAudienceVerifier, err := admissionauth.NewVerifier(ctx, admissionauth.WithRemoteIssuer(issuer, tc.audience+"-other", client))
			if err != nil {
				t.Fatalf("construct wrong-audience verifier: %v", err)
			}
			if err := wrongAudienceVerifier.Verify(ctx, request, token.Status.Token); err == nil || err.Error() != "webhook token verification failed" {
				t.Fatalf("verify token for wrong audience: got %v, want generic denial", err)
			}

			var admissions atomic.Int32
			handler, err := admissionauth.WithTokenVerification(ctx, func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
				admissions.Add(1)
				return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
			}, admissionauth.WithRemoteIssuer(issuer, tc.audience, client))
			if err != nil {
				t.Fatalf("construct admission HTTP handler: %v", err)
			}
			webhook := httptest.NewServer(handler)
			defer webhook.Close()
			postReview := func(review *admissionv1.AdmissionReview, bearer string) (*admissionv1.AdmissionReview, int) {
				body, err := json.Marshal(review)
				if err != nil {
					t.Fatalf("marshal AdmissionReview: %v", err)
				}
				httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, webhook.URL+"/admit", bytes.NewReader(body))
				if err != nil {
					t.Fatalf("create webhook request: %v", err)
				}
				httpRequest.Header.Set("Content-Type", "application/json")
				if bearer != "" {
					httpRequest.Header.Set("Authorization", "Bearer "+bearer)
				}
				response, err := webhook.Client().Do(httpRequest)
				if err != nil {
					t.Fatalf("send webhook request: %v", err)
				}
				defer response.Body.Close()
				if response.StatusCode != http.StatusOK {
					return nil, response.StatusCode
				}
				var decoded admissionv1.AdmissionReview
				if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
					t.Fatalf("decode webhook response: %v", err)
				}
				return &decoded, response.StatusCode
			}
			review := &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Request:  &admissionv1.AdmissionRequest{UID: "real-token-e2e", Kind: request.Kind, Resource: request.Resource, Object: request.Object},
			}
			allowedReview, status := postReview(review, token.Status.Token)
			if status != http.StatusOK || allowedReview == nil || allowedReview.Response == nil || !allowedReview.Response.Allowed || allowedReview.Response.UID != review.Request.UID || admissions.Load() != 1 {
				t.Fatalf("verified HTTP webhook: status=%d response=%+v admissions=%d", status, allowedReview, admissions.Load())
			}
			scale.UID = "real-token-scale"
			review.Request = scale
			allowedScale, status := postReview(review, token.Status.Token)
			if status != http.StatusOK || allowedScale == nil || allowedScale.Response == nil || !allowedScale.Response.Allowed || allowedScale.Response.UID != scale.UID || admissions.Load() != 2 {
				t.Fatalf("legitimate /scale HTTP webhook: status=%d response=%+v admissions=%d", status, allowedScale, admissions.Load())
			}
			review.Request = wrongGroup
			deniedReview, status := postReview(review, token.Status.Token)
			if status != http.StatusOK || deniedReview == nil || deniedReview.Response == nil || deniedReview.Response.Allowed || admissions.Load() != 2 {
				t.Fatalf("unauthorized group HTTP webhook: status=%d response=%+v admissions=%d", status, deniedReview, admissions.Load())
			}
			if _, status := postReview(review, ""); status != http.StatusUnauthorized || admissions.Load() != 2 {
				t.Fatalf("missing bearer HTTP status=%d admissions=%d, want 401 without admission", status, admissions.Load())
			}
		})
	}
}
