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
	"crypto/ecdsa"
	"encoding/json"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/keyutil"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	admissionauth "k8s.io/component-helpers/auth/admission"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

func webhookAuthRequest(t *testing.T, kind metav1.GroupVersionKind, resource metav1.GroupVersionResource, subresource string, operation admissionv1.Operation) *admissionv1.AdmissionRequest {
	t.Helper()
	apiVersion := kind.Version
	if kind.Group != "" {
		apiVersion = kind.Group + "/" + kind.Version
	}
	raw, err := json.Marshal(metav1.TypeMeta{APIVersion: apiVersion, Kind: kind.Kind})
	if err != nil {
		t.Fatalf("marshal admitted object metadata: %v", err)
	}
	req := &admissionv1.AdmissionRequest{
		UID:                "real-token-e2e",
		Kind:               kind,
		Resource:           resource,
		SubResource:        subresource,
		RequestKind:        &kind,
		RequestResource:    &resource,
		RequestSubResource: subresource,
		Operation:          operation,
	}
	switch operation {
	case admissionv1.Create, admissionv1.Connect:
		req.Object.Raw = raw
	case admissionv1.Update:
		req.Object.Raw = raw
		req.OldObject.Raw = bytes.Clone(raw)
	case admissionv1.Delete:
		req.OldObject.Raw = raw
	}
	return req
}

func verifyDiscoveredWebhookKinds(ctx context.Context, t *testing.T, cs clientset.Interface, verifier *admissionauth.Verifier, token, namespace string) {
	t.Helper()
	_, resources, err := cs.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("discover enabled REST APIs: %v", err)
	}
	tested := 0
	for _, list := range resources {
		gv, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			t.Fatal(err)
		}
		for _, resource := range list.APIResources {
			kind := metav1.GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: resource.Kind}
			if resource.Group != "" {
				kind.Group = resource.Group
			}
			if resource.Version != "" {
				kind.Version = resource.Version
			}
			if kind.Group == gv.Group {
				continue
			}
			parent, subresource, _ := strings.Cut(resource.Name, "/")
			gvr := metav1.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: parent}
			var operation admissionv1.Operation
			switch {
			case slices.Contains(resource.Verbs, "create"):
				operation = admissionv1.Create
			case slices.Contains(resource.Verbs, "update") || slices.Contains(resource.Verbs, "patch"):
				operation = admissionv1.Update
			case slices.Contains(resource.Verbs, "delete"):
				operation = admissionv1.Delete
			default:
				t.Fatalf("cross-group resource %s/%s has no classified admission operation: %v", list.GroupVersion, resource.Name, resource.Verbs)
			}
			tested++
			t.Run(list.GroupVersion+"/"+resource.Name, func(t *testing.T) {
				req := webhookAuthRequest(t, kind, gvr, subresource, operation)
				if resource.Namespaced {
					req.Namespace = namespace
				}
				if err := verifier.Verify(ctx, req, token); err != nil {
					t.Fatalf("discovered %s/%s with admitted kind %+v is unhandled: %v", list.GroupVersion, resource.Name, kind, err)
				}
				badKind := metav1.GroupVersionKind{Group: "unexpected.example.com", Version: "v1", Kind: "Thing"}
				invalid := webhookAuthRequest(t, badKind, gvr, subresource, operation)
				invalid.Namespace = req.Namespace
				if err := verifier.Verify(ctx, invalid, token); err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("arbitrary cross-group kind accepted: %v", err)
				}
			})
		}
	}
	if tested == 0 {
		t.Fatal("discovery returned no cross-group admission kinds")
	}
	t.Logf("checked %d discovered cross-group admission kinds", tested)
}

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
			request := webhookAuthRequest(t,
				metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
				metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
				"", admissionv1.Create,
			)
			if err := verifier.Verify(ctx, request, token.Status.Token); err != nil {
				t.Fatalf("verify issued %s token with audience %q: %v", tc.kind, tc.audience, err)
			}
			// The signed attestation covers the resource group, while a /scale
			// subresource legitimately carries an autoscaling Scale object.
			scale := webhookAuthRequest(t,
				metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"},
				request.Resource, "scale", admissionv1.Update,
			)
			if err := verifier.Verify(ctx, scale, token.Status.Token); err != nil {
				t.Fatalf("verify issued token for legitimate /scale subresource: %v", err)
			}

			wrongGroup := webhookAuthRequest(t,
				metav1.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
				metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
				"", admissionv1.Create,
			)
			if err := verifier.Verify(ctx, wrongGroup, token.Status.Token); err == nil || err.Error() != "webhook token verification failed" {
				t.Fatalf("verify token for unauthorized group: got %v, want generic denial", err)
			}
			objectMismatch := request.DeepCopy()
			objectMismatch.Object.Raw = []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)
			oldObjectMismatch := webhookAuthRequest(t, request.Kind, request.Resource, "", admissionv1.Update)
			oldObjectMismatch.OldObject.Raw = []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)
			for _, mismatch := range []struct {
				name string
				req  *admissionv1.AdmissionRequest
			}{
				{name: "object group", req: objectMismatch},
				{name: "old object group", req: oldObjectMismatch},
			} {
				if err := verifier.Verify(ctx, mismatch.req, token.Status.Token); err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verify mismatched %s: got %v, want generic denial", mismatch.name, err)
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
				defer func() { _ = response.Body.Close() }()
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
				Request:  request.DeepCopy(),
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

func TestWebhookAuthLiveAPIServerDispatch(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerWebhookAuthenticationToken, true)

	ctx := ktesting.Init(t)
	var activeToken atomic.Value
	activeToken.Store("")

	var issuer, signingKeyFile string
	kubeClient, kubeConfig, tearDown := framework.StartTestServer(ctx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			issuer = "https://" + opts.SecureServing.Listener.Addr().String()
			signingKeyFile = opts.ServiceAccountSigningKeyFile
			opts.APIEnablement.RuntimeConfig["api/all"] = "true"
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
	discoveryClient, err := rest.HTTPClientFor(kubeConfig)
	if err != nil {
		t.Fatalf("create client for issuer discovery: %v", err)
	}

	// Borrow kube-apiserver's own ServiceAccount signing key so we can also
	// forge adversarial JWTs signed by the exact key advertised at /openid/v1/jwks.
	rawKey, err := keyutil.PrivateKeyFromFile(signingKeyFile)
	if err != nil {
		t.Fatalf("load apiserver signing key %q: %v", signingKeyFile, err)
	}
	ecdsaKey, ok := rawKey.(*ecdsa.PrivateKey)
	if !ok {
		t.Fatalf("expected *ecdsa.PrivateKey in %q, got %T", signingKeyFile, rawKey)
	}
	var jwksDoc jose.JSONWebKeySet
	jwksResp, err := discoveryClient.Get(issuer + "/openid/v1/jwks")
	if err != nil {
		t.Fatalf("fetch live apiserver JWKS: %v", err)
	}
	defer func() { _ = jwksResp.Body.Close() }()
	if err := json.NewDecoder(jwksResp.Body).Decode(&jwksDoc); err != nil || len(jwksDoc.Keys) == 0 {
		t.Fatalf("decode live apiserver JWKS: %v (keys=%d)", err, len(jwksDoc.Keys))
	}
	borrowedSigner, err := jose.NewSigner(
		jose.SigningKey{
			Algorithm: jose.ES256,
			Key:       jose.JSONWebKey{Key: ecdsaKey, KeyID: jwksDoc.Keys[0].KeyID},
		},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("create borrowed apiserver signer: %v", err)
	}
	signWithBorrowedKey := func(claims map[string]any) string {
		payload, err := json.Marshal(claims)
		if err != nil {
			t.Fatalf("marshal forged claims: %v", err)
		}
		jws, err := borrowedSigner.Sign(payload)
		if err != nil {
			t.Fatalf("sign forged claims: %v", err)
		}
		tok, err := jws.CompactSerialize()
		if err != nil {
			t.Fatalf("compact serialize forged JWS: %v", err)
		}
		return tok
	}

	// The proxy injects the test token; automatic apiserver token attachment is
	// outside this test. Token issuance and AdmissionReview dispatch are real.
	var targetHandler atomic.Value
	var lastWebhookStatus atomic.Int32
	targetWebhook := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetHandler.Load().(http.Handler).ServeHTTP(w, r)
	}))
	defer targetWebhook.Close()

	webhookProxy := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		outReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetWebhook.URL+r.URL.RequestURI(), r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		outReq.Header = r.Header.Clone()
		if tok := activeToken.Load().(string); tok != "" {
			outReq.Header.Set("Authorization", "Bearer "+tok)
		}
		resp, err := targetWebhook.Client().Do(outReq)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer func() { _ = resp.Body.Close() }()
		lastWebhookStatus.Store(int32(resp.StatusCode))
		for k, vals := range resp.Header {
			for _, v := range vals {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	}))
	defer webhookProxy.Close()

	webhookURL := webhookProxy.URL + "/validate"
	var admittedCount atomic.Int32
	var lastAdmittedRequest atomic.Pointer[admissionv1.AdmissionRequest]
	verifiedHandler, err := admissionauth.WithTokenVerification(
		ctx,
		func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
			admittedCount.Add(1)
			lastAdmittedRequest.Store(req.DeepCopy())
			return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
		},
		admissionauth.WithRemoteIssuer(issuer, webhookURL, discoveryClient),
	)
	if err != nil {
		t.Fatalf("construct admission.WithTokenVerification handler: %v", err)
	}
	targetHandler.Store(http.Handler(verifiedHandler))

	caBundle := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: webhookProxy.Certificate().Raw,
	})

	ns := framework.CreateNamespaceOrDie(kubeClient, "webhook-live-e2e", t)
	defer framework.DeleteNamespaceOrDie(kubeClient, ns, t)
	sa, deleteSA := createDeleteSvcAcct(t, cs, &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "apiserver-webhook-sa", Namespace: ns.Name},
	})
	defer deleteSA()

	authutil.GrantServiceAccountAuthorization(t, ctx, cs, sa.Name, ns.Name, rbacv1.PolicyRule{
		Verbs:         []string{"attest"},
		APIGroups:     []string{"authentication.k8s.io"},
		Resources:     []string{"admissionReviewAPIGroups"},
		ResourceNames: []string{"apps"},
	})
	authutil.GrantServiceAccountAuthorization(t, ctx, cs, sa.Name, ns.Name, rbacv1.PolicyRule{
		Verbs:     []string{"create"},
		APIGroups: []string{""},
		Resources: []string{"serviceaccounts/token"},
	})

	// Fail blocks early authentication errors; Ignore still honors explicit
	// admission-policy denials for the unauthorized batch group.
	failPolicy := admissionregistrationv1.Fail
	ignorePolicy := admissionregistrationv1.Ignore
	sideEffects := admissionregistrationv1.SideEffectClassNone
	vwc, deleteVWC := createDeleteValidating(t, cs, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "live-kep6060-validating"},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name:         "validate.kep6060.webhook.test",
				ClientConfig: admissionregistrationv1.WebhookClientConfig{URL: &webhookURL, CABundle: caBundle},
				Rules: []admissionregistrationv1.RuleWithOperations{
					{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update, admissionregistrationv1.Delete},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments", "deployments/scale"}},
					},
				},
				FailurePolicy:           &failPolicy,
				SideEffects:             &sideEffects,
				AdmissionReviewVersions: []string{"v1"},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"kubernetes.io/metadata.name": ns.Name},
				},
			},
			{
				Name:         "validate-ignore.kep6060.webhook.test",
				ClientConfig: admissionregistrationv1.WebhookClientConfig{URL: &webhookURL, CABundle: caBundle},
				Rules: []admissionregistrationv1.RuleWithOperations{
					{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{"batch"}, APIVersions: []string{"v1"}, Resources: []string{"jobs"}},
					},
				},
				FailurePolicy:           &ignorePolicy,
				SideEffects:             &sideEffects,
				AdmissionReviewVersions: []string{"v1"},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"kubernetes.io/metadata.name": ns.Name},
				},
			},
		},
	})
	defer deleteVWC()

	// Mint a genuine KEP-6060 token bound to vwc authorizing only the "apps" API group.
	issuedToken, err := cs.CoreV1().ServiceAccounts(ns.Name).CreateToken(ctx, sa.Name, &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			Audiences: []string{webhookURL},
			BoundObjectRef: &authenticationv1.BoundObjectReference{
				Kind:       "ValidatingWebhookConfiguration",
				APIVersion: "admissionregistration.k8s.io/v1",
				Name:       vwc.Name,
			},
			Attestations: map[string]authenticationv1.AttestationValue{
				"admissionReviewAPIGroups": {"apps"},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("issue real webhook-bound token via CreateToken: %v", err)
	}

	makeDeployment := func(name string) *appsv1.Deployment {
		replicas := int32(1)
		return &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns.Name},
			Spec: appsv1.DeploymentSpec{
				Replicas: &replicas,
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": name}},
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": name}},
					Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "pause", Image: "pause"}}},
				},
			},
		}
	}

	// Wait until the webhook informer intercepts an unauthenticated request.
	err = wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		lastWebhookStatus.Store(0)
		_, createErr := cs.AppsV1().Deployments(ns.Name).Create(ctx, makeDeployment("warmup-probe"), metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
		if createErr == nil {
			return false, nil
		}
		return lastWebhookStatus.Load() == http.StatusUnauthorized, nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for webhook registration to intercept Deployment creation: %v", err)
	}
	if admittedCount.Load() != 0 {
		t.Fatalf("unauthenticated warmup requests must never reach downstream handler, got %d", admittedCount.Load())
	}

	// 1. Live positive dispatch: configure the real CreateToken-issued bearer.
	// Creating an apps/v1 Deployment and updating its /scale subresource must both succeed.
	activeToken.Store(issuedToken.Status.Token)
	lastWebhookStatus.Store(0)
	createdDeploy, err := cs.AppsV1().Deployments(ns.Name).Create(ctx, makeDeployment("verified-deploy"), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create Deployment with valid webhook token failed: %v", err)
	}
	if admittedCount.Load() != 1 || lastWebhookStatus.Load() != http.StatusOK {
		t.Fatalf("Deployment create: admissions=%d status=%d, want 1 and 200", admittedCount.Load(), lastWebhookStatus.Load())
	}

	lastWebhookStatus.Store(0)
	_, err = cs.AppsV1().Deployments(ns.Name).UpdateScale(ctx, createdDeploy.Name, &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:            createdDeploy.Name,
			Namespace:       ns.Name,
			ResourceVersion: createdDeploy.ResourceVersion,
		},
		Spec: autoscalingv1.ScaleSpec{Replicas: 2},
	}, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("update Deployment /scale with valid apps webhook token failed: %v", err)
	}
	if admittedCount.Load() != 2 || lastWebhookStatus.Load() != http.StatusOK {
		t.Fatalf("Deployment scale: admissions=%d status=%d, want 2 and 200", admittedCount.Load(), lastWebhookStatus.Load())
	}

	// 2. Unauthorized API group with failurePolicy: Ignore:
	// The token only authorizes "apps", not "batch". Even though the batch webhook
	// rule uses failurePolicy: Ignore, our handler returns HTTP 200 Allowed:false,
	// so kube-apiserver must deny the Job creation.
	lastWebhookStatus.Store(0)
	_, err = cs.BatchV1().Jobs(ns.Name).Create(ctx, &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "unauthorized-job", Namespace: ns.Name},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers:    []corev1.Container{{Name: "job", Image: "busybox"}},
				},
			},
		},
	}, metav1.CreateOptions{})
	if err == nil || !strings.Contains(err.Error(), "webhook token verification failed") || lastWebhookStatus.Load() != http.StatusOK {
		t.Fatalf("apps-only token for Job: error=%v status=%d, want HTTP200 admission denial", err, lastWebhookStatus.Load())
	}
	if admittedCount.Load() != 2 {
		t.Fatalf("unauthorized batch/v1 Job must not reach downstream handler, admittedCount=%d", admittedCount.Load())
	}

	// 3. Ordinary unbound ServiceAccount token minted by kube-apiserver for the
	// exact webhook audience: must be rejected because it has no webhook binding.
	ordinarySAToken, err := cs.CoreV1().ServiceAccounts(ns.Name).CreateToken(ctx, sa.Name, &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			Audiences: []string{webhookURL},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("issue ordinary ServiceAccount token: %v", err)
	}
	activeToken.Store(ordinarySAToken.Status.Token)
	lastWebhookStatus.Store(0)
	_, err = cs.AppsV1().Deployments(ns.Name).Create(ctx, makeDeployment("ordinary-sa-token-deploy"), metav1.CreateOptions{})
	if err == nil || lastWebhookStatus.Load() != http.StatusUnauthorized {
		t.Fatalf("ordinary SA token: error=%v status=%d, want HTTP401 blocked by failurePolicy Fail", err, lastWebhookStatus.Load())
	}
	if admittedCount.Load() != 2 {
		t.Fatalf("ordinary SA token must not reach downstream handler, admittedCount=%d", admittedCount.Load())
	}

	// Signed tokens must still satisfy binding and attestation policy.
	now := time.Now()
	forgedProbes := []struct {
		name   string
		claims map[string]any
	}{
		{
			name: "borrowed-key token bound to both Validating and Mutating webhook configs",
			claims: map[string]any{
				"iss": issuer,
				"sub": "system:serviceaccount:" + ns.Name + ":" + sa.Name,
				"aud": []string{webhookURL},
				"exp": now.Add(5 * time.Minute).Unix(),
				"nbf": now.Add(-1 * time.Minute).Unix(),
				"iat": now.Unix(),
				"kubernetes.io": map[string]any{
					"validatingwebhookconfiguration": map[string]any{"name": vwc.Name, "uid": string(vwc.UID)},
					"mutatingwebhookconfiguration":   map[string]any{"name": "mwc", "uid": "mwc-uid"},
					"attestations":                   map[string]any{"admissionReviewAPIGroups": []string{"apps"}},
				},
			},
		},
		{
			name: "borrowed-key token carrying multiple admissionReviewAPIGroups",
			claims: map[string]any{
				"iss": issuer,
				"sub": "system:serviceaccount:" + ns.Name + ":" + sa.Name,
				"aud": []string{webhookURL},
				"exp": now.Add(5 * time.Minute).Unix(),
				"nbf": now.Add(-1 * time.Minute).Unix(),
				"iat": now.Unix(),
				"kubernetes.io": map[string]any{
					"validatingwebhookconfiguration": map[string]any{"name": vwc.Name, "uid": string(vwc.UID)},
					"attestations":                   map[string]any{"admissionReviewAPIGroups": []string{"apps", "batch"}},
				},
			},
		},
		{
			name: "borrowed-key token with missing webhook UID",
			claims: map[string]any{
				"iss": issuer,
				"sub": "system:serviceaccount:" + ns.Name + ":" + sa.Name,
				"aud": []string{webhookURL},
				"exp": now.Add(5 * time.Minute).Unix(),
				"nbf": now.Add(-1 * time.Minute).Unix(),
				"iat": now.Unix(),
				"kubernetes.io": map[string]any{
					"validatingwebhookconfiguration": map[string]any{"name": vwc.Name, "uid": ""},
					"attestations":                   map[string]any{"admissionReviewAPIGroups": []string{"apps"}},
				},
			},
		},
	}

	for i, fp := range forgedProbes {
		t.Run(fp.name, func(t *testing.T) {
			activeToken.Store(signWithBorrowedKey(fp.claims))
			lastWebhookStatus.Store(0)
			_, err := cs.AppsV1().Deployments(ns.Name).Create(ctx, makeDeployment("forged-deploy-"+string(rune('a'+i))), metav1.CreateOptions{})
			if err == nil || lastWebhookStatus.Load() != http.StatusUnauthorized {
				t.Fatalf("forged probe %q: error=%v status=%d, want HTTP401 rejection", fp.name, err, lastWebhookStatus.Load())
			}
			if admittedCount.Load() != 2 {
				t.Fatalf("forged probe %q reached downstream handler", fp.name)
			}
		})
	}

	activeToken.Store(issuedToken.Status.Token)
	lastWebhookStatus.Store(0)
	if err := cs.AppsV1().Deployments(ns.Name).Delete(ctx, createdDeploy.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete Deployment with authenticated oldObject: %v", err)
	}
	if admittedCount.Load() != 3 || lastWebhookStatus.Load() != http.StatusOK {
		t.Fatalf("Deployment delete: admissions=%d status=%d, want 3 and 200", admittedCount.Load(), lastWebhookStatus.Load())
	}
	deleted := lastAdmittedRequest.Load()
	if deleted == nil || deleted.Operation != admissionv1.Delete || len(deleted.Object.Raw) != 0 || deleted.Object.Object != nil || len(deleted.OldObject.Raw) == 0 {
		t.Fatalf("DELETE admission shape: %+v", deleted)
	}

	t.Run("discovered cross-group admission kinds", func(t *testing.T) {
		authutil.GrantServiceAccountAuthorization(t, ctx, cs, sa.Name, ns.Name, rbacv1.PolicyRule{
			Verbs: []string{"attest"}, APIGroups: []string{"authentication.k8s.io"},
			Resources: []string{authenticationv1.AttestationAdmissionReviewAPIGroups}, ResourceNames: []string{"*"},
		})
		wildcardToken, err := cs.CoreV1().ServiceAccounts(ns.Name).CreateToken(ctx, sa.Name, &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{webhookURL},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind: "ValidatingWebhookConfiguration", APIVersion: "admissionregistration.k8s.io/v1", Name: vwc.Name, UID: vwc.UID,
				},
				Attestations: map[string]authenticationv1.AttestationValue{authenticationv1.AttestationAdmissionReviewAPIGroups: {"*"}},
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("issue discovery test token: %v", err)
		}
		verifier, err := admissionauth.NewVerifier(ctx, admissionauth.WithRemoteIssuer(issuer, webhookURL, discoveryClient))
		if err != nil {
			t.Fatal(err)
		}
		verifyDiscoveredWebhookKinds(ctx, t, cs, verifier, wildcardToken.Status.Token, ns.Name)
	})

	t.Run("Ignore skips early authentication failure", func(t *testing.T) {
		activeToken.Store(ordinarySAToken.Status.Token)
		defer activeToken.Store(issuedToken.Status.Token)
		lastWebhookStatus.Store(0)
		before := admittedCount.Load()
		_, err := cs.BatchV1().Jobs(ns.Name).Create(ctx, &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "ignored-authentication-job", Namespace: ns.Name},
			Spec: batchv1.JobSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				RestartPolicy: corev1.RestartPolicyNever,
				Containers:    []corev1.Container{{Name: "job", Image: "busybox"}},
			}}},
		}, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
		if err != nil || lastWebhookStatus.Load() != http.StatusUnauthorized || admittedCount.Load() != before {
			t.Fatalf("Ignore authentication failure: error=%v status=%d admissions=%d, want skipped401 without callback", err, lastWebhookStatus.Load(), admittedCount.Load())
		}
	})
}
