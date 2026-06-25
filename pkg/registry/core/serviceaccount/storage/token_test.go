/*
Copyright 2024 The Kubernetes Authors.

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

package storage

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"iter"
	"slices"
	"strings"
	"testing"
	"time"

	"gopkg.in/go-jose/go-jose.v2/jwt"

	admissionregistration "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/warning"
	fake "k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	token "k8s.io/kubernetes/pkg/serviceaccount"
)

func TestCreate_Token_WithExpiryCap(t *testing.T) {

	testcases := []struct {
		desc                         string
		extendExpiration             bool
		maxExpirationSeconds         int
		maxExtendedExpirationSeconds int
		expectedTokenAgeSec          int
	}{
		{
			desc:                         "passed expiration respected if less than max",
			extendExpiration:             false,
			maxExpirationSeconds:         5 * 60 * 60,                               // 5h
			maxExtendedExpirationSeconds: token.ExpirationExtensionSeconds,          // 1y
			expectedTokenAgeSec:          token.WarnOnlyBoundTokenExpirationSeconds, // 1h 7s
		},
		{
			desc:                         "maxExtendedExpirationSeconds honoured",
			extendExpiration:             true,
			maxExpirationSeconds:         2 * 60 * 60, // 2h
			maxExtendedExpirationSeconds: 5 * 60 * 60, // 5h
			expectedTokenAgeSec:          5 * 60 * 60, // 5h
		},
		{
			desc:                         "ExpirationExtensionSeconds used for exp",
			extendExpiration:             true,
			maxExpirationSeconds:         2 * 365 * 24 * 60 * 60,           // 2y
			maxExtendedExpirationSeconds: token.ExpirationExtensionSeconds, // 1y
			expectedTokenAgeSec:          token.ExpirationExtensionSeconds, // 1y
		},
		{
			desc:                         "ExpirationSeconds used for exp",
			extendExpiration:             true,
			maxExpirationSeconds:         5 * 60 * 60,                      // 5h
			maxExtendedExpirationSeconds: token.ExpirationExtensionSeconds, // 1y
			expectedTokenAgeSec:          token.ExpirationExtensionSeconds, // 1y
		},
		{
			desc:                         "requested time use with extension disabled",
			extendExpiration:             false,
			maxExpirationSeconds:         5 * 60 * 60, // 5h
			expectedTokenAgeSec:          3607,        // 1h
			maxExtendedExpirationSeconds: token.ExpirationExtensionSeconds,
		},
		{
			desc:                         "maxExpirationSeconds honoured with extension disabled",
			extendExpiration:             false,
			maxExpirationSeconds:         30 * 60, // 30m
			expectedTokenAgeSec:          30 * 60, // 30m
			maxExtendedExpirationSeconds: token.ExpirationExtensionSeconds,
		},
	}

	// Create a test service account
	serviceAccount := validNewServiceAccount("foo")

	// Create a new pod
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: serviceAccount.Namespace,
		},
		Spec: api.PodSpec{
			ServiceAccountName: serviceAccount.Name,
		},
	}
	podGetter := &objectGetter{obj: pod}
	aud := authenticator.Audiences{
		"aud-1",
		"aud-2",
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			storage, server := newTokenStorage(t, testTokenGenerator{"fake"}, aud, podGetter, panicGetter{}, nil)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericregistrytest.NewNamespaceScopeContext(storage.Store, serviceAccount.Namespace)

			// Enable ExternalServiceAccountTokenSigner feature
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalServiceAccountTokenSigner, true)

			// record namespace in the store.
			_, err := storage.Store.Create(ctx, serviceAccount, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed creating test service account: %v", err)
			}

			// add the namespace to the context as it is required
			ctx = request.WithNamespace(ctx, serviceAccount.Namespace)
			storage.Token.extendExpiration = tc.extendExpiration
			storage.Token.maxExpirationSeconds = int64(tc.maxExpirationSeconds)
			storage.Token.maxExtendedExpirationSeconds = int64(tc.maxExtendedExpirationSeconds)

			tokenReqTimeStamp := time.Now()
			out, err := storage.Token.Create(ctx, serviceAccount.Name, &authenticationapi.TokenRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      serviceAccount.Name,
					Namespace: serviceAccount.Namespace,
				},
				Spec: authenticationapi.TokenRequestSpec{
					ExpirationSeconds: 3607,
					BoundObjectRef: &authenticationapi.BoundObjectReference{
						Name:       pod.Name,
						Kind:       "Pod",
						APIVersion: "v1",
					},
					Audiences: aud,
				},
			}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed calling /token endpoint for service account: %v", err)
			}

			tokenReq := out.(*authenticationapi.TokenRequest)
			payload := strings.Split(tokenReq.Status.Token, ".")[1]
			claims, err := base64.RawURLEncoding.DecodeString(payload)
			if err != nil {
				t.Fatalf("failed when decoding payload: %v", err)
			}
			structuredClaim := jwt.Claims{}
			err = json.Unmarshal(claims, &structuredClaim)
			if err != nil {
				t.Fatalf("Error unmarshalling Claims: %v", err)
			}
			structuredClaim.Expiry.Time()
			confidenceInterval := 10 // seconds
			upperBound := tokenReqTimeStamp.Add(time.Duration(tc.expectedTokenAgeSec+confidenceInterval) * time.Second)
			lowerBound := tokenReqTimeStamp.Add(time.Duration(tc.expectedTokenAgeSec-confidenceInterval) * time.Second)

			// check for token expiration with a toleration of +/-10s after tokenReqTimeStamp to make for latencies.
			if structuredClaim.Expiry.Time().After(upperBound) ||
				structuredClaim.Expiry.Time().Before(lowerBound) {
				expiryDiff := structuredClaim.Expiry.Time().Sub(tokenReqTimeStamp)
				t.Fatalf("expected token expiration to be %v (±%ds) in the future, was %v", time.Duration(tc.expectedTokenAgeSec)*time.Second, confidenceInterval, expiryDiff)
			}

		})

	}
}

func TestTokenRequest_ServiceAccountUIDValidation(t *testing.T) {
	testCases := []struct {
		name                    string
		featureGateEnabled      bool
		serviceAccountUID       types.UID
		requestUID              types.UID
		expectError             string
		expectedRecordedWarning string
		expectAuditAnnotations  map[string]string
		expectedResultUID       types.UID
	}{
		{
			name:               "feature gate enabled - matching UID",
			featureGateEnabled: true,
			serviceAccountUID:  "correct-sa-uid-123",
			requestUID:         "correct-sa-uid-123",
			expectedResultUID:  "correct-sa-uid-123",
		},
		{
			name:               "feature gate enabled - mismatched UID",
			featureGateEnabled: true,
			serviceAccountUID:  "correct-sa-uid-123",
			requestUID:         "wrong-sa-uid-456",
			expectError:        `Operation cannot be fulfilled on TokenRequest.authentication.k8s.io "test-sa": the UID in the token request (wrong-sa-uid-456) does not match the UID of the service account (correct-sa-uid-123)`,
		},
		{
			name:                    "feature gate disabled - mismatched UID",
			featureGateEnabled:      false,
			serviceAccountUID:       "correct-sa-uid-123",
			requestUID:              "wrong-sa-uid-456",
			expectedResultUID:       "wrong-sa-uid-456", // No validation, so request UID is used as-is (backwards compatibility)
			expectedRecordedWarning: "the UID in the token request (wrong-sa-uid-456) does not match the UID of the service account (correct-sa-uid-123) but TokenRequestServiceAccountUIDValidation is not enabled. In the future, this will return a conflict error",
			expectAuditAnnotations: map[string]string{
				"authentication.k8s.io/token-request-uid-mismatch": "the UID in the token request (wrong-sa-uid-456) does not match the UID of the service account (correct-sa-uid-123)",
			},
		},
		{
			name:               "empty request UID",
			featureGateEnabled: false,
			serviceAccountUID:  "correct-sa-uid-123",
			requestUID:         "",
			expectedResultUID:  "correct-sa-uid-123",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.TokenRequestServiceAccountUIDValidation, tc.featureGateEnabled)

			serviceAccount := validNewServiceAccount("test-sa")
			serviceAccount.UID = tc.serviceAccountUID

			serviceAccountGetter := &objectGetter{obj: serviceAccount}
			aud := authenticator.Audiences{"test-audience"}

			storage, server := newTokenStorage(t, testTokenGenerator{"fake"}, aud, panicGetter{}, panicGetter{}, nil)
			defer server.Terminate(t)
			defer storage.DestroyFunc()

			storage.Token.svcaccts = serviceAccountGetter

			dc := dummyRecorder{agent: "", text: ""}
			ctx := context.Background()
			ctx = request.WithNamespace(warning.WithWarningRecorder(ctx, &dc), serviceAccount.Namespace)
			// create an audit context to allow recording audit information
			ctx = audit.WithAuditContext(ctx)

			tokenReq := &authenticationapi.TokenRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      serviceAccount.Name,
					Namespace: serviceAccount.Namespace,
					UID:       tc.requestUID,
				},
				Spec: authenticationapi.TokenRequestSpec{
					Audiences:         aud,
					ExpirationSeconds: 3600, // 1 hour
				},
			}

			out, err := storage.Token.Create(ctx, serviceAccount.Name, tokenReq, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})

			if len(tc.expectError) > 0 {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				if err.Error() != tc.expectError {
					t.Errorf("expected error %q, got %q", tc.expectError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("expected no error but got: %v", err)
				}
				result := out.(*authenticationapi.TokenRequest)
				if result.UID != tc.expectedResultUID {
					t.Errorf("expected result UID %q, got %q", tc.expectedResultUID, result.UID)
				}
			}

			if len(tc.expectedRecordedWarning) > 0 && tc.expectedRecordedWarning != dc.getWarning() {
				t.Errorf("expected recorded warning %q, got %q", tc.expectedRecordedWarning, dc.getWarning())
			}

			auditContext := audit.AuditContextFrom(ctx)
			for key, expectedValue := range tc.expectAuditAnnotations {
				actualValue, ok := auditContext.GetEventAnnotation(key)
				if !ok || actualValue != expectedValue {
					t.Errorf("expected audit annotation %q with value %q, got %v", key, expectedValue, actualValue)
				}
			}
		})
	}
}

type objectGetter struct {
	obj runtime.Object
	err error
}

func (f objectGetter) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return f.obj, f.err
}

var _ rest.Getter = objectGetter{}

// A basic fake token generator which always returns a static string
type testTokenGenerator struct {
	staticToken string
}

func (f testTokenGenerator) GenerateToken(ctx context.Context, claims *jwt.Claims, privateClaims interface{}) (string, error) {
	c, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	return f.staticToken + "." + base64.RawURLEncoding.EncodeToString(c) + "." + f.staticToken, nil
}

var _ token.TokenGenerator = testTokenGenerator{}

type dummyRecorder struct {
	agent string
	text  string
}

func (r *dummyRecorder) AddWarning(agent, text string) {
	r.agent = agent
	r.text = text
}

func (r *dummyRecorder) getWarning() string {
	return r.text
}

var _ warning.Recorder = &dummyRecorder{}

func TestValidateWebhookAudience(t *testing.T) {
	hooks := slices.Values([]*webhookFields{
		{config: &admissionregistration.WebhookClientConfig{URL: new("https://validate.example.com")}},
		{config: &admissionregistration.WebhookClientConfig{
			Service: &admissionregistration.ServiceReference{
				Name:      "webhook-svc",
				Namespace: "webhook-ns",
				Port:      new(int32(8443)),
				Path:      new("/validate"),
			},
		}},
		{config: &admissionregistration.WebhookClientConfig{
			Service: &admissionregistration.ServiceReference{
				Name:      "default-port-svc",
				Namespace: "default-ns",
			},
		}},
		{config: &admissionregistration.WebhookClientConfig{
			Service: &admissionregistration.ServiceReference{
				Name:      "no-leading-slash-svc",
				Namespace: "default-ns",
				Port:      new(int32(443)),
				Path:      new("hooks"),
			},
		}},
		{config: &admissionregistration.WebhookClientConfig{
			Service: &admissionregistration.ServiceReference{
				Name:      "empty-path-svc",
				Namespace: "default-ns",
				Port:      new(int32(443)),
				Path:      new(""),
			},
		}},
		{config: &admissionregistration.WebhookClientConfig{
			Service: &admissionregistration.ServiceReference{
				Name:      "slash-path-svc",
				Namespace: "default-ns",
				Port:      new(int32(443)),
				Path:      new("/"),
			},
		}},
	})

	cases := []struct {
		name     string
		audience string
		wantErr  string
	}{
		{name: "URL match", audience: "https://validate.example.com"},
		{name: "service match with port and path", audience: "https://webhook-svc.webhook-ns.svc:8443/validate"},
		{name: "service match with default port (nil path)", audience: "https://default-port-svc.default-ns.svc:443/"},
		{name: "service path without leading slash gets slash prepended", audience: "https://no-leading-slash-svc.default-ns.svc:443/hooks"},
		{name: "service with empty string path treated as slash", audience: "https://empty-path-svc.default-ns.svc:443/"},
		{name: "service with explicit slash path", audience: "https://slash-path-svc.default-ns.svc:443/"},
		{name: "service with trailing slash rejected", audience: "https://webhook-svc.webhook-ns.svc:8443/validate/", wantErr: "audience does not match webhook client config in the bound object"},
		{name: "no match", audience: "https://unknown.example.com", wantErr: "audience does not match webhook client config in the bound object"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateWebhookAudience(hooks, tc.audience)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error %q, got nil", tc.wantErr)
				}
				if got := err.Error(); got != tc.wantErr {
					t.Errorf("expected error:\n\t%s\ngot:\n\t%s", tc.wantErr, got)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestValidateAttestationAPIGroup(t *testing.T) {
	appsHooks := slices.Values([]*webhookFields{
		{rules: []admissionregistration.RuleWithOperations{
			{Rule: admissionregistration.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments"}}},
			{Rule: admissionregistration.Rule{APIGroups: []string{"batch"}, APIVersions: []string{"v1"}, Resources: []string{"jobs"}}},
		}},
	})

	wildcardHooks := slices.Values([]*webhookFields{
		{rules: []admissionregistration.RuleWithOperations{
			{Rule: admissionregistration.Rule{APIGroups: []string{"*"}, APIVersions: []string{"v1"}, Resources: []string{"*"}}},
		}},
	})

	emptyHooks := slices.Values([]*webhookFields{{}})

	cases := []struct {
		name    string
		group   string
		hooks   iter.Seq[*webhookFields]
		wantErr string
	}{
		{name: "exact match apps", group: "apps", hooks: appsHooks},
		{name: "exact match batch", group: "batch", hooks: appsHooks},
		{name: "wildcard attestation skips check", group: "*", hooks: appsHooks},
		{name: "wildcard attestation against empty hooks", group: "*", hooks: emptyHooks},
		{name: "wildcard rule matches any group", group: "networking.k8s.io", hooks: wildcardHooks},
		{name: "no match", group: "networking.k8s.io", hooks: appsHooks, wantErr: "attestation does not match webhook rules"},
		{name: "empty rules", group: "apps", hooks: emptyHooks, wantErr: "attestation does not match webhook rules"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAttestationAPIGroup(tc.hooks, tc.group)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error %q, got nil", tc.wantErr)
				}
				if got := err.Error(); got != tc.wantErr {
					t.Errorf("expected error:\n\t%s\ngot:\n\t%s", tc.wantErr, got)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// recordingAuthorizer records all Authorize calls and returns a configurable decision.
type recordingAuthorizer struct {
	decision authorizer.Decision
	calls    []authorizer.Attributes
}

func (r *recordingAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	r.calls = append(r.calls, a)
	if r.decision == authorizer.DecisionAllow {
		return authorizer.DecisionAllow, "", nil
	}
	return authorizer.DecisionDeny, "denied by test", nil
}

func (r *recordingAuthorizer) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionFromParts(r.Authorize(ctx, a))
}

func (r *recordingAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", nil
}

func newTestTokenREST(t *testing.T, authz authorizer.Authorizer, fakeClient *fake.Clientset) *TokenREST {
	sa := &api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "test-sa", Namespace: "test-ns", UID: "sa-uid-123"},
	}
	return &TokenREST{
		svcaccts: &fakeObjectGetter{obj: sa},
		pods:     nil,
		secrets:  nil,
		nodes:    nil,
		validatingWebhooks: &vwhGetter{
			validatingWebhooks: fakeClient.AdmissionregistrationV1().ValidatingWebhookConfigurations(),
		},
		mutatingWebhooks: &mwhGetter{
			mutatingWebhooks: fakeClient.AdmissionregistrationV1().MutatingWebhookConfigurations(),
		},
		authorizer:           authz,
		issuer:               fakeTokenGenerator{"fake-token"},
		auds:                 authenticator.Audiences{"api"},
		audsSet:              sets.New("api"),
		maxExpirationSeconds: 3600,
	}
}

func testWebhookCreateContext() context.Context {
	ctx := request.WithNamespace(request.NewContext(), "test-ns")
	ctx = request.WithRequestInfo(ctx, &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "create",
		Namespace:         "test-ns",
		Resource:          "serviceaccounts",
		Subresource:       "token",
		Name:              "test-sa",
	})
	return audit.WithAuditContext(ctx)
}

type fakeObjectGetter struct {
	obj runtime.Object
	err error
}

func (f *fakeObjectGetter) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.obj, nil
}

func TestTokenRESTCreateWebhookAuthenticationFlow(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerWebhookAuthenticationToken, true)

	webhookCfg := &admissionregistration.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "my-webhook", UID: "webhook-uid-456"},
		Webhooks: []admissionregistration.ValidatingWebhook{{
			ClientConfig: admissionregistration.WebhookClientConfig{URL: new("https://webhook.example.com")},
			Rules: []admissionregistration.RuleWithOperations{{
				Rule: admissionregistration.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments"}},
			}},
		}},
	}

	validReq := &authenticationapi.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-sa", Namespace: "test-ns"},
		Spec: authenticationapi.TokenRequestSpec{
			Audiences:         []string{"https://webhook.example.com"},
			ExpirationSeconds: 3600,
			BoundObjectRef: &authenticationapi.BoundObjectReference{
				Kind:       "ValidatingWebhookConfiguration",
				APIVersion: "admissionregistration.k8s.io/v1",
				Name:       "my-webhook",
			},
			Attestations: map[string]authenticationapi.AttestationValue{
				"admissionReviewAPIGroups": {"apps"},
			},
		},
	}

	ctx := testWebhookCreateContext()

	t.Run("success", func(t *testing.T) {
		authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
		fakeClient := fake.NewClientset(webhookCfg)
		r := newTestTokenREST(t, authz, fakeClient)

		_, err := r.Create(ctx, "test-sa", validReq.DeepCopy(), nil, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Verify authorizer was called with correct attributes
		if len(authz.calls) != 1 {
			t.Fatalf("expected 1 authorizer call, got %d", len(authz.calls))
		}
		call := authz.calls[0]
		if call.GetVerb() != "attest" {
			t.Errorf("expected verb 'attest', got %q", call.GetVerb())
		}
		if call.GetAPIGroup() != "authentication.k8s.io" {
			t.Errorf("expected API group 'authentication.k8s.io', got %q", call.GetAPIGroup())
		}
		if call.GetResource() != "admissionReviewAPIGroups" {
			t.Errorf("expected resource 'admissionReviewAPIGroups', got %q", call.GetResource())
		}
		if call.GetName() != "apps" {
			t.Errorf("expected name 'apps', got %q", call.GetName())
		}

		// Verify the fake client was called to get the webhook config
		actions := fakeClient.Actions()
		if len(actions) != 1 {
			t.Fatalf("expected 1 client action, got %d: %v", len(actions), actions)
		}
		getAction, ok := actions[0].(ktesting.GetAction)
		if !ok {
			t.Fatalf("expected GetAction, got %T", actions[0])
		}
		if getAction.GetName() != "my-webhook" {
			t.Errorf("expected get of 'my-webhook', got %q", getAction.GetName())
		}
	})

	t.Run("denied by authorizer", func(t *testing.T) {
		authz := &recordingAuthorizer{decision: authorizer.DecisionDeny}
		fakeClient := fake.NewClientset(webhookCfg)
		r := newTestTokenREST(t, authz, fakeClient)

		_, err := r.Create(ctx, "test-sa", validReq.DeepCopy(), nil, &metav1.CreateOptions{})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		wantErr := `admissionReviewAPIGroups.authentication.k8s.io "apps" is forbidden: User "system:serviceaccount:test-ns:test-sa" cannot attest resource "admissionReviewAPIGroups" in API group "authentication.k8s.io" at the cluster scope: denied by test`
		if got := err.Error(); got != wantErr {
			t.Fatalf("expected error:\n\t%s\ngot:\n\t%s", wantErr, got)
		}

		// Authorizer should have been called
		if len(authz.calls) != 1 {
			t.Fatalf("expected 1 authorizer call, got %d", len(authz.calls))
		}
		// Webhook getter should NOT have been called (authz denied before reaching it)
		if len(fakeClient.Actions()) != 0 {
			t.Fatalf("expected 0 client actions, got %d", len(fakeClient.Actions()))
		}
	})

	t.Run("wrong audience", func(t *testing.T) {
		authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
		fakeClient := fake.NewClientset(webhookCfg)
		r := newTestTokenREST(t, authz, fakeClient)

		req := validReq.DeepCopy()
		req.Spec.Audiences = []string{"https://wrong.example.com"}

		_, err := r.Create(ctx, "test-sa", req, nil, &metav1.CreateOptions{})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		wantErr := `serviceaccounts/token "test-sa" is forbidden: token request denied`
		if got := err.Error(); got != wantErr {
			t.Fatalf("expected error:\n\t%s\ngot:\n\t%s", wantErr, got)
		}

		// Webhook getter should have been called (audience check happens after fetch)
		if len(fakeClient.Actions()) != 1 {
			t.Fatalf("expected 1 client action, got %d", len(fakeClient.Actions()))
		}
	})

	t.Run("API group not in webhook rules", func(t *testing.T) {
		authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
		fakeClient := fake.NewClientset(webhookCfg)
		r := newTestTokenREST(t, authz, fakeClient)

		req := validReq.DeepCopy()
		req.Spec.Attestations = map[string]authenticationapi.AttestationValue{
			"admissionReviewAPIGroups": {"networking.k8s.io"},
		}

		_, err := r.Create(ctx, "test-sa", req, nil, &metav1.CreateOptions{})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		wantErr := `serviceaccounts/token "test-sa" is forbidden: token request denied`
		if got := err.Error(); got != wantErr {
			t.Fatalf("expected error:\n\t%s\ngot:\n\t%s", wantErr, got)
		}
	})

	t.Run("feature gate disabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerWebhookAuthenticationToken, false)

		authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
		fakeClient := fake.NewClientset(webhookCfg)
		r := newTestTokenREST(t, authz, fakeClient)

		_, err := r.Create(ctx, "test-sa", validReq.DeepCopy(), nil, &metav1.CreateOptions{})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		wantErr := `cannot bind token to object of type admissionregistration.k8s.io/v1, Kind=ValidatingWebhookConfiguration (feature gate APIServerWebhookAuthenticationToken is disabled)`
		if got := err.Error(); got != wantErr {
			t.Fatalf("expected error:\n\t%s\ngot:\n\t%s", wantErr, got)
		}

		// Neither authorizer nor webhook getter should be called
		if len(authz.calls) != 0 {
			t.Fatalf("expected 0 authorizer calls, got %d", len(authz.calls))
		}
		if len(fakeClient.Actions()) != 0 {
			t.Fatalf("expected 0 client actions, got %d", len(fakeClient.Actions()))
		}
	})
}

func TestTokenRESTCreateWebhookExpirationCap(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerWebhookAuthenticationToken, true)

	webhookCfg := &admissionregistration.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "my-webhook", UID: "webhook-uid-456"},
		Webhooks: []admissionregistration.ValidatingWebhook{{
			ClientConfig: admissionregistration.WebhookClientConfig{URL: new("https://webhook.example.com")},
			Rules: []admissionregistration.RuleWithOperations{{
				Rule: admissionregistration.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments"}},
			}},
		}},
	}

	ctx := testWebhookCreateContext()

	cases := []struct {
		name          string
		requestExp    int64
		wantExpCapped int64
	}{
		{
			name:          "expiration at minimum (equals cap) is preserved",
			requestExp:    600,
			wantExpCapped: 600,
		},
		{
			name:          "expiration over cap is truncated to 10 minutes",
			requestExp:    3600,
			wantExpCapped: 600,
		},
		{
			name:          "expiration slightly over cap is truncated",
			requestExp:    601,
			wantExpCapped: 600,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
			fakeClient := fake.NewClientset(webhookCfg)
			r := newTestTokenREST(t, authz, fakeClient)

			req := &authenticationapi.TokenRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-sa", Namespace: "test-ns"},
				Spec: authenticationapi.TokenRequestSpec{
					Audiences:         []string{"https://webhook.example.com"},
					ExpirationSeconds: tc.requestExp,
					BoundObjectRef: &authenticationapi.BoundObjectReference{
						Kind:       "ValidatingWebhookConfiguration",
						APIVersion: "admissionregistration.k8s.io/v1",
						Name:       "my-webhook",
					},
					Attestations: map[string]authenticationapi.AttestationValue{
						"admissionReviewAPIGroups": {"apps"},
					},
				},
			}

			result, err := r.Create(ctx, "test-sa", req, nil, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			tokenReq := result.(*authenticationapi.TokenRequest)
			if tokenReq.Spec.ExpirationSeconds != tc.wantExpCapped {
				t.Errorf("expected ExpirationSeconds %d, got %d", tc.wantExpCapped, tokenReq.Spec.ExpirationSeconds)
			}
		})
	}
}

func TestTokenRESTCreateWebhookDeletionTimestamp(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerWebhookAuthenticationToken, true)

	ctx := testWebhookCreateContext()

	makeReq := func() *authenticationapi.TokenRequest {
		return &authenticationapi.TokenRequest{
			ObjectMeta: metav1.ObjectMeta{Name: "test-sa", Namespace: "test-ns"},
			Spec: authenticationapi.TokenRequestSpec{
				Audiences:         []string{"https://webhook.example.com"},
				ExpirationSeconds: 600,
				BoundObjectRef: &authenticationapi.BoundObjectReference{
					Kind:       "ValidatingWebhookConfiguration",
					APIVersion: "admissionregistration.k8s.io/v1",
					Name:       "my-webhook",
				},
				Attestations: map[string]authenticationapi.AttestationValue{
					"admissionReviewAPIGroups": {"apps"},
				},
			},
		}
	}

	baseWebhook := func() *admissionregistration.ValidatingWebhookConfiguration {
		return &admissionregistration.ValidatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: "my-webhook", UID: "webhook-uid-456"},
			Webhooks: []admissionregistration.ValidatingWebhook{{
				ClientConfig: admissionregistration.WebhookClientConfig{URL: new("https://webhook.example.com")},
				Rules: []admissionregistration.RuleWithOperations{{
					Rule: admissionregistration.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments"}},
				}},
			}},
		}
	}

	t.Run("nil deletion timestamp succeeds", func(t *testing.T) {
		cfg := baseWebhook()
		// DeletionTimestamp is nil by default
		authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
		fakeClient := fake.NewClientset(cfg)
		r := newTestTokenREST(t, authz, fakeClient)

		_, err := r.Create(ctx, "test-sa", makeReq(), nil, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("deletion timestamp in the past is rejected", func(t *testing.T) {
		cfg := baseWebhook()
		pastTime := metav1.NewTime(time.Now().Add(-1 * time.Hour))
		cfg.DeletionTimestamp = &pastTime

		authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
		fakeClient := fake.NewClientset(cfg)
		r := newTestTokenREST(t, authz, fakeClient)

		_, err := r.Create(ctx, "test-sa", makeReq(), nil, &metav1.CreateOptions{})
		if err == nil {
			t.Fatal("expected error for webhook with past deletion timestamp, got nil")
		}
		wantErr := `serviceaccounts/token "test-sa" is forbidden: token request denied`
		if got := err.Error(); got != wantErr {
			t.Fatalf("expected error:\n\t%s\ngot:\n\t%s", wantErr, got)
		}
	})

	t.Run("deletion timestamp in the future succeeds", func(t *testing.T) {
		cfg := baseWebhook()
		futureTime := metav1.NewTime(time.Now().Add(1 * time.Hour))
		cfg.DeletionTimestamp = &futureTime

		authz := &recordingAuthorizer{decision: authorizer.DecisionAllow}
		fakeClient := fake.NewClientset(cfg)
		r := newTestTokenREST(t, authz, fakeClient)

		_, err := r.Create(ctx, "test-sa", makeReq(), nil, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}
