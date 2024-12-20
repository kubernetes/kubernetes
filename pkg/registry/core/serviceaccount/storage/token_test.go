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
	"strings"
	"testing"
	"time"

	"gopkg.in/go-jose/go-jose.v2/jwt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/warning"
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

			ctx := context.Background()
			// add the namespace to the context as it is required
			ctx = request.WithNamespace(ctx, serviceAccount.Namespace)

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
