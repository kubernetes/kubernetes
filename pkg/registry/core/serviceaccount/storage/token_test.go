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
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	token "k8s.io/kubernetes/pkg/serviceaccount"
)

func TestCreate_Token_WithExpiryCap(t *testing.T) {

	testcases := []struct {
		desc                 string
		extendExpiration     bool
		maxExpirationSeconds int
		expectedTokenAgeSec  int
		isExternal           bool
	}{
		{
			desc:                 "maxExpirationSeconds honoured",
			extendExpiration:     true,
			maxExpirationSeconds: 5 * 60 * 60, // 5h
			expectedTokenAgeSec:  5 * 60 * 60, // 5h
			isExternal:           true,
		},
		{
			desc:                 "ExpirationExtensionSeconds used for exp",
			extendExpiration:     true,
			maxExpirationSeconds: 2 * 365 * 24 * 60 * 60,           // 2 years
			expectedTokenAgeSec:  token.ExpirationExtensionSeconds, // 1y
			isExternal:           true,
		},
		{
			desc:                 "ExpirationExtensionSeconds used for exp",
			extendExpiration:     true,
			maxExpirationSeconds: 5 * 60 * 60,                      // 5h
			expectedTokenAgeSec:  token.ExpirationExtensionSeconds, // 1y
			isExternal:           false,
		},
		{
			desc:                 "requested time use with extension disabled",
			extendExpiration:     false,
			maxExpirationSeconds: 5 * 60 * 60, // 5h
			expectedTokenAgeSec:  3607,        // 1h
			isExternal:           true,
		},
		{
			desc:                 "maxExpirationSeconds honoured with extension disabled",
			extendExpiration:     false,
			maxExpirationSeconds: 30 * 60, // 30m
			expectedTokenAgeSec:  30 * 60, // 30m
			isExternal:           true,
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
			storage.Token.isTokenSignerExternal = tc.isExternal

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
			upperBound := tokenReqTimeStamp.Add(time.Duration(tc.expectedTokenAgeSec+10) * time.Second)
			lowerBound := tokenReqTimeStamp.Add(time.Duration(tc.expectedTokenAgeSec-10) * time.Second)

			// check for token expiration with a toleration of +/-10s after tokenReqTimeStamp to make for latencies.
			if structuredClaim.Expiry.Time().After(upperBound) ||
				structuredClaim.Expiry.Time().Before(lowerBound) {
				t.Fatalf("expected token expiration to be between %v to %v\n was %v", upperBound, lowerBound, structuredClaim.Expiry.Time())
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
