/*
Copyright 2016 The Kubernetes Authors.

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

package bootstrap

import (
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/klog/v2/ktesting"
)

const (
	givenTokenID     = "abc123"
	givenTokenID2    = "def456"
	givenTokenSecret = "tokenSecret"
)

func timeString(delta time.Duration) string {
	return time.Now().Add(delta).Format(time.RFC3339)
}

func TestValidateSecretForSigning(t *testing.T) {
	cases := []struct {
		description string
		tokenID     string
		tokenSecret string
		okToSign    string
		expiration  string
		valid       bool
	}{
		{
			"Signing token with no exp",
			givenTokenID, givenTokenSecret, "true", "", true,
		},
		{
			"Signing token with valid exp",
			givenTokenID, givenTokenSecret, "true", timeString(time.Hour), true,
		},
		{
			"Expired signing token",
			givenTokenID, givenTokenSecret, "true", timeString(-time.Hour), false,
		},
		{
			"Signing token with bad exp",
			givenTokenID, givenTokenSecret, "true", "garbage", false,
		},
		{
			"Signing token without signing bit",
			givenTokenID, givenTokenSecret, "", "garbage", false,
		},
		{
			"Signing token with bad signing bit",
			givenTokenID, givenTokenSecret, "", "", false,
		},
		{
			"Signing token with no ID",
			"", givenTokenSecret, "true", "", false,
		},
		{
			"Signing token with no secret",
			givenTokenID, "", "true", "", false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.description, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			secret := &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       metav1.NamespaceSystem,
					Name:            bootstrapapi.BootstrapTokenSecretPrefix + givenTokenID,
					ResourceVersion: "1",
				},
				Type: bootstrapapi.SecretTypeBootstrapToken,
				Data: map[string][]byte{
					bootstrapapi.BootstrapTokenIDKey:           []byte(tc.tokenID),
					bootstrapapi.BootstrapTokenSecretKey:       []byte(tc.tokenSecret),
					bootstrapapi.BootstrapTokenUsageSigningKey: []byte(tc.okToSign),
					bootstrapapi.BootstrapTokenExpirationKey:   []byte(tc.expiration),
				},
			}

			tokenID, tokenSecret, ok := validateSecretForSigning(ctx, secret)
			if ok != tc.valid {
				t.Errorf("Unexpected validation failure. Expected %v, got %v", tc.valid, ok)
			}
			if ok {
				if tokenID != tc.tokenID {
					t.Errorf("Unexpected Token ID. Expected %q, got %q", givenTokenID, tokenID)
				}
				if tokenSecret != tc.tokenSecret {
					t.Errorf("Unexpected Token Secret. Expected %q, got %q", givenTokenSecret, tokenSecret)
				}
			}
		})
	}

}

func TestValidateSecret(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       metav1.NamespaceSystem,
			Name:            bootstrapapi.BootstrapTokenSecretPrefix + givenTokenID,
			ResourceVersion: "1",
		},
		Type: bootstrapapi.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:           []byte(givenTokenID),
			bootstrapapi.BootstrapTokenSecretKey:       []byte(givenTokenSecret),
			bootstrapapi.BootstrapTokenUsageSigningKey: []byte("true"),
		},
	}

	tokenID, tokenSecret, ok := validateSecretForSigning(ctx, secret)
	if !ok {
		t.Errorf("Unexpected validation failure.")
	}
	if tokenID != givenTokenID {
		t.Errorf("Unexpected Token ID. Expected %q, got %q", givenTokenID, tokenID)
	}
	if tokenSecret != givenTokenSecret {
		t.Errorf("Unexpected Token Secret. Expected %q, got %q", givenTokenSecret, tokenSecret)
	}
}

func TestBadSecretName(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       metav1.NamespaceSystem,
			Name:            givenTokenID,
			ResourceVersion: "1",
		},
		Type: bootstrapapi.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:           []byte(givenTokenID),
			bootstrapapi.BootstrapTokenSecretKey:       []byte(givenTokenSecret),
			bootstrapapi.BootstrapTokenUsageSigningKey: []byte("true"),
		},
	}

	_, _, ok := validateSecretForSigning(ctx, secret)
	if ok {
		t.Errorf("Token validation should fail with bad name")
	}
}

func TestMismatchSecretName(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       metav1.NamespaceSystem,
			Name:            bootstrapapi.BootstrapTokenSecretPrefix + givenTokenID2,
			ResourceVersion: "1",
		},
		Type: bootstrapapi.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:           []byte(givenTokenID),
			bootstrapapi.BootstrapTokenSecretKey:       []byte(givenTokenSecret),
			bootstrapapi.BootstrapTokenUsageSigningKey: []byte("true"),
		},
	}

	_, _, ok := validateSecretForSigning(ctx, secret)
	if ok {
		t.Errorf("Token validation should fail with mismatched name")
	}
}
