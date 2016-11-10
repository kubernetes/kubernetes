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

package util

import (
	"bytes"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestUsingEmptyTokenFails(t *testing.T) {
	// Simulates what happens when you omit --token on the CLI
	s := newSecretsWithToken("")

	given, err := UseGivenTokenIfValid(s)
	if err != nil {
		t.Errorf("UseGivenTokenIfValid returned an error when the token was omitted: %v", err)
	}
	if given {
		t.Error("UseGivenTokenIfValid returned given = true when the token was omitted; expected false")
	}
}

func TestTokenValidationFailures(t *testing.T) {
	invalidTokens := []string{
		"1234567890123456789012",
		"12345.1234567890123456",
		".1234567890123456",
		"123456.1234567890.123456",
	}

	for _, token := range invalidTokens {
		s := newSecretsWithToken(token)
		_, err := UseGivenTokenIfValid(s)

		if err == nil {
			t.Errorf("UseGivenTokenIfValid did not return an error for this invalid token: [%s]", token)
		}
	}
}

func TestValidTokenPopulatesSecrets(t *testing.T) {
	s := newSecretsWithToken("123456.0123456789AbCdEf")
	expectedToken := []byte("0123456789abcdef")
	expectedTokenID := "123456"
	expectedBearerToken := "0123456789abcdef"

	given, err := UseGivenTokenIfValid(s)
	if err != nil {
		t.Errorf("UseGivenTokenIfValid gave an error for a valid token: %v", err)
	}
	if !given {
		t.Error("UseGivenTokenIfValid returned given = false when given a valid token")
	}
	if s.TokenID != expectedTokenID {
		t.Errorf("UseGivenTokenIfValid did not populate the TokenID correctly; expected [%s] but got [%s]", expectedTokenID, s.TokenID)
	}
	if s.BearerToken != expectedBearerToken {
		t.Errorf("UseGivenTokenIfValid did not populate the BearerToken correctly; expected [%s] but got [%s]", expectedBearerToken, s.BearerToken)
	}
	if !bytes.Equal(s.Token, expectedToken) {
		t.Errorf("UseGivenTokenIfValid did not populate the Token correctly; expected %v but got %v", expectedToken, s.Token)
	}
}

func newSecretsWithToken(token string) *kubeadmapi.Secrets {
	s := new(kubeadmapi.Secrets)
	s.GivenToken = token
	return s
}
