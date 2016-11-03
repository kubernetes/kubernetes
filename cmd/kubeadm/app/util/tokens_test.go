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
	"strings"
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
	var tests = []struct {
		t        string
		expected bool
	}{
		{
			t:        "1234567890123456789012",
			expected: false,
		},
		{
			t:        "12345.1234567890123456",
			expected: false,
		},
		{
			t:        ".1234567890123456",
			expected: false,
		},
		{
			t:        "123456.1234567890.123456",
			expected: false,
		},
	}

	for _, rt := range tests {
		s := newSecretsWithToken(rt.t)
		_, err := UseGivenTokenIfValid(s)
		if (err == nil) != rt.expected {
			t.Errorf(
				"failed UseGivenTokenIfValid and did not return an error for this invalid token: [%s]",
				rt.t,
			)
		}
	}
}

func TestValidTokenPopulatesSecrets(t *testing.T) {
	var tests = []struct {
		token               string
		expectedToken       []byte
		expectedTokenID     string
		expectedBearerToken string
	}{
		{
			token:               "123456.0123456789AbCdEf",
			expectedToken:       []byte("0123456789abcdef"),
			expectedTokenID:     "123456",
			expectedBearerToken: "0123456789abcdef",
		},
	}

	for _, rt := range tests {
		s := newSecretsWithToken(rt.token)

		given, err := UseGivenTokenIfValid(s)
		if err != nil {
			t.Errorf("UseGivenTokenIfValid gave an error for a valid token: %v", err)
		}
		if !given {
			t.Error("UseGivenTokenIfValid returned given = false when given a valid token")
		}
		if s.TokenID != rt.expectedTokenID {
			t.Errorf("UseGivenTokenIfValid did not populate the TokenID correctly; expected [%s] but got [%s]", rt.expectedTokenID, s.TokenID)
		}
		if s.BearerToken != rt.expectedBearerToken {
			t.Errorf("UseGivenTokenIfValid did not populate the BearerToken correctly; expected [%s] but got [%s]", rt.expectedBearerToken, s.BearerToken)
		}
		if !bytes.Equal(s.Token, rt.expectedToken) {
			t.Errorf("UseGivenTokenIfValid did not populate the Token correctly; expected %v but got %v", rt.expectedToken, s.Token)
		}
	}
}

func newSecretsWithToken(token string) *kubeadmapi.Secrets {
	s := new(kubeadmapi.Secrets)
	s.GivenToken = token
	return s
}

func TestGenerateToken(t *testing.T) {
	var genTest = []struct {
		s kubeadmapi.Secrets
		l int
		n int
	}{
		{kubeadmapi.Secrets{}, 2, 6},
	}

	for _, rt := range genTest {
		GenerateToken(&rt.s)
		givenToken := strings.Split(strings.ToLower(rt.s.GivenToken), ".")
		if len(givenToken) != rt.l {
			t.Errorf(
				"failed GenerateToken num parts:\n\texpected: %d\n\t  actual: %d",
				rt.l,
				len(givenToken),
			)
		}
		if len(givenToken[0]) != rt.n {
			t.Errorf(
				"failed GenerateToken first part length:\n\texpected: %d\n\t  actual: %d",
				rt.l,
				len(givenToken),
			)
		}
	}
}

func TestUseGivenTokenIfValid(t *testing.T) {
	var tokenTest = []struct {
		s        kubeadmapi.Secrets
		expected bool
	}{
		{kubeadmapi.Secrets{GivenToken: ""}, false},         // GivenToken == ""
		{kubeadmapi.Secrets{GivenToken: "noperiod"}, false}, // not 2-part '.' format
		{kubeadmapi.Secrets{GivenToken: "abcd.a"}, false},   // len(tokenID) != 6
		{kubeadmapi.Secrets{GivenToken: "abcdef.a"}, true},
	}

	for _, rt := range tokenTest {
		actual, _ := UseGivenTokenIfValid(&rt.s)
		if actual != rt.expected {
			t.Errorf(
				"failed UseGivenTokenIfValid:\n\texpected: %t\n\t  actual: %t\n\t token:%s",
				rt.expected,
				actual,
				rt.s.GivenToken,
			)
		}
	}
}

func TestRandBytes(t *testing.T) {
	var randTest = []struct {
		r        int
		l        int
		expected error
	}{
		{0, 0, nil},
		{1, 1, nil},
		{2, 2, nil},
		{3, 3, nil},
		{100, 100, nil},
	}

	for _, rt := range randTest {
		actual, _, err := RandBytes(rt.r)
		if err != rt.expected {
			t.Errorf(
				"failed RandBytes:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				err,
			)
		}
		if len(actual) != rt.l {
			t.Errorf(
				"failed RandBytes:\n\texpected: %d\n\t  actual: %d\n",
				rt.l,
				len(actual),
			)
		}
	}
}
