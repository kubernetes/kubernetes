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
	"testing"

	"github.com/stretchr/testify/assert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestUsingEmptyTokenFails(t *testing.T) {
	// Simulates what happens when you omit --token on the CLI
	s := newSecretsWithToken("")

	ok, err := UseGivenTokenIfValid(s)
	assert.False(t, ok)
	assert.NoError(t, err)
}

func TestTokenValidationFailures(t *testing.T) {
	invalidTokens := []string{
		"1234567890123456789012",
		"12345.1234567890123456",
		".1234567890123456",
		"123456.1234567890.123456",
	}

	for _, token := range(invalidTokens) {
		s := newSecretsWithToken(token)
		ok, err := UseGivenTokenIfValid(s)

		assert.False(t, ok, "expected invalid token to return ok = false: [%s]", token)
		assert.Error(t, err, "expected invalid token to return an error: [%s]", token)
		assert.Contains(t, err.Error(), "<6 characters>.<16 characters>", "expected better validation failure message")
	}
}

func TestValidTokenPopulatesSecrets(t *testing.T) {
	s := newSecretsWithToken("123456.0123456789AbCdEf")

	ok, err := UseGivenTokenIfValid(s)
	assert.True(t, ok)
	assert.NoError(t, err)

	assert.Equal(t, "123456", s.TokenID)
	assert.Equal(t, "0123456789abcdef", s.BearerToken)
	assert.Equal(t, []byte("0123456789abcdef"), s.Token)
}

func newSecretsWithToken(token string) *kubeadmapi.Secrets {
	s := new(kubeadmapi.Secrets)
	s.GivenToken = token
	return s
}
