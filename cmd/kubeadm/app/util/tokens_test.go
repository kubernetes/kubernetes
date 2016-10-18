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
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
)

func TestUsingEmptyTokenFails(t *testing.T) {
	// Simulates what happens when you omit --token on the CLI
	config := newConfigWithToken("")

	ok, err := UseGivenTokenIfValid(config)
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
		config := newConfigWithToken(token)
		ok, err := UseGivenTokenIfValid(config)

		assert.False(t, ok, "expected invalid token to return ok = false: [%s]", token)
		assert.Error(t, err, "expected invalid token to return an error: [%s]", token)
		assert.Contains(t, err.Error(), "<6 characters>.<16 characters>", "expected better validation failure message")
	}
}

func TestValidTokenPopulatesSecrets(t *testing.T) {
	config := newConfigWithToken("123456.0123456789AbCdEf")

	ok, err := UseGivenTokenIfValid(config)
	assert.True(t, ok)
	assert.NoError(t, err)

	assert.Equal(t, "123456", config.Secrets.TokenID)
	assert.Equal(t, "0123456789abcdef", config.Secrets.BearerToken)
	assert.Equal(t, []byte("0123456789abcdef"), config.Secrets.Token)
}

func newConfigWithToken(token string) *kubeadmapi.KubeadmConfig {
	config := new(kubeadmapi.KubeadmConfig)
	config.Secrets.GivenToken = token
	return config
}
