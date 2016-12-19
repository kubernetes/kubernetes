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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestTokenParseErrors(t *testing.T) {
	invalidTokens := []string{
		"1234567890123456789012",
		"12345.1234567890123456",
		".1234567890123456",
		"123456.1234567890.123456",
	}

	for _, token := range invalidTokens {
		if _, _, err := ParseToken(token); err == nil {
			t.Errorf("generateTokenIfNeeded did not return an error for this invalid token: [%s]", token)
		}
	}
}

func TestGenerateToken(t *testing.T) {
	var cfg kubeadmapi.TokenDiscovery

	GenerateToken(&cfg)
	if len(cfg.ID) != 6 {
		t.Errorf("failed GenerateToken first part length:\n\texpected: 6\n\t  actual: %d", len(cfg.ID))
	}
	if len(cfg.Secret) != 16 {
		t.Errorf("failed GenerateToken first part length:\n\texpected: 16\n\t  actual: %d", len(cfg.Secret))
	}
}

func TestRandBytes(t *testing.T) {
	var randTest = []int{
		0,
		1,
		2,
		3,
		100,
	}

	for _, rt := range randTest {
		actual, err := RandBytes(rt)
		if err != nil {
			t.Errorf("failed RandBytes: %v", err)
		}
		if len(actual) != rt*2 {
			t.Errorf("failed RandBytes:\n\texpected: %d\n\t  actual: %d\n", rt*2, len(actual))
		}
	}
}
