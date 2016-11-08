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

package master

import (
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestValidTokenPopulatesSecrets(t *testing.T) {
	s := newSecretsWithToken(t, "123456.0123456789AbCdEf")
	expectedToken := "0123456789abcdef"
	expectedTokenID := "123456"

	err := generateTokenIfNeeded(s)
	if err != nil {
		t.Errorf("generateTokenIfNeeded gave an error for a valid token: %v", err)
	}
	if s.ID != expectedTokenID {
		t.Errorf("generateTokenIfNeeded did not populate the TokenID correctly; expected [%s] but got [%s]", expectedTokenID, s.ID)
	}
	if s.Secret != expectedToken {
		t.Errorf("generateTokenIfNeeded did not populate the Token correctly; expected %v but got %v", expectedToken, s.Secret)
	}
}

func newSecretsWithToken(t *testing.T, token string) *kubeadmapi.TokenDiscovery {
	var err error
	d := &kubeadmapi.TokenDiscovery{}
	if token == "" {
		return d
	}
	d.ID, d.Secret, err = util.ParseToken(token)
	if err != nil {
		t.Errorf("failed to parse token")
	}
	return d
}
