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
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestValidTokenPopulatesSecrets(t *testing.T) {
	t.Run("provided", func(t *testing.T) {
		expectedID := "123456"
		expectedSecret := "0123456789abcdef"
		s := &kubeadmapi.TokenDiscovery{
			ID:     expectedID,
			Secret: expectedSecret,
		}

		err := kubeadmutil.GenerateTokenIfNeeded(s)
		if err != nil {
			t.Errorf("GenerateTokenIfNeeded gave an error for a valid token: %v", err)
		}
		if s.ID != expectedID {
			t.Errorf("GenerateTokenIfNeeded did not populate the TokenID correctly; expected [%s] but got [%s]", expectedID, s.ID)
		}
		if s.Secret != expectedSecret {
			t.Errorf("GenerateTokenIfNeeded did not populate the Token correctly; expected %v but got %v", expectedSecret, s.Secret)
		}
	})

	t.Run("not provided", func(t *testing.T) {
		s := &kubeadmapi.TokenDiscovery{}

		err := kubeadmutil.GenerateTokenIfNeeded(s)
		if err != nil {
			t.Errorf("GenerateTokenIfNeeded gave an error for a valid token: %v", err)
		}
		if s.ID == "" {
			t.Errorf("GenerateTokenIfNeeded did not populate the TokenID correctly; expected ID to be non-empty")
		}
		if s.Secret == "" {
			t.Errorf("GenerateTokenIfNeeded did not populate the Token correctly; expected Secret to be non-empty")
		}
	})
}
