/*
Copyright 2017 The Kubernetes Authors.

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

package token

import (
	"bytes"
	"testing"
	"time"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestEncodeTokenSecretData(t *testing.T) {
	var tests = []struct {
		token *kubeadmapi.TokenDiscovery
		t     time.Duration
	}{
		{token: &kubeadmapi.TokenDiscovery{ID: "foo", Secret: "bar"}},                 // should use default
		{token: &kubeadmapi.TokenDiscovery{ID: "foo", Secret: "bar"}, t: time.Second}, // should use default
	}
	for _, rt := range tests {
		actual := encodeTokenSecretData(rt.token.ID, rt.token.Secret, rt.t, []string{}, "")
		if !bytes.Equal(actual["token-id"], []byte(rt.token.ID)) {
			t.Errorf(
				"failed EncodeTokenSecretData:\n\texpected: %s\n\t  actual: %s",
				rt.token.ID,
				actual["token-id"],
			)
		}
		if !bytes.Equal(actual["token-secret"], []byte(rt.token.Secret)) {
			t.Errorf(
				"failed EncodeTokenSecretData:\n\texpected: %s\n\t  actual: %s",
				rt.token.Secret,
				actual["token-secret"],
			)
		}
		if rt.t > 0 {
			if actual["expiration"] == nil {
				t.Errorf(
					"failed EncodeTokenSecretData, duration was not added to time",
				)
			}
		}
	}
}
