/*
Copyright 2023 The Kubernetes Authors.

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

package node

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
)

func TestUpdateOrCreateTokens(t *testing.T) {
	tests := []struct {
		name         string
		failIfExists bool
		tokens       []bootstraptokenv1.BootstrapToken
		wantErr      bool
	}{
		{
			name:         "token is nil",
			failIfExists: true,
			tokens:       []bootstraptokenv1.BootstrapToken{},
			wantErr:      false,
		},
		{
			name:         "create secret which does not exist",
			failIfExists: true,
			tokens: []bootstraptokenv1.BootstrapToken{
				{
					Token: &bootstraptokenv1.BootstrapTokenString{
						ID:     "token1",
						Secret: "token1data",
					},
				},
			},
			wantErr: false,
		},
		{
			name:         "create multiple secrets which do not exist",
			failIfExists: true,
			tokens: []bootstraptokenv1.BootstrapToken{
				{
					Token: &bootstraptokenv1.BootstrapTokenString{
						ID:     "token1",
						Secret: "token1data",
					},
				},
				{
					Token: &bootstraptokenv1.BootstrapTokenString{
						ID:     "token2",
						Secret: "token2data",
					},
				},
				{
					Token: &bootstraptokenv1.BootstrapTokenString{
						ID:     "token3",
						Secret: "token3data",
					},
				},
			},
			wantErr: false,
		},
		{
			name:         "create secret which exists, failIfExists is false",
			failIfExists: false,
			tokens: []bootstraptokenv1.BootstrapToken{
				{
					Token: &bootstraptokenv1.BootstrapTokenString{
						ID:     "foo",
						Secret: "bar",
					},
				},
			},
			wantErr: false,
		},
		{
			name:         "create secret which exists, failIfExists is true",
			failIfExists: true,
			tokens: []bootstraptokenv1.BootstrapToken{
				{
					Token: &bootstraptokenv1.BootstrapTokenString{
						ID:     "foo",
						Secret: "bar",
					},
				},
			},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := newMockClientForTest(t)
			if err := UpdateOrCreateTokens(client, tc.failIfExists, tc.tokens); (err != nil) != tc.wantErr {
				t.Fatalf("UpdateOrCreateTokens() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func newMockClientForTest(t *testing.T) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()
	_, err := client.CoreV1().Secrets(metav1.NamespaceSystem).Create(context.TODO(), &v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bootstrap-token-foo",
			Labels:    map[string]string{"app": "foo"},
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string][]byte{"foo": {'f', 'o', 'o'}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating sercet: %v", err)
	}
	return client
}
