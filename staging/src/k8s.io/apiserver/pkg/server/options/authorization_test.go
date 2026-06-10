/*
Copyright The Kubernetes Authors.

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

package options

import (
	"testing"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

// TestDelegatingAuthorizationOptionsToAuthorizerAuthorizerName verifies that the union returned
// by DelegatingAuthorizationOptions.toAuthorizer reports an AuthorizerName composed of the
// names of its sub-authorizers in insertion order. Sub-authorizer names are stable identifiers
// owned by each authorizer implementation (privilegedGroupAuthorizer → "authorizer.kubernetes.io/PrivilegedGroups",
// path.NewAuthorizer → AuthorizerFunc → "authorizer.kubernetes.io/AuthorizerFunc",
// webhook from DelegatingAuthorizerConfig.New → "authorizer.kubernetes.io/Webhook").
func TestDelegatingAuthorizationOptionsToAuthorizerAuthorizerName(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	backoff := DefaultAuthWebhookRetryBackoff()

	tests := []struct {
		name          string
		options       *DelegatingAuthorizationOptions
		useFakeClient bool
		wantName      string
	}{
		{
			name:     "empty options, nil client => empty union",
			options:  &DelegatingAuthorizationOptions{},
			wantName: "authorizer.kubernetes.io/Union[]",
		},
		{
			name: "AlwaysAllowGroups only, nil client",
			options: &DelegatingAuthorizationOptions{
				AlwaysAllowGroups: []string{"system:masters"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups]",
		},
		{
			name: "AlwaysAllowPaths only, nil client",
			options: &DelegatingAuthorizationOptions{
				AlwaysAllowPaths: []string{"/healthz"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/AuthorizerFunc]",
		},
		{
			name: "Groups + Paths, nil client",
			options: &DelegatingAuthorizationOptions{
				AlwaysAllowGroups: []string{"system:masters"},
				AlwaysAllowPaths:  []string{"/healthz", "/readyz"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, authorizer.kubernetes.io/AuthorizerFunc]",
		},
		{
			name: "Groups + Paths + webhook client",
			options: &DelegatingAuthorizationOptions{
				AlwaysAllowGroups:   []string{"system:masters"},
				AlwaysAllowPaths:    []string{"/healthz"},
				WebhookRetryBackoff: backoff,
			},
			useFakeClient: true,
			wantName:      "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, authorizer.kubernetes.io/AuthorizerFunc, authorizer.kubernetes.io/Webhook]",
		},
		{
			name: "webhook client only",
			options: &DelegatingAuthorizationOptions{
				WebhookRetryBackoff: backoff,
			},
			useFakeClient: true,
			wantName:      "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/Webhook]",
		},
		{
			name:          "NewDelegatingAuthorizationOptions defaults, with client",
			options:       NewDelegatingAuthorizationOptions(),
			useFakeClient: true,
			wantName:      "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, authorizer.kubernetes.io/AuthorizerFunc, authorizer.kubernetes.io/Webhook]",
		},
		{
			name:     "NewDelegatingAuthorizationOptions defaults, nil client",
			options:  NewDelegatingAuthorizationOptions(),
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, authorizer.kubernetes.io/AuthorizerFunc]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var client kubernetes.Interface
			if tt.useFakeClient {
				client = fakeClient
			}
			authz, err := tt.options.toAuthorizer(client)
			if err != nil {
				t.Fatalf("toAuthorizer: %v", err)
			}
			if got := authz.AuthorizerName(); got != tt.wantName {
				t.Errorf("AuthorizerName() = %q, want %q", got, tt.wantName)
			}
		})
	}
}
