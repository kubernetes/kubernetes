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

package authorizer

import (
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/util/wait"
	authzconfig "k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/node"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
)

// TestReloadableAuthorizerResolverNewForConfigAuthorizerName verifies that the union returned
// by reloadableAuthorizerResolver.newForConfig reports an AuthorizerName composed of the
// superuser group authorizer plus one entry per configured authorizer, using the per-entry
// name from the AuthorizationConfiguration (surfaced through instrumentedAuthorizer).
func TestReloadableAuthorizerResolverNewForConfigAuthorizerName(t *testing.T) {
	// LoadKubeconfig stats the file, so a non-existent path fails. Write a minimal valid
	// kubeconfig to a temp file; the file is only parsed during webhook construction and
	// never actually dialed by the AuthorizerName-only assertions below.
	kubeConfigPath := filepath.Join(t.TempDir(), "webhook-kubeconfig")
	if err := os.WriteFile(kubeConfigPath, []byte(`apiVersion: v1
kind: Config
clusters:
- name: test
  cluster:
    server: https://example.invalid
contexts:
- name: test
  context:
    cluster: test
    user: test
current-context: test
users:
- name: test
  user:
    token: dummy
`), 0o600); err != nil {
		t.Fatalf("writing kubeconfig: %v", err)
	}
	webhookEntry := func(name string) authzconfig.AuthorizerConfiguration {
		return authzconfig.AuthorizerConfiguration{
			Type: authzconfig.AuthorizerType(modes.ModeWebhook),
			Name: name,
			Webhook: &authzconfig.WebhookConfiguration{
				FailurePolicy:              authzconfig.FailurePolicyNoOpinion,
				SubjectAccessReviewVersion: "v1",
				ConnectionInfo: authzconfig.WebhookConnectionInfo{
					Type:           authzconfig.AuthorizationWebhookConnectionInfoTypeKubeConfigFile,
					KubeConfigFile: &kubeConfigPath,
				},
			},
		}
	}

	tests := []struct {
		name     string
		entries  []authzconfig.AuthorizerConfiguration
		wantName string
	}{
		{
			name: "AlwaysAllow",
			entries: []authzconfig.AuthorizerConfiguration{
				{Type: authzconfig.AuthorizerType(modes.ModeAlwaysAllow), Name: "alwaysallow"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, alwaysallow]",
		},
		{
			name: "AlwaysDeny",
			entries: []authzconfig.AuthorizerConfiguration{
				{Type: authzconfig.AuthorizerType(modes.ModeAlwaysDeny), Name: "alwaysdeny"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, alwaysdeny]",
		},
		{
			name: "Node",
			entries: []authzconfig.AuthorizerConfiguration{
				{Type: authzconfig.AuthorizerType(modes.ModeNode), Name: "node"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, node]",
		},
		{
			name: "RBAC",
			entries: []authzconfig.AuthorizerConfiguration{
				{Type: authzconfig.AuthorizerType(modes.ModeRBAC), Name: "rbac"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, rbac]",
		},
		{
			name: "ABAC",
			entries: []authzconfig.AuthorizerConfiguration{
				{Type: authzconfig.AuthorizerType(modes.ModeABAC), Name: "abac"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, abac]",
		},
		{
			name: "mixed Node + RBAC + AlwaysAllow keeps configured order",
			entries: []authzconfig.AuthorizerConfiguration{
				{Type: authzconfig.AuthorizerType(modes.ModeNode), Name: "node"},
				{Type: authzconfig.AuthorizerType(modes.ModeRBAC), Name: "rbac"},
				{Type: authzconfig.AuthorizerType(modes.ModeAlwaysAllow), Name: "fallback-allow"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, node, rbac, fallback-allow]",
		},
		{
			name: "mixed Node + RBAC + Webhook1 + Webhook2 + Deny",
			entries: []authzconfig.AuthorizerConfiguration{
				{Type: authzconfig.AuthorizerType(modes.ModeNode), Name: "node"},
				{Type: authzconfig.AuthorizerType(modes.ModeRBAC), Name: "rbac"},
				webhookEntry("webhook1"),
				webhookEntry("webhook2"),
				{Type: authzconfig.AuthorizerType(modes.ModeAlwaysDeny), Name: "fallback-deny"},
			},
			wantName: "authorizer.kubernetes.io/Union[authorizer.kubernetes.io/PrivilegedGroups, node, rbac, webhook1, webhook2, fallback-deny]",
		},
	}

	// Webhook construction in newForConfig requires a non-nil WebhookRetryBackoff on the
	// resolver's initialConfig. Provide a trivial backoff for the webhook cases; the values
	// are not exercised because no SAR is ever issued.
	backoff := wait.Backoff{Steps: 1}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// newForConfig requires the per-type inner authorizer pointer to be non-nil before
			// dispatching to it. The actual decision logic is not exercised here — we only
			// check the AuthorizerName composition — so a zero-value instance is sufficient.
			r := &reloadableAuthorizerResolver{
				initialConfig: Config{
					WebhookRetryBackoff: &backoff,
				},
				nodeAuthorizer: &node.NodeAuthorizer{},
				rbacAuthorizer: &rbac.RBACAuthorizer{},
				abacAuthorizer: abac.PolicyList{}, // non-nil empty slice
			}
			cfg := &authzconfig.AuthorizationConfiguration{Authorizers: tt.entries}

			authz, _, err := r.newForConfig(cfg)
			if err != nil {
				t.Fatalf("newForConfig: %v", err)
			}
			if got := authz.AuthorizerName(); got != tt.wantName {
				t.Errorf("AuthorizerName() = %q, want %q", got, tt.wantName)
			}
		})
	}
}
