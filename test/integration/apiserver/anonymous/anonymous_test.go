/*
Copyright 2024 The Kubernetes Authors.

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

package anonymous

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	_ "k8s.io/client-go/plugin/pkg/client/auth/oidc"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiserverapptesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	defaultNamespace           = "default"
	defaultRBACRoleName        = "developer-role"
	defaultRBACRoleBindingName = "developer-role-binding"
	anonymousUser              = "system:anonymous"
)

var (
	defaultRole = &rbacv1.Role{
		TypeMeta:   metav1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"},
		ObjectMeta: metav1.ObjectMeta{Name: defaultRBACRoleName, Namespace: defaultNamespace},
		Rules: []rbacv1.PolicyRule{
			{
				Verbs:     []string{"list"},
				Resources: []string{"pods"},
				APIGroups: []string{""},
			},
		},
	}
	defaultRoleBinding = &rbacv1.RoleBinding{
		TypeMeta:   metav1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "RoleBinding"},
		ObjectMeta: metav1.ObjectMeta{Name: defaultRBACRoleBindingName, Namespace: defaultNamespace},
		Subjects: []rbacv1.Subject{
			{
				APIGroup: rbac.GroupName,
				Kind:     rbacv1.UserKind,
				Name:     anonymousUser,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     defaultRBACRoleName,
		},
	}
)

func TestStructuredAuthenticationConfig(t *testing.T) {
	t.Log("Testing anonymous authenticator with authentication config")

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StructuredAuthenticationConfiguration, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AnonymousAuthConfigurableEndpoints, true)

	testCases := []struct {
		desc            string
		authConfig      string
		additionalFlags []string
		assertErrFn     func(t *testing.T, errorToCheck error)
		assertFn        func(t *testing.T, server kubeapiserverapptesting.TestServer)
	}{
		{
			desc: "valid config no conditions",
			authConfig: `
apiVersion: apiserver.config.k8s.io/v1beta1
kind: AuthenticationConfiguration
anonymous:
  enabled: true
`,
			assertErrFn: func(t *testing.T, errorToCheck error) {
				require.NoError(t, errorToCheck)
			},
			assertFn: func(t *testing.T, server kubeapiserverapptesting.TestServer) {
				configureRBAC(t, server)

				client := insecureHTTPClient(t)

				resp, err := client.Get(server.ClientConfig.Host + "/api/v1/namespaces/default/pods")
				require.NoError(t, err)
				defer resp.Body.Close() //nolint:errcheck
				require.Equal(t, 200, resp.StatusCode)

				resp, err = client.Get(server.ClientConfig.Host + "/livez")
				require.NoError(t, err)
				defer resp.Body.Close() //nolint:errcheck
				require.Equal(t, 200, resp.StatusCode)

				resp, err = client.Get(server.ClientConfig.Host + "/healthz")
				require.NoError(t, err)
				defer resp.Body.Close() //nolint:errcheck
				require.Equal(t, 200, resp.StatusCode)

			},
			additionalFlags: nil,
		},
		{
			desc: "valid config with conditions",
			authConfig: `
apiVersion: apiserver.config.k8s.io/v1beta1
kind: AuthenticationConfiguration
anonymous:
  enabled: true
  conditions:
  - path: "/livez"
`,
			assertErrFn: func(t *testing.T, errorToCheck error) {
				require.NoError(t, errorToCheck)
			},
			assertFn: func(t *testing.T, server kubeapiserverapptesting.TestServer) {
				client := insecureHTTPClient(t)

				resp, err := client.Get(server.ClientConfig.Host + "/api/v1/namespaces/default/pods")
				require.NoError(t, err)
				defer resp.Body.Close() //nolint:errcheck
				require.Equal(t, 401, resp.StatusCode)

				resp, err = client.Get(server.ClientConfig.Host + "/livez")
				require.NoError(t, err)
				defer resp.Body.Close() //nolint:errcheck
				require.Equal(t, 200, resp.StatusCode)

				resp, err = client.Get(server.ClientConfig.Host + "/healthz")
				require.NoError(t, err)
				defer resp.Body.Close() //nolint:errcheck
				require.Equal(t, 401, resp.StatusCode)

			},
			additionalFlags: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			flags := []string{"--authorization-mode=RBAC"}
			flags = append(flags, fmt.Sprintf("--authentication-config=%s", writeTempFile(t, tc.authConfig)))
			flags = append(flags, tc.additionalFlags...)

			server, err := kubeapiserverapptesting.StartTestServer(
				t,
				kubeapiserverapptesting.NewDefaultTestServerOptions(),
				flags,
				framework.SharedEtcd(),
			)

			tc.assertErrFn(t, err)

			if tc.assertFn == nil {
				return
			}

			defer server.TearDownFn()

			tc.assertFn(t, server)
		})
	}
}

func TestMain(m *testing.M) {
	// framework.EtcdMain(m.Run)
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()
	file, err := os.CreateTemp("", "anonymous-auth-test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Remove(file.Name()); err != nil {
			t.Fatal(err)
		}
	})
	if err := os.WriteFile(file.Name(), []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	return file.Name()
}

func configureRBAC(t *testing.T, server kubeapiserverapptesting.TestServer) {
	t.Helper()

	kc := kubernetes.NewForConfigOrDie(server.ClientConfig)

	_, err := kc.RbacV1().Roles(defaultNamespace).Create(context.Background(), defaultRole, metav1.CreateOptions{})
	require.NoError(t, err)

	_, err = kc.RbacV1().RoleBindings(defaultNamespace).Create(context.Background(), defaultRoleBinding, metav1.CreateOptions{})
	require.NoError(t, err)

	authutil.WaitForNamedAuthorizationUpdate(
		t,
		context.Background(),
		kc.AuthorizationV1(),
		anonymousUser,
		defaultNamespace, // namespace
		"list",
		"", // resource name
		schema.GroupResource{Group: "", Resource: "pods"},
		true,
	)
}

func insecureHTTPClient(t *testing.T) *http.Client {
	t.Helper()

	return &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
}
