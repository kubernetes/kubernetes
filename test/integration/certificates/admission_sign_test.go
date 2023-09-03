/*
Copyright 2020 The Kubernetes Authors.

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

package certificates

import (
	"context"
	"testing"

	certv1 "k8s.io/api/certificates/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

// Verifies that the CSR approval admission plugin correctly enforces that a
// user has permission to sign CSRs for the named signer
func TestCSRSignerNameSigningPlugin(t *testing.T) {
	tests := map[string]struct {
		allowedSignerName string
		signerName        string
		error             string
	}{
		"should admit when a user has permission for the exact signerName": {
			allowedSignerName: "example.com/something",
			signerName:        "example.com/something",
		},
		"should admit when a user has permission for the wildcard-suffixed signerName": {
			allowedSignerName: "example.com/*",
			signerName:        "example.com/something",
		},
		"should deny if a user does not have permission for the given signerName": {
			allowedSignerName: "example.com/not-something",
			signerName:        "example.com/something",
			error:             `certificatesigningrequests.certificates.k8s.io "csr" is forbidden: user not permitted to sign requests with signerName "example.com/something"`,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// Run an apiserver with the default configuration options.
			s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{"--authorization-mode=RBAC"}, framework.SharedEtcd())
			defer s.TearDownFn()
			client := clientset.NewForConfigOrDie(s.ClientConfig)

			// Drop the default RBAC superuser permissions to rely on the internal superuser authorizer
			if err := client.RbacV1().ClusterRoleBindings().Delete(context.TODO(), "cluster-admin", metav1.DeleteOptions{}); err != nil {
				t.Fatal(err)
			}

			// Grant 'test-user' permission to sign CertificateSigningRequests with the specified signerName.
			const username = "test-user"
			grantUserPermissionToSignFor(t, client, username, test.allowedSignerName)
			// Create a CSR to attempt to sign.
			csr := createTestingCSR(t, client.CertificatesV1().CertificateSigningRequests(), "csr", test.signerName, "")

			// Create a second client, impersonating the 'test-user' for us to test with.
			testuserConfig := restclient.CopyConfig(s.ClientConfig)
			testuserConfig.Impersonate = restclient.ImpersonationConfig{UserName: username}
			testuserClient := clientset.NewForConfigOrDie(testuserConfig)

			// Attempt to 'sign' the certificate.
			// random valid pem
			csr.Status.Certificate = []byte(`
Leading non-PEM content
-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE-----
Intermediate non-PEM content
-----BEGIN CERTIFICATE-----
MIIBqDCCAU6gAwIBAgIUfqZtjoFgczZ+oQZbEC/BDSS2J6wwCgYIKoZIzj0EAwIw
EjEQMA4GA1UEAxMHUm9vdC1DQTAgFw0xNjEwMTEwNTA2MDBaGA8yMTE2MDkxNzA1
MDYwMFowGjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMFkwEwYHKoZIzj0CAQYI
KoZIzj0DAQcDQgAEyWHEMMCctJg8Xa5YWLqaCPbk3MjB+uvXac42JM9pj4k9jedD
kpUJRkWIPzgJI8Zk/3cSzluUTixP6JBSDKtwwaN4MHYwDgYDVR0PAQH/BAQDAgGm
MBMGA1UdJQQMMAoGCCsGAQUFBwMCMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0OBBYE
FF+p0JcY31pz+mjNZnjv0Gum92vZMB8GA1UdIwQYMBaAFB7P6+i4/pfNjqZgJv/b
dgA7Fe4tMAoGCCqGSM49BAMCA0gAMEUCIQCTT1YWQZaAqfQ2oBxzOkJE2BqLFxhz
3smQlrZ5gCHddwIgcvT7puhYOzAgcvMn9+SZ1JOyZ7edODjshCVCRnuHK2c=
-----END CERTIFICATE-----
Trailing non-PEM content
`)

			// superuser should always have permission to sign; dry-run so we don't actually modify the CSR so the non-superuser can attempt as well
			if _, err := client.CertificatesV1().CertificateSigningRequests().UpdateStatus(context.TODO(), csr, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}}); err != nil {
				t.Errorf("expected no superuser error but got: %v", err)
			}

			_, err := testuserClient.CertificatesV1().CertificateSigningRequests().UpdateStatus(context.TODO(), csr, metav1.UpdateOptions{})
			if err != nil && test.error != err.Error() {
				t.Errorf("expected error %q but got: %v", test.error, err)
			}
			if err == nil && test.error != "" {
				t.Errorf("expected to get an error %q but got none", test.error)
			}
		})
	}
}

func grantUserPermissionToSignFor(t *testing.T, client clientset.Interface, username string, signerNames ...string) {
	resourceName := "signername-" + username
	cr := buildSigningClusterRoleForSigners(resourceName, signerNames...)
	crb := buildClusterRoleBindingForUser(resourceName, username, cr.Name)
	if _, err := client.RbacV1().ClusterRoles().Create(context.TODO(), cr, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create test fixtures: %v", err)
	}
	if _, err := client.RbacV1().ClusterRoleBindings().Create(context.TODO(), crb, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create test fixtures: %v", err)
	}
	signRule := cr.Rules[0]
	statusRule := cr.Rules[1]
	authutil.WaitForNamedAuthorizationUpdate(t, context.TODO(), client.AuthorizationV1(), username, "", signRule.Verbs[0], signRule.ResourceNames[0], schema.GroupResource{Group: signRule.APIGroups[0], Resource: signRule.Resources[0]}, true)
	authutil.WaitForNamedAuthorizationUpdate(t, context.TODO(), client.AuthorizationV1(), username, "", statusRule.Verbs[0], "", schema.GroupResource{Group: statusRule.APIGroups[0], Resource: statusRule.Resources[0]}, true)
}

func buildSigningClusterRoleForSigners(name string, signerNames ...string) *rbacv1.ClusterRole {
	return &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Rules: []rbacv1.PolicyRule{
			// must have permission to 'approve' the 'certificatesigners' named
			// 'signerName' to approve CSRs with the given signerName.
			{
				Verbs:         []string{"sign"},
				APIGroups:     []string{certv1.SchemeGroupVersion.Group},
				Resources:     []string{"signers"},
				ResourceNames: signerNames,
			},
			{
				Verbs:     []string{"update"},
				APIGroups: []string{certv1.SchemeGroupVersion.Group},
				Resources: []string{"certificatesigningrequests/status"},
			},
		},
	}
}
