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

	certv1beta1 "k8s.io/api/certificates/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
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

			// Grant 'test-user' permission to sign CertificateSigningRequests with the specified signerName.
			const username = "test-user"
			grantUserPermissionToSignFor(t, client, username, test.allowedSignerName)
			// Create a CSR to attempt to sign.
			csr := createTestingCSR(t, client.CertificatesV1beta1().CertificateSigningRequests(), "csr", test.signerName, "")

			// Create a second client, impersonating the 'test-user' for us to test with.
			testuserConfig := restclient.CopyConfig(s.ClientConfig)
			testuserConfig.Impersonate = restclient.ImpersonationConfig{UserName: username}
			testuserClient := clientset.NewForConfigOrDie(testuserConfig)

			// Attempt to 'sign' the certificate.
			csr.Status.Certificate = []byte("dummy data")
			_, err := testuserClient.CertificatesV1beta1().CertificateSigningRequests().UpdateStatus(context.TODO(), csr, metav1.UpdateOptions{})
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
	waitForNamedAuthorizationUpdate(t, client.AuthorizationV1(), username, "", signRule.Verbs[0], signRule.ResourceNames[0], schema.GroupResource{Group: signRule.APIGroups[0], Resource: signRule.Resources[0]}, true)
	waitForNamedAuthorizationUpdate(t, client.AuthorizationV1(), username, "", statusRule.Verbs[0], "", schema.GroupResource{Group: statusRule.APIGroups[0], Resource: statusRule.Resources[0]}, true)
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
				APIGroups:     []string{certv1beta1.SchemeGroupVersion.Group},
				Resources:     []string{"signers"},
				ResourceNames: signerNames,
			},
			{
				Verbs:     []string{"update"},
				APIGroups: []string{certv1beta1.SchemeGroupVersion.Group},
				Resources: []string{"certificatesigningrequests/status"},
			},
		},
	}
}
