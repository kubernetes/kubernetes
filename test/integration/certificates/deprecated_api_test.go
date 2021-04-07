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
	"time"

	certv1 "k8s.io/api/certificates/v1"
	certv1beta1 "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	capi "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/controller/certificates/approver"
	"k8s.io/kubernetes/test/integration/framework"
)

// TODO this test can be removed once we get to 1.22 and v1beta1 is no longer allowed
// Verifies that the signerName field defaulting is wired up correctly.
// An exhaustive set of test cases for all permutations of the possible
// defaulting cases is written as a unit tests in the
// `pkg/apis/certificates/...` directory.
// This test cases exists to show that the defaulting function is wired up into
// the apiserver correctly.
func TestCSRSignerNameDefaulting(t *testing.T) {
	strPtr := func(s string) *string { return &s }
	tests := map[string]struct {
		csr                certv1beta1.CertificateSigningRequestSpec
		expectedSignerName string
	}{
		"defaults to legacy-unknown if not recognised": {
			csr: certv1beta1.CertificateSigningRequestSpec{
				Request: pemWithGroup(""),
				Usages:  []certv1beta1.KeyUsage{certv1beta1.UsageKeyEncipherment, certv1beta1.UsageDigitalSignature},
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default signerName if an explicit value is provided": {
			csr: certv1beta1.CertificateSigningRequestSpec{
				Request:    pemWithGroup(""),
				Usages:     []certv1beta1.KeyUsage{certv1beta1.UsageKeyEncipherment, certv1beta1.UsageDigitalSignature},
				SignerName: strPtr("example.com/my-custom-signer"),
			},
			expectedSignerName: "example.com/my-custom-signer",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, s, closeFn := framework.RunAMaster(nil)
			defer closeFn()
			client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			csrClient := client.CertificatesV1beta1().CertificateSigningRequests()
			csr := &certv1beta1.CertificateSigningRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "testcsr"},
				Spec:       test.csr,
			}
			csr, err := csrClient.Create(context.TODO(), csr, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create CSR resource: %v", err)
			}
			if *csr.Spec.SignerName != test.expectedSignerName {
				t.Errorf("expected CSR signerName to be %q but it was %q", test.expectedSignerName, *csr.Spec.SignerName)
			}
		})
	}
}

// TODO this test can be removed once we get to 1.22 and v1beta1 is no longer allowed
// Verifies that the CertificateSubjectRestriction admission controller works as expected.
func TestDeprecatedCertificateSubjectRestrictionPlugin(t *testing.T) {
	tests := map[string]struct {
		signerName string
		group      string
		error      string
	}{
		"should admit a request if signerName is NOT kube-apiserver-client and org is system:masters": {
			signerName: certv1beta1.LegacyUnknownSignerName,
			group:      "system:masters",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// Run an apiserver with the default configuration options.
			s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{""}, framework.SharedEtcd())
			defer s.TearDownFn()
			client := clientset.NewForConfigOrDie(s.ClientConfig)

			// Attempt to create the CSR resource.
			csr := buildDeprecatedTestingCSR("csr", test.signerName, test.group)
			_, err := client.CertificatesV1beta1().CertificateSigningRequests().Create(context.TODO(), csr, metav1.CreateOptions{})
			if err != nil && test.error != err.Error() {
				t.Errorf("expected error %q but got: %v", test.error, err)
			}
			if err == nil && test.error != "" {
				t.Errorf("expected to get an error %q but got none", test.error)
			}
		})
	}
}

func buildDeprecatedTestingCSR(name, signerName, groupName string) *certv1beta1.CertificateSigningRequest {
	return &certv1beta1.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: certv1beta1.CertificateSigningRequestSpec{
			SignerName: &signerName,
			Request:    pemWithGroup(groupName),
		},
	}
}

// TODO this test can be removed once we get to 1.22 and v1beta1 is no longer allowed
// Integration tests that verify the behaviour of the CSR auto-approving controller.
func TestDeprecatedController_AutoApproval(t *testing.T) {
	tests := map[string]struct {
		signerName          string
		request             []byte
		usages              []certv1beta1.KeyUsage
		username            string
		autoApproved        bool
		grantNodeClient     bool
		grantSelfNodeClient bool
	}{
		"should not auto-approve CSR that has kube-apiserver-client-kubelet signerName that does not match requirements": {
			signerName:   certv1.KubeAPIServerClientKubeletSignerName,
			request:      pemWithGroup("system:notnodes"),
			autoApproved: false,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// Run an apiserver with the default configuration options.
			s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{""}, framework.SharedEtcd())
			defer s.TearDownFn()
			client := clientset.NewForConfigOrDie(s.ClientConfig)
			informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(s.ClientConfig, "certificatesigningrequest-informers")), time.Second)

			// Register the controller
			c := approver.NewCSRApprovingController(client, informers.Certificates().V1().CertificateSigningRequests())
			// Start the controller & informers
			stopCh := make(chan struct{})
			defer close(stopCh)
			informers.Start(stopCh)
			go c.Run(1, stopCh)

			// Configure appropriate permissions
			if test.grantNodeClient {
				grantUserNodeClientPermissions(t, client, test.username, false)
			}
			if test.grantSelfNodeClient {
				grantUserNodeClientPermissions(t, client, test.username, true)
			}

			// Use a client that impersonates the test case 'username' to ensure the `spec.username`
			// field on the CSR is set correctly.
			impersonationConfig := restclient.CopyConfig(s.ClientConfig)
			impersonationConfig.Impersonate.UserName = test.username
			impersonationClient, err := clientset.NewForConfig(impersonationConfig)
			if err != nil {
				t.Fatalf("Error in create clientset: %v", err)
			}
			csr := &certv1beta1.CertificateSigningRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "csr",
				},
				Spec: certv1beta1.CertificateSigningRequestSpec{
					Request:    test.request,
					Usages:     test.usages,
					SignerName: &test.signerName,
				},
			}
			_, err = impersonationClient.CertificatesV1beta1().CertificateSigningRequests().Create(context.TODO(), csr, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create testing CSR: %v", err)
			}

			if test.autoApproved {
				if err := waitForCertificateRequestApproved(client, csr.Name); err != nil {
					t.Errorf("failed to wait for CSR to be auto-approved: %v", err)
				}
			} else {
				if err := ensureCertificateRequestNotApproved(client, csr.Name); err != nil {
					t.Errorf("failed to ensure that CSR was not auto-approved: %v", err)
				}
			}
		})
	}
}
