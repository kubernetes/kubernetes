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
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"testing"

	capi "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	certclientset "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	restclient "k8s.io/client-go/rest"

	"k8s.io/kubernetes/test/integration/framework"
)

// Verifies that the 'spec.signerName' field can be correctly used as a field selector on LIST requests
func TestCSRSignerNameFieldSelector(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	csrClient := client.CertificatesV1beta1().CertificateSigningRequests()
	csr1 := createTestingCSR(t, csrClient, "csr-1", "example.com/signer-name-1", "")
	csr2 := createTestingCSR(t, csrClient, "csr-2", "example.com/signer-name-2", "")
	// csr3 has the same signerName as csr2 so we can ensure multiple items are returned when running a filtered
	// LIST call.
	csr3 := createTestingCSR(t, csrClient, "csr-3", "example.com/signer-name-2", "")

	signerOneList, err := client.CertificatesV1beta1().CertificateSigningRequests().List(context.TODO(), metav1.ListOptions{FieldSelector: "spec.signerName=example.com/signer-name-1"})
	if err != nil {
		t.Errorf("unable to list CSRs with spec.signerName=example.com/signer-name-1")
		return
	}
	if len(signerOneList.Items) != 1 {
		t.Errorf("expected one CSR to be returned but got %d", len(signerOneList.Items))
	} else if signerOneList.Items[0].Name != csr1.Name {
		t.Errorf("expected CSR named 'csr-1' to be returned but got %q", signerOneList.Items[0].Name)
	}

	signerTwoList, err := client.CertificatesV1beta1().CertificateSigningRequests().List(context.TODO(), metav1.ListOptions{FieldSelector: "spec.signerName=example.com/signer-name-2"})
	if err != nil {
		t.Errorf("unable to list CSRs with spec.signerName=example.com/signer-name-2")
		return
	}
	if len(signerTwoList.Items) != 2 {
		t.Errorf("expected one CSR to be returned but got %d", len(signerTwoList.Items))
	} else if signerTwoList.Items[0].Name != csr2.Name {
		t.Errorf("expected CSR named 'csr-2' to be returned but got %q", signerTwoList.Items[0].Name)
	} else if signerTwoList.Items[1].Name != csr3.Name {
		t.Errorf("expected CSR named 'csr-3' to be returned but got %q", signerTwoList.Items[1].Name)
	}
}

func createTestingCSR(t *testing.T, certClient certclientset.CertificateSigningRequestInterface, name, signerName, groupName string) *capi.CertificateSigningRequest {
	csr, err := certClient.Create(context.TODO(), buildTestingCSR(name, signerName, groupName), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create testing CSR: %v", err)
	}
	return csr
}

func buildTestingCSR(name, signerName, groupName string) *capi.CertificateSigningRequest {
	return &capi.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: capi.CertificateSigningRequestSpec{
			SignerName: &signerName,
			Request:    pemWithGroup(groupName),
		},
	}
}

func pemWithGroup(group string) []byte {
	template := &x509.CertificateRequest{
		Subject: pkix.Name{
			Organization: []string{group},
		},
	}
	return pemWithTemplate(template)
}

func pemWithTemplate(template *x509.CertificateRequest) []byte {
	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		panic(err)
	}

	csrDER, err := x509.CreateCertificateRequest(rand.Reader, template, key)
	if err != nil {
		panic(err)
	}

	csrPemBlock := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csrDER,
	}

	p := pem.EncodeToMemory(csrPemBlock)
	if p == nil {
		panic("invalid pem block")
	}

	return p
}
