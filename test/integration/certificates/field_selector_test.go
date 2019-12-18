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
	csr1 := createTestingCSR(t, csrClient, "csr-1", "example.com/signer-name-1")
	csr2 := createTestingCSR(t, csrClient, "csr-2", "example.com/signer-name-2")
	// csr3 has the same signerName as csr2 so we can ensure multiple items are returned when running a filtered
	// LIST call.
	csr3 := createTestingCSR(t, csrClient, "csr-3", "example.com/signer-name-2")

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

func createTestingCSR(t *testing.T, certClient certclientset.CertificateSigningRequestInterface, name, signerName string) *capi.CertificateSigningRequest {
	csr, err := certClient.Create(context.TODO(), buildTestingCSR(name, signerName), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create testing CSR: %v", err)
	}
	return csr
}

func buildTestingCSR(name, signerName string) *capi.CertificateSigningRequest {
	return &capi.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: capi.CertificateSigningRequestSpec{
			SignerName: &signerName,
			Request:    testCSRPEM,
		},
	}
}

var (
	// The contents of this CSR do not matter, and it is only used to allow the
	// CSR resource submitted during integration tests to pass through
	// validation.
	testCSRPEM = []byte(`-----BEGIN CERTIFICATE REQUEST-----
MIICrzCCAZcCAQAwajENMAsGA1UECAwESE9OSzENMAsGA1UEBwwESE9OSzENMAsG
A1UECgwESE9OSzENMAsGA1UECwwESE9OSzENMAsGA1UEAwwESE9OSzEdMBsGCSqG
SIb3DQEJARYOaG9ua0Bob25rLmhvbmswggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw
ggEKAoIBAQDhw6t5C0Vtzzl4jQVMM9S2epAKOyKZCXRYC50sG8UFectfSALJHPUY
rv3LNfUTSkqg+EJO+5an1PeQS+GK94DUiJ2cUR2hBiTfXenyDAm2fGSDIqLQ/YcZ
fprwlqMu3YfpMH1KyyNORoOgWgsyWP0rBIRoWEFcFNaBu7BazaJHQIYNpcyRkHJC
610It4MV5dUqNFAfYqmxqlkMa4lR0U4f8cCA3J+lajNOMz/GkPotBINU+xX4bVob
Q+ghAatgiZnEvC6pe0LqG788SHaIu7hArSK8ZG7+HcqCwISFLJiA8+A6HE24PhQC
69pGqHePAFO4a09c5/MTPfBfohYkEGX7AgMBAAGgADANBgkqhkiG9w0BAQsFAAOC
AQEAwg/7CWhWZICusSKEeIHJE+rgeSySAgL0S05KJKtwjHK1zf2B8Az4F2pe0aCe
r+mqNyFutmaLOXmNH7H1BJuw0wXeEg8wlT3nknRTJ4EWYf4G0H1dOICk/tB4Mgl1
qgmMcP37QQRCMit5VY9BOKfXo+AHCH9rwmX91mXwzyejY/wO6Y3R6Y+GvMKA259F
zRt2J8VJkeeXOE/H93putfT1KcmayTwO0gTzPFd7ZZzLSVMnpirxCUujkduxy8DK
dDcZdaTZofztqa5ej1gzptxU6fBfVvl3Wevc30yDH5Dum0aiohJbijncgIR6SQx5
6nuYWH340f/Ivm5b1gyEqb12ag==
-----END CERTIFICATE REQUEST-----`)
)
