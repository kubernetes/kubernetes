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

package certificates

import (
	"bytes"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io/ioutil"
	"os"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/cert/triple"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller"
)

type testController struct {
	*CertificateController
	certFile        string
	keyFile         string
	csrStore        cache.Store
	informerFactory informers.SharedInformerFactory
	approver        *fakeAutoApprover
}

func alwaysReady() bool { return true }

func newController(csrs ...runtime.Object) (*testController, error) {
	client := fake.NewSimpleClientset(csrs...)
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	certFile, keyFile, err := createTestCertFiles()
	if err != nil {
		return nil, err
	}

	signer, err := NewCFSSLSigner(certFile, keyFile)
	if err != nil {
		return nil, err
	}

	approver := &fakeAutoApprover{make(chan *certificates.CertificateSigningRequest, 1)}
	controller, err := NewCertificateController(
		client,
		informerFactory.Certificates().V1beta1().CertificateSigningRequests(),
		signer,
		approver,
	)
	if err != nil {
		return nil, err
	}
	controller.csrsSynced = alwaysReady

	return &testController{
		controller,
		certFile,
		keyFile,
		informerFactory.Certificates().V1beta1().CertificateSigningRequests().Informer().GetStore(),
		informerFactory,
		approver,
	}, nil
}

func (c *testController) cleanup() {
	os.Remove(c.certFile)
	os.Remove(c.keyFile)
}

func createTestCertFiles() (string, string, error) {
	keyPair, err := triple.NewCA("test-ca")
	if err != nil {
		return "", "", err
	}

	// Generate cert
	certBuffer := bytes.Buffer{}
	if err := pem.Encode(&certBuffer, &pem.Block{Type: "CERTIFICATE", Bytes: keyPair.Cert.Raw}); err != nil {
		return "", "", err
	}

	// Generate key
	keyBuffer := bytes.Buffer{}
	if err := pem.Encode(&keyBuffer, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(keyPair.Key)}); err != nil {
		return "", "", err
	}

	dir, err := ioutil.TempDir("", "")
	if err != nil {
		return "", "", err
	}

	certFile, err := ioutil.TempFile(dir, "cert")
	if err != nil {
		return "", "", err
	}

	keyFile, err := ioutil.TempFile(dir, "key")
	if err != nil {
		return "", "", err
	}

	_, err = certFile.Write(certBuffer.Bytes())
	if err != nil {
		return "", "", err
	}
	certFile.Close()

	_, err = keyFile.Write(keyBuffer.Bytes())
	if err != nil {
		return "", "", err
	}
	keyFile.Close()

	return certFile.Name(), keyFile.Name(), nil
}

type fakeAutoApprover struct {
	csr chan *certificates.CertificateSigningRequest
}

func (f *fakeAutoApprover) AutoApprove(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
		Type:    certificates.CertificateApproved,
		Reason:  "test reason",
		Message: "test message",
	})
	f.csr <- csr
	return csr, nil
}

// TODO flesh this out to cover things like not being able to find the csr in the cache, not
// auto-approving, etc.
func TestCertificateController(t *testing.T) {
	csrKey, err := cert.NewPrivateKey()
	if err != nil {
		t.Fatalf("error creating private key for csr: %v", err)
	}

	subject := &pkix.Name{
		Organization: []string{"test org"},
		CommonName:   "test cn",
	}
	csrBytes, err := cert.MakeCSR(csrKey, subject, nil, nil)
	if err != nil {
		t.Fatalf("error creating csr: %v", err)
	}

	csr := &certificates.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-csr",
		},
		Spec: certificates.CertificateSigningRequestSpec{
			Request: csrBytes,
			Usages: []certificates.KeyUsage{
				certificates.UsageDigitalSignature,
				certificates.UsageKeyEncipherment,
				certificates.UsageClientAuth,
			},
		},
	}

	controller, err := newController(csr)
	if err != nil {
		t.Fatalf("error creating controller: %v", err)
	}
	defer controller.cleanup()

	received := make(chan struct{})

	controllerSyncHandler := controller.syncHandler
	controller.syncHandler = func(key string) error {
		defer close(received)
		return controllerSyncHandler(key)
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	go controller.Run(1, stopCh)
	go controller.informerFactory.Start(stopCh)

	select {
	case <-received:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out")
	}

	csr = <-controller.approver.csr

	if e, a := 1, len(csr.Status.Conditions); e != a {
		t.Fatalf("expected %d status condition, got %d", e, a)
	}
	if e, a := certificates.CertificateApproved, csr.Status.Conditions[0].Type; e != a {
		t.Errorf("type: expected %v, got %v", e, a)
	}
	if e, a := "test reason", csr.Status.Conditions[0].Reason; e != a {
		t.Errorf("reason: expected %v, got %v", e, a)
	}
	if e, a := "test message", csr.Status.Conditions[0].Message; e != a {
		t.Errorf("message: expected %v, got %v", e, a)
	}
}
