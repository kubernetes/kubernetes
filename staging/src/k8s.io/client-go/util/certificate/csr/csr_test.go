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

package csr

import (
	"fmt"
	"testing"

	certificates "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	certificatesclient "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	certutil "k8s.io/client-go/util/cert"
)

func TestRequestNodeCertificateNoKeyData(t *testing.T) {
	certData, err := RequestNodeCertificate(&fakeClient{}, []byte{}, "fake-node-name")
	if err == nil {
		t.Errorf("Got no error, wanted error an error because there was an empty private key passed in.")
	}
	if certData != nil {
		t.Errorf("Got cert data, wanted nothing as there should have been an error.")
	}
}

func TestRequestNodeCertificateErrorCreatingCSR(t *testing.T) {
	client := &fakeClient{
		failureType: createError,
	}
	privateKeyData, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("Unable to generate a new private key: %v", err)
	}

	certData, err := RequestNodeCertificate(client, privateKeyData, "fake-node-name")
	if err == nil {
		t.Errorf("Got no error, wanted error an error because client.Create failed.")
	}
	if certData != nil {
		t.Errorf("Got cert data, wanted nothing as there should have been an error.")
	}
}

func TestRequestNodeCertificate(t *testing.T) {
	privateKeyData, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("Unable to generate a new private key: %v", err)
	}

	certData, err := RequestNodeCertificate(&fakeClient{}, privateKeyData, "fake-node-name")
	if err != nil {
		t.Errorf("Got %v, wanted no error.", err)
	}
	if certData == nil {
		t.Errorf("Got nothing, expected a CSR.")
	}
}

type FailureType int

const (
	noError FailureType = iota
	createError
	certificateSigningRequestDenied
)

type fakeClient struct {
	certificatesclient.CertificateSigningRequestInterface
	watch       *watch.FakeWatcher
	failureType FailureType
}

func (c *fakeClient) Create(*certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	if c.failureType == createError {
		return nil, fmt.Errorf("fakeClient failed creating request")
	}
	csr := certificates.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			UID:  "fake-uid",
			Name: "fake-certificate-signing-request-name",
		},
	}
	return &csr, nil
}

func (c *fakeClient) List(opts metav1.ListOptions) (*certificates.CertificateSigningRequestList, error) {
	return &certificates.CertificateSigningRequestList{}, nil
}

func (c *fakeClient) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	c.watch = watch.NewFakeWithChanSize(1, false)
	c.watch.Add(c.generateCSR())
	c.watch.Stop()
	return c.watch, nil
}

func (c *fakeClient) generateCSR() *certificates.CertificateSigningRequest {
	var condition certificates.CertificateSigningRequestCondition
	if c.failureType == certificateSigningRequestDenied {
		condition = certificates.CertificateSigningRequestCondition{
			Type: certificates.CertificateDenied,
		}
	} else {
		condition = certificates.CertificateSigningRequestCondition{
			Type: certificates.CertificateApproved,
		}
	}

	csr := certificates.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			UID: "fake-uid",
		},
		Status: certificates.CertificateSigningRequestStatus{
			Conditions: []certificates.CertificateSigningRequestCondition{
				condition,
			},
			Certificate: []byte{},
		},
	}
	return &csr
}
