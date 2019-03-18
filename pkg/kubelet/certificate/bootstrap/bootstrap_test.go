/*
Copyright 2016 The Kubernetes Authors.

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

package bootstrap

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	certificates "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/watch"
	certificatesclient "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/keyutil"
)

func TestLoadRESTClientConfig(t *testing.T) {
	testData := []byte(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: ca-a.crt
    server: https://cluster-a.com
  name: cluster-a
- cluster:
    certificate-authority-data: VGVzdA==
    server: https://cluster-b.com
  name: cluster-b
contexts:
- context:
    cluster: cluster-a
    namespace: ns-a
    user: user-a
  name: context-a
- context:
    cluster: cluster-b
    namespace: ns-b
    user: user-b
  name: context-b
current-context: context-b
users:
- name: user-a
  user:
    token: mytoken-a
- name: user-b
  user:
    token: mytoken-b
`)
	f, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	ioutil.WriteFile(f.Name(), testData, os.FileMode(0755))

	config, err := loadRESTClientConfig(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	expectedConfig := &restclient.Config{
		Host: "https://cluster-b.com",
		TLSClientConfig: restclient.TLSClientConfig{
			CAData: []byte(`Test`),
		},
		BearerToken: "mytoken-b",
	}

	if !reflect.DeepEqual(config, expectedConfig) {
		t.Errorf("Unexpected config: %s", diff.ObjectDiff(config, expectedConfig))
	}
}

func TestRequestNodeCertificateNoKeyData(t *testing.T) {
	certData, err := requestNodeCertificate(&fakeClient{}, []byte{}, "fake-node-name")
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
	privateKeyData, err := keyutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("Unable to generate a new private key: %v", err)
	}

	certData, err := requestNodeCertificate(client, privateKeyData, "fake-node-name")
	if err == nil {
		t.Errorf("Got no error, wanted error an error because client.Create failed.")
	}
	if certData != nil {
		t.Errorf("Got cert data, wanted nothing as there should have been an error.")
	}
}

func TestRequestNodeCertificate(t *testing.T) {
	privateKeyData, err := keyutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("Unable to generate a new private key: %v", err)
	}

	certData, err := requestNodeCertificate(&fakeClient{}, privateKeyData, "fake-node-name")
	if err != nil {
		t.Errorf("Got %v, wanted no error.", err)
	}
	if certData == nil {
		t.Errorf("Got nothing, expected a CSR.")
	}
}

type failureType int

const (
	noError failureType = iota
	createError
	certificateSigningRequestDenied
)

type fakeClient struct {
	certificatesclient.CertificateSigningRequestInterface
	watch       *watch.FakeWatcher
	failureType failureType
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
