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

package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"os"
	"testing"
	"time"

	authnv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestAuthnToKAS(t *testing.T) {
	tCtx := ktesting.Init(t)

	frontProxyCA, frontProxyClient, frontProxyKey, err := newTestCAWithClient(
		pkix.Name{
			CommonName: "test-front-proxy-ca",
		},
		pkix.Name{
			CommonName: "test-aggregated-apiserver",
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	modifyOpts := func(setUIDHeaders bool) func(opts *options.ServerRunOptions) {
		return func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024

			// rewrite the client + request header CA certs with our own content
			frontProxyCAFilename := opts.Authentication.RequestHeader.ClientCAFile
			if err := os.WriteFile(frontProxyCAFilename, frontProxyCA, 0644); err != nil {
				t.Fatal(err)
			}

			opts.Authentication.RequestHeader.AllowedNames = append(opts.Authentication.RequestHeader.AllowedNames, "test-aggregated-apiserver")
			if setUIDHeaders {
				opts.Authentication.RequestHeader.UIDHeaders = []string{"X-Remote-Uid"}
			}
		}
	}

	for _, tt := range []struct {
		name   string
		setUID bool
	}{
		{
			name:   "KAS without UID config",
			setUID: false,
		},
		{
			name:   "KAS with UID config",
			setUID: true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
				ModifyServerRunOptions: modifyOpts(tt.setUID),
			})
			defer tearDownFn()

			// Test an aggregated apiserver client (signed by the new front proxy CA) is authorized
			extensionApiserverClient, err := kubernetes.NewForConfig(&rest.Config{
				Host: kubeConfig.Host,
				TLSClientConfig: rest.TLSClientConfig{
					CAData:     kubeConfig.TLSClientConfig.CAData,
					CAFile:     kubeConfig.TLSClientConfig.CAFile,
					ServerName: kubeConfig.TLSClientConfig.ServerName,
					KeyData:    frontProxyKey,
					CertData:   frontProxyClient,
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			selfInfo := &authnv1.SelfSubjectReview{}
			err = extensionApiserverClient.AuthenticationV1().RESTClient().
				Post().
				Resource("selfsubjectreviews").
				VersionedParams(&metav1.CreateOptions{}, scheme.ParameterCodec).
				Body(&authnv1.SelfSubjectReview{}).
				SetHeader("X-Remote-Uid", "test-uid").
				SetHeader("X-Remote-User", "testuser").
				SetHeader("X-Remote-Groups", "group1", "group2").
				Do(tCtx).
				Into(selfInfo)
			if err != nil {
				t.Fatalf("failed to retrieve self-info: %v", err)
			}

			if selfUID := selfInfo.Status.UserInfo.UID; (len(selfUID) != 0) != tt.setUID {
				t.Errorf("UID should be set: %v, but we got %v", tt.setUID, selfUID)
			}
		})
	}

}

func newTestCAWithClient(caSubject pkix.Name, clientSubject pkix.Name) (caPEMBytes, clientCertPEMBytes, clientKeyPEMBytes []byte, err error) {
	caPrivateKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return nil, nil, nil, err
	}

	newCA, err := certutil.NewSelfSignedCACert(certutil.Config{
		CommonName:   caSubject.CommonName,
		Organization: caSubject.Organization,
		NotBefore:    time.Now().Add(-time.Minute),
	}, caPrivateKey)

	if err != nil {
		return nil, nil, nil, err
	}

	clientCertPrivateKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return nil, nil, nil, err
	}

	clientCertPrivateKeyPEM, err := keyutil.MarshalPrivateKeyToPEM(clientCertPrivateKey)
	if err != nil {
		return nil, nil, nil, err
	}

	clientCert, err := testutils.NewSignedCert(&certutil.Config{
		CommonName:   clientSubject.CommonName,
		Organization: clientSubject.Organization,
		NotBefore:    time.Now().Add(-time.Minute),
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}, clientCertPrivateKey, newCA, caPrivateKey)

	if err != nil {
		return nil, nil, nil, err
	}

	caPEMBytes = testutils.EncodeCertPEM(newCA)
	clientCertPEMBytes = testutils.EncodeCertPEM(clientCert)
	return caPEMBytes,
		clientCertPEMBytes,
		clientCertPrivateKeyPEM,
		nil
}
