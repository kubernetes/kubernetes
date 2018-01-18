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

package apimachinery

import (
	"crypto/x509"
	"io/ioutil"
	"os"

	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/test/e2e/framework"
)

type certContext struct {
	cert        []byte
	key         []byte
	signingCert []byte
}

// Setup the server cert. For example, user apiservers and admission webhooks
// can use the cert to prove their identify to the kube-apiserver
func setupServerCert(namespaceName, serviceName string) *certContext {
	certDir, err := ioutil.TempDir("", "test-e2e-server-cert")
	if err != nil {
		framework.Failf("Failed to create a temp dir for cert generation %v", err)
	}
	defer os.RemoveAll(certDir)
	signingKey, err := cert.NewPrivateKey()
	if err != nil {
		framework.Failf("Failed to create CA private key %v", err)
	}
	signingCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "e2e-server-cert-ca"}, signingKey)
	if err != nil {
		framework.Failf("Failed to create CA cert for apiserver %v", err)
	}
	caCertFile, err := ioutil.TempFile(certDir, "ca.crt")
	if err != nil {
		framework.Failf("Failed to create a temp file for ca cert generation %v", err)
	}
	if err := ioutil.WriteFile(caCertFile.Name(), cert.EncodeCertPEM(signingCert), 0644); err != nil {
		framework.Failf("Failed to write CA cert %v", err)
	}
	key, err := cert.NewPrivateKey()
	if err != nil {
		framework.Failf("Failed to create private key for %v", err)
	}
	signedCert, err := cert.NewSignedCert(
		cert.Config{
			CommonName: serviceName + "." + namespaceName + ".svc",
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		},
		key, signingCert, signingKey,
	)
	if err != nil {
		framework.Failf("Failed to create cert%v", err)
	}
	certFile, err := ioutil.TempFile(certDir, "server.crt")
	if err != nil {
		framework.Failf("Failed to create a temp file for cert generation %v", err)
	}
	keyFile, err := ioutil.TempFile(certDir, "server.key")
	if err != nil {
		framework.Failf("Failed to create a temp file for key generation %v", err)
	}
	if err = ioutil.WriteFile(certFile.Name(), cert.EncodeCertPEM(signedCert), 0600); err != nil {
		framework.Failf("Failed to write cert file %v", err)
	}
	if err = ioutil.WriteFile(keyFile.Name(), cert.EncodePrivateKeyPEM(key), 0644); err != nil {
		framework.Failf("Failed to write key file %v", err)
	}
	return &certContext{
		cert:        cert.EncodeCertPEM(signedCert),
		key:         cert.EncodePrivateKeyPEM(key),
		signingCert: cert.EncodeCertPEM(signingCert),
	}
}
