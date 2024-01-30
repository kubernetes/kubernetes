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
	"os"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/test/utils"

	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
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
	certDir, err := os.MkdirTemp("", "test-e2e-server-cert")
	if err != nil {
		framework.Failf("Failed to create a temp dir for cert generation %v", err)
	}
	defer os.RemoveAll(certDir)
	signingKey, err := utils.NewPrivateKey()
	if err != nil {
		framework.Failf("Failed to create CA private key %v", err)
	}
	signingCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "e2e-server-cert-ca"}, signingKey)
	if err != nil {
		framework.Failf("Failed to create CA cert for apiserver %v", err)
	}
	caCertFile, err := os.CreateTemp(certDir, "ca.crt")
	if err != nil {
		framework.Failf("Failed to create a temp file for ca cert generation %v", err)
	}
	defer utiltesting.CloseAndRemove(&testing.T{}, caCertFile)
	if err := os.WriteFile(caCertFile.Name(), utils.EncodeCertPEM(signingCert), 0644); err != nil {
		framework.Failf("Failed to write CA cert %v", err)
	}
	key, err := utils.NewPrivateKey()
	if err != nil {
		framework.Failf("Failed to create private key for %v", err)
	}
	signedCert, err := utils.NewSignedCert(
		&cert.Config{
			CommonName: serviceName + "." + namespaceName + ".svc",
			AltNames:   cert.AltNames{DNSNames: []string{serviceName + "." + namespaceName + ".svc"}},
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		},
		key, signingCert, signingKey,
	)
	if err != nil {
		framework.Failf("Failed to create cert%v", err)
	}
	certFile, err := os.CreateTemp(certDir, "server.crt")
	if err != nil {
		framework.Failf("Failed to create a temp file for cert generation %v", err)
	}
	defer utiltesting.CloseAndRemove(&testing.T{}, certFile)
	keyFile, err := os.CreateTemp(certDir, "server.key")
	if err != nil {
		framework.Failf("Failed to create a temp file for key generation %v", err)
	}
	if err = os.WriteFile(certFile.Name(), utils.EncodeCertPEM(signedCert), 0600); err != nil {
		framework.Failf("Failed to write cert file %v", err)
	}
	privateKeyPEM, err := keyutil.MarshalPrivateKeyToPEM(key)
	if err != nil {
		framework.Failf("Failed to marshal key %v", err)
	}
	if err = os.WriteFile(keyFile.Name(), privateKeyPEM, 0644); err != nil {
		framework.Failf("Failed to write key file %v", err)
	}
	defer utiltesting.CloseAndRemove(&testing.T{}, keyFile)
	return &certContext{
		cert:        utils.EncodeCertPEM(signedCert),
		key:         privateKeyPEM,
		signingCert: utils.EncodeCertPEM(signingCert),
	}
}
