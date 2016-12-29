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

package certs

import (
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"path"

	certutil "k8s.io/kubernetes/pkg/util/cert"
)

func newCertificateAuthority() (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create private key [%v]", err)
	}

	config := certutil.Config{
		CommonName: "kubernetes",
	}

	cert, err := certutil.NewSelfSignedCACert(config, key)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create self-signed certificate [%v]", err)
	}

	return key, cert, nil
}

func newServerKeyAndCert(caCert *x509.Certificate, caKey *rsa.PrivateKey, altNames certutil.AltNames) (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create private key [%v]", err)
	}

	config := certutil.Config{
		CommonName: "kube-apiserver",
		AltNames:   altNames,
	}
	cert, err := certutil.NewSignedCert(config, key, caCert, caKey)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to sign certificate [%v]", err)
	}

	return key, cert, nil
}

func NewClientKeyAndCert(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create private key [%v]", err)
	}

	config := certutil.Config{
		CommonName: "kubernetes-client",
	}
	cert, err := certutil.NewSignedCert(config, key, caCert, caKey)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to sign certificate [%v]", err)
	}

	return key, cert, nil
}

func writeKeysAndCert(pkiPath string, name string, key *rsa.PrivateKey, cert *x509.Certificate) error {
	publicKeyPath, privateKeyPath, certificatePath := pathsKeysCerts(pkiPath, name)

	if key != nil {
		if err := certutil.WriteKey(privateKeyPath, certutil.EncodePrivateKeyPEM(key)); err != nil {
			return fmt.Errorf("unable to write private key file (%q) [%v]", privateKeyPath, err)
		}
		if pubKey, err := certutil.EncodePublicKeyPEM(&key.PublicKey); err == nil {
			if err := certutil.WriteKey(publicKeyPath, pubKey); err != nil {
				return fmt.Errorf("unable to write public key file (%q) [%v]", publicKeyPath, err)
			}
		} else {
			return fmt.Errorf("unable to encode public key to PEM [%v]", err)
		}
	}

	if cert != nil {
		if err := certutil.WriteCert(certificatePath, certutil.EncodeCertPEM(cert)); err != nil {
			return fmt.Errorf("unable to write certificate file (%q) [%v]", certificatePath, err)
		}
	}

	return nil
}

func pathsKeysCerts(pkiPath, name string) (string, string, string) {
	return path.Join(pkiPath, fmt.Sprintf("%s-pub.pem", name)),
		path.Join(pkiPath, fmt.Sprintf("%s-key.pem", name)),
		path.Join(pkiPath, fmt.Sprintf("%s.pem", name))
}
