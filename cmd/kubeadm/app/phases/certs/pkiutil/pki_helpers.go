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

package pkiutil

import (
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"path"
	"time"

	certutil "k8s.io/kubernetes/pkg/util/cert"
)

// TODO: It should be able to generate different types of private keys: RSA, ECDSA or Ed25519
// TODO: See if it makes sense to move this package directly to pkg/util/cert

func NewCertificateAuthority() (*x509.Certificate, *rsa.PrivateKey, error) {
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

	return cert, key, nil
}

func NewServerKeyAndCert(caCert *x509.Certificate, caKey *rsa.PrivateKey, altNames certutil.AltNames) (*x509.Certificate, *rsa.PrivateKey, error) {
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

	return cert, key, nil
}

func NewClientKeyAndCert(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {
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

	return cert, key, nil
}

func WriteCertAndKey(pkiPath string, name string, cert *x509.Certificate, key *rsa.PrivateKey) error {
	certificatePath, privateKeyPath := pathsForCertAndKey(pkiPath, name)

	if key != nil {
		if err := certutil.WriteKey(privateKeyPath, certutil.EncodePrivateKeyPEM(key)); err != nil {
			return fmt.Errorf("unable to write private key to file %q: [%v]", privateKeyPath, err)
		}
	}

	if cert != nil {
		if err := certutil.WriteCert(certificatePath, certutil.EncodeCertPEM(cert)); err != nil {
			return fmt.Errorf("unable to write certificate to file %q: [%v]", certificatePath, err)
		}
	}

	return nil
}

func TryLoadCertAndKeyFromDisk(pkiPath, name string) (*x509.Certificate, *rsa.PrivateKey, error) {
	certificatePath, privateKeyPath := pathsForCertAndKey(pkiPath, name)

	certs, err := certutil.CertsFromFile(certificatePath)
	if err != nil {
		return nil, nil, fmt.Errorf("couldn't load the certificate file %s: %v", certificatePath, err)
	}

	// We are only putting one certificate in the certificate pem file, so it's safe to just pick the first one
	cert := certs[0]

	// Parse the private key from a file
	privKey, err := certutil.PrivateKeyFromFile(privateKeyPath)
	if err != nil {
		return nil, nil, fmt.Errorf("couldn't load the private key file %s: %v", privateKeyPath, err)
	}
	var key *rsa.PrivateKey
	switch k := privKey.(type) {
	case *rsa.PrivateKey:
		key = k
	case *ecdsa.PrivateKey:
		// TODO: Abstract rsa.PrivateKey away and make certutil.NewSignedCert accept a ecdsa.PrivateKey as well
		// After that, we can support generating kubeconfig files from ecdsa private keys as well
		return nil, nil, fmt.Errorf("the private key file %s isn't in RSA format", privateKeyPath)
	default:
		return nil, nil, fmt.Errorf("the private key file %s isn't in RSA format", privateKeyPath)
	}

	// Check so that the certificate is valid now
	now := time.Now()
	if now.Before(cert.NotBefore) || now.After(cert.NotAfter) {
		return nil, nil, fmt.Errorf("the certificate is not valid now")
	}

	return cert, key, nil
}

func pathsForCertAndKey(pkiPath, name string) (string, string) {
	return path.Join(pkiPath, fmt.Sprintf("%s.crt", name)), path.Join(pkiPath, fmt.Sprintf("%s.key", name))
}
