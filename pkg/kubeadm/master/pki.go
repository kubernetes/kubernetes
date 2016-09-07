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

package master

import (
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"net"
	"path"

	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

/*
func errorf(f string, err error, vargs ...string) error {
	return fmt.Errorf("<master/pki> %s [%s]", fmt.Sprintf(f, v...), err)
}
*/

func newCertificateAuthority() (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create private key [%s]", err)
	}

	config := certutil.CertConfig{
		CommonName: "kubernetes",
	}

	cert, err := certutil.NewSelfSignedCACert(config, key)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create self-singed certificate [%s]", err)
	}

	return key, cert, nil
}

func newServerKeyAndCert(caCert *x509.Certificate, caKey *rsa.PrivateKey, altNames certutil.AltNames) (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unabel to create private key [%s]", err)
	}
	// TODO these are all hardcoded for now, but we need to figure out what shall we do here exactly
	altNames.IPs = append(altNames.IPs, net.ParseIP("10.3.0.1"), net.ParseIP("10.16.0.1"), net.ParseIP("100.64.0.1"))
	altNames.DNSNames = append(altNames.DNSNames,
		"kubernetes",
		"kubernetes.default",
		"kubernetes.default.svc",
		"kubernetes.default.svc.cluster.local",
	)

	config := certutil.CertConfig{
		CommonName: "kube-apiserver",
		AltNames:   altNames,
	}
	cert, err := certutil.NewSignedCert(config, key, caCert, caKey)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to sing certificate [%s]", err)
	}

	return key, cert, nil
}

func newClientKeyAndCert(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create private key [%s]", err)
	}

	config := certutil.CertConfig{
		CommonName: "kubernetes-admin",
	}
	cert, err := certutil.NewSignedCert(config, key, caCert, caKey)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to sign certificate [%s]", err)
	}

	return key, cert, nil
}

func writeKeysAndCert(pkiPath string, name string, key *rsa.PrivateKey, cert *x509.Certificate) error {
	var (
		publicKeyPath   = path.Join(pkiPath, fmt.Sprintf("%s-pub.pem", name))
		privateKeyPath  = path.Join(pkiPath, fmt.Sprintf("%s-key.pem", name))
		certificatePath = path.Join(pkiPath, fmt.Sprintf("%s.pem", name))
	)

	if key != nil {
		if err := certutil.WriteKey(privateKeyPath, certutil.EncodePrivateKeyPEM(key)); err != nil {
			return fmt.Errorf("unable to write private key file (%q) [%s]", privateKeyPath, err)
		}
		if pubKey, err := certutil.EncodePublicKeyPEM(&key.PublicKey); err == nil {
			if err := certutil.WriteKey(publicKeyPath, pubKey); err != nil {
				return fmt.Errorf("unable to write public key file (%q) [%s]", publicKeyPath, err)
			}
		} else {
			return fmt.Errorf("unable to encode public key to PEM [%s]", err)
		}
	}

	if cert != nil {
		if err := certutil.WriteCert(certificatePath, certutil.EncodeCertPEM(cert)); err != nil {
			return fmt.Errorf("unable to write certificate file (%q) [%s]", err)
		}
	}

	return nil
}

func newServiceAccountKey() (*rsa.PrivateKey, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, err
	}
	return key, nil
}

func CreatePKIAssets(params *kubeadmapi.BootstrapParams) (*rsa.PrivateKey, *x509.Certificate, error) {
	var (
		err      error
		altNames certutil.AltNames // TODO actual SANs
	)

	if params.Discovery.ListenIP != nil {
		altNames.IPs = append(altNames.IPs, params.Discovery.ListenIP)
	}

	if params.Discovery.ApiServerDNSName != "" {
		altNames.DNSNames = append(altNames.DNSNames, params.Discovery.ApiServerDNSName)
	}

	pkiPath := path.Join(params.EnvParams["host_pki_path"])

	caKey, caCert, err := newCertificateAuthority()
	if err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while creating CA keys and certificate - %s", err)
	}

	if err := writeKeysAndCert(pkiPath, "ca", caKey, caCert); err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while saving CA keys and certificate - %s", err)
	}

	apiKey, apiCert, err := newServerKeyAndCert(caCert, caKey, altNames)
	if err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while creating API server keys and certificate - %s", err)
	}

	if err := writeKeysAndCert(pkiPath, "apiserver", apiKey, apiCert); err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while saving API server keys and certificate - %s", err)
	}

	saKey, err := newServiceAccountKey()
	if err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while creating service account signing keys [%s]", err)
	}

	if err := writeKeysAndCert(pkiPath, "sa", saKey, nil); err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while saving service account singing keys - %s", err)
	}

	// TODO print a summary of SANs used and checksums (signatures) of each of the certiicates
	fmt.Printf("<master/pki> created keys and certificates in %q\n", params.EnvParams["host_pki_path"])
	return caKey, caCert, nil
}
