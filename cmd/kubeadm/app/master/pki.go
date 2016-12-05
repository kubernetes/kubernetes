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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	ipallocator "k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
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

func newServerKeyAndCert(cfg *kubeadmapi.MasterConfiguration, caCert *x509.Certificate, caKey *rsa.PrivateKey, altNames certutil.AltNames) (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create private key [%v]", err)
	}

	internalAPIServerFQDN := []string{
		"kubernetes",
		"kubernetes.default",
		"kubernetes.default.svc",
		fmt.Sprintf("kubernetes.default.svc.%s", cfg.Networking.DNSDomain),
	}

	_, n, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return nil, nil, fmt.Errorf("error parsing CIDR %q: %v", cfg.Networking.ServiceSubnet, err)
	}
	internalAPIServerVirtualIP, err := ipallocator.GetIndexedIP(n, 1)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to allocate IP address for the API server from the given CIDR (%q) [%v]", &cfg.Networking.ServiceSubnet, err)
	}

	altNames.IPs = append(altNames.IPs, internalAPIServerVirtualIP)
	altNames.DNSNames = append(altNames.DNSNames, internalAPIServerFQDN...)

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

func newClientKeyAndCert(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*rsa.PrivateKey, *x509.Certificate, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create private key [%v]", err)
	}

	config := certutil.Config{
		CommonName: "kubernetes-admin",
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

func newServiceAccountKey() (*rsa.PrivateKey, error) {
	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, err
	}
	return key, nil
}

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// It first generates a self-signed CA certificate, a server certificate (signed by the CA) and a key for
// signing service account tokens. It returns CA key and certificate, which is convenient for use with
// client config funcs.
func CreatePKIAssets(cfg *kubeadmapi.MasterConfiguration) (*rsa.PrivateKey, *x509.Certificate, error) {
	var (
		err      error
		altNames certutil.AltNames
	)

	for _, a := range cfg.API.AdvertiseAddresses {
		if ip := net.ParseIP(a); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else {
			return nil, nil, fmt.Errorf("could not parse ip %q", a)
		}
	}
	altNames.DNSNames = append(altNames.DNSNames, cfg.API.ExternalDNSNames...)

	pkiPath := path.Join(kubeadmapi.GlobalEnvParams.HostPKIPath)

	caKey, caCert, err := newCertificateAuthority()
	if err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while creating CA keys and certificate - %v", err)
	}

	if err := writeKeysAndCert(pkiPath, "ca", caKey, caCert); err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while saving CA keys and certificate - %v", err)
	}
	fmt.Printf("<master/pki> generated Certificate Authority key and certificate:\n%s\n", certutil.FormatCert(caCert))
	pub, prv, cert := pathsKeysCerts(pkiPath, "ca")
	fmt.Printf("Public: %s\nPrivate: %s\nCert: %s\n", pub, prv, cert)

	apiKey, apiCert, err := newServerKeyAndCert(cfg, caCert, caKey, altNames)
	if err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while creating API server keys and certificate - %v", err)
	}

	if err := writeKeysAndCert(pkiPath, "apiserver", apiKey, apiCert); err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while saving API server keys and certificate - %v", err)
	}
	fmt.Printf("<master/pki> generated API Server key and certificate:\n%s\n", certutil.FormatCert(apiCert))
	pub, prv, cert = pathsKeysCerts(pkiPath, "apiserver")
	fmt.Printf("Public: %s\nPrivate: %s\nCert: %s\n", pub, prv, cert)

	saKey, err := newServiceAccountKey()
	if err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while creating service account signing keys [%v]", err)
	}
	if err := writeKeysAndCert(pkiPath, "sa", saKey, nil); err != nil {
		return nil, nil, fmt.Errorf("<master/pki> failure while saving service account signing keys - %v", err)
	}
	fmt.Printf("<master/pki> generated Service Account Signing keys:\n")
	pub, prv, _ = pathsKeysCerts(pkiPath, "sa")
	fmt.Printf("Public: %s\nPrivate: %s\n", pub, prv)

	fmt.Printf("<master/pki> created keys and certificates in %q\n", pkiPath)
	return caKey, caCert, nil
}
