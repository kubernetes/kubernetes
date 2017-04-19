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
	"net"
	"os"

	"encoding/base64"
	"encoding/pem"
	setutil "k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// TODO: Integration test cases
// no files exist => create all four files
// valid ca.{crt,key} exists => create apiserver.{crt,key}
// valid ca.{crt,key} and apiserver.{crt,key} exists => do nothing
// invalid ca.{crt,key} exists => error
// only one of the .crt or .key file exists => error

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// It generates a self-signed CA certificate and a server certificate (signed by the CA)
func CreatePKIAssets(cfg *kubeadmapi.MasterConfiguration) error {
	if cfg.MasterCertificates != nil {
		return saveConfiguredPKIAssets(cfg)
	}
	pkiDir := cfg.CertificatesDir
	hostname, err := os.Hostname()
	if err != nil {
		return fmt.Errorf("couldn't get the hostname: %v", err)
	}

	_, svcSubnet, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return fmt.Errorf("error parsing CIDR %q: %v", cfg.Networking.ServiceSubnet, err)
	}

	// Build the list of SANs
	altNames := getAltNames(cfg.APIServerCertSANs, hostname, cfg.Networking.DNSDomain, svcSubnet)
	// Append the address the API Server is advertising
	altNames.IPs = append(altNames.IPs, net.ParseIP(cfg.API.AdvertiseAddress))

	if cfg.PublicAddress != cfg.API.AdvertiseAddress {
		if ip := net.ParseIP(cfg.PublicAddress); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else if len(validation.IsDNS1123Subdomain(cfg.PublicAddress)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, cfg.PublicAddress)
		}
	}

	var caCert *x509.Certificate
	var caKey *rsa.PrivateKey
	// If at least one of them exists, we should try to load them
	// In the case that only one exists, there will most likely be an error anyway
	if pkiutil.CertOrKeyExist(pkiDir, kubeadmconstants.CACertAndKeyBaseName) {
		// Try to load ca.crt and ca.key from the PKI directory
		caCert, caKey, err = pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.CACertAndKeyBaseName)
		if err != nil || caCert == nil || caKey == nil {
			return fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		// The certificate and key could be loaded, but the certificate is not a CA
		if !caCert.IsCA {
			return fmt.Errorf("certificate and key could be loaded but the certificate is not a CA")
		}

		fmt.Println("[certificates] Using the existing CA certificate and key.")
	} else {
		// The certificate and the key did NOT exist, let's generate them now
		caCert, caKey, err = pkiutil.NewCertificateAuthority()
		if err != nil {
			return fmt.Errorf("failure while generating CA certificate and key [%v]", err)
		}

		if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.CACertAndKeyBaseName, caCert, caKey); err != nil {
			return fmt.Errorf("failure while saving CA certificate and key [%v]", err)
		}
		fmt.Println("[certificates] Generated CA certificate and key.")
	}

	// If at least one of them exists, we should try to load them
	// In the case that only one exists, there will most likely be an error anyway
	if pkiutil.CertOrKeyExist(pkiDir, kubeadmconstants.APIServerCertAndKeyBaseName) {
		// Try to load apiserver.crt and apiserver.key from the PKI directory
		apiCert, apiKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.APIServerCertAndKeyBaseName)
		if err != nil || apiCert == nil || apiKey == nil {
			return fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		fmt.Println("[certificates] Using the existing API Server certificate and key.")
	} else {
		// The certificate and the key did NOT exist, let's generate them now
		// TODO: Add a test case to verify that this cert has the x509.ExtKeyUsageServerAuth flag
		config := certutil.Config{
			CommonName: "kube-apiserver",
			AltNames:   altNames,
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		}
		apiCert, apiKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
		if err != nil {
			return fmt.Errorf("failure while creating API server key and certificate [%v]", err)
		}

		if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.APIServerCertAndKeyBaseName, apiCert, apiKey); err != nil {
			return fmt.Errorf("failure while saving API server certificate and key [%v]", err)
		}
		fmt.Println("[certificates] Generated API server certificate and key.")
		fmt.Printf("[certificates] API Server serving cert is signed for DNS names %v and IPs %v\n", altNames.DNSNames, altNames.IPs)
	}

	// If at least one of them exists, we should try to load them
	// In the case that only one exists, there will most likely be an error anyway
	if pkiutil.CertOrKeyExist(pkiDir, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName) {
		// Try to load apiserver-kubelet-client.crt and apiserver-kubelet-client.key from the PKI directory
		apiCert, apiKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName)
		if err != nil || apiCert == nil || apiKey == nil {
			return fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		fmt.Println("[certificates] Using the existing API Server kubelet client certificate and key.")
	} else {
		// The certificate and the key did NOT exist, let's generate them now
		// TODO: Add a test case to verify that this cert has the x509.ExtKeyUsageClientAuth flag
		config := certutil.Config{
			CommonName:   "kube-apiserver-kubelet-client",
			Organization: []string{kubeadmconstants.MastersGroup},
			Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}
		apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
		if err != nil {
			return fmt.Errorf("failure while creating API server kubelet client key and certificate [%v]", err)
		}

		if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName, apiClientCert, apiClientKey); err != nil {
			return fmt.Errorf("failure while saving API server kubelet client certificate and key [%v]", err)
		}
		fmt.Println("[certificates] Generated API server kubelet client certificate and key.")
	}

	// If the key exists, we should try to load it
	if pkiutil.CertOrKeyExist(pkiDir, kubeadmconstants.ServiceAccountKeyBaseName) {
		// Try to load sa.key from the PKI directory
		_, err := pkiutil.TryLoadKeyFromDisk(pkiDir, kubeadmconstants.ServiceAccountKeyBaseName)
		if err != nil {
			return fmt.Errorf("certificate and/or key existed but they could not be loaded properly [%v]", err)
		}

		fmt.Println("[certificates] Using the existing service account token signing key.")
	} else {
		// The key does NOT exist, let's generate it now
		saTokenSigningKey, err := certutil.NewPrivateKey()
		if err != nil {
			return fmt.Errorf("failure while creating service account token signing key [%v]", err)
		}

		if err = pkiutil.WriteKey(pkiDir, kubeadmconstants.ServiceAccountKeyBaseName, saTokenSigningKey); err != nil {
			return fmt.Errorf("failure while saving service account token signing key [%v]", err)
		}

		if err = pkiutil.WritePublicKey(pkiDir, kubeadmconstants.ServiceAccountKeyBaseName, &saTokenSigningKey.PublicKey); err != nil {
			return fmt.Errorf("failure while saving service account token signing public key [%v]", err)
		}
		fmt.Println("[certificates] Generated service account token signing key and public key.")
	}

	// front proxy CA and client certs are used to secure a front proxy authenticator which is used to assert identity
	// without the client cert, you cannot make use of the front proxy and the kube-aggregator uses this connection
	// so we generate and enable it unconditionally
	// This is a separte CA, so that front proxy identities cannot hit the API and normal client certs cannot be used
	// as front proxies.
	var frontProxyCACert *x509.Certificate
	var frontProxyCAKey *rsa.PrivateKey
	// If at least one of them exists, we should try to load them
	// In the case that only one exists, there will most likely be an error anyway
	if pkiutil.CertOrKeyExist(pkiDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName) {
		// Try to load front-proxy-ca.crt and front-proxy-ca.key from the PKI directory
		frontProxyCACert, frontProxyCAKey, err = pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName)
		if err != nil || frontProxyCACert == nil || frontProxyCAKey == nil {
			return fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		// The certificate and key could be loaded, but the certificate is not a CA
		if !frontProxyCACert.IsCA {
			return fmt.Errorf("certificate and key could be loaded but the certificate is not a CA")
		}

		fmt.Println("[certificates] Using the existing front-proxy CA certificate and key.")
	} else {
		// The certificate and the key did NOT exist, let's generate them now
		frontProxyCACert, frontProxyCAKey, err = pkiutil.NewCertificateAuthority()
		if err != nil {
			return fmt.Errorf("failure while generating front-proxy CA certificate and key [%v]", err)
		}

		if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, frontProxyCACert, frontProxyCAKey); err != nil {
			return fmt.Errorf("failure while saving front-proxy CA certificate and key [%v]", err)
		}
		fmt.Println("[certificates] Generated front-proxy CA certificate and key.")
	}

	// At this point we have a front proxy CA signing key.  We can use that create the front proxy client cert if
	// it doesn't already exist.
	// If at least one of them exists, we should try to load them
	// In the case that only one exists, there will most likely be an error anyway
	if pkiutil.CertOrKeyExist(pkiDir, kubeadmconstants.FrontProxyClientCertAndKeyBaseName) {
		// Try to load apiserver-kubelet-client.crt and apiserver-kubelet-client.key from the PKI directory
		apiCert, apiKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.FrontProxyClientCertAndKeyBaseName)
		if err != nil || apiCert == nil || apiKey == nil {
			return fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		fmt.Println("[certificates] Using the existing front-proxy client certificate and key.")
	} else {
		// The certificate and the key did NOT exist, let's generate them now
		// TODO: Add a test case to verify that this cert has the x509.ExtKeyUsageClientAuth flag
		config := certutil.Config{
			CommonName: "front-proxy-client",
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}
		apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(frontProxyCACert, frontProxyCAKey, config)
		if err != nil {
			return fmt.Errorf("failure while creating front-proxy client key and certificate [%v]", err)
		}

		if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.FrontProxyClientCertAndKeyBaseName, apiClientCert, apiClientKey); err != nil {
			return fmt.Errorf("failure while saving front-proxy client certificate and key [%v]", err)
		}
		fmt.Println("[certificates] Generated front-proxy client certificate and key.")
	}

	fmt.Printf("[certificates] Valid certificates and keys now exist in %q\n", pkiDir)

	return nil
}

func saveConfiguredPKIAssets(cfg *kubeadmapi.MasterConfiguration) error {
	fmt.Println("[certificates] Using the provided CA certificate and key.")
	pkiDir := cfg.CertificatesDir
	caKey, caCert, err := loadCertFromConf(cfg.MasterCertificates.CAKeyPem, cfg.MasterCertificates.CACertPem)
	if err != nil {
		return fmt.Errorf("failure while load CA certificate and key [%v]", err)
	}
	if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.CACertAndKeyBaseName, caCert, caKey); err != nil {
		return fmt.Errorf("failure while saving CA certificate and key [%v]", err)
	}
	fmt.Println("[certificates] Save CA certificate and key.")

	apiKey, apiCert, err := loadCertFromConf(cfg.MasterCertificates.APIServerKeyPem, cfg.MasterCertificates.APIServerCertPem)
	if err != nil {
		return fmt.Errorf("failure while load API server certificate and key [%v]", err)
	}
	if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.APIServerCertAndKeyBaseName, apiCert, apiKey); err != nil {
		return fmt.Errorf("failure while saving API server certificate and key [%v]", err)
	}
	fmt.Println("[certificates] Save API server certificate and key.")

	apiClientKey, apiClientCert, err := loadCertFromConf(cfg.MasterCertificates.APIClientServerKeyPem, cfg.MasterCertificates.APIClientServerCertPem)
	if err != nil {
		return fmt.Errorf("failure while load API server kubelet client key and certificate [%v]", err)
	}

	if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName, apiClientCert, apiClientKey); err != nil {
		return fmt.Errorf("failure while saving API server kubelet client certificate and key [%v]", err)
	}
	fmt.Println("[certificates] Save API server kubelet client certificate and key.")

	sakeyPem, err := base64.StdEncoding.DecodeString(cfg.MasterCertificates.SAKeyPem)
	if err != nil {
		return fmt.Errorf("failure while load service account token key [%v]", err)
	}
	saTokenSigningKey, _, err := parseKeyCertPEM(sakeyPem, nil)

	if err != nil {
		return fmt.Errorf("failure while parse service account token signing key [%v]", err)
	}

	if err = pkiutil.WriteKey(pkiDir, kubeadmconstants.ServiceAccountKeyBaseName, saTokenSigningKey); err != nil {
		return fmt.Errorf("failure while saving service account token signing key [%v]", err)
	}

	if err = pkiutil.WritePublicKey(pkiDir, kubeadmconstants.ServiceAccountKeyBaseName, &saTokenSigningKey.PublicKey); err != nil {
		return fmt.Errorf("failure while saving service account token signing public key [%v]", err)
	}
	fmt.Println("[certificates] Save service account token signing key and public key.")

	frontKey, frontCert, err := loadCertFromConf(cfg.MasterCertificates.FrontProxyKeyPem, cfg.MasterCertificates.FrontProxyCertPem)
	if err != nil {
		return fmt.Errorf("failure while load front-proxy CA certificate and key  [%v]", err)
	}
	if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, frontCert, frontKey); err != nil {
		return fmt.Errorf("failure while saving front-proxy CA certificate and key [%v]", err)
	}
	fmt.Println("[certificates] Save front-proxy CA certificate and key.")

	frontClientKey, frontClientCert, err := loadCertFromConf(cfg.MasterCertificates.FrontProxyClientKeyPem, cfg.MasterCertificates.FrontProxyClientCertPem)
	if err != nil {
		return fmt.Errorf("failure while load front-proxy client CA certificate and key  [%v]", err)
	}
	if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.FrontProxyClientCertAndKeyBaseName, frontClientCert, frontClientKey); err != nil {
		return fmt.Errorf("failure while saving front-proxy client certificate and key [%v]", err)
	}
	fmt.Println("[certificates] Generated front-proxy client certificate and key.")

	fmt.Printf("[certificates] Valid certificates and keys now exist in %q\n", pkiDir)

	return nil
}

// checkAltNamesExist verifies that the cert is valid for all IPs and DNS names it should be valid for
func checkAltNamesExist(IPs []net.IP, DNSNames []string, altNames certutil.AltNames) bool {
	dnsset := setutil.NewString(DNSNames...)

	for _, dnsNameThatShouldExist := range altNames.DNSNames {
		if !dnsset.Has(dnsNameThatShouldExist) {
			return false
		}
	}

	for _, ipThatShouldExist := range altNames.IPs {
		found := false
		for _, ip := range IPs {
			if ip.Equal(ipThatShouldExist) {
				found = true
				break
			}
		}

		if !found {
			return false
		}
	}
	return true
}

// getAltNames builds an AltNames object for the certutil to use when generating the certificates
func getAltNames(cfgAltNames []string, hostname, dnsdomain string, svcSubnet *net.IPNet) certutil.AltNames {
	altNames := certutil.AltNames{
		DNSNames: []string{
			hostname,
			"kubernetes",
			"kubernetes.default",
			"kubernetes.default.svc",
			fmt.Sprintf("kubernetes.default.svc.%s", dnsdomain),
		},
	}

	// Populate IPs/DNSNames from AltNames
	for _, altname := range cfgAltNames {
		if ip := net.ParseIP(altname); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else if len(validation.IsDNS1123Subdomain(altname)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, altname)
		}
	}

	// and lastly, extract the internal IP address for the API server
	internalAPIServerVirtualIP, err := ipallocator.GetIndexedIP(svcSubnet, 1)
	if err != nil {
		fmt.Printf("[certs] WARNING: Unable to get first IP address from the given CIDR (%s): %v\n", svcSubnet.String(), err)
	}
	altNames.IPs = append(altNames.IPs, internalAPIServerVirtualIP)
	return altNames
}

func parseKeyCertPEM(keyPem []byte, certPem []byte) (key *rsa.PrivateKey, cert *x509.Certificate, err error) {
	var ok bool

	if keyPem != nil {
		KeyI, err := certutil.ParsePrivateKeyPEM(keyPem)
		if err != nil {
			return nil, nil, err
		}
		key, ok = KeyI.(*rsa.PrivateKey)

		if !ok {
			return nil, nil, err
		}
	}
	if certPem != nil {
		certA, err := certutil.ParseCertsPEM(certPem)

		if err != nil {
			return nil, nil, err
		}
		cert = certA[0]
	}
	return key, cert, nil
}

func loadCertFromConf(key, cert string) (*rsa.PrivateKey, *x509.Certificate, error) {
	keyPem, err := base64.StdEncoding.DecodeString(key)
	if err != nil {
		return nil, nil, err
	}
	certPem, err := base64.StdEncoding.DecodeString(cert)
	if err != nil {
		return nil, nil, err
	}
	return parseKeyCertPEM(keyPem, certPem)
}

func GeneratePKIAssets(cfg *kubeadmapi.MasterConfiguration) error {
	hostname, err := os.Hostname()
	if err != nil {
		return fmt.Errorf("couldn't get the hostname: %v", err)
	}

	_, svcSubnet, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return fmt.Errorf("error parsing CIDR %q: %v", cfg.Networking.ServiceSubnet, err)
	}

	altNames := getAltNames(cfg.APIServerCertSANs, hostname, cfg.Networking.DNSDomain, svcSubnet)
	if cfg.API.AdvertiseAddress != "" {
		altNames.IPs = append(altNames.IPs, net.ParseIP(cfg.API.AdvertiseAddress))
	}
	if cfg.PublicAddress != cfg.API.AdvertiseAddress {
		if ip := net.ParseIP(cfg.PublicAddress); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else if len(validation.IsDNS1123Subdomain(cfg.PublicAddress)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, cfg.PublicAddress)
		}
	}

	var caCert *x509.Certificate
	var caKey *rsa.PrivateKey
	caCert, caKey, err = pkiutil.NewCertificateAuthority()
	if err != nil {
		return fmt.Errorf("failure while generating CA certificate and key [%v]", err)
	}
	if _, cfg.MasterCertificates.CAKeyPem, cfg.MasterCertificates.CACertPem, err = encodeKeysAndCert(caKey, caCert); err != nil {
		return fmt.Errorf("failure while encoding CA certificate and key [%v]", err)
	}
	config := certutil.Config{
		CommonName: "kube-apiserver",
		AltNames:   altNames,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	apiCert, apiKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return fmt.Errorf("failure while creating API server key and certificate [%v]", err)
	}

	if _, cfg.MasterCertificates.APIServerKeyPem, cfg.MasterCertificates.APIServerCertPem, err = encodeKeysAndCert(apiKey, apiCert); err != nil {
		return fmt.Errorf("failure while encoding API server certificate and key [%v]", err)
	}
	config = certutil.Config{
		CommonName:   "kube-apiserver-kubelet-client",
		Organization: []string{kubeadmconstants.MastersGroup},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return fmt.Errorf("failure while creating API server kubelet client key and certificate [%v]", err)
	}

	if _, cfg.MasterCertificates.APIClientServerKeyPem, cfg.MasterCertificates.APIClientServerCertPem, err = encodeKeysAndCert(apiClientKey, apiClientCert); err != nil {
		return fmt.Errorf("failure while encoding API server certificate and key [%v]", err)
	}

	saTokenSigningKey, err := certutil.NewPrivateKey()
	if err != nil {
		return fmt.Errorf("failure while creating service account token signing key [%v]", err)
	}
	if _, cfg.MasterCertificates.SAKeyPem, _, err = encodeKeysAndCert(saTokenSigningKey, nil); err != nil {
		return fmt.Errorf("failure while encoding service account token signing key [%v]", err)
	}

	var frontProxyCACert *x509.Certificate
	var frontProxyCAKey *rsa.PrivateKey

	// The certificate and the key did NOT exist, let's generate them now
	frontProxyCACert, frontProxyCAKey, err = pkiutil.NewCertificateAuthority()
	if err != nil {
		return fmt.Errorf("failure while generating front-proxy CA certificate and key [%v]", err)
	}

	if _, cfg.MasterCertificates.FrontProxyKeyPem, cfg.MasterCertificates.FrontProxyCertPem, err = encodeKeysAndCert(frontProxyCAKey, frontProxyCACert); err != nil {
		return fmt.Errorf("failure while encoding front-proxy CA certificate and key [%v]", err)
	}

	config = certutil.Config{
		CommonName: "front-proxy-client",
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	apiClientCert, apiClientKey, err = pkiutil.NewCertAndKey(frontProxyCACert, frontProxyCAKey, config)
	if err != nil {
		return fmt.Errorf("failure while creating front-proxy client key and certificate [%v]", err)
	}

	if _, cfg.MasterCertificates.FrontProxyClientKeyPem, cfg.MasterCertificates.FrontProxyClientCertPem, err = encodeKeysAndCert(apiClientKey, apiClientCert); err != nil {
		return fmt.Errorf("failure while encoding front-proxy client certificate and key [%v]", err)
	}

	return nil
}

func encodeKeysAndCert(key *rsa.PrivateKey, cert *x509.Certificate) (pubkeyPemString, prkeyPemString, certPemString string, err error) {
	if key != nil {
		prkeyPem := encodePrivateKeyPEM(key)
		prkeyPemString = base64.StdEncoding.EncodeToString(prkeyPem)
		if pubkeyPem, err := encodePublicKeyPEM(&key.PublicKey); err != nil {
			return "", "", "", fmt.Errorf("unable to encode public key to PEM [%v]", err)
		} else {
			pubkeyPemString = base64.StdEncoding.EncodeToString(pubkeyPem)
		}
	}

	if cert != nil {
		certPem := encodeCertPEM(cert)
		certPemString = base64.StdEncoding.EncodeToString(certPem)
	}

	return
}
func encodePrivateKeyPEM(key *rsa.PrivateKey) []byte {
	block := pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	}
	return pem.EncodeToMemory(&block)
}

// EncodeCertPEM returns PEM-endcoded certificate data
func encodeCertPEM(cert *x509.Certificate) []byte {
	block := pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert.Raw,
	}
	return pem.EncodeToMemory(&block)
}
func encodePublicKeyPEM(key *rsa.PublicKey) ([]byte, error) {
	der, err := x509.MarshalPKIXPublicKey(key)
	if err != nil {
		return []byte{}, err
	}
	block := pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: der,
	}
	return pem.EncodeToMemory(&block), nil
}
