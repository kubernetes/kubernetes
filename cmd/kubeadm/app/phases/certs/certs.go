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

	"k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// NewCACertAndKey will generate a self signed CA.
func NewCACertAndKey() (*x509.Certificate, *rsa.PrivateKey, error) {

	caCert, caKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		return nil, nil, fmt.Errorf("failure while generating CA certificate and key: %v", err)
	}

	return caCert, caKey, nil
}

// NewAPIServerCertAndKey generate CA certificate for apiserver, signed by the given CA.
func NewAPIServerCertAndKey(cfg *kubeadmapi.MasterConfiguration, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	altNames, err := getAltNames(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while composing altnames for API server: %v", err)
	}

	config := certutil.Config{
		CommonName: kubeadmconstants.APIServerCertCommonName,
		AltNames:   *altNames,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	apiCert, apiKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating API server key and certificate: %v", err)
	}

	return apiCert, apiKey, nil
}

// NewAPIServerKubeletClientCertAndKey generate CA certificate for the apiservers to connect to the kubelets securely, signed by the given CA.
func NewAPIServerKubeletClientCertAndKey(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	config := certutil.Config{
		CommonName:   kubeadmconstants.APIServerKubeletClientCertCommonName,
		Organization: []string{kubeadmconstants.MastersGroup},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating API server kubelet client key and certificate: %v", err)
	}

	return apiClientCert, apiClientKey, nil
}

// NewServiceAccountSigningKey generate public/private key pairs for signing service account tokens.
func NewServiceAccountSigningKey() (*rsa.PrivateKey, error) {

	// The key does NOT exist, let's generate it now
	saSigningKey, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, fmt.Errorf("failure while creating service account token signing key: %v", err)
	}

	return saSigningKey, nil
}

// NewFrontProxyCACertAndKey generate a self signed front proxy CA.
// Front proxy CA and client certs are used to secure a front proxy authenticator which is used to assert identity
// without the client cert.
// This is a separte CA, so that front proxy identities cannot hit the API and normal client certs cannot be used
// as front proxies.
func NewFrontProxyCACertAndKey() (*x509.Certificate, *rsa.PrivateKey, error) {

	frontProxyCACert, frontProxyCAKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		return nil, nil, fmt.Errorf("failure while generating front-proxy CA certificate and key: %v", err)
	}

	return frontProxyCACert, frontProxyCAKey, nil
}

// NewFrontProxyClientCertAndKey generate CA certificate for proxy server client, signed by the given front proxy CA.
func NewFrontProxyClientCertAndKey(frontProxyCACert *x509.Certificate, frontProxyCAKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	config := certutil.Config{
		CommonName: kubeadmconstants.FrontProxyClientCertCommonName,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	frontProxyClientCert, frontProxyClientKey, err := pkiutil.NewCertAndKey(frontProxyCACert, frontProxyCAKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating front-proxy client key and certificate: %v", err)
	}

	return frontProxyClientCert, frontProxyClientKey, nil
}

// getAltNames builds an AltNames object for to be used when generating apiserver certificate
func getAltNames(cfg *kubeadmapi.MasterConfiguration) (*certutil.AltNames, error) {

	// advertise address
	advertiseAddress := net.ParseIP(cfg.API.AdvertiseAddress)
	if advertiseAddress == nil {
		return nil, fmt.Errorf("error parsing API AdvertiseAddress %v: is not a valid textual representation of an IP address", cfg.API.AdvertiseAddress)
	}

	// internal IP address for the API server
	_, svcSubnet, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return nil, fmt.Errorf("error parsing CIDR %q: %v", cfg.Networking.ServiceSubnet, err)
	}

	internalAPIServerVirtualIP, err := ipallocator.GetIndexedIP(svcSubnet, 1)
	if err != nil {
		return nil, fmt.Errorf("unable to get first IP address from the given CIDR (%s): %v", svcSubnet.String(), err)
	}

	// create AltNames with defaults DNSNames/IPs
	altNames := &certutil.AltNames{
		DNSNames: []string{
			cfg.NodeName,
			"kubernetes",
			"kubernetes.default",
			"kubernetes.default.svc",
			fmt.Sprintf("kubernetes.default.svc.%s", cfg.Networking.DNSDomain),
		},
		IPs: []net.IP{
			internalAPIServerVirtualIP,
			advertiseAddress,
		},
	}

	// adds additional SAN
	for _, altname := range cfg.APIServerCertSANs {
		if ip := net.ParseIP(altname); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else if len(validation.IsDNS1123Subdomain(altname)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, altname)
		}
	}

	return altNames, nil
}
