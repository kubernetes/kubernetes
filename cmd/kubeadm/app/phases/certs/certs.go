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

	"k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// CreateCertFunc define the signature of methods implementing an create cert actions.
// Each Create cert actions execute an "atomic" crate task
type CreateCertFunc func(cfg *kubeadmapi.MasterConfiguration) (*CreateCertResult, error)

// CreateCertResult represents outcomes of an CreateCert action
type CreateCertResult struct {
	// GeneratedCertAndKeyBaseName contains the base name of generated files.
	// Please note that this attribute can be used also as identifier of the action performed in case of
	// bulk CreateCert actions
	GeneratedCertAndKeyBaseName string
	// UsedExistingCert is true if an exixting certificate file was used (instead of creating a new one)
	UsedExistingCert bool
	// AdditionalMsg contains the list of extra msg documenting the performed CreateCert action
	AdditionalMsgs []string
}

// BulkCreateCertFunc define the signature of methods implementing bulk create cert actions.
// Bulck create cert function are implemented composing "atomic" create cert actions
type BulkCreateCertFunc func(cfg *kubeadmapi.MasterConfiguration) (BulkCreateCertResult, error)

type BulkCreateCertResult []*CreateCertResult

// NewCreateCertResult create a new instance of CreateCertResult
func NewCreateCertResult(baseName string, usedExistingCert bool, additionalMsg ...string) *CreateCertResult {
	return &CreateCertResult{
		GeneratedCertAndKeyBaseName: baseName,
		UsedExistingCert:            usedExistingCert,
		AdditionalMsgs:              additionalMsg,
	}
}

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// Please note that this action is a bulk action calling all the atomic CreateCert actions
func CreatePKIAssets(cfg *kubeadmapi.MasterConfiguration) (BulkCreateCertResult, error) {
	var (
		results BulkCreateCertResult
		r       *CreateCertResult
		err     error
	)

	// create and write to disk a self signed CA (if it does not exists)
	r, err = CreateCACertAndKey(cfg)
	if err != nil {
		return results, fmt.Errorf("Error creating CA certificate: %v", err)
	}
	results = append(results, r)

	// create and write to disk CA certificate for apiserver, signed by the CA (if it does not exists)
	r, err = CreateAPIServerCertAndKey(cfg)
	if err != nil {
		return results, fmt.Errorf("Error creating API Server certificate: %v", err)
	}
	results = append(results, r)

	// create and write to disk CA certificate for kubelets calling apiserver, signed by the CA (if it does not exists)
	r, err = CreateAPIServerKubeletClientCertAndKey(cfg)
	if err != nil {
		return results, fmt.Errorf("Error creating API Server kubelet client certificate: %v", err)
	}
	results = append(results, r)

	// create and write to disk public/private key pairs for signing service account user (if it does not exists)
	r, err = CreateServiceAccountKeyAndPublicKey(cfg)
	if err != nil {
		return results, fmt.Errorf("Error creating service account token signing key and public key: %v", err)
	}
	results = append(results, r)

	// create and write to disk a self signed front proxy CA (if it does not exists)
	r, err = CreateFrontProxyCACertAndKey(cfg)
	if err != nil {
		return results, fmt.Errorf("Error creating front-proxy CA certificate: %v", err)
	}
	results = append(results, r)

	// create and write to disk CA certificate for proxy server client, signed by the front proxy CA (if it does not exists)
	r, err = CreateFrontProxyClientCertAndKey(cfg)
	if err != nil {
		return results, fmt.Errorf("Error creating front-proxy client certificate: %v", err)
	}
	results = append(results, r)

	return results, nil
}

// CreateCACertAndKey will create and write to disk a self signed CA (if it does not exists)
func CreateCACertAndKey(cfg *kubeadmapi.MasterConfiguration) (*CreateCertResult, error) {

	// If at least one of CA Cert or Key exists, we should try to load them
	// In the case that only one exists, there will be an error anyway
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName) {
		// Try to load ca.crt and ca.key from the PKI directory
		_, _, err := tryLoadCreateCACertAndKey(cfg)
		if err != nil {
			return nil, err
		}

		return NewCreateCertResult(kubeadmconstants.CACertAndKeyBaseName, true), nil
	}

	// The certificate and the key did NOT exist, let's generate them now
	caCert, caKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		return nil, fmt.Errorf("failure while generating CA certificate and key [%v]", err)
	}

	if err = pkiutil.WriteCertAndKey(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, caCert, caKey); err != nil {
		return nil, fmt.Errorf("failure while saving CA certificate and key [%v]", err)
	}

	return NewCreateCertResult(kubeadmconstants.CACertAndKeyBaseName, false), nil
}

// CreateAPIServerCertAndKey will create and write to disk CA certificate for apiserver,
// signed by the CA (if it does not exists).
// It assumes the CA certificates should exists into the CertificatesDir
func CreateAPIServerCertAndKey(cfg *kubeadmapi.MasterConfiguration) (*CreateCertResult, error) {

	var caCert *x509.Certificate
	var caKey *rsa.PrivateKey
	var err error

	// Try to load ca.crt and ca.key from the PKI directory
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName) {
		caCert, caKey, err = tryLoadCreateCACertAndKey(cfg)
		if err != nil {
			return nil, fmt.Errorf("Error loading CA certificate and key: %v", err)
		}
	} else {
		return nil, fmt.Errorf("CA certificate and key does not exits in certificate dir")
	}

	// If at least one of API Server Cert or Key exists, we should try to load them
	// In the case that only one exists, there will be an error anyway
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.APIServerCertAndKeyBaseName) {
		// Try to load apiserver.crt and apiserver.key from the PKI directory
		apiCert, apiKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.APIServerCertAndKeyBaseName)
		if err != nil || apiCert == nil || apiKey == nil {
			return nil, fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		return NewCreateCertResult(kubeadmconstants.APIServerCertAndKeyBaseName, true), nil
	}

	// The certificate and the key did NOT exist, let's generate them now
	// Build the list of SANs
	hostname, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("couldn't get the hostname: %v", err)
	}

	_, svcSubnet, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return nil, fmt.Errorf("error parsing CIDR %q: %v", cfg.Networking.ServiceSubnet, err)
	}

	altNames := getAltNames(cfg.APIServerCertSANs, hostname, cfg.Networking.DNSDomain, svcSubnet)

	// Append the address the API Server is advertising
	altNames.IPs = append(altNames.IPs, net.ParseIP(cfg.API.AdvertiseAddress))

	// Geneate the certificate
	config := certutil.Config{
		CommonName: "kube-apiserver",
		AltNames:   altNames,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	apiCert, apiKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, fmt.Errorf("failure while creating API server key and certificate [%v]", err)
	}

	if err = pkiutil.WriteCertAndKey(cfg.CertificatesDir, kubeadmconstants.APIServerCertAndKeyBaseName, apiCert, apiKey); err != nil {
		return nil, fmt.Errorf("failure while saving API server certificate and key [%v]", err)
	}

	signatureMsg := fmt.Sprintf("API Server serving cert is signed for DNS names %v and IPs %v", altNames.DNSNames, altNames.IPs)
	return NewCreateCertResult(kubeadmconstants.APIServerCertAndKeyBaseName, false, signatureMsg), nil
}

// CreateAPIServerKubeletClientCertAndKey will create and write to disk CA certificate for the apiservers
// to connect to the kubelets securely, signed by the CA (if it does not exists).
// It assumes the CA certificates should exists into the CertificatesDir
func CreateAPIServerKubeletClientCertAndKey(cfg *kubeadmapi.MasterConfiguration) (*CreateCertResult, error) {

	var caCert *x509.Certificate
	var caKey *rsa.PrivateKey
	var err error

	// Try to load ca.crt and ca.key from the PKI directory
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName) {
		caCert, caKey, err = tryLoadCreateCACertAndKey(cfg)
		if err != nil {
			return nil, fmt.Errorf("error loading CA certificate and key: %v", err)
		}
	} else {
		return nil, fmt.Errorf("CA certificate and key does not exits in certificate dir")
	}

	// If at least one of APIServerKubeletClient Cert or Key exists, we should try to load them
	// In the case that only one exists, there will be an error anyway
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName) {
		// Try to load apiserver-kubelet-client.crt and apiserver-kubelet-client.key from the PKI directory
		apiCert, apiKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName)
		if err != nil || apiCert == nil || apiKey == nil {
			return nil, fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		return NewCreateCertResult(kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName, true), nil
	}

	// The certificate and the key did NOT exist, let's generate them now
	config := certutil.Config{
		CommonName:   "kube-apiserver-kubelet-client",
		Organization: []string{kubeadmconstants.MastersGroup},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, fmt.Errorf("failure while creating API server kubelet client key and certificate [%v]", err)
	}

	if err = pkiutil.WriteCertAndKey(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName, apiClientCert, apiClientKey); err != nil {
		return nil, fmt.Errorf("failure while saving API server kubelet client certificate and key [%v]", err)
	}

	return NewCreateCertResult(kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName, false), nil
}

// CreateServiceAccountKeyAndPublicKey will create and write to disk public/private key pairs for
// signing service account tokens (if it does not exists)
func CreateServiceAccountKeyAndPublicKey(cfg *kubeadmapi.MasterConfiguration) (*CreateCertResult, error) {

	// If the key exists, we should try to load it
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.ServiceAccountKeyBaseName) {
		// Try to load sa.key from the PKI directory
		_, err := pkiutil.TryLoadKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.ServiceAccountKeyBaseName)
		if err != nil {
			return nil, fmt.Errorf("certificate and/or key existed but they could not be loaded properly [%v]", err)
		}

		return NewCreateCertResult(kubeadmconstants.ServiceAccountKeyBaseName, true), nil
	}

	// The key does NOT exist, let's generate it now
	saTokenSigningKey, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, fmt.Errorf("failure while creating service account token signing key [%v]", err)
	}

	if err = pkiutil.WriteKey(cfg.CertificatesDir, kubeadmconstants.ServiceAccountKeyBaseName, saTokenSigningKey); err != nil {
		return nil, fmt.Errorf("failure while saving service account token signing key [%v]", err)
	}

	if err = pkiutil.WritePublicKey(cfg.CertificatesDir, kubeadmconstants.ServiceAccountKeyBaseName, &saTokenSigningKey.PublicKey); err != nil {
		return nil, fmt.Errorf("failure while saving service account token signing public key [%v]", err)
	}

	return NewCreateCertResult(kubeadmconstants.ServiceAccountKeyBaseName, false), nil
}

// CreateFrontProxyCACertAndKey will create and write to disk a self signed front proxy CA (if it does not exists).
// Front proxy CA and client certs are used to secure a front proxy authenticator which is used to assert identity
// without the client cert.
// This is a separte CA, so that front proxy identities cannot hit the API and normal client certs cannot be used
// as front proxies.
func CreateFrontProxyCACertAndKey(cfg *kubeadmapi.MasterConfiguration) (*CreateCertResult, error) {

	// If at least one of FrontProxyCA Cert or Key exists, we should try to load them
	// In the case that only one exists, there will be an error anyway
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName) {
		// Try to load front-proxy-ca.crt and front-proxy-ca.key from the PKI directory
		_, _, err := tryLoadFrontProxyCACertAndKey(cfg)
		if err != nil {
			return nil, err
		}

		return NewCreateCertResult(kubeadmconstants.FrontProxyCACertAndKeyBaseName, true), nil
	}

	// The certificate and the key did NOT exist, let's generate them now
	frontProxyCACert, frontProxyCAKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		return nil, fmt.Errorf("failure while generating front-proxy CA certificate and key [%v]", err)
	}

	if err = pkiutil.WriteCertAndKey(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, frontProxyCACert, frontProxyCAKey); err != nil {
		return nil, fmt.Errorf("failure while saving front-proxy CA certificate and key [%v]", err)
	}

	return NewCreateCertResult(kubeadmconstants.FrontProxyCACertAndKeyBaseName, false), nil
}

// CreateFrontProxyClientCertAndKey will create and write to disk CA certificate for proxy server client, signed by the front proxy CA (if it does not exists)
// It assumes the front proxy CA certificates should exists into the CertificatesDir
func CreateFrontProxyClientCertAndKey(cfg *kubeadmapi.MasterConfiguration) (*CreateCertResult, error) {

	var frontProxyCACert *x509.Certificate
	var frontProxyCAKey *rsa.PrivateKey
	var err error

	// Try to load front-proxy-ca.crt and front-proxy-ca.key from the PKI directory
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName) {
		frontProxyCACert, frontProxyCAKey, err = tryLoadFrontProxyCACertAndKey(cfg)
		if err != nil {
			return nil, fmt.Errorf("error loading front proxy CA certificate and key: %v", err)
		}
	} else {
		return nil, fmt.Errorf("front proxy CA certificate and key does not exits in certificate dir")
	}

	// If at least one of CreateFrontProxyClient Cert or Key exists, we should try to load them
	// In the case that only one exists, there will be an error anyway
	if pkiutil.CertOrKeyExist(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientCertAndKeyBaseName) {
		// Try to load apiserver-kubelet-client.crt and apiserver-kubelet-client.key from the PKI directory
		apiCert, apiKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientCertAndKeyBaseName)
		if err != nil || apiCert == nil || apiKey == nil {
			return nil, fmt.Errorf("certificate and/or key existed but they could not be loaded properly")
		}

		return NewCreateCertResult(kubeadmconstants.FrontProxyClientCertAndKeyBaseName, true), nil
	}

	// The certificate and the key did NOT exist, let's generate them now
	// TODO: Add a test case to verify that this cert has the x509.ExtKeyUsageClientAuth flag
	config := certutil.Config{
		CommonName: "front-proxy-client",
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(frontProxyCACert, frontProxyCAKey, config)
	if err != nil {
		return nil, fmt.Errorf("failure while creating front-proxy client key and certificate [%v]", err)
	}

	if err = pkiutil.WriteCertAndKey(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientCertAndKeyBaseName, apiClientCert, apiClientKey); err != nil {
		return nil, fmt.Errorf("failure while saving front-proxy client certificate and key [%v]", err)
	}

	return NewCreateCertResult(kubeadmconstants.FrontProxyClientCertAndKeyBaseName, false), nil
}

// tryLoadCreateCACertAndKey will try to load CACertAndKey from disk and validate them
func tryLoadCreateCACertAndKey(cfg *kubeadmapi.MasterConfiguration) (*x509.Certificate, *rsa.PrivateKey, error) {

	// Try to load ca.crt and ca.key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil || caCert == nil || caKey == nil {
		return nil, nil, fmt.Errorf("CA certificate and/or key existed but they could not be loaded properly")
	}

	// The certificate and key could be loaded, but the certificate is not a CA
	if !caCert.IsCA {
		return nil, nil, fmt.Errorf("CA certificate and key could be loaded but the certificate is not a CA")
	}

	return caCert, caKey, nil
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

// tryLoadFrontProxyCACertAndKey will try to load FrontProxyCACACertAndKey from disk and validate them
func tryLoadFrontProxyCACertAndKey(cfg *kubeadmapi.MasterConfiguration) (*x509.Certificate, *rsa.PrivateKey, error) {

	// Try to load front-proxy-ca.crt and front-proxy-ca.key from the PKI directory
	frontProxyCACert, frontProxyCAKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName)
	if err != nil || frontProxyCACert == nil || frontProxyCAKey == nil {
		return nil, nil, fmt.Errorf("front-proxy CA certificate and/or key existed but they could not be loaded properly")
	}

	// The certificate and key could be loaded, but the certificate is not a CA
	if !frontProxyCACert.IsCA {
		return nil, nil, fmt.Errorf("front-proxy CA certificate and key could be loaded but the certificate is not a CA")
	}

	return frontProxyCACert, frontProxyCAKey, nil
}

// String methods generates string representation of BulkCreateCertResult
// TODO: this is an UX concern that should be shared betwee kubeadm init and kubeadm phase certs;
// consider review package structure (e.g. move into cmd/utils). see https://github.com/kubernetes/kubeadm/issues/267
func (results BulkCreateCertResult) String() string {
	var s string

	// Prints results to UX
	for _, r := range results {
		// UX message about create/use existing
		if r.UsedExistingCert {
			s += fmt.Sprintf("[certificates] Using the existing %s and key.\n", getUXDesciption(r.GeneratedCertAndKeyBaseName))
		} else {
			s += fmt.Sprintf("[certificates] Generated %s and key.\n", getUXDesciption(r.GeneratedCertAndKeyBaseName))
		}

		// UX additional messages
		for _, msg := range r.AdditionalMsgs {
			s += fmt.Sprintf("[certificates] %s\n", msg)
		}
	}

	return s
}

// getUXDesciption return UX description for the given GeneratedCertAndKeyBaseName
func getUXDesciption(generatedCertAndKeyBaseName string) string {
	switch generatedCertAndKeyBaseName {
	case kubeadmconstants.CACertAndKeyBaseName:
		return "CA certificate and key"
	case kubeadmconstants.APIServerCertAndKeyBaseName:
		return "API Server certificate"
	case kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName:
		return "API Server kubelet client certificate and key"
	case kubeadmconstants.ServiceAccountKeyBaseName:
		return "service account token signing key and public key"
	case kubeadmconstants.FrontProxyCACertAndKeyBaseName:
		return "front-proxy CA certificate and key"
	case kubeadmconstants.FrontProxyClientCertAndKeyBaseName:
		return "front-proxy CA certificate and key"
	default:
		return generatedCertAndKeyBaseName
	}
}
