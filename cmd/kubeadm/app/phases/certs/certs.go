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
	"crypto/x509"
	"fmt"
	"net"
	"os"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	setutil "k8s.io/kubernetes/pkg/util/sets"
)

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// It generates a self-signed CA certificate and a server certificate (signed by the CA)
// TODO: We should create apiserver -> kubelet certs somehow here also...
func CreatePKIAssets(cfg *kubeadmapi.MasterConfiguration, pkiDir string) (*x509.Certificate, error) {
	altNames := certutil.AltNames{}

	// First, define all domains this cert should be signed for
	internalAPIServerFQDN := []string{
		"kubernetes",
		"kubernetes.default",
		"kubernetes.default.svc",
		fmt.Sprintf("kubernetes.default.svc.%s", cfg.Networking.DNSDomain),
	}
	hostname, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("couldn't get the hostname: %v", err)
	}
	altNames.DNSNames = append(cfg.API.ExternalDNSNames, hostname)
	altNames.DNSNames = append(altNames.DNSNames, internalAPIServerFQDN...)

	// then, add all IP addresses we're bound to
	for _, a := range cfg.API.AdvertiseAddresses {
		if ip := net.ParseIP(a); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else {
			return nil, fmt.Errorf("could not parse ip %q", a)
		}
	}
	// and lastly, extract the internal IP address for the API server
	_, n, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return nil, fmt.Errorf("error parsing CIDR %q: %v", cfg.Networking.ServiceSubnet, err)
	}
	internalAPIServerVirtualIP, err := ipallocator.GetIndexedIP(n, 1)
	if err != nil {
		return nil, fmt.Errorf("unable to allocate IP address for the API server from the given CIDR (%q) [%v]", &cfg.Networking.ServiceSubnet, err)
	}

	altNames.IPs = append(altNames.IPs, internalAPIServerVirtualIP)

	// Try to load ca.crt and ca.key from the PKI directory
	caCert, caKey, oneOfThemExists, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.CACertAndKeyBaseName)

	if err == nil && caCert != nil && caKey != nil {
		// The certificate and key already exist, skip the generation
		if !caCert.IsCA {
			return nil, fmt.Errorf("certificate and key existed but the certificate is not a certificate authority")
		}

		fmt.Println("[certificates] Using the existing CA certificate and key.")
	} else if !oneOfThemExists {
		// The certificate and the key did NOT exist, let's generate them now
		caCert, caKey, err = pkiutil.NewCertificateAuthority()
		if err != nil {
			return nil, fmt.Errorf("failure while creating CA key and certificate [%v]", err)
		}

		if err = pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.CACertAndKeyBaseName, caCert, caKey); err != nil {
			return nil, fmt.Errorf("failure while saving CA key and certificate [%v]", err)
		}
		fmt.Println("[certificates] Generated CA key and certificate.")
	} else {
		// An unhandled error occured
		return nil, fmt.Errorf("at least one of the CA certificate or the CA key exists but was not parseable")
	}

	// Try to load ca.crt and ca.key from the PKI directory
	apiCert, apiKey, oneOfThemExists, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.APIServerCertAndKeyBaseName)

	if err == nil && apiCert != nil && apiKey != nil {
		// The certificate and key already exist, skip the generation
		fmt.Println("[certificates] Using the existing apiserver certificate and key.")
	} else if !oneOfThemExists {
		// The certificate and the key did NOT exist, let's generate them now
		apiCert, apiKey, err := pkiutil.NewServerKeyAndCert(caCert, caKey, altNames)
		if err != nil {
			return nil, fmt.Errorf("failure while creating API server key and certificate [%v]", err)
		}

		if err := pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.APIServerCertAndKeyBaseName, apiCert, apiKey); err != nil {
			return nil, fmt.Errorf("failure while saving API server key and certificate [%v]", err)
		}
		fmt.Println("[certificates] Generated API Server key and certificate")
	} else {
		// An unhandled error occured
		return nil, fmt.Errorf("at least one of the apiserver certificate or the apiserver key exists but was not parseable")
	}

	fmt.Printf("[certificates] Valid certificates and keys now exist in %q\n", pkiDir)

	// TODO: Public phase methods should not return values for kubeadm to consume later except for errors.
	return caCert, nil
}

// Verify that the cert is valid for all IPs and DNS names it should be valid for
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
