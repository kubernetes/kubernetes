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
	certconstants "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	certutil "k8s.io/kubernetes/pkg/util/cert"
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
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, certconstants.CACertAndKeyBaseName)

	if err == nil && caCert != nil && caCert.IsCA && caKey != nil {
		fmt.Println("[certificates] Using the already-existing CA certificate and key.")
	} else {
		// An error occured while reading the certificate and the key (might simply not exist). That means we should generate our own CA.
		caCert, caKey, err = pkiutil.NewCertificateAuthority()
		if err != nil {
			return nil, fmt.Errorf("failure while creating CA keys and certificate [%v]", err)
		}

		if err = pkiutil.WriteCertAndKey(pkiDir, certconstants.CACertAndKeyBaseName, caCert, caKey); err != nil {
			return nil, fmt.Errorf("failure while saving CA keys and certificate [%v]", err)
		}
		fmt.Println("[certificates] Generated Certificate Authority key and certificate.")
	}

	// Try to load ca.crt and ca.key from the PKI directory
	apiCert, apiKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, certconstants.APIServerCertAndKeyBaseName)

	if err == nil && apiCert != nil && apiKey != nil && checkAltNamesExist(apiCert.IPAddresses, apiCert.DNSNames, altNames) {
		fmt.Println("[certificates] Using the already-existing apiserver certificate and key.")
	} else {
		// An error occured while reading the certificate and the key (might simply not exist). That means we should generate our own certificate.
		apiCert, apiKey, err := pkiutil.NewServerKeyAndCert(caCert, caKey, altNames)
		if err != nil {
			return nil, fmt.Errorf("failure while creating API server keys and certificate [%v]", err)
		}

		if err := pkiutil.WriteCertAndKey(pkiDir, certconstants.APIServerCertAndKeyBaseName, apiCert, apiKey); err != nil {
			return nil, fmt.Errorf("failure while saving API server keys and certificate [%v]", err)
		}
		fmt.Println("[certificates] Generated API Server key and certificate")
	}

	fmt.Printf("[certificates] Valid certificates and keys now exist in %q\n", pkiDir)

	// TODO: Public phase methods should not return anything.
	return caCert, nil
}

// Verify that the cert is valid for all IPs and DNS names it should be valid for
func checkAltNamesExist(IPs []net.IP, DNSNames []string, altNames certutil.AltNames) bool {
	for _, dnsNameThatShouldExist := range altNames.DNSNames {
		found := false
		for _, dnsName := range DNSNames {
			if dnsName == dnsNameThatShouldExist {
				found = true
			}
		}

		if !found {
			return false
		}
	}

	for _, ipThatShouldExist := range altNames.IPs {
		found := false
		for _, ip := range IPs {
			if ip.Equal(ipThatShouldExist) {
				found = true
			}
		}

		if !found {
			return false
		}
	}
	return true
}
