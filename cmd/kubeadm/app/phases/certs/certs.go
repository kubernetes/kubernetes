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
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// It first generates a self-signed CA certificate, a server certificate (signed by the CA) and a key for
// signing service account tokens. It returns CA key and certificate, which is convenient for use with
// client config funcs.
func CreatePKIAssets(cfg *kubeadmapi.MasterConfiguration, pkiPath string) (*x509.Certificate, error) {
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

	caKey, caCert, err := newCertificateAuthority()
	if err != nil {
		return nil, fmt.Errorf("failure while creating CA keys and certificate [%v]", err)
	}

	if err := writeKeysAndCert(pkiPath, "ca", caKey, caCert); err != nil {
		return nil, fmt.Errorf("failure while saving CA keys and certificate [%v]", err)
	}
	fmt.Println("[certificates] Generated Certificate Authority key and certificate.")

	apiKey, apiCert, err := newServerKeyAndCert(caCert, caKey, altNames)
	if err != nil {
		return nil, fmt.Errorf("failure while creating API server keys and certificate [%v]", err)
	}

	if err := writeKeysAndCert(pkiPath, "apiserver", apiKey, apiCert); err != nil {
		return nil, fmt.Errorf("failure while saving API server keys and certificate [%v]", err)
	}
	fmt.Println("[certificates] Generated API Server key and certificate")

	// Generate a private key for service accounts
	saKey, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, fmt.Errorf("failure while creating service account signing keys [%v]", err)
	}
	if err := writeKeysAndCert(pkiPath, "sa", saKey, nil); err != nil {
		return nil, fmt.Errorf("failure while saving service account signing keys [%v]", err)
	}
	fmt.Println("[certificates] Generated Service Account signing keys")
	fmt.Printf("[certificates] Created keys and certificates in %q\n", pkiPath)
	return caCert, nil
}
