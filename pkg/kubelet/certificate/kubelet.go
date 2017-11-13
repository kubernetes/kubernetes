/*
Copyright 2017 The Kubernetes Authors.

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

package certificate

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"net"
	"sync"

	"github.com/prometheus/client_golang/prometheus"

	certificates "k8s.io/api/certificates/v1beta1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	clientcertificates "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// NewKubeletServerCertificateManager creates a certificate manager for the kubelet when retrieving a server certificate
// or returns an error.
func NewKubeletServerCertificateManager(kubeClient clientset.Interface, kubeCfg *kubeletconfig.KubeletConfiguration, nodeName types.NodeName, ips []net.IP, hostnames []string, certDirectory string) (certificate.Manager, error) {
	var certSigningRequestClient clientcertificates.CertificateSigningRequestInterface
	if kubeClient != nil && kubeClient.Certificates() != nil {
		certSigningRequestClient = kubeClient.Certificates().CertificateSigningRequests()
	}
	certificateStore, err := certificate.NewFileStore(
		"kubelet-server",
		certDirectory,
		certDirectory,
		kubeCfg.TLSCertFile,
		kubeCfg.TLSPrivateKeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize server certificate store: %v", err)
	}
	var certificateExpiration = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: metrics.KubeletSubsystem,
			Subsystem: "certificate_manager",
			Name:      "server_expiration_seconds",
			Help:      "Gauge of the lifetime of a certificate. The value is the date the certificate will expire in seconds since January 1, 1970 UTC.",
		},
	)
	prometheus.MustRegister(certificateExpiration)

	m, err := certificate.NewManager(&certificate.Config{
		CertificateSigningRequestClient: certSigningRequestClient,
		Template: &x509.CertificateRequest{
			Subject: pkix.Name{
				CommonName:   fmt.Sprintf("system:node:%s", nodeName),
				Organization: []string{"system:nodes"},
			},
			DNSNames:    hostnames,
			IPAddresses: ips,
		},
		Usages: []certificates.KeyUsage{
			// https://tools.ietf.org/html/rfc5280#section-4.2.1.3
			//
			// Digital signature allows the certificate to be used to verify
			// digital signatures used during TLS negotiation.
			certificates.UsageDigitalSignature,
			// KeyEncipherment allows the cert/key pair to be used to encrypt
			// keys, including the symmetric keys negotiated during TLS setup
			// and used for data transfer.
			certificates.UsageKeyEncipherment,
			// ServerAuth allows the cert to be used by a TLS server to
			// authenticate itself to a TLS client.
			certificates.UsageServerAuth,
		},
		CertificateStore:      certificateStore,
		CertificateExpiration: certificateExpiration,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize server certificate manager: %v", err)
	}
	return m, nil
}

// NewKubeletClientCertificateManager sets up a certificate manager without a
// client that can be used to sign new certificates (or rotate). It answers with
// whatever certificate it is initialized with. If a CSR client is set later, it
// may begin rotating/renewing the client cert
func NewKubeletClientCertificateManager(certDirectory string, nodeName types.NodeName, certData []byte, keyData []byte, certFile string, keyFile string) (certificate.Manager, error) {
	certificateStore, err := certificate.NewFileStore(
		"kubelet-client",
		certDirectory,
		certDirectory,
		certFile,
		keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize client certificate store: %v", err)
	}
	var certificateExpiration = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: metrics.KubeletSubsystem,
			Subsystem: "certificate_manager",
			Name:      "client_expiration_seconds",
			Help:      "Gauge of the lifetime of a certificate. The value is the date the certificate will expire in seconds since January 1, 1970 UTC.",
		},
	)
	prometheus.MustRegister(certificateExpiration)

	m, err := certificate.NewManager(&certificate.Config{
		Template: &x509.CertificateRequest{
			Subject: pkix.Name{
				CommonName:   fmt.Sprintf("system:node:%s", nodeName),
				Organization: []string{"system:nodes"},
			},
		},
		Usages: []certificates.KeyUsage{
			// https://tools.ietf.org/html/rfc5280#section-4.2.1.3
			//
			// DigitalSignature allows the certificate to be used to verify
			// digital signatures including signatures used during TLS
			// negotiation.
			certificates.UsageDigitalSignature,
			// KeyEncipherment allows the cert/key pair to be used to encrypt
			// keys, including the symmetric keys negotiated during TLS setup
			// and used for data transfer..
			certificates.UsageKeyEncipherment,
			// ClientAuth allows the cert to be used by a TLS client to
			// authenticate itself to the TLS server.
			certificates.UsageClientAuth,
		},
		CertificateStore:        certificateStore,
		BootstrapCertificatePEM: certData,
		BootstrapKeyPEM:         keyData,
		CertificateExpiration:   certificateExpiration,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize client certificate manager: %v", err)
	}
	return m, nil
}

// Updater updates kubelet certifcate information if and dns/ip information changes
type Updater interface {
	UpdateCertInfo(nodeAddrs []v1.NodeAddress) error
	Current() *tls.Certificate
}

type updater struct {
	ipList         []net.IP
	dnsList        []string
	host           string
	needsCertRegen bool

	certMutex sync.RWMutex
	cert      *tls.Certificate
}

// NewUpdater sets up a certificate updater that can regenerate certficates if any dns/ip information changes
func NewUpdater(hostName string, certFile, keyFile string, iplist []net.IP, dnslist []string) (Updater, error) {
	c, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}
	return &updater{
		ipList:         iplist,
		dnsList:        dnslist,
		host:           hostName,
		cert:           &c,
		needsCertRegen: false,
	}, nil
}

func (u *updater) UpdateCertInfo(nodeAddrs []v1.NodeAddress) error {
	ips := []net.IP{}
	dnsNames := []string{}
	for _, addr := range nodeAddrs {
		switch addr.Type {
		case v1.NodeHostName:
			if u.host != addr.Address {
				u.needsCertRegen = true
			}
			u.host = addr.Address
		case v1.NodeExternalIP, v1.NodeInternalIP:
			ips = append(ips, net.ParseIP(addr.Address))
		case v1.NodeExternalDNS, v1.NodeInternalDNS:
			dnsNames = append(dnsNames, addr.Address)
		}
	}

	u.updateIPList(ips)
	u.updateDNSList(dnsNames)
	if u.needsCertRegen {
		cert, key, err := certutil.GenerateSelfSignedCertKey(u.host, u.ipList, u.dnsList)
		if err != nil {
			return err
		}
		newCert, err := tls.X509KeyPair(cert, key)
		if err != nil {
			return err
		}
		u.cert = &newCert
		u.needsCertRegen = false
	}
	return nil
}

func (u *updater) Current() *tls.Certificate {
	return u.cert
}

func (u *updater) updateIPList(ips []net.IP) {
	if len(u.ipList) != len(ips) {
		// Length is different so we need to regen cert
		u.needsCertRegen = true
	} else {
		// Lengths are the same, so verify all entries are the same
		for _, ip := range ips {
			var existing net.IP
			for _, existing = range u.ipList {
				if bytes.Equal(ip, existing) {
					break
				}
			}
			if !bytes.Equal(existing, ip) {
				// We found a difference, so no need to continue evaluating
				u.needsCertRegen = true
				break
			}
		}
	}
	u.ipList = ips
}

func (u *updater) updateDNSList(dnsNames []string) {
	if len(u.dnsList) != len(dnsNames) {
		// Length is different so we need to regen cert
		u.needsCertRegen = true
	} else {
		// Lengths are the same, so verify all entries are the same
		for _, dnsName := range dnsNames {
			var existing string
			for _, existing = range u.dnsList {
				if dnsName == existing {
					break
				}
			}
			if existing != dnsName {
				// We found a difference, so no need to continue evaluating
				u.needsCertRegen = true
				break
			}
		}
	}
	u.dnsList = dnsNames
}
