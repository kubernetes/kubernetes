/*
Copyright 2019 The Kubernetes Authors.

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

package cert

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/url"
	"strings"
)

// GetClientCANames gets the CA names for client certs that a server accepts.  This is useful when inspecting the
// state of particular servers.  apiHost is "host:port"
func GetClientCANames(apiHost string) ([]string, error) {
	// when we run this the second time, we know which one we are expecting
	acceptableCAs := []string{}
	tlsConfig := &tls.Config{
		InsecureSkipVerify: true, // this is insecure to always get to the GetClientCertificate
		GetClientCertificate: func(hello *tls.CertificateRequestInfo) (*tls.Certificate, error) {
			acceptableCAs = []string{}
			for _, curr := range hello.AcceptableCAs {
				acceptableCAs = append(acceptableCAs, string(curr))
			}
			return &tls.Certificate{}, nil
		},
	}

	conn, err := tls.Dial("tcp", apiHost, tlsConfig)
	if err != nil {
		return nil, err
	}
	if err := conn.Close(); err != nil {
		return nil, err
	}

	return acceptableCAs, nil
}

// GetClientCANamesForURL is GetClientCANames against a URL string like we use in kubeconfigs
func GetClientCANamesForURL(kubeConfigURL string) ([]string, error) {
	apiserverURL, err := url.Parse(kubeConfigURL)
	if err != nil {
		return nil, err
	}
	return GetClientCANames(apiserverURL.Host)
}

// GetServingCertificates returns the x509 certs used by a server as certificates and pem encoded bytes.
// The serverName is optional for specifying a different name to get SNI certificates.  apiHost is "host:port"
func GetServingCertificates(apiHost, serverName string) ([]*x509.Certificate, [][]byte, error) {
	tlsConfig := &tls.Config{
		InsecureSkipVerify: true, // this is insecure so that we always get connected
	}
	// if a name is specified for SNI, set it.
	if len(serverName) > 0 {
		tlsConfig.ServerName = serverName
	}

	conn, err := tls.Dial("tcp", apiHost, tlsConfig)
	if err != nil {
		return nil, nil, err
	}
	if err = conn.Close(); err != nil {
		return nil, nil, fmt.Errorf("failed to close connection : %v", err)
	}

	peerCerts := conn.ConnectionState().PeerCertificates
	peerCertBytes := [][]byte{}
	for _, a := range peerCerts {
		actualCert, err := EncodeCertificates(a)
		if err != nil {
			return nil, nil, err
		}
		peerCertBytes = append(peerCertBytes, []byte(strings.TrimSpace(string(actualCert))))
	}

	return peerCerts, peerCertBytes, err
}

// GetServingCertificatesForURL is GetServingCertificates against a URL string like we use in kubeconfigs
func GetServingCertificatesForURL(kubeConfigURL, serverName string) ([]*x509.Certificate, [][]byte, error) {
	apiserverURL, err := url.Parse(kubeConfigURL)
	if err != nil {
		return nil, nil, err
	}
	return GetServingCertificates(apiserverURL.Host, serverName)
}
