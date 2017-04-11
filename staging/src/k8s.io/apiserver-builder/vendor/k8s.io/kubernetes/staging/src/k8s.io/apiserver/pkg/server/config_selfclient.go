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

package server

import (
	"crypto/x509"
	"fmt"
	"net"

	restclient "k8s.io/client-go/rest"
)

// LoopbackClientServerNameOverride is passed to the apiserver from the loopback client in order to
// select the loopback certificate via SNI if TLS is used.
const LoopbackClientServerNameOverride = "apiserver-loopback-client"

func (s *SecureServingInfo) NewLoopbackClientConfig(token string, loopbackCert []byte) (*restclient.Config, error) {
	if s == nil || (s.Cert == nil && len(s.SNICerts) == 0) {
		return nil, nil
	}

	host, port, err := LoopbackHostPort(s.BindAddress)
	if err != nil {
		return nil, err
	}

	return &restclient.Config{
		// Increase QPS limits. The client is currently passed to all admission plugins,
		// and those can be throttled in case of higher load on apiserver - see #22340 and #22422
		// for more details. Once #22422 is fixed, we may want to remove it.
		QPS:         50,
		Burst:       100,
		Host:        "https://" + net.JoinHostPort(host, port),
		BearerToken: token,
		// override the ServerName to select our loopback certificate via SNI. This name is also
		// used by the client to compare the returns server certificate against.
		TLSClientConfig: restclient.TLSClientConfig{
			ServerName: LoopbackClientServerNameOverride,
			CAData:     loopbackCert,
		},
	}, nil
}

func trustedChain(chain []*x509.Certificate) bool {
	intermediates := x509.NewCertPool()
	for _, cert := range chain[1:] {
		intermediates.AddCert(cert)
	}
	_, err := chain[0].Verify(x509.VerifyOptions{
		Intermediates: intermediates,
		KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	})
	return err == nil
}

func parseChain(bss [][]byte) ([]*x509.Certificate, error) {
	var result []*x509.Certificate
	for _, bs := range bss {
		x509Cert, err := x509.ParseCertificate(bs)
		if err != nil {
			return nil, err
		}
		result = append(result, x509Cert)
	}

	return result, nil
}

func findCA(chain []*x509.Certificate) (*x509.Certificate, error) {
	for _, cert := range chain {
		if cert.IsCA {
			return cert, nil
		}
	}

	return nil, fmt.Errorf("no certificate with CA:TRUE found in chain")
}

// LoopbackHostPort returns the host and port loopback REST clients should use
// to contact the server.
func LoopbackHostPort(bindAddress string) (string, string, error) {
	host, port, err := net.SplitHostPort(bindAddress)
	if err != nil {
		// should never happen
		return "", "", fmt.Errorf("invalid server bind address: %q", bindAddress)
	}

	// Value is expected to be an IP or DNS name, not "0.0.0.0".
	if host == "0.0.0.0" {
		// compare MaybeDefaultWithSelfSignedCerts which adds "localhost" to the cert as alternateDNS
		host = "localhost"
	}
	return host, port, nil
}

func certMatchesName(cert *x509.Certificate, name string) bool {
	for _, certName := range cert.DNSNames {
		if certName == name {
			return true
		}
	}

	return false
}

func certMatchesIP(cert *x509.Certificate, ip string) bool {
	for _, certIP := range cert.IPAddresses {
		if certIP.String() == ip {
			return true
		}
	}

	return false
}
