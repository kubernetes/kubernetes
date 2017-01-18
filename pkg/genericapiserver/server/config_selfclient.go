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

package genericapiserver

import (
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"net"

	"k8s.io/kubernetes/pkg/client/restclient"

	"github.com/golang/glog"
)

// NewSelfClientConfig returns a clientconfig which can be used to talk to this apiserver.
func NewSelfClientConfig(secureServingInfo *SecureServingInfo, insecureServingInfo *ServingInfo, token string) (*restclient.Config, error) {
	cfg, err := secureServingInfo.NewSelfClientConfig(token)
	if cfg != nil && err == nil {
		return cfg, nil
	}
	if err != nil {
		if insecureServingInfo == nil {
			// be fatal if insecure port is not available
			return nil, err
		}

		glog.Warningf("Failed to create secure local client, falling back to insecure local connection: %v", err)
	}
	if cfg, err := insecureServingInfo.NewSelfClientConfig(token); err != nil || cfg != nil {
		return cfg, err
	}

	return nil, errors.New("Unable to set url for apiserver local client")
}

func (s *SecureServingInfo) NewSelfClientConfig(token string) (*restclient.Config, error) {
	if s == nil || (s.Cert == nil && len(s.SNICerts) == 0) {
		return nil, nil
	}

	host, port, err := net.SplitHostPort(s.ServingInfo.BindAddress)
	if err != nil {
		// should never happen
		return nil, fmt.Errorf("invalid secure bind address: %q", s.ServingInfo.BindAddress)
	}
	if host == "0.0.0.0" {
		// compare MaybeDefaultWithSelfSignedCerts which adds "localhost" to the cert as alternateDNS
		host = "localhost"
	}

	clientConfig := &restclient.Config{
		// Increase QPS limits. The client is currently passed to all admission plugins,
		// and those can be throttled in case of higher load on apiserver - see #22340 and #22422
		// for more details. Once #22422 is fixed, we may want to remove it.
		QPS:         50,
		Burst:       100,
		Host:        "https://" + net.JoinHostPort(host, port),
		BearerToken: token,
	}

	// find certificate for host: either explicitly given, from the server cert bundle or one of the SNI certs,
	// but only return CA:TRUE certificates.
	var derCA []byte
	if s.CACert != nil {
		derCA = s.CACert.Certificate[0]
	}
	if derCA == nil && net.ParseIP(host) == nil {
		if cert, found := s.SNICerts[host]; found {
			chain, err := parseChain(cert.Certificate)
			if err != nil {
				return nil, fmt.Errorf("failed to parse SNI certificate for host %q: %v", host, err)
			}

			if trustedChain(chain) {
				return clientConfig, nil
			}

			ca, err := findCA(chain)
			if err != nil {
				return nil, fmt.Errorf("no CA certificate found in SNI server certificate bundle for host %q: %v", host, err)
			}
			derCA = ca.Raw
		}
	}
	if derCA == nil && s.Cert != nil {
		chain, err := parseChain(s.Cert.Certificate)
		if err != nil {
			return nil, fmt.Errorf("failed to parse server certificate: %v", err)
		}

		if (net.ParseIP(host) != nil && certMatchesIP(chain[0], host)) || certMatchesName(chain[0], host) {
			if trustedChain(chain) {
				return clientConfig, nil
			}

			ca, err := findCA(chain)
			if err != nil {
				return nil, fmt.Errorf("no CA certificate found in server certificate bundle: %v", err)
			}
			derCA = ca.Raw
		}
	}
	if derCA == nil {
		return nil, fmt.Errorf("failed to find certificate which matches %q", host)
	}
	pemCA := bytes.Buffer{}
	if err := pem.Encode(&pemCA, &pem.Block{Type: "CERTIFICATE", Bytes: derCA}); err != nil {
		return nil, err
	}
	clientConfig.CAData = pemCA.Bytes()

	return clientConfig, nil
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

func (s *ServingInfo) NewSelfClientConfig(token string) (*restclient.Config, error) {
	if s == nil {
		return nil, nil
	}
	return &restclient.Config{
		Host: s.BindAddress,
		// Increase QPS limits. The client is currently passed to all admission plugins,
		// and those can be throttled in case of higher load on apiserver - see #22340 and #22422
		// for more details. Once #22422 is fixed, we may want to remove it.
		QPS:   50,
		Burst: 100,
	}, nil
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
