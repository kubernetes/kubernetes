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
	if cfg, err := secureServingInfo.NewSelfClientConfig(token); err != nil || cfg != nil {
		if insecureServingInfo == nil {
			// be fatal if insecure port is not available
			return cfg, err
		} else {
			glog.Warningf("Failed to create secure local client, falling back to insecure local connection: %v", err)
		}
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

	// find certificate for host: either explicitly given, from the server cert bundle or one of the SNI certs
	var derCA []byte
	if s.CACert != nil {
		derCA = s.CACert.Certificate[0]
	}
	if derCA == nil && s.Cert != nil {
		x509Cert, err := x509.ParseCertificate(s.Cert.Certificate[0])
		if err != nil {
			return nil, fmt.Errorf("failed to parse server certificate: %v", err)
		}

		if (net.ParseIP(host) != nil && certMatchesIP(x509Cert, host)) || certMatchesName(x509Cert, host) {
			derCA = s.Cert.Certificate[0]
		}
	}
	if derCA == nil && net.ParseIP(host) == nil {
		if cert, found := s.SNICerts[host]; found {
			derCA = cert.Certificate[0]
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
