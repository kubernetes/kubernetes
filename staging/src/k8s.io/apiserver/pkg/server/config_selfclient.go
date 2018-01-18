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

	host, port, err := LoopbackHostPort(s.Listener.Addr().String())
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

// LoopbackHostPort returns the host and port loopback REST clients should use
// to contact the server.
func LoopbackHostPort(bindAddress string) (string, string, error) {
	host, port, err := net.SplitHostPort(bindAddress)
	if err != nil {
		// should never happen
		return "", "", fmt.Errorf("invalid server bind address: %q", bindAddress)
	}

	// Value is expected to be an IP or DNS name, not "0.0.0.0".
	if host == "0.0.0.0" || host == "::" {
		host = "localhost"
		// Get ip of local interface, but fall back to "localhost".
		// Note that "localhost" is resolved with the external nameserver first with Go's stdlib.
		// So if localhost.<yoursearchdomain> resolves, we don't get a 127.0.0.1 as expected.
		addrs, err := net.InterfaceAddrs()
		if err == nil {
			for _, address := range addrs {
				if ipnet, ok := address.(*net.IPNet); ok && ipnet.IP.IsLoopback() {
					host = ipnet.IP.String()
					break
				}
			}
		}
	}
	return host, port, nil
}
