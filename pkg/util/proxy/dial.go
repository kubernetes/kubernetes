/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package proxy

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/url"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/third_party/golang/netutil"
)

func DialURL(url *url.URL, transport http.RoundTripper) (net.Conn, error) {
	dialAddr := netutil.CanonicalAddr(url)

	dialer, _ := util.Dialer(transport)

	switch url.Scheme {
	case "http":
		if dialer != nil {
			return dialer("tcp", dialAddr)
		}
		return net.Dial("tcp", dialAddr)
	case "https":
		// Get the tls config from the transport if we recognize it
		var tlsConfig *tls.Config
		var tlsConn *tls.Conn
		var err error
		tlsConfig, _ = util.TLSClientConfig(transport)

		if dialer != nil {
			// We have a dialer; use it to open the connection, then
			// create a tls client using the connection.
			netConn, err := dialer("tcp", dialAddr)
			if err != nil {
				return nil, err
			}
			if tlsConfig == nil {
				// tls.Client requires non-nil config
				glog.Warningf("using custom dialer with no TLSClientConfig. Defaulting to InsecureSkipVerify")
				// tls.Handshake() requires ServerName or InsecureSkipVerify
				tlsConfig = &tls.Config{
					InsecureSkipVerify: true,
				}
			} else if len(tlsConfig.ServerName) == 0 && !tlsConfig.InsecureSkipVerify {
				// tls.Handshake() requires ServerName or InsecureSkipVerify
				// infer the ServerName from the hostname we're connecting to.
				inferredHost := dialAddr
				if host, _, err := net.SplitHostPort(dialAddr); err == nil {
					inferredHost = host
				}
				// Make a copy to avoid polluting the provided config
				tlsConfigCopy := *tlsConfig
				tlsConfigCopy.ServerName = inferredHost
				tlsConfig = &tlsConfigCopy
			}
			tlsConn = tls.Client(netConn, tlsConfig)
			if err := tlsConn.Handshake(); err != nil {
				netConn.Close()
				return nil, err
			}

		} else {
			// Dial
			tlsConn, err = tls.Dial("tcp", dialAddr, tlsConfig)
			if err != nil {
				return nil, err
			}
		}

		// Return if we were configured to skip validation
		if tlsConfig != nil && tlsConfig.InsecureSkipVerify {
			return tlsConn, nil
		}

		// Verify
		host, _, _ := net.SplitHostPort(dialAddr)
		if err := tlsConn.VerifyHostname(host); err != nil {
			tlsConn.Close()
			return nil, err
		}

		return tlsConn, nil
	default:
		return nil, fmt.Errorf("Unknown scheme: %s", url.Scheme)
	}
}
