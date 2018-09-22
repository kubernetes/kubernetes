/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/golang/glog"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/third_party/forked/golang/netutil"
)

// DialRequest implements proxy support, see spdy:
// https://github.com/kubernetes/kubernetes/blob/5e8e570dbda6ed89af9bc2e0a05e3d94bfdfcb61/staging/src/k8s.io/apimachinery/pkg/util/httpstream/spdy/roundtripper.go#L111
func DialRequest(req *http.Request, transport http.RoundTripper) (net.Conn, error) {
	glog.V(5).Info("received dial request")
	ctx := req.Context()
	proxier := utilnet.NewProxierWithNoProxyCIDR(http.ProxyFromEnvironment)
	proxyURL, err := proxier(req)
	if err != nil {
		return nil, err
	}

	if proxyURL == nil {
		glog.V(5).Info("Dialing without proxy")
		return DialURLWithoutProxy(ctx, req.URL, transport)
	}

	targetHost := netutil.CanonicalAddr(req.URL)
	glog.V(5).Infof("dialing with proxy %s to target: %s", proxyURL, targetHost)

	proxyReq := http.Request{
		Method: "CONNECT",
		URL:    &url.URL{},
		Host:   targetHost,
	}

	if pa := proxyAuth(proxyURL); pa != "" {
		proxyReq.Header = http.Header{}
		proxyReq.Header.Set("Proxy-Authorization", pa)
	}

	proxyDialConn, err := DialURLWithoutProxy(ctx, proxyURL, transport)
	if err != nil {
		return nil, err
	}

	proxyClientConn := httputil.NewProxyClientConn(proxyDialConn, nil)
	glog.V(5).Infof("proxy request: %#v", proxyReq)
	_, err = proxyClientConn.Do(&proxyReq)
	if err != nil && err != httputil.ErrPersistEOF {
		return nil, err
	}

	glog.V(5).Infof("hijacking request: %#v", proxyClientConn)
	rwc, _ := proxyClientConn.Hijack()

	if req.URL.Scheme != "https" {
		return rwc, nil
	}

	host, _, err := net.SplitHostPort(targetHost)
	if err != nil {
		return nil, err
	}

	tlsConfig, err := utilnet.TLSClientConfig(transport)
	if err != nil {
		glog.V(5).Infof("Unable to unwrap transport %T to get at TLS config: %v", transport, err)
	}

	switch {
	case tlsConfig == nil:
		tlsConfig = &tls.Config{ServerName: host}
	case len(tlsConfig.ServerName) == 0:
		tlsConfig = tlsConfig.Clone()
		tlsConfig.ServerName = host
	}

	tlsConn := tls.Client(rwc, tlsConfig)

	// need to manually call Handshake() so we can call VerifyHostname() below
	if err := tlsConn.Handshake(); err != nil {
		return nil, err
	}

	// Return if we were configured to skip validation
	if tlsConfig.InsecureSkipVerify {
		return tlsConn, nil
	}

	if err := tlsConn.VerifyHostname(tlsConfig.ServerName); err != nil {
		return nil, err
	}

	return tlsConn, nil
}

// see proxyAuth from spdy
// https://github.com/kubernetes/kubernetes/blob/5e8e570dbda6ed89af9bc2e0a05e3d94bfdfcb61/staging/src/k8s.io/apimachinery/pkg/util/httpstream/spdy/roundtripper.go#L236
func proxyAuth(proxyURL *url.URL) string {
	if proxyURL == nil || proxyURL.User == nil {
		return ""
	}
	credentials := proxyURL.User.String()
	encodedAuth := base64.StdEncoding.EncodeToString([]byte(credentials))
	return fmt.Sprintf("Basic %s", encodedAuth)
}

func DialURLWithoutProxy(ctx context.Context, url *url.URL, transport http.RoundTripper) (net.Conn, error) {
	dialAddr := netutil.CanonicalAddr(url)

	dialer, err := utilnet.DialerFor(transport)
	if err != nil {
		glog.V(5).Infof("Unable to unwrap transport %T to get dialer: %v", transport, err)
	}

	switch url.Scheme {
	case "http":
		if dialer != nil {
			return dialer(ctx, "tcp", dialAddr)
		}
		var d net.Dialer
		return d.DialContext(ctx, "tcp", dialAddr)
	case "https":
		// Get the tls config from the transport if we recognize it
		var tlsConfig *tls.Config
		var tlsConn *tls.Conn
		var err error
		tlsConfig, err = utilnet.TLSClientConfig(transport)
		if err != nil {
			glog.V(5).Infof("Unable to unwrap transport %T to get at TLS config: %v", transport, err)
		}

		if dialer != nil {
			// We have a dialer; use it to open the connection, then
			// create a tls client using the connection.
			netConn, err := dialer(ctx, "tcp", dialAddr)
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
				tlsConfigCopy := tlsConfig.Clone()
				tlsConfigCopy.ServerName = inferredHost
				tlsConfig = tlsConfigCopy
			}
			tlsConn = tls.Client(netConn, tlsConfig)
			if err := tlsConn.Handshake(); err != nil {
				netConn.Close()
				return nil, err
			}

		} else {
			// Dial. This Dial method does not allow to pass a context unfortunately
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
		if tlsConfig != nil && len(tlsConfig.ServerName) > 0 {
			host = tlsConfig.ServerName
		}
		if err := tlsConn.VerifyHostname(host); err != nil {
			tlsConn.Close()
			return nil, err
		}

		return tlsConn, nil
	default:
		return nil, fmt.Errorf("Unknown scheme: %s", url.Scheme)
	}
}
