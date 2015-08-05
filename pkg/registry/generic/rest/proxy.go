/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package rest

import (
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/proxy"

	"k8s.io/kubernetes/third_party/golang/netutil"
	"github.com/golang/glog"
	"github.com/mxk/go-flowrate/flowrate"
)

// UpgradeAwareProxyHandler is a handler for proxy requests that may require an upgrade
type UpgradeAwareProxyHandler struct {
	UpgradeRequired bool
	Location        *url.URL
	Transport       http.RoundTripper
	FlushInterval   time.Duration
	MaxBytesPerSec  int64
	err             error
}

const defaultFlushInterval = 200 * time.Millisecond

// NewUpgradeAwareProxyHandler creates a new proxy handler with a default flush interval
func NewUpgradeAwareProxyHandler(location *url.URL, transport http.RoundTripper, upgradeRequired bool) *UpgradeAwareProxyHandler {
	return &UpgradeAwareProxyHandler{
		Location:        location,
		Transport:       transport,
		UpgradeRequired: upgradeRequired,
		FlushInterval:   defaultFlushInterval,
	}
}

// RequestError returns an error that occurred while handling request
func (h *UpgradeAwareProxyHandler) RequestError() error {
	return h.err
}

// ServeHTTP handles the proxy request
func (h *UpgradeAwareProxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	h.err = nil
	if len(h.Location.Scheme) == 0 {
		h.Location.Scheme = "http"
	}
	if h.tryUpgrade(w, req) {
		return
	}
	if h.UpgradeRequired {
		h.err = errors.NewBadRequest("Upgrade request required")
		return
	}

	loc := *h.Location
	loc.RawQuery = req.URL.RawQuery

	// If original request URL ended in '/', append a '/' at the end of the
	// of the proxy URL
	if !strings.HasSuffix(loc.Path, "/") && strings.HasSuffix(req.URL.Path, "/") {
		loc.Path += "/"
	}

	// From pkg/apiserver/proxy.go#ServeHTTP:
	// Redirect requests with an empty path to a location that ends with a '/'
	// This is essentially a hack for https://github.com/GoogleCloudPlatform/kubernetes/issues/4958.
	// Note: Keep this code after tryUpgrade to not break that flow.
	if len(loc.Path) == 0 {
		var queryPart string
		if len(req.URL.RawQuery) > 0 {
			queryPart = "?" + req.URL.RawQuery
		}
		w.Header().Set("Location", req.URL.Path+"/"+queryPart)
		w.WriteHeader(http.StatusMovedPermanently)
		return
	}

	if h.Transport == nil {
		h.Transport = h.defaultProxyTransport(req.URL)
	}

	newReq, err := http.NewRequest(req.Method, loc.String(), req.Body)
	if err != nil {
		h.err = err
		return
	}
	newReq.Header = req.Header

	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: h.Location.Scheme, Host: h.Location.Host})
	proxy.Transport = h.Transport
	proxy.FlushInterval = h.FlushInterval
	proxy.ServeHTTP(w, newReq)
}

// tryUpgrade returns true if the request was handled.
func (h *UpgradeAwareProxyHandler) tryUpgrade(w http.ResponseWriter, req *http.Request) bool {
	if !httpstream.IsUpgradeRequest(req) {
		return false
	}

	backendConn, err := h.dialURL()
	if err != nil {
		h.err = err
		return true
	}
	defer backendConn.Close()

	requestHijackedConn, _, err := w.(http.Hijacker).Hijack()
	if err != nil {
		h.err = err
		return true
	}
	defer requestHijackedConn.Close()

	newReq, err := http.NewRequest(req.Method, h.Location.String(), req.Body)
	if err != nil {
		h.err = err
		return true
	}
	newReq.Header = req.Header

	if err = newReq.Write(backendConn); err != nil {
		h.err = err
		return true
	}

	wg := &sync.WaitGroup{}
	wg.Add(2)

	go func() {
		var writer io.WriteCloser
		if h.MaxBytesPerSec > 0 {
			writer = flowrate.NewWriter(backendConn, h.MaxBytesPerSec)
		} else {
			writer = backendConn
		}
		_, err := io.Copy(writer, requestHijackedConn)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error proxying data from client to backend: %v", err)
		}
		wg.Done()
	}()

	go func() {
		var reader io.ReadCloser
		if h.MaxBytesPerSec > 0 {
			reader = flowrate.NewReader(backendConn, h.MaxBytesPerSec)
		} else {
			reader = backendConn
		}
		_, err := io.Copy(requestHijackedConn, reader)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error proxying data from backend to client: %v", err)
		}
		wg.Done()
	}()

	wg.Wait()
	return true
}

func (h *UpgradeAwareProxyHandler) dialURL() (net.Conn, error) {
	dialAddr := netutil.CanonicalAddr(h.Location)

	var dialer func(network, addr string) (net.Conn, error)
	if httpTransport, ok := h.Transport.(*http.Transport); ok && httpTransport.Dial != nil {
		dialer = httpTransport.Dial
	}

	switch h.Location.Scheme {
	case "http":
		if dialer != nil {
			return dialer("tcp", dialAddr)
		}
		return net.Dial("tcp", dialAddr)
	case "https":
		// TODO: this TLS logic can probably be cleaned up; it's messy in an attempt
		// to preserve behavior that we don't know for sure is exercised.

		// Get the tls config from the transport if we recognize it
		var tlsConfig *tls.Config
		var tlsConn *tls.Conn
		var err error
		if h.Transport != nil {
			httpTransport, ok := h.Transport.(*http.Transport)
			if ok {
				tlsConfig = httpTransport.TLSClientConfig
			}
		}
		if dialer != nil {
			// We have a dialer; use it to open the connection, then
			// create a tls client using the connection.
			netConn, err := dialer("tcp", dialAddr)
			if err != nil {
				return nil, err
			}
			// tls.Client requires non-nil config
			if tlsConfig == nil {
				glog.Warningf("using custom dialer with no TLSClientConfig. Defaulting to InsecureSkipVerify")
				tlsConfig = &tls.Config{
					InsecureSkipVerify: true,
				}
			}
			tlsConn = tls.Client(netConn, tlsConfig)
			if err := tlsConn.Handshake(); err != nil {
				return nil, err
			}

		} else {
			// Dial
			tlsConn, err = tls.Dial("tcp", dialAddr, tlsConfig)
			if err != nil {
				return nil, err
			}
		}

		// Verify
		host, _, _ := net.SplitHostPort(dialAddr)
		if err := tlsConn.VerifyHostname(host); err != nil {
			tlsConn.Close()
			return nil, err
		}

		return tlsConn, nil
	default:
		return nil, fmt.Errorf("unknown scheme: %s", h.Location.Scheme)
	}
}

func (h *UpgradeAwareProxyHandler) defaultProxyTransport(url *url.URL) http.RoundTripper {
	scheme := url.Scheme
	host := url.Host
	suffix := h.Location.Path
	if strings.HasSuffix(url.Path, "/") && !strings.HasSuffix(suffix, "/") {
		suffix += "/"
	}
	pathPrepend := strings.TrimSuffix(url.Path, suffix)
	internalTransport := &proxy.Transport{
		Scheme:      scheme,
		Host:        host,
		PathPrepend: pathPrepend,
	}
	return &corsRemovingTransport{
		RoundTripper: internalTransport,
	}
}

// corsRemovingTransport is a wrapper for an internal transport. It removes CORS headers
// from the internal response.
type corsRemovingTransport struct {
	http.RoundTripper
}

func (p *corsRemovingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := p.RoundTripper.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	removeCORSHeaders(resp)
	return resp, nil

}

// removeCORSHeaders strip CORS headers sent from the backend
// This should be called on all responses before returning
func removeCORSHeaders(resp *http.Response) {
	resp.Header.Del("Access-Control-Allow-Credentials")
	resp.Header.Del("Access-Control-Allow-Headers")
	resp.Header.Del("Access-Control-Allow-Methods")
	resp.Header.Del("Access-Control-Allow-Origin")
}
