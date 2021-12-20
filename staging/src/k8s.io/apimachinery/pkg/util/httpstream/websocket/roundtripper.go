/*
Copyright 2020 The Kubernetes Authors.

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

package websocket

import (
	"crypto/tls"
	"github.com/gorilla/websocket"
	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"net/http"
	"net/url"
)

// RoundTripper stores dialer information and knows how
// to establish a connect to the remote websocket endpoint.  WebsocketRoundTripper
// implements the UpgradeRoundTripper interface.
type RoundTripper struct {
	http.RoundTripper
	//tlsConfig holds the TLS configuration settings to use when connecting
	//to the remote server.
	tlsConfig *tls.Config

	// websocket connection
	Conn *websocket.Conn

	// proxier knows which proxy to use given a request, defaults to http.ProxyFromEnvironment
	// Used primarily for mocking the proxy discovery in tests.
	proxier func(req *http.Request) (*url.URL, error)

	// followRedirects indicates if the round tripper should examine responses for redirects and
	// follow them.
	followRedirects bool
	// requireSameHostRedirects restricts redirect following to only follow redirects to the same host
	// as the original request.
	requireSameHostRedirects bool
}

var _ utilnet.TLSClientConfigHolder = &RoundTripper{}
var _ httpstream.UpgradeRoundTripper = &RoundTripper{}

// NewRoundTripper creates a new WsRoundTripper that will use the specified
// tlsConfig.
func NewRoundTripper(tlsConfig *tls.Config, followRedirects, requireSameHostRedirects bool) *RoundTripper {
	return NewRoundTripperWithProxy(tlsConfig, followRedirects, requireSameHostRedirects, utilnet.NewProxierWithNoProxyCIDR(http.ProxyFromEnvironment))
}

// NewRoundTripperWithProxy creates a new WsRoundTripper that will use the
// specified tlsConfig and proxy func.
func NewRoundTripperWithProxy(tlsConfig *tls.Config, followRedirects, requireSameHostRedirects bool, proxier func(*http.Request) (*url.URL, error)) *RoundTripper {
	return &RoundTripper{
		tlsConfig:                tlsConfig,
		followRedirects:          followRedirects,
		requireSameHostRedirects: requireSameHostRedirects,
		proxier:                  proxier,
	}
}

// TLSClientConfig implements pkg/util/net.TLSClientConfigHolder for proper TLS checking during
// proxying with a spdy roundtripper.
func (s *RoundTripper) TLSClientConfig() *tls.Config {
	return s.tlsConfig
}

// // proxyAuth returns, for a given proxy URL, the value to be used for the Proxy-Authorization header
// func (s *RoundTripper) proxyAuth(proxyURL *url.URL) string {
// 	if proxyURL == nil || proxyURL.User == nil {
// 		return ""
// 	}
// 	credentials := proxyURL.User.String()
// 	encodedAuth := base64.StdEncoding.EncodeToString([]byte(credentials))
// 	return fmt.Sprintf("Basic %s", encodedAuth)
// }

// NewConnection doesn't do anything right now
func (wsRoundTripper *RoundTripper) NewConnection(resp *http.Response) (httpstream.Connection, error) {
	return &Connection{Conn: wsRoundTripper.Conn}, nil
}

// RoundTrip connects to the remote websocket using the headers in the request and the TLS
// configuration from the config
func (wsRoundTripper *RoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {

	// set the protocol version directly on the dialer from the header
	protocolVersions := request.Header[httpstream.HeaderProtocolVersion]

	// there's no need for the headers for the protocol version anymore
	if protocolVersions != nil {
		request.Header.Del((httpstream.HeaderProtocolVersion))
	}

	// create a dialer
	// TODO: add proxy support
	dialer := websocket.Dialer{
		TLSClientConfig: wsRoundTripper.tlsConfig,
		Subprotocols:    protocolVersions,
		Proxy:           wsRoundTripper.proxier,
	}

	wsCon, resp, err := dialer.Dial(request.URL.String(), request.Header)

	if err != nil {
		return nil, err
	}

	// for safe keeping
	wsRoundTripper.Conn = wsCon

	return resp, nil
}
