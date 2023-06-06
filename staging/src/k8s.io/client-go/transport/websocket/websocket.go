/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"net/http"

	gwebsocket "github.com/gorilla/websocket"

	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
)

// Header to identify this WebSocket client can do stream translation.
const (
	HeaderWebSocketKey             = "K8s-Websocket-Protocol"
	HeaderWebSocketStreamTranslate = "stream-translate"
)

func RoundTripperFor(config *restclient.Config) (http.RoundTripper, *RoundTripper, error) {
	transportCfg, err := config.TransportConfig()
	if err != nil {
		return nil, nil, err
	}
	tlsConfig, err := transport.TLSConfigFor(transportCfg)
	if err != nil {
		return nil, nil, err
	}
	proxy := config.Proxy
	if proxy == nil {
		proxy = utilnet.NewProxierWithNoProxyCIDR(http.ProxyFromEnvironment)
	}

	upgradeRoundTripper := &RoundTripper{
		TLSConfig: tlsConfig,
		Proxier:   proxy,
	}
	wrapper, err := transport.HTTPWrappersForConfig(transportCfg, upgradeRoundTripper)
	if err != nil {
		return nil, nil, err
	}
	return wrapper, upgradeRoundTripper, nil
}

// Negotiate opens a connection to a remote server and attempts to negotiate
// a WebSocket connection. Upon success, it returns the negotiated connection.
// The round tripper rt must use the WebSocket round tripper wsRt - see RoundTripperFor.
func Negotiate(rt http.RoundTripper, wsRt *RoundTripper, req *http.Request, protocols ...string) (*gwebsocket.Conn, error) {
	req.Header[HeaderWebSocketKey] = []string{HeaderWebSocketStreamTranslate}
	req.Header[httpstream.HeaderProtocolVersion] = protocols
	resp, err := rt.RoundTrip(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %v", err)
	}
	err = resp.Body.Close()
	if err != nil {
		wsRt.Conn.Close()
		return nil, fmt.Errorf("error closing response body: %v", err)
	}
	return wsRt.Conn, nil
}
