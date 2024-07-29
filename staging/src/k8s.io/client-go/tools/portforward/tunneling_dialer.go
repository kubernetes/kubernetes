/*
Copyright 2023 The Kubernetes Authors.

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

package portforward

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	constants "k8s.io/apimachinery/pkg/util/portforward"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport/websocket"
	"k8s.io/klog/v2"
)

const PingPeriod = 10 * time.Second

// tunnelingDialer implements "httpstream.Dial" interface
type tunnelingDialer struct {
	ctx       context.Context
	url       *url.URL
	transport http.RoundTripper
	holder    websocket.ConnectionHolder
}

// NewTunnelingDialer creates and returns the tunnelingDialer structure which implemements the "httpstream.Dialer"
// interface. The dialer can upgrade a websocket request, creating a websocket connection. This function
// returns an error if one occurs.
//
// TODO (https://github.com/kubernetes/kubernetes/issues/126379): logcheck:context // NewSPDYOverWebsocketDialerWithContext should be used instead of NewSPDYOverWebsocketDialer in code which supports contextual logging.
func NewSPDYOverWebsocketDialer(url *url.URL, config *restclient.Config) (httpstream.Dialer, error) {
	return NewSPDYOverWebsocketDialerWithContext(context.Background(), url, config)
}

// NewTunnelingDialerWithContext creates and returns the tunnelingDialer structure which implemements the "httpstream.Dialer"
// interface. The dialer can upgrade a websocket request, creating a websocket connection. This function
// returns an error if one occurs.
func NewSPDYOverWebsocketDialerWithContext(ctx context.Context, url *url.URL, config *restclient.Config) (httpstream.Dialer, error) {
	transport, holder, err := websocket.RoundTripperFor(config)
	if err != nil {
		return nil, err
	}
	return &tunnelingDialer{
		ctx:       ctx,
		url:       url,
		transport: transport,
		holder:    holder,
	}, nil
}

// Dial upgrades to a tunneling streaming connection, returning a SPDY connection
// containing a WebSockets connection (which implements "net.Conn"). Also
// returns the protocol negotiated, or an error.
func (d *tunnelingDialer) Dial(protocols ...string) (httpstream.Connection, string, error) {
	// Websockets requires "GET" method: RFC 6455 Sec. 4.1 (page 17).
	req, err := http.NewRequestWithContext(d.ctx, "GET", d.url.String(), nil)
	if err != nil {
		return nil, "", err
	}
	logger := klog.FromContext(d.ctx)
	// Add the spdy tunneling prefix to the requested protocols. The tunneling
	// handler will know how to negotiate these protocols.
	tunnelingProtocols := []string{}
	for _, protocol := range protocols {
		tunnelingProtocol := constants.WebsocketsSPDYTunnelingPrefix + protocol
		tunnelingProtocols = append(tunnelingProtocols, tunnelingProtocol)
	}
	logger.V(4).Info("Before WebSocket Upgrade Connection...")
	conn, err := websocket.Negotiate(d.transport, d.holder, req, tunnelingProtocols...)
	if err != nil {
		return nil, "", err
	}
	if conn == nil {
		return nil, "", fmt.Errorf("negotiated websocket connection is nil")
	}
	protocol := conn.Subprotocol()
	protocol = strings.TrimPrefix(protocol, constants.WebsocketsSPDYTunnelingPrefix)
	logger.V(4).Info("Protocol negotiated completed", "protocol", protocol)

	// Wrap the websocket connection which implements "net.Conn".
	tConn := NewTunnelingConnectionWithContext(d.ctx, "client", conn)
	// Create SPDY connection injecting the previously created tunneling connection.
	spdyConn, err := spdy.NewClientConnectionWithPings(tConn, PingPeriod)

	return spdyConn, protocol, err
}
