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
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/portforward"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport/websocket"
	"k8s.io/klog/v2"
)

// wsDialer implements "httpstream.Dial" interface
type wsDialer struct {
	url       *url.URL
	transport http.RoundTripper
	holder    websocket.ConnectionHolder
	ports     []string
}

// NewWebSocketDialer creates and returns the wsDialer structure which implemements the "httpstream.Dialer"
// interface. The dialer can upgrade a websocket request, creating a websocket connection. This function
// returns an error if one occurs.
func NewWebSocketDialer(url *url.URL, config *restclient.Config, ports []string) (httpstream.Dialer, error) {
	if len(ports) == 0 {
		return nil, fmt.Errorf("you must specify at least 1 port")
	}
	transport, holder, err := websocket.RoundTripperFor(config)
	if err != nil {
		return nil, err
	}
	return &wsDialer{
		url:       url,
		transport: transport,
		holder:    holder,
		ports:     ports,
	}, nil
}

// Dial upgrades a websocket request, returning a websocket connection (wrapped
// by an "httpstream.Connection"), the negotiated protocol, or an error if one occurred.
func (d *wsDialer) Dial(protocols ...string) (httpstream.Connection, string, error) {
	// There is no passed context, so skip the context when creating request for now.
	// Websockets requires "GET" method: RFC 6455 Sec. 4.1 (page 17).
	req, err := http.NewRequest("GET", d.url.String(), nil)
	if err != nil {
		return nil, "", err
	}
	// Add the port(s) as request query params.
	forwardedPorts, err := parsePorts(d.ports)
	if err != nil {
		return nil, "", err
	}
	for _, port := range forwardedPorts {
		query := req.URL.Query()
		remotePort := int(port.Remote)
		klog.V(4).Infof("Remote Port: %d", remotePort)
		query.Set("ports", strconv.Itoa(remotePort))
		req.URL.RawQuery = query.Encode()
	}
	// Hard-code the v2 portforward protocol for the websocket dialer for now.
	websocketProtocols := []string{portforward.PortForwardV2Name}
	klog.V(4).Infoln("Before WebSocket Upgrade Connection...")
	conn, err := websocket.Negotiate(d.transport, d.holder, req, websocketProtocols...)
	if err != nil {
		return nil, "", err
	}
	if conn == nil {
		return nil, "", fmt.Errorf("negotiated websocket connection is nil")
	}
	protocol := conn.Subprotocol()
	klog.V(4).Infof("negotiated protocol: %s", protocol)

	// Encapsulate the websocket connection and start the reading loop. This client
	// endpoint does not throttle stream reading/writing.
	wsConn := NewWebsocketConnection(conn, 0)
	go func() {
		heartbeatPeriod := 5 * time.Second
		heartbeatDeadline := heartbeatPeriod * 5
		// nil channel means this endpoint does not handle the "StreamCreate" signal.
		wsConn.Start(nil, BufferSize, heartbeatPeriod, heartbeatDeadline)
	}()

	return wsConn, protocol, nil
}
