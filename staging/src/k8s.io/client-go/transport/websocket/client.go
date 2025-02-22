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

package websocket

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	gwebsocket "github.com/gorilla/websocket"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
)

const dataBufferSize = 32 * 1024

var (
	_ utilnet.TLSClientConfigHolder = &Client{}
	_ http.RoundTripper             = &wsRoundtripper{}
)

var (
	statusScheme = runtime.NewScheme()
	statusCodecs = serializer.NewCodecFactory(statusScheme)
)

func init() {
	statusScheme.AddUnversionedTypes(metav1.SchemeGroupVersion,
		&metav1.Status{},
	)
}

// ConnectionHolder defines functions for structure providing
// access to the websocket connection.
type ConnectionHolder interface {
	DataBufferSize() int
	Connection() *gwebsocket.Conn
}

type wsRoundtripper struct {
	dialer *gwebsocket.Dialer
	conn   *gwebsocket.Conn
}

func (rt *wsRoundtripper) RoundTrip(request *http.Request) (response *http.Response, retErr error) {
	defer func() {
		if request.Body != nil {
			err := request.Body.Close()
			if retErr == nil {
				retErr = err
			}
		}
	}()

	switch request.URL.Scheme {
	case "https":
		request.URL.Scheme = "wss"
	case "http":
		request.URL.Scheme = "ws"
	default:
		return nil, fmt.Errorf("unknown url scheme: %s", request.URL.Scheme)
	}
	wsConn, resp, err := rt.dialer.DialContext(request.Context(), request.URL.String(), request.Header)
	if err != nil {
		// BadHandshake error becomes an "UpgradeFailureError" (used for streaming fallback).
		if errors.Is(err, gwebsocket.ErrBadHandshake) {
			cause := err
			// Enhance the error message with the error response if possible.
			if resp != nil && len(resp.Status) > 0 {
				defer resp.Body.Close()                         //nolint:errcheck
				cause = fmt.Errorf("%w (%s)", err, resp.Status) // Always add the response status
				responseError := ""
				responseErrorBytes, readErr := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
				if readErr != nil {
					cause = fmt.Errorf("%w: unable to read error from server response", cause)
				} else {
					// If returned error can be decoded as "metav1.Status", return a "StatusError".
					responseError = strings.TrimSpace(string(responseErrorBytes))
					if len(responseError) > 0 {
						if obj, _, decodeErr := statusCodecs.UniversalDecoder().Decode(responseErrorBytes, nil, &metav1.Status{}); decodeErr == nil {
							if status, ok := obj.(*metav1.Status); ok {
								cause = &apierrors.StatusError{ErrStatus: *status}
							}
						} else {
							// Otherwise, append the responseError string.
							cause = fmt.Errorf("%w: %s", cause, responseError)
						}
					}
				}
			}
			return nil, &httpstream.UpgradeFailureError{Cause: cause}
		}
		return nil, err
	}

	// Ensure we got back a protocol we understand
	foundProtocol := false
	for _, protocolVersion := range rt.dialer.Subprotocols {
		if protocolVersion == wsConn.Subprotocol() {
			foundProtocol = true
			break
		}
	}
	if !foundProtocol {
		wsConn.Close() // nolint:errcheck
		return nil, &httpstream.UpgradeFailureError{Cause: fmt.Errorf("invalid protocol, expected one of %q, got %q", rt.dialer.Subprotocols, wsConn.Subprotocol())}
	}

	rt.conn = wsConn

	return resp, nil
}

// Client knows how to establish a connection to a remote WebSocket endpoint and make it available for use.
type Client struct {
	rt        *wsRoundtripper
	rtUpgrade http.RoundTripper
}

// Connection returns the stored websocket connection.
func (c *Client) Connection() *gwebsocket.Conn {
	return c.rt.conn
}

// DataBufferSize returns the size of buffers for the
// websocket connection.
func (c *Client) DataBufferSize() int {
	return dataBufferSize
}

// TLSClientConfig implements pkg/util/net.TLSClientConfigHolder.
func (c *Client) TLSClientConfig() *tls.Config {
	return c.rt.dialer.TLSClientConfig
}

// NewClient from the restclient.Config.
// The websocket Client contains the websocket connection after the connection is established.
// Returns an error if there is a problem creating the round trippers.
func NewClient(config *restclient.Config) (*Client, error) {
	transportCfg, err := config.TransportConfig()
	if err != nil {
		return nil, err
	}
	tlsConfig, err := transport.TLSConfigFor(transportCfg)
	if err != nil {
		return nil, err
	}
	proxy := config.Proxy
	if proxy == nil {
		proxy = utilnet.NewProxierWithNoProxyCIDR(http.ProxyFromEnvironment)
	}

	dialer := gwebsocket.Dialer{
		Proxy:           proxy,
		TLSClientConfig: tlsConfig,
		ReadBufferSize:  dataBufferSize + 1024, // add space for the protocol byte indicating which channel the data is for
		WriteBufferSize: dataBufferSize + 1024, // add space for the protocol byte indicating which channel the data is for
	}

	if config.Dial != nil {
		dialer.NetDialContext = config.Dial
	}

	rt := &wsRoundtripper{
		dialer: &dialer,
	}

	wrapper, err := transport.HTTPWrappersForConfig(transportCfg, rt)
	if err != nil {
		return nil, err
	}

	client := &Client{
		rt:        rt,
		rtUpgrade: wrapper,
	}

	return client, nil
}

// Connect opens a connection to a remote server and attempts to negotiate
// a WebSocket connection. Upon success, it returns the negotiated connection.
// The round tripper rt must use the WebSocket round tripper wsRt - see RoundTripperFor.
func (c *Client) Connect(ctx context.Context, url string, protocols ...string) (*gwebsocket.Conn, error) {
	// fake request to trigger the websocket upgrade
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	c.rt.dialer.Subprotocols = protocols
	// it executes all the wrapped roundtrippers and end dialing
	// using the gorilla websocket dialer
	resp, err := c.rtUpgrade.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	err = resp.Body.Close()
	if err != nil {
		c.rt.conn.Close()
		return nil, fmt.Errorf("error closing response body: %v", err)
	}
	return c.rt.conn, nil
}
