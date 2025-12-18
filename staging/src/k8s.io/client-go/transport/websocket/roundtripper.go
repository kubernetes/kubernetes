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
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/coder/websocket"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
)

var (
	_ utilnet.TLSClientConfigHolder = &RoundTripper{}
	_ http.RoundTripper             = &RoundTripper{}
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

// Conn wraps a *websocket.Conn and stores connection addresses for net.Conn compatibility.
type Conn struct {
	*websocket.Conn
	localAddr  net.Addr
	remoteAddr net.Addr
}

// LocalAddr returns the local network address.
func (c *Conn) LocalAddr() net.Addr {
	return c.localAddr
}

// RemoteAddr returns the remote network address.
func (c *Conn) RemoteAddr() net.Addr {
	return c.remoteAddr
}

// ConnectionHolder defines functions for structure providing
// access to the websocket connection.
type ConnectionHolder interface {
	DataBufferSize() int
	Connection() *Conn
}

// RoundTripper knows how to establish a connection to a remote WebSocket endpoint and make it available for use.
// RoundTripper must not be reused.
type RoundTripper struct {
	// TLSConfig holds the TLS configuration settings to use when connecting
	// to the remote server.
	TLSConfig *tls.Config

	// Proxier specifies a function to return a proxy for a given
	// Request. If the function returns a non-nil error, the
	// request is aborted with the provided error.
	// If Proxy is nil or returns a nil *URL, no proxy is used.
	Proxier func(req *http.Request) (*url.URL, error)

	// Conn holds the WebSocket connection after a round trip.
	Conn *Conn
}

// Connection returns the stored websocket connection.
func (rt *RoundTripper) Connection() *Conn {
	return rt.Conn
}

// DataBufferSize returns the size of buffers for the
// websocket connection.
func (rt *RoundTripper) DataBufferSize() int {
	return 32 * 1024
}

// TLSClientConfig implements pkg/util/net.TLSClientConfigHolder.
func (rt *RoundTripper) TLSClientConfig() *tls.Config {
	return rt.TLSConfig
}

// RoundTrip connects to the remote websocket using the headers in the request and the TLS
// configuration from the config
func (rt *RoundTripper) RoundTrip(request *http.Request) (retResp *http.Response, retErr error) {
	defer func() {
		if request.Body != nil {
			err := request.Body.Close()
			if retErr == nil {
				retErr = err
			}
		}
	}()

	// set the protocol version directly on the dialer from the header
	protocolVersions := request.Header[wsstream.WebSocketProtocolHeader]
	delete(request.Header, wsstream.WebSocketProtocolHeader)

	// Convert URL scheme for websocket
	wsURL := *request.URL
	switch request.URL.Scheme {
	case "https":
		wsURL.Scheme = "wss"
	case "http":
		wsURL.Scheme = "ws"
	default:
		return nil, fmt.Errorf("unknown url scheme: %s", request.URL.Scheme)
	}

	// Create a transport wrapper that captures connection addresses
	addrCapture := &addrCaptureTransport{}
	transport := &http.Transport{
		Proxy:           rt.Proxier,
		TLSClientConfig: rt.TLSConfig,
		DialContext:     addrCapture.dialContext,
	}
	addrCapture.Transport = transport

	httpClient := &http.Client{
		Transport: transport,
	}

	opts := &websocket.DialOptions{
		HTTPClient:   httpClient,
		HTTPHeader:   request.Header,
		Subprotocols: protocolVersions,
	}

	ctx := request.Context()
	wsConn, resp, err := websocket.Dial(ctx, wsURL.String(), opts)
	if err != nil {
		// Check if we got a response that indicates upgrade failure
		if resp != nil && resp.StatusCode != http.StatusSwitchingProtocols {
			cause := fmt.Errorf("websocket upgrade failed: %w", err)
			// Enhance the error message with the error response if possible.
			if len(resp.Status) > 0 {
				defer resp.Body.Close()                           //nolint:errcheck
				cause = fmt.Errorf("%w (%s)", cause, resp.Status) // Always add the response status
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
	for _, protocolVersion := range protocolVersions {
		if protocolVersion == wsConn.Subprotocol() {
			foundProtocol = true
			break
		}
	}
	if !foundProtocol {
		wsConn.Close(websocket.StatusProtocolError, "invalid protocol") // nolint:errcheck
		return nil, &httpstream.UpgradeFailureError{Cause: fmt.Errorf("invalid protocol, expected one of %q, got %q", protocolVersions, wsConn.Subprotocol())}
	}

	// Get captured addresses from the transport wrapper
	localAddr, remoteAddr := addrCapture.getAddrs()

	// Set read limit to handle large data transfers.
	// -1 disables the limit entirely to align with previous gorilla/websocket behavior
	wsConn.SetReadLimit(-1)

	rt.Conn = &Conn{
		Conn:       wsConn,
		localAddr:  localAddr,
		remoteAddr: remoteAddr,
	}

	return resp, nil
}

// addrCaptureTransport wraps an http.Transport to capture the local and remote
// addresses of the underlying connection. This is needed because coder/websocket
// doesn't expose the underlying net.Conn directly.
type addrCaptureTransport struct {
	*http.Transport
	mu         sync.Mutex
	localAddr  net.Addr
	remoteAddr net.Addr
}

// dialContext is the dial function that captures connection addresses.
// It's assigned to http.Transport.DialContext.
func (t *addrCaptureTransport) dialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	dialer := &net.Dialer{}
	conn, err := dialer.DialContext(ctx, network, addr)
	if err != nil {
		return nil, err
	}
	// Capture addresses immediately upon successful dial
	t.mu.Lock()
	t.localAddr = conn.LocalAddr()
	t.remoteAddr = conn.RemoteAddr()
	t.mu.Unlock()
	return conn, nil
}

func (t *addrCaptureTransport) getAddrs() (local, remote net.Addr) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.localAddr, t.remoteAddr
}

// RoundTripperFor transforms the passed rest config into a wrapped roundtripper, as well
// as a pointer to the websocket RoundTripper. The websocket RoundTripper contains the
// websocket connection after RoundTrip() on the wrapper. Returns an error if there is
// a problem creating the round trippers.
func RoundTripperFor(config *restclient.Config) (http.RoundTripper, ConnectionHolder, error) {
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
func Negotiate(rt http.RoundTripper, connectionInfo ConnectionHolder, req *http.Request, protocols ...string) (*Conn, error) {
	// Plumb protocols to RoundTripper#RoundTrip
	req.Header[wsstream.WebSocketProtocolHeader] = protocols
	resp, err := rt.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	conn := connectionInfo.Connection()
	if conn == nil {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		return nil, fmt.Errorf("websocket connection was not established")
	}
	if resp != nil && resp.Body != nil {
		err = resp.Body.Close()
		if err != nil {
			conn.Close(websocket.StatusNormalClosure, "")
			return nil, fmt.Errorf("error closing response body: %v", err)
		}
	}
	return conn, nil
}
