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
	"bytes"
	"crypto/tls"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"

	gwebsocket "github.com/gorilla/websocket"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
)

var (
	_ utilnet.TLSClientConfigHolder = &RoundTripper{}
	_ http.RoundTripper             = &RoundTripper{}

	// statusScheme is private scheme for the decoding here until someone fixes the TODO in NewConnection
	statusScheme = runtime.NewScheme()
	// statusCodecs knows about query parameters used with the meta v1 API spec.
	statusCodecs = serializer.NewCodecFactory(statusScheme)
)

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
	Conn *gwebsocket.Conn
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
	protocolVersions := request.Header[httpstream.HeaderProtocolVersion]
	delete(request.Header, httpstream.HeaderProtocolVersion)

	dialer := gwebsocket.Dialer{
		Proxy:           rt.Proxier,
		TLSClientConfig: rt.TLSConfig,
		Subprotocols:    protocolVersions,
	}
	switch request.URL.Scheme {
	case "https":
		request.URL.Scheme = "wss"
	case "http":
		request.URL.Scheme = "ws"
	}
	wsConn, resp, err := dialer.DialContext(request.Context(), request.URL.String(), request.Header)
	if err != nil {
		if err != gwebsocket.ErrBadHandshake {
			return nil, err
		}
		// resp.Body contains (a part of) the response when err == gwebsocket.ErrBadHandshake
		responseErrorBytes, bodyErr := ioutil.ReadAll(resp.Body)
		if bodyErr != nil {
			return nil, err
		}
		// TODO: I don't belong here, I should be abstracted from this class
		if obj, _, err := statusCodecs.UniversalDecoder().Decode(responseErrorBytes, nil, &metav1.Status{}); err == nil {
			if status, ok := obj.(*metav1.Status); ok {
				return nil, &apierrors.StatusError{ErrStatus: *status}
			}
		}
		return nil, fmt.Errorf("unable to upgrade connection: %s", bytes.TrimSpace(responseErrorBytes))
	}

	rt.Conn = wsConn

	return resp, nil
}
