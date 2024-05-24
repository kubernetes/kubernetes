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
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
)

func TestWebSocketRoundTripper_RoundTripperSucceeds(t *testing.T) {
	// Create fake WebSocket server.
	websocketServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		conns, err := webSocketServerStreams(req, w)
		if err != nil {
			t.Fatalf("error on webSocketServerStreams: %v", err)
		}
		defer conns.conn.Close()
	}))
	defer websocketServer.Close()

	// Create the wrapped roundtripper and websocket upgrade roundtripper and call "RoundTrip()".
	websocketLocation, err := url.Parse(websocketServer.URL)
	require.NoError(t, err)
	req, err := http.NewRequestWithContext(context.Background(), "GET", websocketServer.URL, nil)
	require.NoError(t, err)
	rt, wsRt, err := RoundTripperFor(&restclient.Config{Host: websocketLocation.Host})
	require.NoError(t, err)
	requestedProtocol := remotecommand.StreamProtocolV5Name
	req.Header[wsstream.WebSocketProtocolHeader] = []string{requestedProtocol}
	_, err = rt.RoundTrip(req)
	require.NoError(t, err)
	// WebSocket Connection is stored in websocket RoundTripper.
	// Compare the expected negotiated subprotocol with the actual subprotocol.
	actualProtocol := wsRt.Connection().Subprotocol()
	assert.Equal(t, requestedProtocol, actualProtocol)

}

func TestWebSocketRoundTripper_RoundTripperFails(t *testing.T) {
	// Create fake WebSocket server.
	websocketServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Bad handshake means websocket server will not completely initialize.
		_, err := webSocketServerStreams(req, w)
		require.Error(t, err)
		assert.True(t, strings.Contains(err.Error(), "websocket server finished before becoming ready"))
	}))
	defer websocketServer.Close()

	// Create the wrapped roundtripper and websocket upgrade roundtripper and call "RoundTrip()".
	websocketLocation, err := url.Parse(websocketServer.URL)
	require.NoError(t, err)
	req, err := http.NewRequestWithContext(context.Background(), "GET", websocketServer.URL, nil)
	require.NoError(t, err)
	rt, _, err := RoundTripperFor(&restclient.Config{Host: websocketLocation.Host})
	require.NoError(t, err)
	// Requested subprotocol version 1 is not supported by test websocket server.
	requestedProtocol := remotecommand.StreamProtocolV1Name
	req.Header[wsstream.WebSocketProtocolHeader] = []string{requestedProtocol}
	_, err = rt.RoundTrip(req)
	// Ensure a "bad handshake" error is returned, since requested protocol is not supported.
	require.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "bad handshake"))
	assert.True(t, httpstream.IsUpgradeFailure(err))
}

func TestWebSocketRoundTripper_NegotiateCreatesConnection(t *testing.T) {
	// Create fake WebSocket server.
	websocketServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		conns, err := webSocketServerStreams(req, w)
		if err != nil {
			t.Fatalf("error on webSocketServerStreams: %v", err)
		}
		defer conns.conn.Close()
	}))
	defer websocketServer.Close()

	// Create the websocket roundtripper and call "Negotiate" to create websocket connection.
	websocketLocation, err := url.Parse(websocketServer.URL)
	require.NoError(t, err)
	req, err := http.NewRequestWithContext(context.Background(), "GET", websocketServer.URL, nil)
	require.NoError(t, err)
	rt, wsRt, err := RoundTripperFor(&restclient.Config{Host: websocketLocation.Host})
	require.NoError(t, err)
	requestedProtocol := remotecommand.StreamProtocolV5Name
	conn, err := Negotiate(rt, wsRt, req, requestedProtocol)
	require.NoError(t, err)
	// Compare the expected negotiated subprotocol with the actual subprotocol.
	actualProtocol := conn.Subprotocol()
	assert.Equal(t, requestedProtocol, actualProtocol)
}

// websocketStreams contains the WebSocket connection and streams from a server.
type websocketStreams struct {
	conn io.Closer
}

func webSocketServerStreams(req *http.Request, w http.ResponseWriter) (*websocketStreams, error) {
	conn := wsstream.NewConn(map[string]wsstream.ChannelProtocolConfig{
		remotecommand.StreamProtocolV5Name: {
			Binary:   true,
			Channels: []wsstream.ChannelType{},
		},
	})
	conn.SetIdleTimeout(4 * time.Hour)
	// Opening the connection responds to WebSocket client, negotiating
	// the WebSocket upgrade connection and the subprotocol.
	_, _, err := conn.Open(w, req)
	if err != nil {
		return nil, err
	}
	return &websocketStreams{conn: conn}, nil
}
