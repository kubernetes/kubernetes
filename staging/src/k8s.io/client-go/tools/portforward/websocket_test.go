/*
Copyright 2024 The Kubernetes Authors.

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
	"bytes"
	"crypto/rand"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	gwebsocket "github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/httpstream"
	constants "k8s.io/apimachinery/pkg/util/portforward"
	"k8s.io/client-go/rest"
)

// TestWebSocketConnection_CreateSteramPropagatedToServer ensures the "CreateStream"
// call on the client endpoint of the websocket server propagates the newly created
// websocket stream to the server. Validates random data written to the client
// websocket stream is received on the websocket server stream.
func TestWebSocketConnection_CreateStreamPropagatedToServer(t *testing.T) {
	// Create fake WebSocket server with a stream creation channel.
	streamCh := make(chan httpstream.Stream)
	closeCh := make(chan bool)
	websocketServer := createWebSocketServer(streamCh, closeCh)
	require.NotNil(t, websocketServer)
	defer func() {
		close(closeCh)
		websocketServer.Close()
	}()
	// Create the WebSocketDialer.
	websocketLocation, err := url.Parse(websocketServer.URL)
	require.NoError(t, err)
	wsDialer, err := NewWebSocketDialer(websocketLocation, &rest.Config{Host: websocketLocation.Host}, []string{"80"})
	require.NoError(t, err)
	// Create the websocket client connection by dialing to the websocket server.
	wsClientConn, protocol, err := wsDialer.Dial(constants.PortForwardV2Name)
	require.NoError(t, err)
	require.NotNil(t, wsClientConn)
	defer wsClientConn.Close() //nolint:errcheck
	// Validate the negotiated portforward protocol is V2.
	assert.Equal(t, constants.PortForwardV2Name, protocol)
	// Create a stream on the client websocket connection.
	headers := http.Header{}
	headers.Set("foo", "bar")
	headers.Set("requestID", "88888")
	clientWsStream, err := wsClientConn.CreateStream(headers)
	require.NoError(t, err)
	defer wsClientConn.RemoveStreams(clientWsStream)
	// Retrieve the stream created at the websocket server, and validate it is the same
	// stream created on the websocket client connection endpoint.
	serverWsStream := <-streamCh
	require.NotNil(t, serverWsStream)
	assert.Equal(t, clientWsStream.Identifier(), serverWsStream.Identifier())
	assert.Equal(t, clientWsStream.Headers(), serverWsStream.Headers())
	// Send random data on the client stream, and validate the server stream receives it.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	_, err = rand.Read(randomData)
	require.NoError(t, err)
	randReader := bytes.NewReader(randomData)
	go func() {
		// Write the random data into the client websocket stream.
		if _, err = io.Copy(clientWsStream, randReader); err != nil {
			t.Errorf("Error during io.Copy()")
			return
		}
		clientWsStream.Close() //nolint:errcheck
	}()
	// Read the server websocket stream, and validate the data is the same
	// as from the client stream.
	serverReceived, err := io.ReadAll(serverWsStream)
	assert.NoError(t, err)
	assert.Equal(t, randomData, serverReceived)
}

// TestWebSocketDialer_DialCreatesWebSocketClient tests the Dialer can
// successfully dial to create the client websocket connection.
func TestWebSocketDialer_DialCreatesWebSocketClient(t *testing.T) {
	// Create fake WebSocket server without stream creation channel.
	closeCh := make(chan bool)
	websocketServer := createWebSocketServer(nil, closeCh)
	require.NotNil(t, websocketServer)
	defer func() {
		close(closeCh)
		websocketServer.Close()
	}()
	// Create the WebSocketDialer.
	websocketLocation, err := url.Parse(websocketServer.URL)
	require.NoError(t, err)
	wsDialer, err := NewWebSocketDialer(websocketLocation, &rest.Config{Host: websocketLocation.Host}, []string{"80"})
	require.NoError(t, err)
	// Create the websocket client connection by dialing to the websocket server.
	wsClientConn, protocol, err := wsDialer.Dial(constants.PortForwardV2Name)
	require.NoError(t, err)
	require.NotNil(t, wsClientConn)
	defer wsClientConn.Close() //nolint:errcheck
	// Validate the negotiated portforward protocol is V2.
	assert.Equal(t, constants.PortForwardV2Name, protocol)
}

// TestWebSocketDialoer_MissingPortParameterIsError validates the port parameter to the
// websocket dialer is correctly checked.
func TestWebSocketDialer_MissingPortParameterIsError(t *testing.T) {
	websocketServerURL := "http://127.0.0.1:8080"
	websocketLocation, err := url.Parse(websocketServerURL)
	require.NoError(t, err)
	emptyPortParameters := []string{}
	_, err = NewWebSocketDialer(websocketLocation, &rest.Config{Host: websocketLocation.Host}, emptyPortParameters)
	require.Error(t, err, "expected error from empty port parameter in websocket dialer constructor")
}

// TestWebSocketDialer_InvalidPortParameterIsError ensures correct validation
// of the websocket dialer passed port parameter.
func TestWebSocketDialer_InvalidPortParameterIsError(t *testing.T) {
	// Create fake WebSocket server without stream creation channel.
	closeCh := make(chan bool)
	websocketServer := createWebSocketServer(nil, closeCh)
	defer func() {
		close(closeCh)
		websocketServer.Close()
	}()
	// Create the WebSocketDialer.
	websocketServerURL := "http://127.0.0.1:8080"
	websocketLocation, err := url.Parse(websocketServerURL)
	require.NoError(t, err)
	wsDialer, err := NewWebSocketDialer(websocketLocation, &rest.Config{Host: websocketLocation.Host}, []string{"INVALID_PORT"})
	require.NoError(t, err)
	// Create the websocket client connection by dialing to the websocket server--error
	wsClientConn, protocol, err := wsDialer.Dial(constants.PortForwardV2Name)
	require.Error(t, err)
	require.Nil(t, wsClientConn)
	assert.Equal(t, "", protocol)
}

// createWebSocketServer takes two channels, and constructs a test Server as
// the websocket endpoint. The two channels are: 1) the stream creation channel, and 2)
// the websocket server close channel. The stream creation channel will receive the
// websocket stream created by the websocket server read loop, handling the "StreamCreate"
// signal. Closing the "closeCh" will stop the websocket server. The heartbeat is not created
// on the server side of the websocket connection.
func createWebSocketServer(streamCh chan httpstream.Stream, closeCh chan bool) *httptest.Server {
	return createWebSocketServerWithProtocols(streamCh, closeCh, []string{constants.PortForwardV2Name})
}

// createWebSocketServerWithProtocols creates a websocket server that will successfully negotiate
// the passed protocols; otherwise an "UpgradeFailure" error will be returned when attempting
// to upgrade the http connection.
func createWebSocketServerWithProtocols(streamCh chan httpstream.Stream, closeCh chan bool, protocols []string) *httptest.Server {
	websocketServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var upgrader = gwebsocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Accepting all requests
			},
			Subprotocols: protocols,
		}
		conn, err := upgrader.Upgrade(w, req, nil)
		if err != nil {
			return
		}
		defer conn.Close()                              //nolint:errcheck
		wsServerConn := NewWebsocketConnection(conn, 0) // zero means no throttling stream read/writes.
		defer wsServerConn.Close()                      //nolint:errcheck
		// Start the websocket connection reading loop.
		go func() {
			wsServerConn.Start(streamCh, BufferSize, 0, 0) // no hearbeat on server-side endpoint
		}()
		<-closeCh // Wait on the closing of the channel.
	}))
	return websocketServer
}
