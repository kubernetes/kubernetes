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
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/onsi/gomega"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	"k8s.io/ktesting"
	"k8s.io/streaming/pkg/httpstream"
	"k8s.io/streaming/pkg/httpstream/wsstream"
)

func TestWebSocketRoundTripper_RoundTripperSucceeds(t *testing.T) {
	tCtx := ktesting.Init(t)
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
	tCtx.ExpectNoError(err)
	req, err := http.NewRequestWithContext(context.Background(), "GET", websocketServer.URL, nil)
	tCtx.ExpectNoError(err)
	rt, wsRt, err := RoundTripperFor(&restclient.Config{Host: websocketLocation.Host})
	tCtx.ExpectNoError(err)
	requestedProtocol := remotecommand.StreamProtocolV5Name
	req.Header[wsstream.WebSocketProtocolHeader] = []string{requestedProtocol}
	_, err = rt.RoundTrip(req)
	tCtx.ExpectNoError(err)
	// WebSocket Connection is stored in websocket RoundTripper.
	// Compare the expected negotiated subprotocol with the actual subprotocol.
	actualProtocol := wsRt.Connection().Subprotocol()
	tCtx.Assert(actualProtocol).Should(gomega.Equal(requestedProtocol))

}

func TestWebSocketRoundTripper_RoundTripperFails(t *testing.T) {
	testCases := map[string]struct {
		statusCode    int
		body          string
		status        *metav1.Status
		expectedError string
	}{
		"Empty response status still returns basic websocket error": {
			statusCode:    -1,
			body:          "",
			expectedError: "websocket: bad handshake",
		},
		"Empty response body still returns status": {
			statusCode:    http.StatusForbidden,
			body:          "",
			expectedError: "(403 Forbidden)",
		},
		"Error response body returned as string when can not be cast as metav1.Status": {
			statusCode:    http.StatusForbidden,
			body:          "RBAC violated",
			expectedError: "(403 Forbidden): RBAC violated",
		},
		"Error returned as metav1.Status within response body": {
			statusCode: http.StatusBadRequest,
			body:       "",
			status: &metav1.Status{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "meta.k8s.io/v1",
					Kind:       "Status",
				},
				Status: "Failure",
				Reason: "Unable to negotiate sub-protocol",
				Code:   http.StatusBadRequest,
			},
		},
	}
	encoder := statusCodecs.LegacyCodec(metav1.SchemeGroupVersion)
	tCtx := ktesting.Init(t)
	for testName, testCase := range testCases {
		tCtx.Run(testName, func(tCtx ktesting.TContext) {
			// Create fake WebSocket server.
			websocketServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if testCase.statusCode > 0 {
					w.WriteHeader(testCase.statusCode)
				}
				if testCase.status != nil {
					statusBytes, err := runtime.Encode(encoder, testCase.status)
					if err != nil {
						tCtx.Errorf("unexpected error %v", err)
						return
					}
					_, err = w.Write(statusBytes)
					if err != nil {
						tCtx.Errorf("unexpected error %v", err)
						return
					}
				} else if len(testCase.body) > 0 {
					_, err := w.Write([]byte(testCase.body))
					if err != nil {
						tCtx.Errorf("unexpected error %v", err)
						return
					}
				}
			}))
			defer websocketServer.Close()

			// Create the wrapped roundtripper and websocket upgrade roundtripper and call "RoundTrip()".
			websocketLocation, err := url.Parse(websocketServer.URL)
			tCtx.ExpectNoError(err)
			req, err := http.NewRequestWithContext(context.Background(), "GET", websocketServer.URL, nil)
			tCtx.ExpectNoError(err)
			rt, _, err := RoundTripperFor(&restclient.Config{Host: websocketLocation.Host})
			tCtx.ExpectNoError(err)
			_, err = rt.RoundTrip(req)
			tCtx.Require(err).Should(gomega.HaveOccurred())
			tCtx.Assert(httpstream.IsUpgradeFailure(err)).Should(gomega.BeTrueBecause("error should be a websocket upgrade failure"))
			if testCase.status != nil {
				upgradeErr := &httpstream.UpgradeFailureError{}
				validErr := errors.As(err, &upgradeErr)
				tCtx.Assert(validErr).Should(gomega.BeTrueBecause("could not cast error as httpstream.UpgradeFailureError"))
				statusErr := upgradeErr.Cause
				apiErr := &apierrors.StatusError{}
				validErr = errors.As(statusErr, &apiErr)
				tCtx.Assert(validErr).Should(gomega.BeTrueBecause("could not cast error as apierrors.StatusError"))
				tCtx.Assert(apiErr.ErrStatus).Should(gomega.Equal(*testCase.status))
			} else {
				tCtx.Assert(err.Error()).Should(gomega.ContainSubstring(testCase.expectedError),
					"expected (%s), got (%s)", testCase.expectedError, err.Error())
			}
		})
	}
}

func TestWebSocketRoundTripper_NegotiateCreatesConnection(t *testing.T) {
	tCtx := ktesting.Init(t)
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
	tCtx.ExpectNoError(err)
	req, err := http.NewRequestWithContext(context.Background(), "GET", websocketServer.URL, nil)
	tCtx.ExpectNoError(err)
	rt, wsRt, err := RoundTripperFor(&restclient.Config{Host: websocketLocation.Host})
	tCtx.ExpectNoError(err)
	requestedProtocol := remotecommand.StreamProtocolV5Name
	conn, err := Negotiate(rt, wsRt, req, requestedProtocol)
	tCtx.ExpectNoError(err)
	// Compare the expected negotiated subprotocol with the actual subprotocol.
	actualProtocol := conn.Subprotocol()
	tCtx.Assert(actualProtocol).Should(gomega.Equal(requestedProtocol))
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
