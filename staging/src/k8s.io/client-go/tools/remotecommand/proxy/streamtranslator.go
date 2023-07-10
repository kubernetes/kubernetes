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

package proxy

import (
	"fmt"
	"net/http"
	"net/url"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	apimachineryproxy "k8s.io/apimachinery/pkg/util/proxy"
	constants "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/client-go/util/exec"
	"k8s.io/klog/v2"
)

// StreamTranslatorHandler is a handler which translates WebSocket stream data
// to SPDY to proxy to kubelet (and ContainerRuntime).
type StreamTranslatorHandler struct {
	// Location is the location of the upstream proxy. It is used as the location to Dial on the upstream server
	// for upgrade requests.
	Location *url.URL
	// Transport provides an optional round tripper to use to proxy. If nil, the default proxy transport is used
	Transport http.RoundTripper
	// Responder is passed errors that occur while setting up proxying.
	Responder apimachineryproxy.ErrorResponder
	// Options define the requested streams (e.g. stdin, stdout).
	Options Options
}

// NewStreamTranslatorHandler creates a new proxy handler. Responder is required for returning
// errors to the caller.
func NewStreamTranslatorHandler(location *url.URL, transport http.RoundTripper, responder apimachineryproxy.ErrorResponder, opts Options) *StreamTranslatorHandler {
	return &StreamTranslatorHandler{
		Location:  location,
		Transport: transport,
		Responder: responder,
		Options:   opts,
	}
}

func (h *StreamTranslatorHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Create WebSocket server, including particular streams requested.
	websocketStreams, err := webSocketServerStreams(req, w, h.Options)
	if err != nil {
		klog.Errorf("websocket createStreams error: %v", err)
		h.Responder.Error(w, req, err)
		return
	}
	defer websocketStreams.conn.Close()

	// Create the SPDY client connection proxied through kubelet.
	tlsConfig, err := utilnet.TLSClientConfig(h.Transport)
	if tlsConfig != nil && err == nil {
		// Manually set NextProtos to "http/1.1", since http/2.0 will NOT upgrade a connection.
		tlsConfig.NextProtos = []string{"http/1.1"}
	} else {
		klog.Infof("Unable to unwrap transport %T to get at TLS config: %v", h.Transport, err)
	}
	spdyRoundTripper := spdy.NewRoundTripper(tlsConfig)
	spdyExecutor, err := remotecommand.NewSPDYExecutorForTransports(spdyRoundTripper, spdyRoundTripper, "POST", h.Location)
	if err != nil {
		klog.Errorf("websocket createStreams error: %v", err)
		h.Responder.Error(w, req, err)
		return
	}

	// Wire the WebSocket server streams output to the SPDY client input.
	opts := remotecommand.StreamOptions{}
	if h.Options.Stdin {
		opts.Stdin = websocketStreams.stdinStream
	}
	if h.Options.Stdout {
		opts.Stdout = websocketStreams.stdoutStream
	}
	if h.Options.Stderr {
		opts.Stderr = websocketStreams.stderrStream
	}
	if h.Options.Tty {
		opts.Tty = true
		opts.TerminalSizeQueue = &translatorSizeQueue{resizeChan: websocketStreams.resizeChan}
	}
	// Start the SPDY client with connected streams. Output from the WebSocket server
	// streams will be forwarded into the SPDY client.
	err = spdyExecutor.StreamWithContext(req.Context(), opts)
	if err != nil {
		klog.Errorf("spdyExecutor StreamWithContext() error: %v", err)
		// Report error back to the WebSocket client.
		if exitErr, ok := err.(exec.CodeExitError); ok && exitErr.Exited() {
			rc := exitErr.ExitStatus()
			websocketStreams.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
				Status: metav1.StatusFailure,
				Reason: constants.NonZeroExitCodeReason,
				Details: &metav1.StatusDetails{
					Causes: []metav1.StatusCause{
						{
							Type:    constants.ExitCodeCauseType,
							Message: fmt.Sprintf("%d", rc),
						},
					},
				},
				Message: fmt.Sprintf("command terminated with non-zero exit code: %v", exitErr),
			}})
		} else {
			websocketStreams.writeStatus(apierrors.NewInternalError(err))
		}
		return
	}

	// Write the success status back to the WebSocket client.
	websocketStreams.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
		Status: metav1.StatusSuccess,
	}})
}

// translatorSizeQueue feeds the size events from the WebSocket
// resizeChan into the SPDY client input. Implements TerminalSizeQueue
// interface.
type translatorSizeQueue struct {
	resizeChan chan remotecommand.TerminalSize
}

func (t *translatorSizeQueue) Next() *remotecommand.TerminalSize {
	size, ok := <-t.resizeChan
	if !ok {
		return nil
	}
	return &size
}
