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
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy2"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/klog/v2"
)

const defaultIdleConnectionTimeout = 4 * time.Hour

// StreamTranslatorHandler is a handler which translates WebSocket stream data
// to SPDY to proxy to kubelet (and ContainerRuntime).
type StreamTranslatorHandler struct {
	// Location is the location of the upstream proxy. It is used as the location to Dial on the upstream server
	// for upgrade requests.
	Location *url.URL
	// Transport provides an optional round tripper to use to proxy. If nil, the default proxy transport is used
	Transport http.RoundTripper
	// FlushInterval controls how often the standard HTTP proxy will flush content from the upstream.
	FlushInterval time.Duration
	// MaxBytesPerSec controls the maximum rate for an upstream connection. No rate is imposed if the value is zero.
	MaxBytesPerSec int64
	// Responder is passed errors that occur while setting up proxying.
	Responder ErrorResponder
}

// NewStreamTranslatorHandler creates a new proxy handler with a default flush interval. Responder is required for returning
// errors to the caller.
func NewStreamTranslatorHandler(location *url.URL, transport http.RoundTripper, responder ErrorResponder) *StreamTranslatorHandler {
	return &StreamTranslatorHandler{
		Location:      normalizeLocation(location),
		Transport:     transport,
		FlushInterval: defaultFlushInterval,
		Responder:     responder,
	}
}

func (h *StreamTranslatorHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Create WebSocket server, including particular streams requested.
	streamOpts, _ := streamOptionsFromRequest(req)
	ctx, ok := webSocketServerStreams(req, w, streamOpts, defaultIdleConnectionTimeout)
	if !ok {
		err := fmt.Errorf("Unable to create websocket streams")
		klog.Infof("websocket createStreams error: %v", err)
		h.Responder.Error(w, req, err)
		return
	}
	defer ctx.conn.Close()

	// Create the SPDY client connection proxied through kubelet.
	tlsConfig, err := utilnet.TLSClientConfig(h.Transport)
	if err == nil {
		// Manually set NextProtos to "http/1.1", since http/2.0 will NOT upgrade a connection.
		tlsConfig.NextProtos = []string{"http/1.1"}
	} else {
		klog.Infof("Unable to unwrap transport %T to get at TLS config: %v", h.Transport, err)
	}
	spdyRoundtripper := spdy.NewRoundTripper(tlsConfig)
	spdyExecutor, err := spdy2.NewSPDYExecutorForTransports(spdyRoundtripper, spdyRoundtripper, "POST", h.Location)
	if err != nil {
		klog.Infof("create SPDY executor error: %v", err)
		h.Responder.Error(w, req, err)
		return
	}

	// Wire the WebSocket server stream output to the SPDY client input.
	opts := spdy2.StreamOptions{}
	if streamOpts.Stdin {
		opts.Stdin = ctx.stdinStream
	}
	if streamOpts.Stdout {
		opts.Stdout = ctx.stdoutStream
	}
	if streamOpts.Stderr {
		opts.Stderr = ctx.stderrStream
	}
	if streamOpts.TTY {
		opts.Tty = true
		opts.TerminalSizeQueue = &translatorSizeQueue{resizeChan: ctx.resizeChan}
	}
	// Start the SPDY client with connected streams.
	err = spdyExecutor.Stream(opts)
	if err != nil {
		klog.Infof("spdyExecutor Stream() error: %v", err)
		h.Responder.Error(w, req, err)
		return
	}

	// Write the status back to the WebSocket client.
	ctx.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
		Status: metav1.StatusSuccess,
	}})
}

// translatorSizeQueue feeds the size events from the WebSocket
// resizeChan into the SPDY client input. Implements TerminalSizeQueue
// interface.
type translatorSizeQueue struct {
	resizeChan chan spdy2.TerminalSize
}

func (t *translatorSizeQueue) Next() *spdy2.TerminalSize {
	size, ok := <-t.resizeChan
	if !ok {
		return nil
	}
	return &size
}
