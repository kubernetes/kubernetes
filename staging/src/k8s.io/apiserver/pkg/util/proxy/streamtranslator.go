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
	"strconv"
	"time"

	"github.com/mxk/go-flowrate/flowrate"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	constants "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apiserver/pkg/util/proxy/metrics"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/client-go/util/exec"
)

// StreamTranslatorHandler is a handler which translates WebSocket stream data
// to SPDY to proxy to kubelet (and ContainerRuntime).
type StreamTranslatorHandler struct {
	// Location is the location of the upstream proxy. It is used as the location to Dial on the upstream server
	// for upgrade requests.
	Location *url.URL
	// Transport provides an optional round tripper to use to proxy. If nil, the default proxy transport is used
	Transport http.RoundTripper
	// MaxBytesPerSec throttles stream Reader/Writer if necessary
	MaxBytesPerSec int64
	// Options define the requested streams (e.g. stdin, stdout).
	Options Options
}

// NewStreamTranslatorHandler creates a new proxy handler. Responder is required for returning
// errors to the caller.
func NewStreamTranslatorHandler(location *url.URL, transport http.RoundTripper, maxBytesPerSec int64, opts Options) *StreamTranslatorHandler {
	return &StreamTranslatorHandler{
		Location:       location,
		Transport:      transport,
		MaxBytesPerSec: maxBytesPerSec,
		Options:        opts,
	}
}

func (h *StreamTranslatorHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Create WebSocket server, including particular streams requested. If this websocket
	// endpoint is not able to be upgraded, the websocket library will return errors
	// to the client.
	websocketStreams, err := webSocketServerStreams(req, w, h.Options)
	if err != nil {
		// Client error increments bad request status code.
		metrics.IncStreamTranslatorRequest(req.Context(), strconv.Itoa(http.StatusBadRequest))
		return
	}
	defer websocketStreams.conn.Close()

	// Creating SPDY executor, ensuring redirects are not followed.
	spdyRoundTripper, err := spdy.NewRoundTripperWithConfig(spdy.RoundTripperConfig{UpgradeTransport: h.Transport, PingPeriod: 5 * time.Second})
	if err != nil {
		websocketStreams.writeStatus(apierrors.NewInternalError(err)) //nolint:errcheck
		metrics.IncStreamTranslatorRequest(req.Context(), strconv.Itoa(http.StatusInternalServerError))
		return
	}
	spdyExecutor, err := remotecommand.NewSPDYExecutorRejectRedirects(spdyRoundTripper, spdyRoundTripper, "POST", h.Location)
	if err != nil {
		websocketStreams.writeStatus(apierrors.NewInternalError(err)) //nolint:errcheck
		metrics.IncStreamTranslatorRequest(req.Context(), strconv.Itoa(http.StatusInternalServerError))
		return
	}

	// Wire the WebSocket server streams output to the SPDY client input. The stdin/stdout/stderr streams
	// can be throttled if the transfer rate exceeds the "MaxBytesPerSec" (zero means unset). Throttling
	// the streams instead of the underlying connection *may* not perform the same if two streams
	// traveling the same direction (e.g. stdout, stderr) are being maxed out.
	opts := remotecommand.StreamOptions{}
	if h.Options.Stdin {
		stdin := websocketStreams.stdinStream
		if h.MaxBytesPerSec > 0 {
			stdin = flowrate.NewReader(stdin, h.MaxBytesPerSec)
		}
		opts.Stdin = stdin
	}
	if h.Options.Stdout {
		stdout := websocketStreams.stdoutStream
		if h.MaxBytesPerSec > 0 {
			stdout = flowrate.NewWriter(stdout, h.MaxBytesPerSec)
		}
		opts.Stdout = stdout
	}
	if h.Options.Stderr {
		stderr := websocketStreams.stderrStream
		if h.MaxBytesPerSec > 0 {
			stderr = flowrate.NewWriter(stderr, h.MaxBytesPerSec)
		}
		opts.Stderr = stderr
	}
	if h.Options.Tty {
		opts.Tty = true
		opts.TerminalSizeQueue = &translatorSizeQueue{resizeChan: websocketStreams.resizeChan}
	}
	// Start the SPDY client with connected streams. Output from the WebSocket server
	// streams will be forwarded into the SPDY client. Report SPDY execution errors
	// through the websocket error stream.
	err = spdyExecutor.StreamWithContext(req.Context(), opts)
	if err != nil {
		//nolint:errcheck   // Ignore writeStatus returned error
		if statusErr, ok := err.(*apierrors.StatusError); ok {
			websocketStreams.writeStatus(statusErr)
			// Increment status code returned within status error.
			metrics.IncStreamTranslatorRequest(req.Context(), strconv.Itoa(int(statusErr.Status().Code)))
		} else if exitErr, ok := err.(exec.CodeExitError); ok && exitErr.Exited() {
			websocketStreams.writeStatus(codeExitToStatusError(exitErr))
			// Returned an exit code from the container, so not an error in
			// stream translator--add StatusOK to metrics.
			metrics.IncStreamTranslatorRequest(req.Context(), strconv.Itoa(http.StatusOK))
		} else {
			websocketStreams.writeStatus(apierrors.NewInternalError(err))
			metrics.IncStreamTranslatorRequest(req.Context(), strconv.Itoa(http.StatusInternalServerError))
		}
		return
	}

	// Write the success status back to the WebSocket client.
	//nolint:errcheck
	websocketStreams.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
		Status: metav1.StatusSuccess,
	}})
	metrics.IncStreamTranslatorRequest(req.Context(), strconv.Itoa(http.StatusOK))
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

// codeExitToStatusError converts a passed CodeExitError to the type necessary
// to send through an error stream using "writeStatus".
func codeExitToStatusError(exitErr exec.CodeExitError) *apierrors.StatusError {
	rc := exitErr.ExitStatus()
	return &apierrors.StatusError{
		ErrStatus: metav1.Status{
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
		},
	}
}
