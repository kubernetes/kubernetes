/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"

	"k8s.io/klog"
)

func handleHTTPStreams(req *http.Request, w http.ResponseWriter, portForwarder PortForwarder, podName string, uid types.UID, supportedPortForwardProtocols []string, idleTimeout, streamCreationTimeout time.Duration) error {
	_, err := httpstream.Handshake(req, w, supportedPortForwardProtocols)
	// negotiated protocol isn't currently used server side, but could be in the future
	if err != nil {
		// Handshake writes the error to the client
		return err
	}
	streamChan := make(chan httpstream.Stream, 1)

	klog.V(5).Infof("Upgrading port forward response")
	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(w, req, httpStreamReceived(streamChan))
	if conn == nil {
		return errors.New("Unable to upgrade httpstream connection")
	}
	defer conn.Close()

	klog.V(5).Infof("(conn=%p) setting port forwarding streaming connection idle timeout to %v", conn, idleTimeout)
	conn.SetIdleTimeout(idleTimeout)

	h := &httpStreamHandler{
		conn:                  conn,
		streamChan:            streamChan,
		streamPairs:           make(map[string]*httpStreamPair),
		streamCreationTimeout: streamCreationTimeout,
		pod:                   podName,
		uid:                   uid,
		forwarder:             portForwarder,
	}
	h.run()

	return nil
}

// httpStreamReceived is the httpstream.NewStreamHandler for port
// forward streams. It checks each stream's port and stream type headers,
// rejecting any streams that with missing or invalid values. Each valid
// stream is sent to the streams channel.
func httpStreamReceived(streams chan httpstream.Stream) func(httpstream.Stream, <-chan struct{}) error {
	return func(stream httpstream.Stream, replySent <-chan struct{}) error {
		// make sure it has a valid port header
		portString := stream.Headers().Get(api.PortHeader)
		if len(portString) == 0 {
			return fmt.Errorf("%q header is required", api.PortHeader)
		}
		port, err := strconv.ParseUint(portString, 10, 16)
		if err != nil {
			return fmt.Errorf("unable to parse %q as a port: %v", portString, err)
		}
		if port < 1 {
			return fmt.Errorf("port %q must be > 0", portString)
		}

		// make sure it has a valid stream type header
		streamType := stream.Headers().Get(api.StreamType)
		if len(streamType) == 0 {
			return fmt.Errorf("%q header is required", api.StreamType)
		}
		if streamType != api.StreamTypeError && streamType != api.StreamTypeData {
			return fmt.Errorf("invalid stream type %q", streamType)
		}

		streams <- stream
		return nil
	}
}

// httpStreamHandler is capable of processing multiple port forward
// requests over a single httpstream.Connection.
type httpStreamHandler struct {
	conn                  httpstream.Connection
	streamChan            chan httpstream.Stream
	streamPairsLock       sync.RWMutex
	streamPairs           map[string]*httpStreamPair
	streamCreationTimeout time.Duration
	pod                   string
	uid                   types.UID
	forwarder             PortForwarder
}

// getStreamPair returns a httpStreamPair for requestID. This creates a
// new pair if one does not yet exist for the requestID. The returned bool is
// true if the pair was created.
func (h *httpStreamHandler) getStreamPair(requestID string) (*httpStreamPair, bool) {
	h.streamPairsLock.Lock()
	defer h.streamPairsLock.Unlock()

	if p, ok := h.streamPairs[requestID]; ok {
		klog.V(5).Infof("(conn=%p, request=%s) found existing stream pair", h.conn, requestID)
		return p, false
	}

	klog.V(5).Infof("(conn=%p, request=%s) creating new stream pair", h.conn, requestID)

	p := newPortForwardPair(requestID)
	h.streamPairs[requestID] = p

	return p, true
}

// monitorStreamPair waits for the pair to receive both its error and data
// streams, or for the timeout to expire (whichever happens first), and then
// removes the pair.
func (h *httpStreamHandler) monitorStreamPair(p *httpStreamPair, timeout <-chan time.Time) {
	select {
	case <-timeout:
		err := fmt.Errorf("(conn=%v, request=%s) timed out waiting for streams", h.conn, p.requestID)
		utilruntime.HandleError(err)
		p.printError(err.Error())
	case <-p.complete:
		klog.V(5).Infof("(conn=%v, request=%s) successfully received error and data streams", h.conn, p.requestID)
	}
	h.removeStreamPair(p.requestID)
}

// hasStreamPair returns a bool indicating if a stream pair for requestID
// exists.
func (h *httpStreamHandler) hasStreamPair(requestID string) bool {
	h.streamPairsLock.RLock()
	defer h.streamPairsLock.RUnlock()

	_, ok := h.streamPairs[requestID]
	return ok
}

// removeStreamPair removes the stream pair identified by requestID from streamPairs.
func (h *httpStreamHandler) removeStreamPair(requestID string) {
	h.streamPairsLock.Lock()
	defer h.streamPairsLock.Unlock()

	delete(h.streamPairs, requestID)
}

// requestID returns the request id for stream.
func (h *httpStreamHandler) requestID(stream httpstream.Stream) string {
	requestID := stream.Headers().Get(api.PortForwardRequestIDHeader)
	if len(requestID) == 0 {
		klog.V(5).Infof("(conn=%p) stream received without %s header", h.conn, api.PortForwardRequestIDHeader)
		// If we get here, it's because the connection came from an older client
		// that isn't generating the request id header
		// (https://github.com/kubernetes/kubernetes/blob/843134885e7e0b360eb5441e85b1410a8b1a7a0c/pkg/client/unversioned/portforward/portforward.go#L258-L287)
		//
		// This is a best-effort attempt at supporting older clients.
		//
		// When there aren't concurrent new forwarded connections, each connection
		// will have a pair of streams (data, error), and the stream IDs will be
		// consecutive odd numbers, e.g. 1 and 3 for the first connection. Convert
		// the stream ID into a pseudo-request id by taking the stream type and
		// using id = stream.Identifier() when the stream type is error,
		// and id = stream.Identifier() - 2 when it's data.
		//
		// NOTE: this only works when there are not concurrent new streams from
		// multiple forwarded connections; it's a best-effort attempt at supporting
		// old clients that don't generate request ids.  If there are concurrent
		// new connections, it's possible that 1 connection gets streams whose IDs
		// are not consecutive (e.g. 5 and 9 instead of 5 and 7).
		streamType := stream.Headers().Get(api.StreamType)
		switch streamType {
		case api.StreamTypeError:
			requestID = strconv.Itoa(int(stream.Identifier()))
		case api.StreamTypeData:
			requestID = strconv.Itoa(int(stream.Identifier()) - 2)
		}

		klog.V(5).Infof("(conn=%p) automatically assigning request ID=%q from stream type=%s, stream ID=%d", h.conn, requestID, streamType, stream.Identifier())
	}
	return requestID
}

// run is the main loop for the httpStreamHandler. It processes new
// streams, invoking portForward for each complete stream pair. The loop exits
// when the httpstream.Connection is closed.
func (h *httpStreamHandler) run() {
	klog.V(5).Infof("(conn=%p) waiting for port forward streams", h.conn)
Loop:
	for {
		select {
		case <-h.conn.CloseChan():
			klog.V(5).Infof("(conn=%p) upgraded connection closed", h.conn)
			break Loop
		case stream := <-h.streamChan:
			requestID := h.requestID(stream)
			streamType := stream.Headers().Get(api.StreamType)
			klog.V(5).Infof("(conn=%p, request=%s) received new stream of type %s", h.conn, requestID, streamType)

			p, created := h.getStreamPair(requestID)
			if created {
				go h.monitorStreamPair(p, time.After(h.streamCreationTimeout))
			}
			if complete, err := p.add(stream); err != nil {
				msg := fmt.Sprintf("error processing stream for request %s: %v", requestID, err)
				utilruntime.HandleError(errors.New(msg))
				p.printError(msg)
			} else if complete {
				go h.portForward(p)
			}
		}
	}
}

// portForward invokes the httpStreamHandler's forwarder.PortForward
// function for the given stream pair.
func (h *httpStreamHandler) portForward(p *httpStreamPair) {
	defer p.dataStream.Close()
	defer p.errorStream.Close()

	portString := p.dataStream.Headers().Get(api.PortHeader)
	port, _ := strconv.ParseInt(portString, 10, 32)

	klog.V(5).Infof("(conn=%p, request=%s) invoking forwarder.PortForward for port %s", h.conn, p.requestID, portString)
	err := h.forwarder.PortForward(h.pod, h.uid, int32(port), p.dataStream)
	klog.V(5).Infof("(conn=%p, request=%s) done invoking forwarder.PortForward for port %s", h.conn, p.requestID, portString)

	if err != nil {
		msg := fmt.Errorf("error forwarding port %d to pod %s, uid %v: %v", port, h.pod, h.uid, err)
		utilruntime.HandleError(msg)
		fmt.Fprint(p.errorStream, msg.Error())
	}
}

// httpStreamPair represents the error and data streams for a port
// forwarding request.
type httpStreamPair struct {
	lock        sync.RWMutex
	requestID   string
	dataStream  httpstream.Stream
	errorStream httpstream.Stream
	complete    chan struct{}
}

// newPortForwardPair creates a new httpStreamPair.
func newPortForwardPair(requestID string) *httpStreamPair {
	return &httpStreamPair{
		requestID: requestID,
		complete:  make(chan struct{}),
	}
}

// add adds the stream to the httpStreamPair. If the pair already
// contains a stream for the new stream's type, an error is returned. add
// returns true if both the data and error streams for this pair have been
// received.
func (p *httpStreamPair) add(stream httpstream.Stream) (bool, error) {
	p.lock.Lock()
	defer p.lock.Unlock()

	switch stream.Headers().Get(api.StreamType) {
	case api.StreamTypeError:
		if p.errorStream != nil {
			return false, errors.New("error stream already assigned")
		}
		p.errorStream = stream
	case api.StreamTypeData:
		if p.dataStream != nil {
			return false, errors.New("data stream already assigned")
		}
		p.dataStream = stream
	}

	complete := p.errorStream != nil && p.dataStream != nil
	if complete {
		close(p.complete)
	}
	return complete, nil
}

// printError writes s to p.errorStream if p.errorStream has been set.
func (p *httpStreamPair) printError(s string) {
	p.lock.RLock()
	defer p.lock.RUnlock()
	if p.errorStream != nil {
		fmt.Fprint(p.errorStream, s)
	}
}
