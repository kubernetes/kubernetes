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
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	gwebsocket "github.com/gorilla/websocket"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	constants "k8s.io/apimachinery/pkg/util/portforward"
	"k8s.io/client-go/tools/portforward"
	spdytransport "k8s.io/client-go/transport/spdy"
	"k8s.io/klog/v2"
)

const (
	// garbage collection to remove invalid subrequest resources.
	gcPeriod = 5 * time.Second
	// time to wait for second subrequest stream to arrive.
	streamCreateTimeout = 10 * time.Second
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
	// Server-side of upgraded websocket connection
	wsConnection *portforward.WebsocketConnection
	// Client-side of upstream SPDY connection
	spdyConnection httpstream.Connection
	// Map of pending subrequests to the streams associated with subrequest.
	requestStreams *requestStreams
}

// NewStreamTranslatorHandler creates a new proxy handler, which translates websocket
// streams/data into SPDY stream/data upstream.
func NewStreamTranslatorHandler(location *url.URL, transport http.RoundTripper, maxBytesPerSec int64) *StreamTranslatorHandler {
	return &StreamTranslatorHandler{
		Location:       location,
		Transport:      transport,
		MaxBytesPerSec: maxBytesPerSec,
		requestStreams: newRequestStreams(),
	}
}

// ServeHTTP called to run stream translator proxy. HTTP request is upgraded to websocket
// streaming connection. An SPDY connection is also created upstream to the kubelet.
// Responds to websocket streams created on websocket connection by creating and copying
// data on an associated upstream SPDY stream.
func (h *StreamTranslatorHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// First, dial upstream server to establish SPDY connection.
	spdyRoundTripper, err := spdy.NewRoundTripperWithConfig(spdy.RoundTripperConfig{UpgradeTransport: h.Transport})
	if err != nil {
		klog.Errorf("error creating spdy roundtripper: %v", err)
		return
	}
	dialer := spdytransport.NewDialer(spdyRoundTripper, &http.Client{Transport: spdyRoundTripper}, req.Method, h.Location)
	spdyConn, spdyProtocol, err := dialer.Dial(constants.PortForwardV1Name)
	if err != nil {
		klog.Errorf("error spdy upgrading connection: %v", err)
		return
	}
	defer spdyConn.Close() //nolint:errcheck
	klog.V(3).Infof("upstream sdpy connection created: %s", spdyProtocol)
	h.spdyConnection = spdyConn

	// If SPDY upstream connection was successfully established, then
	// upgrade the current request to a websocket server connection.
	var upgrader = gwebsocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Accepting all requests
		},
		Subprotocols: []string{
			constants.PortForwardV2Name,
		},
	}
	conn, err := upgrader.Upgrade(w, req, nil)
	if err != nil {
		klog.Errorf("error upgrading websocket connection: %v", err)
		return
	}
	defer conn.Close() //nolint:errcheck
	klog.V(3).Infof("websocket connection created: %s", conn.Subprotocol())
	h.wsConnection = portforward.NewWebsocketConnection(conn, h.MaxBytesPerSec)

	// Channel for communicating websocket streams.
	streamCreateCh := make(chan httpstream.Stream)
	klog.V(3).Infoln("websocket connection read loop starting...")
	go h.wsConnection.Start(streamCreateCh, portforward.BufferSize, 0, 0) // no hearbeat sent on server-side endpoint

	klog.V(3).Infoln("stream translator starting garbage collection loop...")
	stopCh := make(chan struct{})
	go h.garbageCollect(stopCh, gcPeriod, streamCreateTimeout)

	// Loop iterating over websocket streams received from the stream create
	// channel, until the websocket connection is closed.
	klog.V(3).Infoln("stream translator starting websocket stream channel reception loop...")
	for {
		select {
		case wsStream := <-streamCreateCh:
			go func() {
				requestID, err := getRequestID(wsStream)
				if err != nil {
					klog.Errorf("websocket stream created with invalid requestID: %v", err)
					h.wsConnection.RemoveStreams(wsStream)
					return
				}
				klog.V(5).Infof("websocket stream received from channel: %d", requestID)
				// Retrieve streams associated with request (or create the structure if it does not exist).
				streams := h.requestStreams.get(requestID)
				// Create spdy stream that will be connected to websocket stream.
				spdyStream, err := h.spdyConnection.CreateStream(wsStream.Headers())
				if err != nil {
					klog.Errorf("error creating spdy stream: %v", err)
					h.requestStreams.remove(requestID)
					h.spdyConnection.RemoveStreams(streams.getSpdyStreams()...)
					h.wsConnection.RemoveStreams(append(streams.getWebsocketStreams(), wsStream)...)
					return
				}
				ready, err := streams.addStreamPair(wsStream, spdyStream)
				if err != nil {
					klog.Errorf("error adding websocket/spdy data streams: %v", err)
					h.requestStreams.remove(requestID)
					h.spdyConnection.RemoveStreams(append(streams.getSpdyStreams(), spdyStream)...)
					h.wsConnection.RemoveStreams(append(streams.getWebsocketStreams(), wsStream)...)
					return
				}
				// If both the data stream and error stream for a subrequest have been
				// created, then begin streaming the portforward.
				if ready {
					klog.V(4).Infof("stream pairs complete for request (%d)...portforwarding", requestID)
					h.requestStreams.remove(requestID) // no longer pending--remove
					streams.portForward()
					h.spdyConnection.RemoveStreams(streams.getSpdyStreams()...)
					h.wsConnection.RemoveStreams(streams.getWebsocketStreams()...)
				}
			}()
		case <-h.wsConnection.CloseChan():
			klog.V(3).Infof("channel closed--port forward connections closing")
			close(stopCh)            // stop garbage collection
			h.spdyConnection.Close() //nolint:errcheck
			h.wsConnection.Close()   //nolint:errcheck
			return
		}
	}
}

// garbageCollect loops through currently stored subrequests, removing resources
// for subrequests in an invalid state. "gcPeriod" defines the time between garbage
// collections, while "streamCreateTimeout" is the amount of time to wait for
// the second subrequest stream to arrive. Should run in its own goroutine.
func (h *StreamTranslatorHandler) garbageCollect(stopCh chan struct{}, gcPeriod time.Duration, streamCreateTimeout time.Duration) {
	ticker := time.NewTicker(gcPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			klog.V(8).Infof("stream translator garbarge collection...")
			streamPairs := h.requestStreams.cleanup(streamCreateTimeout)
			for _, sp := range streamPairs {
				h.wsConnection.RemoveStreams(sp.getWebsocketStreams()...)
				h.spdyConnection.RemoveStreams(sp.getSpdyStreams()...)
			}
		case <-stopCh:
			return
		}
	}
}

// requestStreams encapsulates the mapping between a *pending* subrequest
// (identified by requestID) and the incomplete streams (data and error)
// associated with the subrequest. Once all the requests streams have arrived,
// it is no longer pending, and it is safe to remove the entry.
type requestStreams struct {
	requestStreamsMap  map[int]*streamPairs
	requestStreamsLock sync.Mutex
}

// newRequestStreams returns a pointer to a new, initialized "requestStreams" struct.
func newRequestStreams() *requestStreams {
	return &requestStreams{
		requestStreamsMap:  map[int]*streamPairs{},
		requestStreamsLock: sync.Mutex{},
	}
}

// get returns the streamPairs struct associated with the requestID. If
// the struct does not exist, it is created.
func (rs *requestStreams) get(requestID int) *streamPairs {
	rs.requestStreamsLock.Lock()
	defer rs.requestStreamsLock.Unlock()
	_, exists := rs.requestStreamsMap[requestID]
	if !exists {
		rs.requestStreamsMap[requestID] = &streamPairs{}
	}
	return rs.requestStreamsMap[requestID]
}

// remove deletes the subrequest from the requestStreamsMap.
func (rs *requestStreams) remove(requestID int) {
	rs.requestStreamsLock.Lock()
	defer rs.requestStreamsLock.Unlock()
	delete(rs.requestStreamsMap, requestID)
}

// cleanup loops through the requestStreamsMap, removing subrequests that have
// not received both sets of streams necessary to portforward before the
// "streamCreateTimeout" has passed. Returns an array of the streamPairs structs
// that are timed out (for further cleanup if necessary).
func (rs *requestStreams) cleanup(streamCreateTimeout time.Duration) []*streamPairs {
	rs.requestStreamsLock.Lock()
	defer rs.requestStreamsLock.Unlock()
	klog.V(8).Infof("starting request streams cleanup...")
	streams := []*streamPairs{}
	for requestID, streamPairs := range rs.requestStreamsMap {
		if streamPairs.isTimedOut(streamCreateTimeout) {
			klog.Errorf("timeout waiting for second set of streams for request: %d", requestID)
			delete(rs.requestStreamsMap, requestID)
			streams = append(streams, streamPairs)
		}
	}
	klog.V(8).Infof("request streams cleanup: %d requests removed", len(streams))
	return streams
}

// streamPairs group streams associated with the same portforward subrequest.
// Each port-forward subrequest requires a data stream and an error stream,
// and there is both a downstream websocket stream and and upstream
// spdy stream for each of these stream types.
type streamPairs struct {
	// mutually exclusive access to these streams
	streamPairsLock sync.Mutex
	// time last pair of streams was added in microseconds since unix epoch
	lastUpdate int64
	// websocket/spdy data streams
	wsDataStream   httpstream.Stream
	spdyDataStream httpstream.Stream
	// websocket/spdy error streams
	wsErrorStream   httpstream.Stream
	spdyErrorStream httpstream.Stream
}

// getWebsocketStreams returns the websocket streams stored in the streamPair.
func (sp *streamPairs) getWebsocketStreams() []httpstream.Stream {
	sp.streamPairsLock.Lock()
	defer sp.streamPairsLock.Unlock()
	wsStreams := []httpstream.Stream{}
	if sp.wsDataStream != nil {
		wsStreams = append(wsStreams, sp.wsDataStream)
	}
	if sp.wsErrorStream != nil {
		wsStreams = append(wsStreams, sp.wsErrorStream)
	}
	return wsStreams
}

// getSpdyStreams returns the spdy streams stored in the streamPair.
func (sp *streamPairs) getSpdyStreams() []httpstream.Stream {
	sp.streamPairsLock.Lock()
	defer sp.streamPairsLock.Unlock()
	spdyStreams := []httpstream.Stream{}
	if sp.spdyDataStream != nil {
		spdyStreams = append(spdyStreams, sp.spdyDataStream)
	}
	if sp.wsErrorStream != nil {
		spdyStreams = append(spdyStreams, sp.spdyErrorStream)
	}
	return spdyStreams
}

// addStreamPair stores a websocket and spdy stream *of the same type* (e.g.
// data or error) returning true if the all four streams are now present
// (and therefore able to begin portforwarding). Returns false and the error
// if an error occurs.
func (sp *streamPairs) addStreamPair(wsStream httpstream.Stream, spdyStream httpstream.Stream) (bool, error) {
	sp.streamPairsLock.Lock()
	defer sp.streamPairsLock.Unlock()
	streamType := wsStream.Headers().Get(v1.StreamType)
	if streamType != spdyStream.Headers().Get(v1.StreamType) {
		return false, fmt.Errorf("streams added to streamPair are not the same type")
	}
	if streamType == v1.StreamTypeData {
		if sp.wsDataStream != nil || sp.spdyDataStream != nil {
			requestID := wsStream.Headers().Get(v1.PortForwardRequestIDHeader)
			return false, fmt.Errorf("duplicate data streams added to streamPair: %s", requestID)
		}
		sp.wsDataStream = wsStream
		sp.spdyDataStream = spdyStream
	} else if streamType == v1.StreamTypeError {
		if sp.wsErrorStream != nil || sp.spdyErrorStream != nil {
			requestID := wsStream.Headers().Get(v1.PortForwardRequestIDHeader)
			return false, fmt.Errorf("duplicate error streams added to streamPair: %s", requestID)
		}
		sp.wsErrorStream = wsStream
		sp.spdyErrorStream = spdyStream
	} else {
		return false, fmt.Errorf("unknown stream type: %s", streamType)
	}
	sp.lastUpdate = time.Now().UnixMicro()
	// If all four streams are present, then return complete = true.
	return sp.isComplete(), nil
}

// isTimedOut returns true if the streamPairs does not have a full
// complement of streams, and streamCreateTimeout has passed; false
// otherwise.
func (sp *streamPairs) isTimedOut(streamCreateTimeout time.Duration) bool {
	sp.streamPairsLock.Lock()
	defer sp.streamPairsLock.Unlock()
	// skip streamPairs that have not had at least one stream update.
	if sp.lastUpdate != int64(0) && !sp.isComplete() {
		lastUpdated := time.UnixMicro(sp.lastUpdate)
		timeout := lastUpdated.Add(streamCreateTimeout)
		if time.Now().UnixMicro() > timeout.UnixMicro() {
			return true
		}
	}
	return false
}

// isComplete returns true if all four streams necessary for portforwarding
// are present; false otherwise.
func (sp *streamPairs) isComplete() bool {
	return (sp.wsDataStream != nil && sp.wsErrorStream != nil &&
		sp.spdyDataStream != nil && sp.spdyErrorStream != nil)
}

// portForward connects the websocket and spdy data and error streams, copying
// in both directions for the data stream, and only one direction (from
// upstream spdy to downstream websocket) for the error stream. Completes
// when either 1) an error occurs writing upstream, or 2) the read from
// spdy to websocket streams completes.
func (sp *streamPairs) portForward() {

	readingDataDone := make(chan struct{})
	writingDataError := make(chan struct{})

	go func() {
		// Copy error from the upstream spdy side to the websocket client.
		if _, err := io.Copy(sp.wsErrorStream, sp.spdyErrorStream); err != nil && !isClosedConnectionError(err) {
			klog.Errorf("error copying upstream spdy error stream to websocket stream: %v", err)
			return
		}
		klog.V(2).Infoln("error stream reading complete")
	}()
	go func() {
		// Copy from upstream spdy to the websocket data stream
		if _, err := io.Copy(sp.wsDataStream, sp.spdyDataStream); err != nil && !isClosedConnectionError(err) {
			klog.Errorf("error copying upstream spdy data stream to websocket stream: %v", err)
			return
		}
		// Inform the select below that the copy is done
		klog.V(2).Infoln("datastream reading complete...closing readingDataDone channel")
		close(readingDataDone)
	}()
	go func() {
		// Inform upstream spdy server we're not sending any more data after copy unblocks
		defer func() {
			klog.V(3).Infoln("writing websocket data stream to spdy data stream -- complete")
			sp.spdyDataStream.Close() //nolint:errcheck
		}()
		// Copy from the websocket stream to the upstream spdy endpoint.
		if _, err := io.Copy(sp.spdyDataStream, sp.wsDataStream); err != nil && !isClosedConnectionError(err) {
			klog.Errorf("error writing websocket stream data to sdpy data stream: %v", err)
			// break out of the select below without waiting for the other copy to finish
			close(writingDataError)
		}
	}()

	// Wait for either a websocket->spdy writing error or for
	// copying from spdy to websocket data streams to finish.
	select {
	case <-readingDataDone:
	case <-writingDataError:
	}
}

// getRequestID returns the requestID stored in the stream's headers,
// or an error if it not possible to parse the requestID header.
func getRequestID(s httpstream.Stream) (int, error) {
	headers := s.Headers()
	requestIDStr := headers.Get(v1.PortForwardRequestIDHeader)
	return strconv.Atoi(requestIDStr)
}

// isClosedConnectionError returns true if the error is for using
// a closed network connection.
func isClosedConnectionError(err error) bool {
	if err == nil {
		return false
	}
	if strings.Contains(err.Error(), "use of closed network connection") {
		return true
	}
	return false
}
