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

package portforward

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"

	gwebsocket "github.com/gorilla/websocket"
	"github.com/mxk/go-flowrate/flowrate"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/klog/v2"
)

// streamType constants used in websocket stream headers
const (
	StreamData   = 1
	StreamClose  = 255
	StreamCreate = 254

	BufferSize = 32 * 1024
)

var _ httpstream.Connection = &WebsocketConnection{}

// WebsocketConnection implements the "httpstream.Connection" interface, wrapping
// a websocket connection and its streams.
type WebsocketConnection struct {
	conn           *gwebsocket.Conn
	connWriteLock  sync.Mutex
	streams        map[int]*wsStream
	streamsMu      sync.Mutex
	streamID       int
	streamIDLock   sync.Mutex
	closeChan      chan bool
	maxBytesPerSec int64
}

// NewWebsocketConnection wraps the passed gorilla/websockets connection
// with the WebsocketConnection struct (implementing httpstream.Connection).
func NewWebsocketConnection(conn *gwebsocket.Conn, maxBytesPerSec int64) *WebsocketConnection {
	closeChan := make(chan bool)
	wsConn := &WebsocketConnection{
		conn:           conn,
		streams:        map[int]*wsStream{},
		closeChan:      closeChan,
		maxBytesPerSec: maxBytesPerSec,
	}
	// Close channel when detecting close connection.
	closeHandler := conn.CloseHandler()
	conn.SetCloseHandler(func(code int, text string) error {
		klog.V(3).Infof("websocket conn close: %d--%s", code, text)
		close(closeChan)
		err := closeHandler(code, text)
		return err
	})
	return wsConn
}

func (c *WebsocketConnection) getStream(id int) *wsStream {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	return c.streams[id]
}

func (c *WebsocketConnection) setStream(id int, s *wsStream) {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	c.streams[id] = s
}

func (c *WebsocketConnection) removeStream(s httpstream.Stream) {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	delete(c.streams, int(s.Identifier()))
}

// nextStreamID generates a monotonically increasing set of
// unique identifers for websocket streams.
func (c *WebsocketConnection) nextStreamID() int {
	c.streamIDLock.Lock()
	defer c.streamIDLock.Unlock()
	id := c.streamID
	c.streamID++
	return id
}

// CreateStream uses id from passed headers to create a stream over "c.conn" connection.
// Returns a Stream structure or nil and an error if one occurred.
func (c *WebsocketConnection) CreateStream(headers http.Header) (httpstream.Stream, error) {
	c.connWriteLock.Lock()
	defer c.connWriteLock.Unlock()
	s, err := c.createStream(headers)
	if err != nil {
		return nil, err
	}
	// Signal the other connection endpoint that a stream was created.
	klog.V(5).Infof("Signaling StreamCreate to other endpoint: %d", s.id)
	_, err = s.writeWithHeaders([]byte{}, StreamCreate, headers)
	return s, err
}

// createStream creates the websocket stream (which implements httpstream.Stream)
// structure, storing metadata within the connection. Returns an error if one occurs.
func (c *WebsocketConnection) createStream(headers http.Header) (*wsStream, error) {
	id := c.nextStreamID()
	if s := c.getStream(id); s != nil {
		return nil, fmt.Errorf("duplicate stream for type %d", id)
	}
	klog.V(4).Infof("CreateStream: %d", id)
	reader, writer := io.Pipe()
	s := &wsStream{
		id:             id,
		headers:        headers,
		readPipe:       reader,
		writePipe:      writer,
		conn:           c.conn,
		connWriteLock:  &c.connWriteLock,
		maxBytesPerSec: c.maxBytesPerSec,
	}
	c.setStream(id, s)
	return s, nil
}

func (c *WebsocketConnection) Close() error {
	klog.V(4).Infof("Connection Close(); closing streams...")
	c.closeAllStreamReaders(nil)
	// Signal other endpoint that websocket connection is closing.
	c.conn.WriteControl(gwebsocket.CloseMessage, []byte{}, time.Now().Add(writeDeadline)) //nolint:errcheck
	return c.conn.Close()
}

func (c *WebsocketConnection) CloseChan() <-chan bool {
	return c.closeChan
}

func (c *WebsocketConnection) SetIdleTimeout(timeout time.Duration) {}

func (c *WebsocketConnection) RemoveStreams(streams ...httpstream.Stream) {
	for _, stream := range streams {
		klog.V(4).Infof("RemoveStream: %d", stream.Identifier())
		stream.Close() //nolint:errcheck
		c.removeStream(stream)
	}
}

// Start is the reading processor for this endpoint of the websocket
// connection. This loop reads the connection, and demultiplexes the data
// into one of the individual stream pipes (by checking the stream id). This
// loop can *not* be run concurrently, because there can only be one websocket
// connection reader at a time (a read mutex would provide no benefit). The passed
// stream creation channel is used to communicate dynamically created streams,
// if the websocket connection detects a stream creation signal.
func (c *WebsocketConnection) Start(streamCreateCh chan httpstream.Stream, bufferSize int, period time.Duration, deadline time.Duration) {
	// Initialize and start the ping/pong heartbeat, if necessary. Only client-side connection
	// needs to run the heartbeat.
	if period > 0 && deadline > 0 {
		h := newHeartbeat(c.conn, period, deadline)
		// Set initial timeout for websocket connection reading.
		if err := c.conn.SetReadDeadline(time.Now().Add(deadline)); err != nil {
			klog.Errorf("Websocket initial setting read deadline failed %v", err)
			return
		}
		go h.start()
	}
	// Buffer size must correspond to the same size allocated
	// for the read buffer during websocket client creation. A
	// difference can cause incomplete connection reads.
	readBuffer := make([]byte, bufferSize)
	for {
		// NextReader() only returns data messages (BinaryMessage or Text
		// Message). Even though this call will never return control frames
		// such as ping, pong, or close, this call is necessary for these
		// message types to be processed. There can only be one reader
		// at a time, so this reader loop must *not* be run concurrently;
		// there is no lock for reading. Calling "NextReader()" before the
		// current reader has been processed will close the current reader.
		// If the heartbeat read deadline times out, this "NextReader()" will
		// return an i/o error, and error handling will clean up.
		messageType, r, err := c.conn.NextReader()
		if err != nil {
			var wsCloseErr *gwebsocket.CloseError
			if !errors.As(err, &wsCloseErr) || wsCloseErr.Code != gwebsocket.CloseNormalClosure {
				c.closeAllStreamReaders(fmt.Errorf("next reader: %w", err))
			}
			return
		}
		// Throttle reading from the websocket connection (note: only one Reader
		// on the connection can be active at a time).
		if c.maxBytesPerSec > 0 {
			r = flowrate.NewReader(r, c.maxBytesPerSec)
		}
		// All remote command protocols send/receive only binary data messages.
		if messageType != gwebsocket.BinaryMessage {
			c.closeAllStreamReaders(fmt.Errorf("unexpected message type: %d", messageType))
			return
		}
		// Initially, read the websocket stream headers.
		wsStreamHeaders, err := readWsStreamHeaders(r)
		if err != nil {
			c.closeAllStreamReaders(fmt.Errorf("read stream id: %w", err))
			return
		}
		klog.V(5).Infof("websocket stream headers read: %s", wsStreamHeaders.String())
		streamType := wsStreamHeaders.StreamType
		streamID := wsStreamHeaders.StreamID
		// StreamCreate signal means the other websocket connection endpoint
		// has created a new stream.
		if streamType == StreamCreate {
			klog.V(4).Infof("stream create signal: %d", streamID)
			// Create the stream, but do not send StreamCreate signal.
			stream, err := c.createStream(wsStreamHeaders.Headers)
			if err != nil {
				c.closeAllStreamReaders(fmt.Errorf("creating websocket stream: %w", err))
				return
			}
			// Queue newly created websocket stream onto channel.
			klog.V(5).Infof("queueing stream on channel: %d", streamID)
			streamCreateCh <- stream
			continue
		} else if streamType == StreamClose {
			// Read the next byte, which is the stream id.
			klog.V(5).Infof("stream close signal: %d", streamID)
			s := c.getStream(streamID)
			if s != nil {
				s.writePipe.Close() //nolint:errcheck
			} else {
				klog.V(6).Infof("Unknown stream id during close %d--discarding message", streamID)
			}
			continue
		}
		// Retrieve stream from connection. If stream is nil (not found)
		// we *must* still drain the Reader before the next iteration.
		klog.V(5).Infof("received stream %d", streamID)
		s := c.getStream(streamID)
		for {
			klog.V(6).Infof("reading into buffer (%d)", streamID)
			nr, errRead := r.Read(readBuffer)
			if nr > 0 {
				if s != nil {
					klog.V(6).Infof("writing into stream pipe (%d)", streamID)
					_, errWrite := s.writePipe.Write(readBuffer[:nr])
					if errWrite != nil {
						// Pipe must have been closed by the stream user.
						// Nothing to do, discard the message.
						break
					}
				} else {
					klog.Errorf("Unknown stream id %d, discarding message", streamID)
				}
			}
			if errRead != nil {
				if errRead == io.EOF {
					break
				}
				c.closeAllStreamReaders(fmt.Errorf("read message: %w", err))
				return
			}
		}
	}
}

// closeAllStreamReaders closes readers in all streams.
// This unblocks all stream.Read() calls.
func (c *WebsocketConnection) closeAllStreamReaders(err error) {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	for _, s := range c.streams {
		// Closing writePipe unblocks all readPipe.Read() callers and prevents any future writes.
		s.writePipe.CloseWithError(err) //nolint:errcheck
	}
}

// TODO(seans): Refactor the "heartbeat" code which is shared with RemoteCommand.

// heartbeat encasulates data necessary for the websocket ping/pong heartbeat. This
// heartbeat works by setting a read deadline on the websocket connection, then
// pushing this deadline into the future for every successful heartbeat. If the
// heartbeat "pong" fails to respond within the deadline, then the "NextReader()" call
// inside the "readDemuxLoop" will return an i/o error prompting a connection close
// and cleanup.
type heartbeat struct {
	conn *gwebsocket.Conn
	// period defines how often a "ping" heartbeat message is sent to the other endpoint
	period time.Duration
	// closing the "closer" channel will clean up the heartbeat timers
	closer chan struct{}
	// optional data to send with "ping" message
	message []byte
	// optionally received data message with "pong" message, same as sent with ping
	pongMessage []byte
}

// newHeartbeat creates heartbeat structure encapsulating fields necessary to
// run the websocket connection ping/pong mechanism and sets up handlers on
// the websocket connection.
func newHeartbeat(conn *gwebsocket.Conn, period time.Duration, deadline time.Duration) *heartbeat {
	h := &heartbeat{
		conn:   conn,
		period: period,
		closer: make(chan struct{}),
	}
	// Set up handler for receiving returned "pong" message from other endpoint
	// by pushing the read deadline into the future. The "msg" received could
	// be empty.
	h.conn.SetPongHandler(func(msg string) error {
		// Push the read deadline into the future.
		klog.V(8).Infof("Pong message received (%s)--resetting read deadline", msg)
		err := h.conn.SetReadDeadline(time.Now().Add(deadline))
		if err != nil {
			klog.Errorf("Websocket setting read deadline failed %v", err)
			return err
		}
		if len(msg) > 0 {
			h.pongMessage = []byte(msg)
		}
		return nil
	})
	// Set up handler to cleanup timers when this endpoint receives "Close" message.
	closeHandler := h.conn.CloseHandler()
	h.conn.SetCloseHandler(func(code int, text string) error {
		close(h.closer)
		return closeHandler(code, text)
	})
	return h
}

// TODO(sean): uncomment this unused method when adding unit tests.
// setMessage is optional data sent with "ping" heartbeat. According to the websocket RFC
// this data sent with "ping" message should be returned in "pong" message.
// func (h *heartbeat) setMessage(msg string) {
// 	h.message = []byte(msg)
// }

// start the heartbeat by setting up necesssary handlers and looping by sending "ping"
// message every "period" until the "closer" channel is closed.
func (h *heartbeat) start() {
	// Loop to continually send "ping" message through websocket connection every "period".
	t := time.NewTicker(h.period)
	defer t.Stop()
	for {
		select {
		case <-h.closer:
			klog.V(8).Infof("closed channel--returning")
			return
		case <-t.C:
			// "WriteControl" does not need to be protected by a mutex. According to
			// gorilla/websockets library docs: "The Close and WriteControl methods can
			// be called concurrently with all other methods."
			if err := h.conn.WriteControl(gwebsocket.PingMessage, h.message, time.Now().Add(writeDeadline)); err == nil {
				klog.V(8).Infof("Websocket Ping succeeeded")
			} else {
				klog.Errorf("Websocket Ping failed: %v", err)
				var netErr net.Error
				if errors.Is(err, gwebsocket.ErrCloseSent) {
					// we continue because c.conn.CloseChan will manage closing the connection already
					continue
				} else if errors.As(err, &netErr) && netErr.Timeout() {
					// Continue, in case this is a transient failure.
					// c.conn.CloseChan above will tell us when the connection is
					// actually closed.
					// If Temporary function hadn't been deprecated, we would have used it.
					// But most of temporary errors are timeout errors anyway.
					continue
				}
				return
			}
		}
	}
}
