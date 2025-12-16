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

package remotecommand

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	wsTransport "k8s.io/client-go/transport/websocket"
	"k8s.io/klog/v2"
)

// writeDeadline defines the time that a client-side write to the websocket
// connection must complete before an i/o timeout occurs.
const writeDeadline = 60 * time.Second

var (
	_ Executor          = &wsStreamExecutor{}
	_ streamCreator     = &wsStreamCreator{}
	_ httpstream.Stream = &stream{}

	streamType2streamID = map[string]byte{
		v1.StreamTypeStdin:  remotecommand.StreamStdIn,
		v1.StreamTypeStdout: remotecommand.StreamStdOut,
		v1.StreamTypeStderr: remotecommand.StreamStdErr,
		v1.StreamTypeError:  remotecommand.StreamErr,
		v1.StreamTypeResize: remotecommand.StreamResize,
	}
)

// wsStreamExecutor handles transporting standard shell streams over an httpstream connection.
type wsStreamExecutor struct {
	transport http.RoundTripper
	upgrader  wsTransport.ConnectionHolder
	method    string
	url       string
	// requested protocols in priority order (e.g. v5.channel.k8s.io before v4.channel.k8s.io).
	protocols []string
	// selected protocol from the handshake process; could be empty string if handshake fails.
	negotiated string
}

func NewWebSocketExecutor(config *restclient.Config, method, url string) (Executor, error) {
	// Only supports V5 protocol for correct version skew functionality.
	// Previous api servers will proxy upgrade requests to legacy websocket
	// servers on container runtimes which support V1-V4. These legacy
	// websocket servers will not handle the new CLOSE signal.
	return NewWebSocketExecutorForProtocols(config, method, url, remotecommand.StreamProtocolV5Name)
}

// NewWebSocketExecutorForProtocols allows to execute commands via a WebSocket connection.
func NewWebSocketExecutorForProtocols(config *restclient.Config, method, url string, protocols ...string) (Executor, error) {
	transport, upgrader, err := wsTransport.RoundTripperFor(config)
	if err != nil {
		return nil, fmt.Errorf("error creating websocket transports: %v", err)
	}
	return &wsStreamExecutor{
		transport: transport,
		upgrader:  upgrader,
		method:    method,
		url:       url,
		protocols: protocols,
	}, nil
}

// Deprecated: use StreamWithContext instead to avoid possible resource leaks.
// See https://github.com/kubernetes/kubernetes/pull/103177 for details.
func (e *wsStreamExecutor) Stream(options StreamOptions) error {
	return e.StreamWithContext(context.Background(), options)
}

// StreamWithContext upgrades an HTTPRequest to a WebSocket connection, and starts the various
// goroutines to implement the necessary streams over the connection. The "options" parameter
// defines which streams are requested. Returns an error if one occurred. This method is NOT
// safe to run concurrently with the same executor (because of the state stored in the upgrader).
func (e *wsStreamExecutor) StreamWithContext(ctx context.Context, options StreamOptions) error {
	req, err := http.NewRequestWithContext(ctx, e.method, e.url, nil)
	if err != nil {
		return err
	}
	conn, err := wsTransport.Negotiate(e.transport, e.upgrader, req, e.protocols...)
	if err != nil {
		return err
	}
	if conn == nil {
		panic(fmt.Errorf("websocket connection is nil"))
	}
	defer conn.Close(websocket.StatusNormalClosure, "")
	e.negotiated = conn.Subprotocol()
	klog.V(4).Infof("The subprotocol is %s", e.negotiated)

	var streamer streamProtocolHandler
	switch e.negotiated {
	case remotecommand.StreamProtocolV5Name:
		streamer = newStreamProtocolV5(options)
	case remotecommand.StreamProtocolV4Name:
		streamer = newStreamProtocolV4(options)
	case remotecommand.StreamProtocolV3Name:
		streamer = newStreamProtocolV3(options)
	case remotecommand.StreamProtocolV2Name:
		streamer = newStreamProtocolV2(options)
	case "":
		klog.V(4).Infof("The server did not negotiate a streaming protocol version. Falling back to %s", remotecommand.StreamProtocolV1Name)
		fallthrough
	case remotecommand.StreamProtocolV1Name:
		streamer = newStreamProtocolV1(options)
	}

	panicChan := make(chan any, 1)
	errorChan := make(chan error, 1)
	go func() {
		defer func() {
			if p := recover(); p != nil {
				panicChan <- p
			}
		}()

		readyChan := make(chan struct{})
		creator := newWSStreamCreator(ctx, conn.Conn)
		go func() {
			select {
			// Wait until all streams have been created before starting the readDemuxLoop.
			// This is to avoid a race condition where the readDemuxLoop receives a message
			// for a stream that has not yet been created.
			case <-readyChan:
			case <-ctx.Done():
				creator.closeAllStreamReaders(ctx.Err())
				return
			}

			creator.readDemuxLoop(e.upgrader.DataBufferSize())
		}()
		errorChan <- streamer.stream(creator, readyChan)
	}()

	select {
	case p := <-panicChan:
		panic(p)
	case err := <-errorChan:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

type wsStreamCreator struct {
	conn *websocket.Conn
	ctx  context.Context
	// Protects writing to websocket connection; reading is lock-free
	connWriteLock sync.Mutex
	// map of stream id to stream; multiple streams read/write the connection
	streams   map[byte]*stream
	streamsMu sync.Mutex
	// setStreamErr holds the error to return to anyone calling setStreams.
	// this is populated in closeAllStreamReaders
	setStreamErr error
	// readLoopOnce ensures readDemuxLoop is only run once, preventing
	// accidental concurrent reads which would corrupt the websocket connection.
	readLoopOnce sync.Once
}

func newWSStreamCreator(ctx context.Context, conn *websocket.Conn) *wsStreamCreator {
	return &wsStreamCreator{
		conn:    conn,
		ctx:     ctx,
		streams: map[byte]*stream{},
	}
}

func (c *wsStreamCreator) getStream(id byte) *stream {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	return c.streams[id]
}

func (c *wsStreamCreator) setStream(id byte, s *stream) error {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	if c.setStreamErr != nil {
		return c.setStreamErr
	}
	c.streams[id] = s
	return nil
}

// CreateStream uses id from passed headers to create a stream over "c.conn" connection.
// Returns a Stream structure or nil and an error if one occurred.
func (c *wsStreamCreator) CreateStream(headers http.Header) (httpstream.Stream, error) {
	streamType := headers.Get(v1.StreamType)
	id, ok := streamType2streamID[streamType]
	if !ok {
		return nil, fmt.Errorf("unknown stream type: %s", streamType)
	}
	if s := c.getStream(id); s != nil {
		return nil, fmt.Errorf("duplicate stream for type %s", streamType)
	}
	reader, writer := io.Pipe()
	s := &stream{
		headers:       headers,
		readPipe:      reader,
		writePipe:     writer,
		conn:          c.conn,
		ctx:           c.ctx,
		connWriteLock: &c.connWriteLock,
		id:            id,
	}
	if err := c.setStream(id, s); err != nil {
		_ = s.writePipe.Close()
		_ = s.readPipe.Close()
		return nil, err
	}
	return s, nil
}

// readDemuxLoop is the lock-free reading processor for this endpoint of the websocket
// connection. This loop reads the connection, and demultiplexes the data
// into one of the individual stream pipes (by checking the stream id). This
// loop can *not* be run concurrently, because there can only be one websocket
// connection reader at a time (a read mutex would provide no benefit).
// The sync.Once guard ensures this method only executes once per wsStreamCreator instance.
func (c *wsStreamCreator) readDemuxLoop(bufferSize int) {
	c.readLoopOnce.Do(func() {
		c.doReadDemuxLoop(bufferSize)
	})
}

// doReadDemuxLoop is the actual implementation of the read demux loop.
// It should only be called via readDemuxLoop to ensure single execution.
func (c *wsStreamCreator) doReadDemuxLoop(bufferSize int) {
	// Buffer size must correspond to the same size allocated
	// for the read buffer during websocket client creation. A
	// difference can cause incomplete connection reads.
	readBuffer := make([]byte, bufferSize)
	for {
		// Reader() only returns data messages (Binary or Text).
		// Control frames (ping, pong, close) are handled automatically by coder/websocket.
		// There can only be one reader at a time, so this reader loop must *not* be
		// run concurrently; there is no lock for reading.
		messageType, r, err := c.conn.Reader(c.ctx)
		if err != nil {
			// Check for normal closure
			if websocket.CloseStatus(err) == websocket.StatusNormalClosure {
				err = nil // readers will get io.EOF as it's a normal closure
			} else {
				err = fmt.Errorf("next reader: %w", err)
			}
			c.closeAllStreamReaders(err)
			return
		}
		// All remote command protocols send/receive only binary data messages.
		if messageType != websocket.MessageBinary {
			c.closeAllStreamReaders(fmt.Errorf("unexpected message type: %d", messageType))
			return
		}
		// It's ok to read just a single byte because the underlying library wraps the actual
		// connection with a buffered reader anyway.
		_, err = io.ReadFull(r, readBuffer[:1])
		if err != nil {
			c.closeAllStreamReaders(fmt.Errorf("read stream id: %w", err))
			return
		}
		streamID := readBuffer[0]
		s := c.getStream(streamID)
		if s == nil {
			klog.Errorf("Unknown stream id %d, discarding message", streamID)
			continue
		}
		for {
			nr, errRead := r.Read(readBuffer)
			if nr > 0 {
				// Write the data to the stream's pipe. This can block.
				_, errWrite := s.writePipe.Write(readBuffer[:nr])
				if errWrite != nil {
					// Pipe must have been closed by the stream user.
					// Nothing to do, discard the message.
					break
				}
			}
			if errRead != nil {
				if errRead == io.EOF {
					break
				}
				c.closeAllStreamReaders(fmt.Errorf("read message: %w", errRead))
				return
			}
		}
	}
}

// closeAllStreamReaders closes readers in all streams.
// This unblocks all stream.Read() calls, and keeps any future streams from being created.
func (c *wsStreamCreator) closeAllStreamReaders(err error) {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	for _, s := range c.streams {
		// Closing writePipe unblocks all readPipe.Read() callers and prevents any future writes.
		_ = s.writePipe.CloseWithError(err)
	}
	// ensure callers to setStreams receive an error after this point
	if err != nil {
		c.setStreamErr = err
	} else {
		c.setStreamErr = fmt.Errorf("closed all streams")
	}
}

type stream struct {
	headers   http.Header
	readPipe  *io.PipeReader
	writePipe *io.PipeWriter
	// conn is used for writing directly into the connection.
	// Is nil after Close() / Reset() to prevent future writes.
	conn *websocket.Conn
	ctx  context.Context
	// connWriteLock protects conn against concurrent write operations. There must be a single writer and a single reader only.
	// The mutex is shared across all streams because the underlying connection is shared.
	connWriteLock *sync.Mutex
	id            byte
	// writeBuf is a reusable buffer for building [streamID][data] messages.
	// This avoids allocating a new buffer for every Write() call.
	writeBuf []byte
}

func (s *stream) Read(p []byte) (n int, err error) {
	return s.readPipe.Read(p)
}

// Write writes directly to the underlying WebSocket connection.
// The message is sent atomically with the stream ID prepended.
func (s *stream) Write(p []byte) (n int, err error) {
	klog.V(8).Infof("Write() on stream %d", s.id)
	defer klog.V(8).Infof("Write() done on stream %d", s.id)
	s.connWriteLock.Lock()
	defer s.connWriteLock.Unlock()
	if s.conn == nil {
		return 0, fmt.Errorf("write on closed stream %d", s.id)
	}

	// Create a context with write deadline for the entire write operation.
	ctx, cancel := context.WithTimeout(s.ctx, writeDeadline)
	defer cancel()

	// Build message with stream ID prefix, reusing buffer to avoid allocations.
	// We must combine [streamID][data] into a single buffer because coder/websocket
	// sends each Write() as a separate frame immediately (unlike gorilla/websocket
	// which buffered until Close()). The server expects streamID and data together
	// in the same message.
	needed := 1 + len(p)
	if cap(s.writeBuf) < needed {
		s.writeBuf = make([]byte, needed)
	} else {
		s.writeBuf = s.writeBuf[:needed]
	}
	s.writeBuf[0] = s.id
	copy(s.writeBuf[1:], p)

	err = s.conn.Write(ctx, websocket.MessageBinary, s.writeBuf)
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

// Close half-closes the stream, indicating this side is finished with the stream.
func (s *stream) Close() error {
	klog.V(6).Infof("Close() on stream %d", s.id)
	defer klog.V(6).Infof("Close() done on stream %d", s.id)
	s.connWriteLock.Lock()
	defer s.connWriteLock.Unlock()
	if s.conn == nil {
		return fmt.Errorf("Close() on already closed stream %d", s.id)
	}

	ctx, cancel := context.WithTimeout(s.ctx, writeDeadline)
	defer cancel()

	// Communicate the CLOSE stream signal to the other websocket endpoint.
	err := s.conn.Write(ctx, websocket.MessageBinary, []byte{remotecommand.StreamClose, s.id})
	s.conn = nil
	return err
}

func (s *stream) Reset() error {
	klog.V(4).Infof("Reset() on stream %d", s.id)
	defer klog.V(4).Infof("Reset() done on stream %d", s.id)
	s.Close()
	return s.writePipe.Close()
}

func (s *stream) Headers() http.Header {
	return s.headers
}

func (s *stream) Identifier() uint32 {
	return uint32(s.id)
}
