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

	gwebsocket "github.com/gorilla/websocket"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport/websocket"
	"k8s.io/klog/v2"
)

const (
	pingReadDeadline = 60 * time.Second
	// should be less than pingReadDeadline
	pingWriteDeadline = 10 * time.Second
	PingPeriod        = 5 * time.Second
)

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
	upgrader  websocket.ConnectionHolder
	method    string
	url       string
	protocols []string
}

// NewWebSocketExecutor allows to execute commands via a WebSocket connection.
func NewWebSocketExecutor(config *restclient.Config, method, url string) (Executor, error) {
	transport, upgrader, err := websocket.RoundTripperFor(config)
	if err != nil {
		return nil, fmt.Errorf("error creating websocket transports: %v", err)
	}
	return &wsStreamExecutor{
		transport: transport,
		upgrader:  upgrader,
		method:    method,
		url:       url,
		// Only supports V5 protocol for correct version skew functionality.
		// Previous api servers will proxy upgrade requests to legacy websocket
		// servers on container runtimes which support V1-V4. These legacy
		// websocket servers will not handle the new CLOSE signal.
		protocols: []string{remotecommand.StreamProtocolV5Name},
	}, nil
}

// Deprecated: use StreamWithContext instead to avoid possible resource leaks.
// See https://github.com/kubernetes/kubernetes/pull/103177 for details.
func (e *wsStreamExecutor) Stream(options StreamOptions) error {
	return e.StreamWithContext(context.Background(), options)
}

func (e *wsStreamExecutor) StreamWithContext(ctx context.Context, options StreamOptions) error {
	req, err := http.NewRequestWithContext(ctx, e.method, e.url, nil)
	if err != nil {
		return err
	}
	conn, err := websocket.Negotiate(e.transport, e.upgrader, req, e.protocols...)
	if err != nil {
		return err
	}
	if conn == nil {
		panic(fmt.Errorf("websocket connection is nil"))
	}
	defer conn.Close()
	protocol := conn.Subprotocol()
	klog.V(4).Infof("The subprotocol is %s", protocol)

	var streamer streamProtocolHandler
	switch protocol {
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
		creator := newWSStreamCreator(conn)
		go creator.run(e.upgrader.DataBufferSize()) // connection read/stream write loop in its own goroutine.
		errorChan <- streamer.stream(creator)
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
	conn      *gwebsocket.Conn
	connMu    sync.Mutex
	streams   map[byte]*stream
	streamsMu sync.Mutex
}

func newWSStreamCreator(conn *gwebsocket.Conn) *wsStreamCreator {
	ws := wsStreamCreator{
		conn:    conn,
		streams: map[byte]*stream{},
	}

	go ws.sendPings(PingPeriod) // start heartbeat

	return &ws
}

func (c *wsStreamCreator) getStream(id byte) *stream {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	return c.streams[id]
}

func (c *wsStreamCreator) setStream(id byte, s *stream) {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	c.streams[id] = s
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
		headers:   headers,
		readPipe:  reader,
		writePipe: writer,
		conn:      c.conn,
		connMu:    &c.connMu,
		id:        id,
	}
	c.setStream(id, s)
	return s, nil
}

// sendPings starts heartbeat sending a "ping" from the endpoint every "period".
func (c *wsStreamCreator) sendPings(period time.Duration) {
	c.connMu.Lock()
	closedCh := make(chan struct{})
	closeHandler := c.conn.CloseHandler()
	c.conn.SetCloseHandler(func(code int, text string) error {
		close(closedCh)
		return closeHandler(code, text)
	})
	c.connMu.Unlock()
	t := time.NewTicker(period)
	defer t.Stop()
	for {
		select {
		case <-closedCh:
			return
		case <-t.C:
			c.connMu.Lock()
			if err := c.conn.WriteControl(gwebsocket.PingMessage, nil, time.Now().Add(period)); err != nil {
				klog.V(7).Infof("Websocket Ping failed: %v", err)
				// Continue, in case this is a transient failure.
				// c.conn.CloseChan above will tell us when the connection is
				// actually closed.
			} else {
				klog.V(7).Infof("Websocket Ping succeeeded")
			}
			c.connMu.Unlock()
		}
	}
}

// run is executed in its own goroutine, reading the connection and demultiplexing
// the data messages into individual streams.
func (c *wsStreamCreator) run(bufferSize int) {
	// Buffer size must correspond to the same size allocated
	// for the read buffer during websocket client creation. A
	// difference can cause incomplete connection reads.
	readBuffer := make([]byte, bufferSize)
	// Set up handler for ping/pong heartbeat.
	if err := c.conn.SetReadDeadline(time.Now().Add(pingReadDeadline)); err != nil {
		klog.V(7).Infof("Websocket setting read deadline failed %v", err)
	}
	c.conn.SetPongHandler(func(string) error {
		err := c.conn.SetReadDeadline(time.Now().Add(pingReadDeadline))
		if err != nil {
			klog.V(7).Infof("Websocket setting read deadline failed %v", err)
		}
		return nil
	})
	for {
		messageType, r, err := c.conn.NextReader()
		if err != nil {
			websocketErr, ok := err.(*gwebsocket.CloseError)
			if ok && websocketErr.Code == gwebsocket.CloseNormalClosure {
				err = nil // readers will get io.EOF as it's a normal closure
			} else {
				err = fmt.Errorf("next reader: %w", err)
			}
			c.closeAllStreamReaders(err)
			return
		}
		if messageType != gwebsocket.BinaryMessage {
			c.closeAllStreamReaders(fmt.Errorf("unexpected message type: %d", messageType))
			return
		}
		// it's ok to read just a single byte because the underlying library wraps the actual connection with
		// a buffered reader anyway.
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
				_, errWrite := s.writePipe.Write(readBuffer[:nr])
				if errWrite != nil {
					// Pipe must have been closed by the stream user. Nothing to do, discard the message.
					break
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
func (c *wsStreamCreator) closeAllStreamReaders(err error) {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()
	for _, s := range c.streams {
		// Closing writePipe unblocks all readPipe.Read() callers and prevents any future writes.
		_ = s.writePipe.CloseWithError(err)
	}
}

type stream struct {
	headers   http.Header
	readPipe  *io.PipeReader
	writePipe *io.PipeWriter
	// conn is used for writing directly into the connection.
	// Is nil after Close() / Reset() to prevent future writes.
	conn *gwebsocket.Conn
	// connMu protects conn against concurrent write operations. There must be a single writer and a single reader only.
	// The mutex is shared across all streams because the underlying connection is shared.
	connMu *sync.Mutex
	id     byte
}

func (s *stream) Read(p []byte) (n int, err error) {
	return s.readPipe.Read(p)
}

// Write writes directly to the underlying WebSocket connection.
func (s *stream) Write(p []byte) (n int, err error) {
	klog.V(4).Infof("Write() on stream %d", s.id)
	defer klog.V(4).Infof("Write() done on stream %d", s.id)
	s.connMu.Lock()
	defer s.connMu.Unlock()
	if s.conn == nil {
		return 0, fmt.Errorf("write on closed stream %d", s.id)
	}
	err = s.conn.SetWriteDeadline(time.Now().Add(pingWriteDeadline))
	if err != nil {
		klog.V(7).Infof("Websocket setting write deadline failed %v", err)
		return 0, err
	}
	// Message writer buffers the message data, so we don't need to do that ourselves.
	// Just write id and the data as two separate writes to avoid allocating an intermediate buffer.
	w, err := s.conn.NextWriter(gwebsocket.BinaryMessage)
	if err != nil {
		return 0, err
	}
	_, err = w.Write([]byte{s.id})
	if err != nil {
		_ = w.Close()
		return 0, err
	}
	n, err = w.Write(p)
	if err != nil {
		_ = w.Close()
		return n, err
	}
	return n, w.Close()
}

// Close half-closes the stream, indicating this side is finished with the stream.
func (s *stream) Close() error {
	klog.V(4).Infof("Close() on stream %d", s.id)
	defer klog.V(4).Infof("Close() done on stream %d", s.id)
	s.connMu.Lock()
	defer s.connMu.Unlock()
	if s.conn == nil {
		return fmt.Errorf("Close() on already closed stream %d", s.id)
	}
	s.conn.WriteMessage(gwebsocket.BinaryMessage, []byte{remotecommand.StreamClose, s.id})
	s.conn = nil
	return nil
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
