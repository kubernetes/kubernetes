/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"io"
	"net/http"
	"sync"

	gwebsocket "github.com/gorilla/websocket"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport/websocket"
	"k8s.io/klog/v2"
)

const (
	// These constants match:
	// - pkg/kubelet/cri/streaming/remotecommand/websocket.go
	// - staging/src/k8s.io/apiserver/pkg/util/wsstream/conn.go

	streamStdIn  = 0
	streamStdOut = 1
	streamStdErr = 2
	streamErr    = 3
	streamResize = 4
)

var (
	_ Executor          = &wsStreamExecutor{}
	_ streamCreator     = &wsStreamCreator{}
	_ httpstream.Stream = &stream{}

	streamType2streamID = map[string]byte{
		v1.StreamTypeStdin:  streamStdIn,
		v1.StreamTypeStdout: streamStdOut,
		v1.StreamTypeStderr: streamStdErr,
		v1.StreamTypeError:  streamErr,
		v1.StreamTypeResize: streamResize,
	}
)

// wsStreamExecutor handles transporting standard shell streams over an httpstream connection.
type wsStreamExecutor struct {
	config    *restclient.Config
	method    string
	url       string
	protocols []string
}

// NewWebSocketExecutor allows to execute commands via a WebSocket connection.
func NewWebSocketExecutor(config *restclient.Config, method, url string) Executor {
	return &wsStreamExecutor{
		config:    config,
		method:    method,
		url:       url,
		protocols: []string{remotecommand.StreamProtocolV4Name, remotecommand.StreamProtocolV1Name},
	}
}

func (e *wsStreamExecutor) Stream(options StreamOptions) error {
	req, err := http.NewRequest(e.method, e.url, nil)
	if err != nil {
		return err
	}
	rt, wsRt, err := websocket.RoundTripperFor(e.config)
	if err != nil {
		return err
	}
	conn, err := websocket.Negotiate(rt, wsRt, req, e.protocols...)
	if err != nil {
		return err
	}
	defer conn.Close()

	var streamer streamProtocolHandler

	streamingProto := conn.Subprotocol()
	klog.V(4).Infof("The protocol is %s", streamingProto)

	switch streamingProto {
	case remotecommand.StreamProtocolV4Name:
		streamer = newStreamProtocolV4(options)
	case "":
		klog.V(4).Infof("The server did not negotiate a streaming protocol version. Falling back to %s", remotecommand.StreamProtocolV1Name)
		fallthrough
	case remotecommand.StreamProtocolV1Name:
		streamer = newStreamProtocolV1(options)
	default:
		return fmt.Errorf("unsupported streaming protocol: %q", streamingProto)
	}

	return streamer.stream(newWSStreamCreator(conn))
}

type wsStreamCreator struct {
	conn    *gwebsocket.Conn
	connMu  sync.Mutex
	streams map[byte]*stream
}

func newWSStreamCreator(conn *gwebsocket.Conn) *wsStreamCreator {
	return &wsStreamCreator{
		conn:    conn,
		streams: map[byte]*stream{},
	}
}

func (c *wsStreamCreator) CreateStream(headers http.Header) (httpstream.Stream, error) {
	streamType := headers.Get(v1.StreamType)
	id, ok := streamType2streamID[streamType]
	if !ok {
		return nil, fmt.Errorf("unknown stream type: %s", streamType)
	}
	if c.streams[id] != nil {
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
	c.streams[id] = s
	return s, nil
}

func (c *wsStreamCreator) Run() {
	readBuffer := make([]byte, 32*1024) // same as io.Copy()
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
		s := c.streams[streamID]
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
	klog.Infof("Write() on stream %d", s.id)
	defer klog.Infof("Write() done on stream %d", s.id)
	s.connMu.Lock()
	defer s.connMu.Unlock()
	if s.conn == nil {
		return 0, fmt.Errorf("write on closed stream %d", s.id)
	}
	// TODO s.conn.SetWriteDeadline()
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
	klog.Infof("Close() on stream %d", s.id)
	defer klog.Infof("Close() done on stream %d", s.id)
	s.connMu.Lock()
	defer s.connMu.Unlock()
	s.conn = nil
	// TODO See https://github.com/kubernetes/kubernetes/issues/89899#issuecomment-1132502190.
	return nil
}

func (s *stream) Reset() error {
	klog.Infof("Reset() on stream %d", s.id)
	defer klog.Infof("Reset() done on stream %d", s.id)
	s.connMu.Lock()
	defer s.connMu.Unlock()
	s.conn = nil
	// TODO send half-close, interrupt and reads/writes on pipes, and start discarding any incoming messages for this stream.
	return s.writePipe.Close()
}

func (s *stream) Headers() http.Header {
	return s.headers
}

func (s *stream) Identifier() uint32 {
	return uint32(s.id)
}
