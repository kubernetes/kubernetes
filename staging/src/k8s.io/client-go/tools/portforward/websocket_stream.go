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
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	gwebsocket "github.com/gorilla/websocket"
	"github.com/mxk/go-flowrate/flowrate"

	"k8s.io/klog/v2"
)

const writeDeadline = 2 * time.Second

type wsStream struct {
	id        int
	headers   http.Header
	readPipe  *io.PipeReader
	writePipe *io.PipeWriter
	// conn is used for writing directly into the connection.
	// Is nil after Close() / Reset() to prevent future writes.
	conn *gwebsocket.Conn
	// connWriteLock protects conn against concurrent write operations. There must be a single writer and a single reader only.
	// The mutex is shared across all streams because the underlying connection is shared.
	connWriteLock *sync.Mutex
	// Used to throttle stream writing.
	maxBytesPerSec int64
}

const (
	// headerSizeBytes is the number of bytes to store the encoded header size.
	// A uint16 requires 2 bytes.
	headerSizeBytes = 2
	// maxHeaderSize equals the largest websocket stream header allowed.
	maxHeaderSize = 4 * 1024
)

// wsStreamHeader encloses the metadata prepended to a websocket binary message.
type wsStreamHeader struct {
	StreamType int // Data, Close, or Create
	StreamID   int
	Headers    http.Header
}

// String returns the websocket stream header as a string.
func (wsh *wsStreamHeader) String() string {
	return fmt.Sprintf("%d/%d", wsh.StreamID, wsh.StreamType)
}

// encodeStreamHeaders returns an encoded wsStreamHeader struct as a byte
// array. The first two bytes of the array are an integer representing the size
// of the encodeWsStreamHeaders. Returns an error if one occurs.
func encodeWsStreamHeaders(streamType int, streamID int, headers http.Header) ([]byte, error) {
	// Encode the wsStreamHeader struct using the passed parameters.
	var buf bytes.Buffer
	wsHeader := wsStreamHeader{
		StreamType: streamType,
		StreamID:   streamID,
		Headers:    headers,
	}
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(&wsHeader)
	if err != nil {
		klog.Errorf("Error encoding websocket stream headers: %v", err)
		return []byte{}, err
	}
	numBytes := uint16(len(buf.Bytes()))
	klog.V(5).Infof("Websocket stream header size: %d", numBytes)
	headerBuffer := make([]byte, headerSizeBytes)
	binary.LittleEndian.PutUint16(headerBuffer, numBytes)
	// Prepend the size of the encoded wsStreamHeader as a two byte
	// integer to the returned byte array.
	headerBuffer = append(headerBuffer, buf.Bytes()...)
	return headerBuffer, nil
}

// readWsStreamHeaders reads, decodes, and returns a pointer
// to a wsStreamHeader struct (or an error if one occurs). If
// successful, the reader parameter "r" points past the websocket
// stream header after the function returns (points to the beginning
// of the websocket binary data message).
func readWsStreamHeaders(r io.Reader) (*wsStreamHeader, error) {
	// Read the first two bytes containing an integer representing the
	// size of the encoded wsStreamHeader struct.
	headerBuffer := make([]byte, maxHeaderSize)
	_, err := io.ReadFull(r, headerBuffer[:headerSizeBytes])
	if err != nil {
		return nil, err
	}
	headerSize := binary.LittleEndian.Uint16(headerBuffer)
	// Read the next "headerSize" bytes into the headerBuffer
	_, err = io.ReadFull(r, headerBuffer[:headerSize])
	if err != nil {
		return nil, err
	}
	reader := bytes.NewReader(headerBuffer)
	dec := gob.NewDecoder(reader)
	// Decode these headerBuffer bytes into a wsStreamHeader struct.
	var wsHeaders wsStreamHeader
	err = dec.Decode(&wsHeaders)
	if err != nil {
		klog.Errorf("Error decoding wsStreamHeader: %v", err)
		return nil, err
	}
	return &wsHeaders, nil
}

// Read fills the passed "p" byte slice by reading from the stream's pipe. Another goroutine
// fills the pipe by writing the to the pipe in connection read loop after we have determined
// what stream to write to by inspecting the prepended stream id. Returns the number of
// bytes read or an error if one occurred reading the pipe.
func (s *wsStream) Read(p []byte) (n int, err error) {
	klog.V(5).Infof("Read() stream %d", s.id)
	defer klog.V(5).Infof("Read() done on stream %d", s.id)
	return s.readPipe.Read(p)
}

// Write writes directly to the underlying WebSocket connection.
func (s *wsStream) Write(p []byte) (n int, err error) {
	klog.V(5).Infof("Write() on stream %d", s.id)
	defer klog.V(5).Infof("Write() done on stream %d", s.id)
	s.connWriteLock.Lock()
	defer s.connWriteLock.Unlock()
	return s.writeWithHeaders(p, StreamData, http.Header{})
}

// writeWithHeaders writes the (possibly empty) byte array "p" plus stream headers including
// "messageType" (e.g. StreamData, StreamCreate) and "headers". Returns the number
// of bytes written (not including headers) or an error if one occurs. Important: There
// can only be one websocket connection writer active at a time.
func (s *wsStream) writeWithHeaders(p []byte, messageType int, headers http.Header) (n int, err error) {
	if s.conn == nil {
		return 0, fmt.Errorf("write on closed stream %d", s.id)
	}
	err = s.conn.SetWriteDeadline(time.Now().Add(writeDeadline))
	if err != nil {
		klog.V(4).Infof("Websocket setting write deadline failed %v", err)
		return 0, err
	}
	// Message writer buffers the message data, so we don't need to do that ourselves.
	// Just write id and the data as two separate writes to avoid allocating an
	// intermediate buffer. Only one writer can be active at a time.
	w, err := s.conn.NextWriter(gwebsocket.BinaryMessage)
	if err != nil {
		return 0, err
	}
	defer func() {
		if w != nil {
			w.Close() //nolint:errcheck
		}
	}()
	// Throttle rate of writing to websocket connection (there is only one Writer
	// that can be active on the connection at a time).
	if s.maxBytesPerSec > 0 {
		w = flowrate.NewWriter(w, s.maxBytesPerSec)
	}
	// Prepend the websocket stream headers to the writer.
	header, err := encodeWsStreamHeaders(messageType, s.id, headers)
	if err != nil {
		return 0, err
	}
	_, err = w.Write(header)
	if err != nil {
		return 0, err
	}
	// Next, write the passed data in "p".
	n, err = w.Write(p)
	if err != nil {
		return n, err
	}
	err = w.Close()
	w = nil
	return n, err
}

// Close half-closes the stream, indicating this side is finished with the stream.
func (s *wsStream) Close() error {
	klog.V(5).Infof("Close() on stream %d", s.id)
	defer klog.V(5).Infof("Close() done on stream %d", s.id)
	s.connWriteLock.Lock()
	defer s.connWriteLock.Unlock()
	if s.conn == nil {
		return fmt.Errorf("Close() on already closed stream %d", s.id)
	}
	// Send close signal to other endpoint.
	s.writeWithHeaders([]byte{}, StreamClose, http.Header{}) //nolint:errcheck
	s.conn = nil
	return s.writePipe.Close()
}

func (s *wsStream) Reset() error {
	klog.V(5).Infof("Reset() on stream %d", s.id)
	defer klog.V(5).Infof("Reset() done on stream %d", s.id)
	s.Close() //nolint:errcheck
	return s.writePipe.Close()
}

func (s *wsStream) Headers() http.Header {
	return s.headers
}

func (s *wsStream) Identifier() uint32 {
	return uint32(s.id)
}
