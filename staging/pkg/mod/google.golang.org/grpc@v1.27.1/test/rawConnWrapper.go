/*
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package test

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
)

type listenerWrapper struct {
	net.Listener
	mu  sync.Mutex
	rcw *rawConnWrapper
}

func listenWithConnControl(network, address string) (net.Listener, error) {
	l, err := net.Listen(network, address)
	if err != nil {
		return nil, err
	}
	return &listenerWrapper{Listener: l}, nil
}

// Accept blocks until Dial is called, then returns a net.Conn for the server
// half of the connection.
func (l *listenerWrapper) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	l.mu.Lock()
	l.rcw = newRawConnWrapperFromConn(c)
	l.mu.Unlock()
	return c, nil
}

func (l *listenerWrapper) getLastConn() *rawConnWrapper {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.rcw
}

type dialerWrapper struct {
	c   net.Conn
	rcw *rawConnWrapper
}

func (d *dialerWrapper) dialer(target string, t time.Duration) (net.Conn, error) {
	c, err := net.DialTimeout("tcp", target, t)
	d.c = c
	d.rcw = newRawConnWrapperFromConn(c)
	return c, err
}

func (d *dialerWrapper) getRawConnWrapper() *rawConnWrapper {
	return d.rcw
}

type rawConnWrapper struct {
	cc io.ReadWriteCloser
	fr *http2.Framer

	// writing headers:
	headerBuf bytes.Buffer
	hpackEnc  *hpack.Encoder

	// reading frames:
	frc    chan http2.Frame
	frErrc chan error
}

func newRawConnWrapperFromConn(cc io.ReadWriteCloser) *rawConnWrapper {
	rcw := &rawConnWrapper{
		cc:     cc,
		frc:    make(chan http2.Frame, 1),
		frErrc: make(chan error, 1),
	}
	rcw.hpackEnc = hpack.NewEncoder(&rcw.headerBuf)
	rcw.fr = http2.NewFramer(cc, cc)
	rcw.fr.ReadMetaHeaders = hpack.NewDecoder(4096 /*initialHeaderTableSize*/, nil)

	return rcw
}

func (rcw *rawConnWrapper) Close() error {
	return rcw.cc.Close()
}

func (rcw *rawConnWrapper) encodeHeaderField(k, v string) error {
	err := rcw.hpackEnc.WriteField(hpack.HeaderField{Name: k, Value: v})
	if err != nil {
		return fmt.Errorf("HPACK encoding error for %q/%q: %v", k, v, err)
	}
	return nil
}

// encodeRawHeader is for usage on both client and server side to construct header based on the input
// key, value pairs.
func (rcw *rawConnWrapper) encodeRawHeader(headers ...string) []byte {
	if len(headers)%2 == 1 {
		panic("odd number of kv args")
	}

	rcw.headerBuf.Reset()

	pseudoCount := map[string]int{}
	var keys []string
	vals := map[string][]string{}

	for len(headers) > 0 {
		k, v := headers[0], headers[1]
		headers = headers[2:]
		if _, ok := vals[k]; !ok {
			keys = append(keys, k)
		}
		if strings.HasPrefix(k, ":") {
			pseudoCount[k]++
			if pseudoCount[k] == 1 {
				vals[k] = []string{v}
			} else {
				// Allows testing of invalid headers w/ dup pseudo fields.
				vals[k] = append(vals[k], v)
			}
		} else {
			vals[k] = append(vals[k], v)
		}
	}
	for _, k := range keys {
		for _, v := range vals[k] {
			rcw.encodeHeaderField(k, v)
		}
	}
	return rcw.headerBuf.Bytes()
}

// encodeHeader is for usage on client side to write request header.
//
// encodeHeader encodes headers and returns their HPACK bytes. headers
// must contain an even number of key/value pairs.  There may be
// multiple pairs for keys (e.g. "cookie").  The :method, :path, and
// :scheme headers default to GET, / and https.
func (rcw *rawConnWrapper) encodeHeader(headers ...string) []byte {
	if len(headers)%2 == 1 {
		panic("odd number of kv args")
	}

	rcw.headerBuf.Reset()

	if len(headers) == 0 {
		// Fast path, mostly for benchmarks, so test code doesn't pollute
		// profiles when we're looking to improve server allocations.
		rcw.encodeHeaderField(":method", "GET")
		rcw.encodeHeaderField(":path", "/")
		rcw.encodeHeaderField(":scheme", "https")
		return rcw.headerBuf.Bytes()
	}

	if len(headers) == 2 && headers[0] == ":method" {
		// Another fast path for benchmarks.
		rcw.encodeHeaderField(":method", headers[1])
		rcw.encodeHeaderField(":path", "/")
		rcw.encodeHeaderField(":scheme", "https")
		return rcw.headerBuf.Bytes()
	}

	pseudoCount := map[string]int{}
	keys := []string{":method", ":path", ":scheme"}
	vals := map[string][]string{
		":method": {"GET"},
		":path":   {"/"},
		":scheme": {"https"},
	}
	for len(headers) > 0 {
		k, v := headers[0], headers[1]
		headers = headers[2:]
		if _, ok := vals[k]; !ok {
			keys = append(keys, k)
		}
		if strings.HasPrefix(k, ":") {
			pseudoCount[k]++
			if pseudoCount[k] == 1 {
				vals[k] = []string{v}
			} else {
				// Allows testing of invalid headers w/ dup pseudo fields.
				vals[k] = append(vals[k], v)
			}
		} else {
			vals[k] = append(vals[k], v)
		}
	}
	for _, k := range keys {
		for _, v := range vals[k] {
			rcw.encodeHeaderField(k, v)
		}
	}
	return rcw.headerBuf.Bytes()
}

func (rcw *rawConnWrapper) writeHeaders(p http2.HeadersFrameParam) error {
	if err := rcw.fr.WriteHeaders(p); err != nil {
		return fmt.Errorf("error writing HEADERS: %v", err)
	}
	return nil
}

func (rcw *rawConnWrapper) writeRSTStream(streamID uint32, code http2.ErrCode) error {
	if err := rcw.fr.WriteRSTStream(streamID, code); err != nil {
		return fmt.Errorf("error writing RST_STREAM: %v", err)
	}
	return nil
}

func (rcw *rawConnWrapper) writeGoAway(maxStreamID uint32, code http2.ErrCode, debugData []byte) error {
	if err := rcw.fr.WriteGoAway(maxStreamID, code, debugData); err != nil {
		return fmt.Errorf("error writing GoAway: %v", err)
	}
	return nil
}

func (rcw *rawConnWrapper) writeRawFrame(t http2.FrameType, flags http2.Flags, streamID uint32, payload []byte) error {
	if err := rcw.fr.WriteRawFrame(t, flags, streamID, payload); err != nil {
		return fmt.Errorf("error writing Raw Frame: %v", err)
	}
	return nil
}
