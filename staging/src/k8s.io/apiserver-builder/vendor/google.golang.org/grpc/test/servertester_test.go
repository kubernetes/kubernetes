/*
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package grpc_test

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
)

// This is a subset of http2's serverTester type.
//
// serverTester wraps a io.ReadWriter (acting like the underlying
// network connection) and provides utility methods to read and write
// http2 frames.
//
// NOTE(bradfitz): this could eventually be exported somewhere. Others
// have asked for it too. For now I'm still experimenting with the
// API and don't feel like maintaining a stable testing API.

type serverTester struct {
	cc io.ReadWriteCloser // client conn
	t  testing.TB
	fr *http2.Framer

	// writing headers:
	headerBuf bytes.Buffer
	hpackEnc  *hpack.Encoder

	// reading frames:
	frc       chan http2.Frame
	frErrc    chan error
	readTimer *time.Timer
}

func newServerTesterFromConn(t testing.TB, cc io.ReadWriteCloser) *serverTester {
	st := &serverTester{
		t:      t,
		cc:     cc,
		frc:    make(chan http2.Frame, 1),
		frErrc: make(chan error, 1),
	}
	st.hpackEnc = hpack.NewEncoder(&st.headerBuf)
	st.fr = http2.NewFramer(cc, cc)
	st.fr.ReadMetaHeaders = hpack.NewDecoder(4096 /*initialHeaderTableSize*/, nil)

	return st
}

func (st *serverTester) readFrame() (http2.Frame, error) {
	go func() {
		fr, err := st.fr.ReadFrame()
		if err != nil {
			st.frErrc <- err
		} else {
			st.frc <- fr
		}
	}()
	t := time.NewTimer(2 * time.Second)
	defer t.Stop()
	select {
	case f := <-st.frc:
		return f, nil
	case err := <-st.frErrc:
		return nil, err
	case <-t.C:
		return nil, errors.New("timeout waiting for frame")
	}
}

// greet initiates the client's HTTP/2 connection into a state where
// frames may be sent.
func (st *serverTester) greet() {
	st.writePreface()
	st.writeInitialSettings()
	st.wantSettings()
	st.writeSettingsAck()
	for {
		f, err := st.readFrame()
		if err != nil {
			st.t.Fatal(err)
		}
		switch f := f.(type) {
		case *http2.WindowUpdateFrame:
			// grpc's transport/http2_server sends this
			// before the settings ack. The Go http2
			// server uses a setting instead.
		case *http2.SettingsFrame:
			if f.IsAck() {
				return
			}
			st.t.Fatalf("during greet, got non-ACK settings frame")
		default:
			st.t.Fatalf("during greet, unexpected frame type %T", f)
		}
	}
}

func (st *serverTester) writePreface() {
	n, err := st.cc.Write([]byte(http2.ClientPreface))
	if err != nil {
		st.t.Fatalf("Error writing client preface: %v", err)
	}
	if n != len(http2.ClientPreface) {
		st.t.Fatalf("Writing client preface, wrote %d bytes; want %d", n, len(http2.ClientPreface))
	}
}

func (st *serverTester) writeInitialSettings() {
	if err := st.fr.WriteSettings(); err != nil {
		st.t.Fatalf("Error writing initial SETTINGS frame from client to server: %v", err)
	}
}

func (st *serverTester) writeSettingsAck() {
	if err := st.fr.WriteSettingsAck(); err != nil {
		st.t.Fatalf("Error writing ACK of server's SETTINGS: %v", err)
	}
}

func (st *serverTester) wantSettings() *http2.SettingsFrame {
	f, err := st.readFrame()
	if err != nil {
		st.t.Fatalf("Error while expecting a SETTINGS frame: %v", err)
	}
	sf, ok := f.(*http2.SettingsFrame)
	if !ok {
		st.t.Fatalf("got a %T; want *SettingsFrame", f)
	}
	return sf
}

func (st *serverTester) wantSettingsAck() {
	f, err := st.readFrame()
	if err != nil {
		st.t.Fatal(err)
	}
	sf, ok := f.(*http2.SettingsFrame)
	if !ok {
		st.t.Fatalf("Wanting a settings ACK, received a %T", f)
	}
	if !sf.IsAck() {
		st.t.Fatal("Settings Frame didn't have ACK set")
	}
}

// wait for any activity from the server
func (st *serverTester) wantAnyFrame() http2.Frame {
	f, err := st.fr.ReadFrame()
	if err != nil {
		st.t.Fatal(err)
	}
	return f
}

func (st *serverTester) encodeHeaderField(k, v string) {
	err := st.hpackEnc.WriteField(hpack.HeaderField{Name: k, Value: v})
	if err != nil {
		st.t.Fatalf("HPACK encoding error for %q/%q: %v", k, v, err)
	}
}

// encodeHeader encodes headers and returns their HPACK bytes. headers
// must contain an even number of key/value pairs.  There may be
// multiple pairs for keys (e.g. "cookie").  The :method, :path, and
// :scheme headers default to GET, / and https.
func (st *serverTester) encodeHeader(headers ...string) []byte {
	if len(headers)%2 == 1 {
		panic("odd number of kv args")
	}

	st.headerBuf.Reset()

	if len(headers) == 0 {
		// Fast path, mostly for benchmarks, so test code doesn't pollute
		// profiles when we're looking to improve server allocations.
		st.encodeHeaderField(":method", "GET")
		st.encodeHeaderField(":path", "/")
		st.encodeHeaderField(":scheme", "https")
		return st.headerBuf.Bytes()
	}

	if len(headers) == 2 && headers[0] == ":method" {
		// Another fast path for benchmarks.
		st.encodeHeaderField(":method", headers[1])
		st.encodeHeaderField(":path", "/")
		st.encodeHeaderField(":scheme", "https")
		return st.headerBuf.Bytes()
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
			st.encodeHeaderField(k, v)
		}
	}
	return st.headerBuf.Bytes()
}

func (st *serverTester) writeHeadersGRPC(streamID uint32, path string) {
	st.writeHeaders(http2.HeadersFrameParam{
		StreamID: streamID,
		BlockFragment: st.encodeHeader(
			":method", "POST",
			":path", path,
			"content-type", "application/grpc",
			"te", "trailers",
		),
		EndStream:  false,
		EndHeaders: true,
	})
}

func (st *serverTester) writeHeaders(p http2.HeadersFrameParam) {
	if err := st.fr.WriteHeaders(p); err != nil {
		st.t.Fatalf("Error writing HEADERS: %v", err)
	}
}

func (st *serverTester) writeData(streamID uint32, endStream bool, data []byte) {
	if err := st.fr.WriteData(streamID, endStream, data); err != nil {
		st.t.Fatalf("Error writing DATA: %v", err)
	}
}

func (st *serverTester) writeRSTStream(streamID uint32, code http2.ErrCode) {
	if err := st.fr.WriteRSTStream(streamID, code); err != nil {
		st.t.Fatalf("Error writing RST_STREAM: %v", err)
	}
}
