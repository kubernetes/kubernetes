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

package framer

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	netutil "k8s.io/apimachinery/pkg/util/net"
)

func TestRead(t *testing.T) {
	data := []byte{
		0x00, 0x00, 0x00, 0x04,
		0x01, 0x02, 0x03, 0x04,
		0x00, 0x00, 0x00, 0x03,
		0x05, 0x06, 0x07,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01,
		0x08,
	}
	b := bytes.NewBuffer(data)
	r := NewLengthDelimitedFrameReader(io.NopCloser(b))
	buf := make([]byte, 1)
	if n, err := r.Read(buf); err != io.ErrShortBuffer && n != 1 && bytes.Equal(buf, []byte{0x01}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer && n != 1 && bytes.Equal(buf, []byte{0x02}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read the remaining frame
	buf = make([]byte, 2)
	if n, err := r.Read(buf); err != nil && n != 2 && bytes.Equal(buf, []byte{0x03, 0x04}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read with buffer equal to frame
	buf = make([]byte, 3)
	if n, err := r.Read(buf); err != nil && n != 3 && bytes.Equal(buf, []byte{0x05, 0x06, 0x07}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read empty frame
	buf = make([]byte, 3)
	if n, err := r.Read(buf); err != nil && n != 0 && bytes.Equal(buf, []byte{}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read with larger buffer than frame
	buf = make([]byte, 3)
	if n, err := r.Read(buf); err != nil && n != 1 && bytes.Equal(buf, []byte{0x08}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read EOF
	if n, err := r.Read(buf); err != io.EOF && n != 0 {
		t.Fatalf("unexpected: %v %d", err, n)
	}
}

func TestReadLarge(t *testing.T) {
	data := []byte{
		0x00, 0x00, 0x00, 0x04,
		0x01, 0x02, 0x03, 0x04,
		0x00, 0x00, 0x00, 0x03,
		0x05, 0x06, 0x07,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01,
		0x08,
	}
	b := bytes.NewBuffer(data)
	r := NewLengthDelimitedFrameReader(io.NopCloser(b))
	buf := make([]byte, 40)
	if n, err := r.Read(buf); err != nil && n != 4 && bytes.Equal(buf, []byte{0x01, 0x02, 0x03, 0x04}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil && n != 3 && bytes.Equal(buf, []byte{0x05, 0x06, 0x7}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil && n != 0 && bytes.Equal(buf, []byte{}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil && n != 1 && bytes.Equal(buf, []byte{0x08}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read EOF
	if n, err := r.Read(buf); err != io.EOF && n != 0 {
		t.Fatalf("unexpected: %v %d", err, n)
	}
}

func TestReadInvalidFrame(t *testing.T) {
	data := []byte{
		0x00, 0x00, 0x00, 0x04,
		0x01, 0x02,
	}
	b := bytes.NewBuffer(data)
	r := NewLengthDelimitedFrameReader(io.NopCloser(b))
	buf := make([]byte, 1)
	if n, err := r.Read(buf); err != io.ErrShortBuffer && n != 1 && bytes.Equal(buf, []byte{0x01}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read the remaining frame
	buf = make([]byte, 3)
	if n, err := r.Read(buf); err != io.ErrUnexpectedEOF && n != 1 && bytes.Equal(buf, []byte{0x02}) {
		t.Fatalf("unexpected: %v %d %v", err, n, buf)
	}
	// read EOF
	if n, err := r.Read(buf); err != io.EOF && n != 0 {
		t.Fatalf("unexpected: %v %d", err, n)
	}
}

func TestReadClientTimeout(t *testing.T) {
	header := []byte{
		0x00, 0x00, 0x00, 0x04,
	}
	data := []byte{
		0x01, 0x02, 0x03, 0x04,
		0x00, 0x00, 0x00, 0x03,
		0x05, 0x06, 0x07,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01,
		0x08,
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(header)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		time.Sleep(1 * time.Second)
		_, _ = w.Write(data)
	}))
	defer server.Close()

	client := &http.Client{
		Timeout: 500 * time.Millisecond,
	}

	resp, err := client.Get(server.URL)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	r := NewLengthDelimitedFrameReader(resp.Body)
	buf := make([]byte, 1)
	if n, err := r.Read(buf); err == nil || !netutil.IsTimeout(err) {
		t.Fatalf("unexpected: %v %d", err, n)
	}
}

func TestJSONFrameReader(t *testing.T) {
	b := bytes.NewBufferString("{\"test\":true}\n1\n[\"a\"]")
	r := NewJSONFramedReader(io.NopCloser(b))
	buf := make([]byte, 20)
	if n, err := r.Read(buf); err != nil || n != 13 || string(buf[:n]) != `{"test":true}` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 1 || string(buf[:n]) != `1` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 5 || string(buf[:n]) != `["a"]` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.EOF || n != 0 {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
}

func TestJSONFrameReaderShortBuffer(t *testing.T) {
	b := bytes.NewBufferString("{\"test\":true}\n1\n[\"a\"]")
	r := NewJSONFramedReader(io.NopCloser(b))
	buf := make([]byte, 3)

	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `{"t` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `est` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `":t` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `rue` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 1 || string(buf[:n]) != `}` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}

	if n, err := r.Read(buf); err != nil || n != 1 || string(buf[:n]) != `1` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}

	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `["a` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 2 || string(buf[:n]) != `"]` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}

	if n, err := r.Read(buf); err != io.EOF || n != 0 {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
}

func TestJSONFrameReaderShortBufferNoUnderlyingArrayReuse(t *testing.T) {
	b := bytes.NewBufferString("{}")
	r := NewJSONFramedReader(io.NopCloser(b))
	buf := make([]byte, 1, 2) // cap(buf) > len(buf) && cap(buf) <= len("{}")

	if n, err := r.Read(buf); !errors.Is(err, io.ErrShortBuffer) || n != 1 || string(buf[:n]) != "{" {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	buf = append(buf, make([]byte, 1)...) // stomps the second byte of the backing array
	if n, err := r.Read(buf[1:]); err != nil || n != 1 || string(buf[1:1+n]) != "}" {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
}

func TestLineDelimitedFrameReader(t *testing.T) {
	b := bytes.NewBufferString("{\"test\":true}\n1\n[\"a\"]\n")
	r := NewLineDelimitedFrameReader(io.NopCloser(b))
	buf := make([]byte, 20)
	if n, err := r.Read(buf); err != nil || n != 13 || string(buf[:n]) != `{"test":true}` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 1 || string(buf[:n]) != `1` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 5 || string(buf[:n]) != `["a"]` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.EOF || n != 0 {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
}

func TestLineDelimitedFrameReaderShortBuffer(t *testing.T) {
	b := bytes.NewBufferString("{\"test\":true}\n1\n[\"a\"]\n")
	r := NewLineDelimitedFrameReader(io.NopCloser(b))
	buf := make([]byte, 3)

	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `{"t` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `est` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `":t` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `rue` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 1 || string(buf[:n]) != `}` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 1 || string(buf[:n]) != `1` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.ErrShortBuffer || n != 3 || string(buf[:n]) != `["a` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != nil || n != 2 || string(buf[:n]) != `"]` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
	if n, err := r.Read(buf); err != io.EOF || n != 0 {
		t.Fatalf("unexpected: %v %d %q", err, n, buf)
	}
}

func TestLineDelimitedFrameReaderEmptyLines(t *testing.T) {
	// Stray empty lines between frames must not surface as empty frames.
	b := bytes.NewBufferString("\n\n{\"a\":1}\n\n{\"b\":2}\n\n")
	r := NewLineDelimitedFrameReader(io.NopCloser(b))
	buf := make([]byte, 16)
	if n, err := r.Read(buf); err != nil || string(buf[:n]) != `{"a":1}` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf[:n])
	}
	if n, err := r.Read(buf); err != nil || string(buf[:n]) != `{"b":2}` {
		t.Fatalf("unexpected: %v %d %q", err, n, buf[:n])
	}
	if n, err := r.Read(buf); err != io.EOF || n != 0 {
		t.Fatalf("unexpected: %v %d", err, n)
	}
}

func TestLineDelimitedFrameReaderTrailingNoNewline(t *testing.T) {
	// A trailing frame without a terminating `\n` is returned as a
	// complete frame at EOF (matches bufio.ScanLines and is needed because
	// the apiserver's JSON encoder omits the trailing newline on
	// single-frame responses). The next call returns io.EOF.
	b := bytes.NewBufferString(`{"a":1}` + "\n" + `{"b":2}`)
	r := NewLineDelimitedFrameReader(io.NopCloser(b))
	buf := make([]byte, 16)
	if n, err := r.Read(buf); err != nil || string(buf[:n]) != `{"a":1}` {
		t.Fatalf("first frame: unexpected: %v %d %q", err, n, buf[:n])
	}
	if n, err := r.Read(buf); err != nil || string(buf[:n]) != `{"b":2}` {
		t.Fatalf("trailing frame: unexpected: %v %d %q", err, n, buf[:n])
	}
	if n, err := r.Read(buf); err != io.EOF || n != 0 {
		t.Fatalf("eof: unexpected: %v %d", err, n)
	}
}
