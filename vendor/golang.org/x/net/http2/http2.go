// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package http2 implements the HTTP/2 protocol.
//
// This package is low-level and intended to be used directly by very
// few people. Most users will use it indirectly through the automatic
// use by the net/http package (from Go 1.6 and later).
// For use in earlier Go versions see ConfigureServer. (Transport support
// requires Go 1.6 or later)
//
// See https://http2.github.io/ for more information on HTTP/2.
//
// See https://http2.golang.org/ for a test server running this code.
package http2

import (
	"bufio"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/net/lex/httplex"
)

var (
	VerboseLogs    bool
	logFrameWrites bool
	logFrameReads  bool
)

func init() {
	e := os.Getenv("GODEBUG")
	if strings.Contains(e, "http2debug=1") {
		VerboseLogs = true
	}
	if strings.Contains(e, "http2debug=2") {
		VerboseLogs = true
		logFrameWrites = true
		logFrameReads = true
	}
}

const (
	// ClientPreface is the string that must be sent by new
	// connections from clients.
	ClientPreface = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"

	// SETTINGS_MAX_FRAME_SIZE default
	// http://http2.github.io/http2-spec/#rfc.section.6.5.2
	initialMaxFrameSize = 16384

	// NextProtoTLS is the NPN/ALPN protocol negotiated during
	// HTTP/2's TLS setup.
	NextProtoTLS = "h2"

	// http://http2.github.io/http2-spec/#SettingValues
	initialHeaderTableSize = 4096

	initialWindowSize = 65535 // 6.9.2 Initial Flow Control Window Size

	defaultMaxReadFrameSize = 1 << 20
)

var (
	clientPreface = []byte(ClientPreface)
)

type streamState int

const (
	stateIdle streamState = iota
	stateOpen
	stateHalfClosedLocal
	stateHalfClosedRemote
	stateResvLocal
	stateResvRemote
	stateClosed
)

var stateName = [...]string{
	stateIdle:             "Idle",
	stateOpen:             "Open",
	stateHalfClosedLocal:  "HalfClosedLocal",
	stateHalfClosedRemote: "HalfClosedRemote",
	stateResvLocal:        "ResvLocal",
	stateResvRemote:       "ResvRemote",
	stateClosed:           "Closed",
}

func (st streamState) String() string {
	return stateName[st]
}

// Setting is a setting parameter: which setting it is, and its value.
type Setting struct {
	// ID is which setting is being set.
	// See http://http2.github.io/http2-spec/#SettingValues
	ID SettingID

	// Val is the value.
	Val uint32
}

func (s Setting) String() string {
	return fmt.Sprintf("[%v = %d]", s.ID, s.Val)
}

// Valid reports whether the setting is valid.
func (s Setting) Valid() error {
	// Limits and error codes from 6.5.2 Defined SETTINGS Parameters
	switch s.ID {
	case SettingEnablePush:
		if s.Val != 1 && s.Val != 0 {
			return ConnectionError(ErrCodeProtocol)
		}
	case SettingInitialWindowSize:
		if s.Val > 1<<31-1 {
			return ConnectionError(ErrCodeFlowControl)
		}
	case SettingMaxFrameSize:
		if s.Val < 16384 || s.Val > 1<<24-1 {
			return ConnectionError(ErrCodeProtocol)
		}
	}
	return nil
}

// A SettingID is an HTTP/2 setting as defined in
// http://http2.github.io/http2-spec/#iana-settings
type SettingID uint16

const (
	SettingHeaderTableSize      SettingID = 0x1
	SettingEnablePush           SettingID = 0x2
	SettingMaxConcurrentStreams SettingID = 0x3
	SettingInitialWindowSize    SettingID = 0x4
	SettingMaxFrameSize         SettingID = 0x5
	SettingMaxHeaderListSize    SettingID = 0x6
)

var settingName = map[SettingID]string{
	SettingHeaderTableSize:      "HEADER_TABLE_SIZE",
	SettingEnablePush:           "ENABLE_PUSH",
	SettingMaxConcurrentStreams: "MAX_CONCURRENT_STREAMS",
	SettingInitialWindowSize:    "INITIAL_WINDOW_SIZE",
	SettingMaxFrameSize:         "MAX_FRAME_SIZE",
	SettingMaxHeaderListSize:    "MAX_HEADER_LIST_SIZE",
}

func (s SettingID) String() string {
	if v, ok := settingName[s]; ok {
		return v
	}
	return fmt.Sprintf("UNKNOWN_SETTING_%d", uint16(s))
}

var (
	errInvalidHeaderFieldName  = errors.New("http2: invalid header field name")
	errInvalidHeaderFieldValue = errors.New("http2: invalid header field value")
)

// validWireHeaderFieldName reports whether v is a valid header field
// name (key). See httplex.ValidHeaderName for the base rules.
//
// Further, http2 says:
//   "Just as in HTTP/1.x, header field names are strings of ASCII
//   characters that are compared in a case-insensitive
//   fashion. However, header field names MUST be converted to
//   lowercase prior to their encoding in HTTP/2. "
func validWireHeaderFieldName(v string) bool {
	if len(v) == 0 {
		return false
	}
	for _, r := range v {
		if !httplex.IsTokenRune(r) {
			return false
		}
		if 'A' <= r && r <= 'Z' {
			return false
		}
	}
	return true
}

var httpCodeStringCommon = map[int]string{} // n -> strconv.Itoa(n)

func init() {
	for i := 100; i <= 999; i++ {
		if v := http.StatusText(i); v != "" {
			httpCodeStringCommon[i] = strconv.Itoa(i)
		}
	}
}

func httpCodeString(code int) string {
	if s, ok := httpCodeStringCommon[code]; ok {
		return s
	}
	return strconv.Itoa(code)
}

// from pkg io
type stringWriter interface {
	WriteString(s string) (n int, err error)
}

// A gate lets two goroutines coordinate their activities.
type gate chan struct{}

func (g gate) Done() { g <- struct{}{} }
func (g gate) Wait() { <-g }

// A closeWaiter is like a sync.WaitGroup but only goes 1 to 0 (open to closed).
type closeWaiter chan struct{}

// Init makes a closeWaiter usable.
// It exists because so a closeWaiter value can be placed inside a
// larger struct and have the Mutex and Cond's memory in the same
// allocation.
func (cw *closeWaiter) Init() {
	*cw = make(chan struct{})
}

// Close marks the closeWaiter as closed and unblocks any waiters.
func (cw closeWaiter) Close() {
	close(cw)
}

// Wait waits for the closeWaiter to become closed.
func (cw closeWaiter) Wait() {
	<-cw
}

// bufferedWriter is a buffered writer that writes to w.
// Its buffered writer is lazily allocated as needed, to minimize
// idle memory usage with many connections.
type bufferedWriter struct {
	w  io.Writer     // immutable
	bw *bufio.Writer // non-nil when data is buffered
}

func newBufferedWriter(w io.Writer) *bufferedWriter {
	return &bufferedWriter{w: w}
}

var bufWriterPool = sync.Pool{
	New: func() interface{} {
		// TODO: pick something better? this is a bit under
		// (3 x typical 1500 byte MTU) at least.
		return bufio.NewWriterSize(nil, 4<<10)
	},
}

func (w *bufferedWriter) Write(p []byte) (n int, err error) {
	if w.bw == nil {
		bw := bufWriterPool.Get().(*bufio.Writer)
		bw.Reset(w.w)
		w.bw = bw
	}
	return w.bw.Write(p)
}

func (w *bufferedWriter) Flush() error {
	bw := w.bw
	if bw == nil {
		return nil
	}
	err := bw.Flush()
	bw.Reset(nil)
	bufWriterPool.Put(bw)
	w.bw = nil
	return err
}

func mustUint31(v int32) uint32 {
	if v < 0 || v > 2147483647 {
		panic("out of range")
	}
	return uint32(v)
}

// bodyAllowedForStatus reports whether a given response status code
// permits a body. See RFC 2616, section 4.4.
func bodyAllowedForStatus(status int) bool {
	switch {
	case status >= 100 && status <= 199:
		return false
	case status == 204:
		return false
	case status == 304:
		return false
	}
	return true
}

type httpError struct {
	msg     string
	timeout bool
}

func (e *httpError) Error() string   { return e.msg }
func (e *httpError) Timeout() bool   { return e.timeout }
func (e *httpError) Temporary() bool { return true }

var errTimeout error = &httpError{msg: "http2: timeout awaiting response headers", timeout: true}

type connectionStater interface {
	ConnectionState() tls.ConnectionState
}

var sorterPool = sync.Pool{New: func() interface{} { return new(sorter) }}

type sorter struct {
	v []string // owned by sorter
}

func (s *sorter) Len() int           { return len(s.v) }
func (s *sorter) Swap(i, j int)      { s.v[i], s.v[j] = s.v[j], s.v[i] }
func (s *sorter) Less(i, j int) bool { return s.v[i] < s.v[j] }

// Keys returns the sorted keys of h.
//
// The returned slice is only valid until s used again or returned to
// its pool.
func (s *sorter) Keys(h http.Header) []string {
	keys := s.v[:0]
	for k := range h {
		keys = append(keys, k)
	}
	s.v = keys
	sort.Sort(s)
	return keys
}

func (s *sorter) SortStrings(ss []string) {
	// Our sorter works on s.v, which sorter owners, so
	// stash it away while we sort the user's buffer.
	save := s.v
	s.v = ss
	sort.Sort(s)
	s.v = save
}
