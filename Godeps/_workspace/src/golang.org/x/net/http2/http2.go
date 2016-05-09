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
	"strconv"
	"strings"
	"sync"
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

// validHeaderFieldName reports whether v is a valid header field name (key).
//  RFC 7230 says:
//   header-field   = field-name ":" OWS field-value OWS
//   field-name     = token
//   tchar = "!" / "#" / "$" / "%" / "&" / "'" / "*" / "+" / "-" / "." /
//           "^" / "_" / "
// Further, http2 says:
//   "Just as in HTTP/1.x, header field names are strings of ASCII
//   characters that are compared in a case-insensitive
//   fashion. However, header field names MUST be converted to
//   lowercase prior to their encoding in HTTP/2. "
func validHeaderFieldName(v string) bool {
	if len(v) == 0 {
		return false
	}
	for _, r := range v {
		if int(r) >= len(isTokenTable) || ('A' <= r && r <= 'Z') {
			return false
		}
		if !isTokenTable[byte(r)] {
			return false
		}
	}
	return true
}

// validHeaderFieldValue reports whether v is a valid header field value.
//
// RFC 7230 says:
//  field-value    = *( field-content / obs-fold )
//  obj-fold       =  N/A to http2, and deprecated
//  field-content  = field-vchar [ 1*( SP / HTAB ) field-vchar ]
//  field-vchar    = VCHAR / obs-text
//  obs-text       = %x80-FF
//  VCHAR          = "any visible [USASCII] character"
//
// http2 further says: "Similarly, HTTP/2 allows header field values
// that are not valid. While most of the values that can be encoded
// will not alter header field parsing, carriage return (CR, ASCII
// 0xd), line feed (LF, ASCII 0xa), and the zero character (NUL, ASCII
// 0x0) might be exploited by an attacker if they are translated
// verbatim. Any request or response that contains a character not
// permitted in a header field value MUST be treated as malformed
// (Section 8.1.2.6). Valid characters are defined by the
// field-content ABNF rule in Section 3.2 of [RFC7230]."
//
// This function does not (yet?) properly handle the rejection of
// strings that begin or end with SP or HTAB.
func validHeaderFieldValue(v string) bool {
	for i := 0; i < len(v); i++ {
		if b := v[i]; b < ' ' && b != '\t' || b == 0x7f {
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
// permits a body. See RFC2616, section 4.4.
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

var isTokenTable = [127]bool{
	'!':  true,
	'#':  true,
	'$':  true,
	'%':  true,
	'&':  true,
	'\'': true,
	'*':  true,
	'+':  true,
	'-':  true,
	'.':  true,
	'0':  true,
	'1':  true,
	'2':  true,
	'3':  true,
	'4':  true,
	'5':  true,
	'6':  true,
	'7':  true,
	'8':  true,
	'9':  true,
	'A':  true,
	'B':  true,
	'C':  true,
	'D':  true,
	'E':  true,
	'F':  true,
	'G':  true,
	'H':  true,
	'I':  true,
	'J':  true,
	'K':  true,
	'L':  true,
	'M':  true,
	'N':  true,
	'O':  true,
	'P':  true,
	'Q':  true,
	'R':  true,
	'S':  true,
	'T':  true,
	'U':  true,
	'W':  true,
	'V':  true,
	'X':  true,
	'Y':  true,
	'Z':  true,
	'^':  true,
	'_':  true,
	'`':  true,
	'a':  true,
	'b':  true,
	'c':  true,
	'd':  true,
	'e':  true,
	'f':  true,
	'g':  true,
	'h':  true,
	'i':  true,
	'j':  true,
	'k':  true,
	'l':  true,
	'm':  true,
	'n':  true,
	'o':  true,
	'p':  true,
	'q':  true,
	'r':  true,
	's':  true,
	't':  true,
	'u':  true,
	'v':  true,
	'w':  true,
	'x':  true,
	'y':  true,
	'z':  true,
	'|':  true,
	'~':  true,
}

type connectionStater interface {
	ConnectionState() tls.ConnectionState
}
